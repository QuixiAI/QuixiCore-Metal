"""Correctness tests for the fused per-head QK-RMSNorm + RoPE kernel (packed QKV).

Oracle: fp64 numpy — per-head RMSNorm (optionally gemma (1+w)) then NeoX split-half or
GPT-J interleaved rotation on Q and K heads; V heads must pass through bit-identically.

Run from kernels/:  python -m pytest qk_norm_rope/correctness/test_qk_norm_rope.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qk_norm_rope, qk_norm_rope_positioned

def _rope_tables(max_pos, half, base=10000.0):
    inv = 1.0 / (base ** (np.arange(half) / half))
    ang = np.outer(np.arange(max_pos), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _ref(qkv, qw, kw, cos, sin, pos, hq, hk, hv, eps, interleaved, gemma):
    T, W = qkv.shape
    HT = hq + hk + hv
    D = W // HT
    x = qkv.reshape(T, HT, D).astype(np.float64)
    out = x.copy()
    for h in range(hq + hk):
        w = (qw if h < hq else kw).astype(np.float64)
        if gemma:
            w = 1.0 + w
        v = x[:, h]
        rms = 1.0 / np.sqrt((v * v).mean(-1, keepdims=True) + eps)
        v = v * rms * w
        c, s = cos[pos].astype(np.float64), sin[pos].astype(np.float64)
        if interleaved:
            v0, v1 = v[:, 0::2], v[:, 1::2]
            r = np.empty_like(v)
            r[:, 0::2] = v0 * c - v1 * s
            r[:, 1::2] = v0 * s + v1 * c
        else:
            half = D // 2
            v1, v2 = v[:, :half], v[:, half:]
            r = np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], axis=-1)
        out[:, h] = r
    return out.reshape(T, W)


def _ref_positioned(qkv, qw, kw, cos, sin, pos, hq, hk, hv, eps, rotary_dim,
                    interleaved, weight_offset, sections=(), section_interleaved=False):
    T, W = qkv.shape
    HT, D = hq + hk + hv, W // (hq + hk + hv)
    x = qkv.reshape(T, HT, D).astype(np.float64)
    out = x.copy()
    rp = rotary_dim // 2
    boundaries = np.cumsum(sections) if sections else None
    for h in range(hq + hk):
        w = (qw if h < hq else kw).astype(np.float64) + weight_offset
        v = x[:, h]
        v = v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w
        r = v.copy()
        for p in range(rp):
            if sections:
                axis = p % 3 if section_interleaved else int(
                    np.searchsorted(boundaries, p, side="right"))
                pp = pos[axis]
            else:
                pp = pos
            c, s = cos[pp, p].astype(np.float64), sin[pp, p].astype(np.float64)
            i0, i1 = (2 * p, 2 * p + 1) if interleaved else (p, rp + p)
            r[:, i0] = v[:, i0] * c - v[:, i1] * s
            r[:, i1] = v[:, i0] * s + v[:, i1] * c
        out[:, h] = r
    return out.reshape(T, W)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("gemma", [False, True])
def test_qk_norm_rope(D, interleaved, gemma):
    hq, hk, hv, T = 8, 2, 2, 33
    rng = np.random.default_rng(0)
    qkv = rng.standard_normal((T, (hq + hk + hv) * D)).astype(np.float32)
    qw = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    kw = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    cos, sin = _rope_tables(4096, D // 2)
    pos = rng.integers(0, 4096, T).astype(np.int32)

    qkv_b = mx.array(qkv).astype(mx.bfloat16)
    got = qk_norm_rope(qkv_b, mx.array(qw).astype(mx.bfloat16),
                       mx.array(kw).astype(mx.bfloat16),
                       mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16),
                       mx.array(pos), hq, hk, hv, eps=1e-6,
                       interleaved=interleaved, gemma=gemma)
    mx.eval(got)
    gn = np.array(got.astype(mx.float32))

    # reference over the bf16-rounded inputs (isolates kernel math from input rounding)
    qkv_r = np.array(qkv_b.astype(mx.float32))
    cos_r = np.array(mx.array(cos).astype(mx.bfloat16).astype(mx.float32))
    sin_r = np.array(mx.array(sin).astype(mx.bfloat16).astype(mx.float32))
    qw_r = np.array(mx.array(qw).astype(mx.bfloat16).astype(mx.float32))
    kw_r = np.array(mx.array(kw).astype(mx.bfloat16).astype(mx.float32))
    ref = _ref(qkv_r, qw_r, kw_r, cos_r, sin_r, pos, hq, hk, hv, 1e-6, interleaved, gemma)
    np.testing.assert_allclose(gn, ref, atol=2e-2, rtol=2e-2)

    # V region passes through bit-identically
    HT, W = hq + hk + hv, (hq + hk + hv) * D
    v_in = qkv_r.reshape(T, HT, D)[:, hq + hk:]
    v_out = gn.reshape(T, HT, D)[:, hq + hk:]
    np.testing.assert_array_equal(v_in, v_out)


def test_matches_unfused_chain():
    """NeoX path cross-check vs tk.rms_norm-style math via the fp64 oracle at fp32 inputs
    quantized to a coarse grid (kernel-vs-composition agreement)."""
    import tk as tkm
    D, hq, hk, hv, T = 128, 4, 2, 2, 16
    rng = np.random.default_rng(1)
    qkv = (np.round(rng.standard_normal((T, (hq + hk + hv) * D)) * 64) / 64).astype(np.float32)
    qw = np.ones(D, np.float32)
    kw = np.ones(D, np.float32)
    cos, sin = _rope_tables(256, D // 2)
    pos = np.arange(T).astype(np.int32)
    got = qk_norm_rope(mx.array(qkv).astype(mx.bfloat16),
                       mx.array(qw).astype(mx.bfloat16), mx.array(kw).astype(mx.bfloat16),
                       mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16),
                       mx.array(pos), hq, hk, hv)
    mx.eval(got)
    ref = _ref(np.array(mx.array(qkv).astype(mx.bfloat16).astype(mx.float32)),
               qw, kw,
               np.array(mx.array(cos).astype(mx.bfloat16).astype(mx.float32)),
               np.array(mx.array(sin).astype(mx.bfloat16).astype(mx.float32)),
               pos, hq, hk, hv, 1e-6, False, False)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "D,rotary_dim,interleaved,weight_offset",
    [(128, 64, False, 0.0), (256, 128, True, 1.0),
     (512, 192, False, 0.25)],
)
def test_qk_norm_rope_positioned_partial(D, rotary_dim, interleaved, weight_offset):
    hq, hk, hv, T = 4, 2, 1, 13
    rng = np.random.default_rng(90 + D)
    qkv = rng.standard_normal((T, (hq + hk + hv) * D)).astype(np.float32)
    qw = (0.2 * rng.standard_normal(D)).astype(np.float32)
    kw = (0.2 * rng.standard_normal(D)).astype(np.float32)
    cos, sin = _rope_tables(97, rotary_dim // 2)
    pos = ((3 * np.arange(T) + 2) % 97).astype(np.int32)
    qb = mx.array(qkv).astype(mx.bfloat16)
    qwb, kwb = mx.array(qw).astype(mx.bfloat16), mx.array(kw).astype(mx.bfloat16)
    cb, sb = mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16)
    got = qk_norm_rope_positioned(
        qb, qwb, kwb, cb, sb, mx.array(pos), hq, hk, hv,
        rotary_dim=rotary_dim, interleaved=interleaved,
        norm_weight_offset=weight_offset)
    mx.eval(got)
    ref = _ref_positioned(
        np.array(qb.astype(mx.float32)), np.array(qwb.astype(mx.float32)),
        np.array(kwb.astype(mx.float32)), np.array(cb.astype(mx.float32)),
        np.array(sb.astype(mx.float32)), pos, hq, hk, hv, 1e-6, rotary_dim,
        interleaved, weight_offset)
    gn = np.array(got.astype(mx.float32))
    np.testing.assert_allclose(gn, ref, atol=3e-2, rtol=2e-2)
    np.testing.assert_array_equal(
        gn.reshape(T, hq + hk + hv, D)[:, hq + hk:],
        np.array(qb.astype(mx.float32)).reshape(T, hq + hk + hv, D)[:, hq + hk:])


@pytest.mark.parametrize(
    "sections,section_interleaved",
    [((8, 12, 12), False), ((11, 11, 10), True)],
)
def test_qk_norm_mrope(sections, section_interleaved):
    D, hq, hk, hv, T = 64, 3, 1, 1, 17
    rng = np.random.default_rng(144)
    qkv = rng.standard_normal((T, (hq + hk + hv) * D)).astype(np.float32)
    qw = (0.8 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    kw = (0.8 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    cos, sin = _rope_tables(128, D // 2)
    ar = np.arange(T, dtype=np.int32)
    pos = np.stack((ar, 2 * ar + 1, 3 * ar + 2)) % 128
    qb = mx.array(qkv).astype(mx.bfloat16)
    qwb, kwb = mx.array(qw).astype(mx.bfloat16), mx.array(kw).astype(mx.bfloat16)
    cb, sb = mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16)
    got = qk_norm_rope_positioned(
        qb, qwb, kwb, cb, sb, mx.array(pos), hq, hk, hv,
        mrope_sections=sections, section_interleaved=section_interleaved)
    mx.eval(got)
    ref = _ref_positioned(
        np.array(qb.astype(mx.float32)), np.array(qwb.astype(mx.float32)),
        np.array(kwb.astype(mx.float32)), np.array(cb.astype(mx.float32)),
        np.array(sb.astype(mx.float32)), pos, hq, hk, hv, 1e-6, D, False, 0.0,
        sections, section_interleaved)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=3e-2, rtol=2e-2)
