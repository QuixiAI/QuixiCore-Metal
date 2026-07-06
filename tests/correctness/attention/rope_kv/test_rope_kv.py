"""Correctness test for the fused RoPE + paged-KV insert Metal kernel.

Rotates K (split-half / GPT-NeoX) and writes rotated K + (unrotated) V into the
paged cache at slot_mapping[token]; un-inserted slots pass through unchanged.

Run from kernels/:  python -m pytest rope_kv/correctness/test_rope_kv.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import rope_kv_insert, rope_kv_insert_norm, rope_q


def _cos_sin(P, D):
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _rope_half(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _bf(x):
    """Round a numpy array through bf16 (what the kernel sees on load)."""
    return np.array(mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))


def _round(x, md):
    return np.array(mx.array(x.astype(np.float32)).astype(md).astype(mx.float32))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H_KV", [1, 2])
def test_rope_kv_insert(dtype, D, H_KV):
    md = _MX[dtype]
    rng = np.random.default_rng(3 + D + H_KV)
    num_blocks, block_size = 4, 4
    num_tokens = 5
    P = num_blocks * block_size

    k = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    cos, sin = _cos_sin(P, D)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot_mapping = np.array([0, 5, -1, 6, 11], dtype=np.int64)  # token 2 is padding
    kc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)

    kc, vc = rope_kv_insert(
        mx.array(k).astype(md), mx.array(v).astype(md),
        mx.array(cos).astype(md), mx.array(sin).astype(md),
        mx.array(positions), mx.array(slot_mapping),
        mx.array(kc0).astype(md), mx.array(vc0).astype(md),
    )
    mx.eval(kc, vc)
    assert kc.dtype == md

    kb, vb, cb, sb = _round(k, md), _round(v, md), _round(cos, md), _round(sin, md)
    ref_k, ref_v = _round(kc0, md), _round(vc0, md)
    for t in range(num_tokens):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        blk, boff = slot // block_size, slot % block_size
        for h in range(H_KV):
            ref_k[blk, boff, h] = _rope_half(kb[t, h], cb[positions[t]], sb[positions[t]])
            ref_v[blk, boff, h] = vb[t, h]

    np.testing.assert_allclose(np.array(kc.astype(mx.float32)), ref_k, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.array(vc.astype(mx.float32)), ref_v, atol=2e-2, rtol=2e-2)


def _rmsnorm(k_row, w, eps, gemma):
    ms = (k_row * k_row).mean()
    weff = (1.0 + w) if gemma else w
    return k_row / np.sqrt(ms + eps) * weff


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gemma", [False, True])
def test_rope_kv_insert_norm(D, gemma):
    rng = np.random.default_rng(4 + D + int(gemma))
    num_blocks, block_size, num_tokens, H_KV = 4, 4, 5, 2
    P = num_blocks * block_size
    eps = 1e-5
    k = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    cos, sin = _cos_sin(P, D)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot_mapping = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    w = rng.normal(size=(D,)).astype(np.float32)
    kc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)

    kc, vc = rope_kv_insert_norm(
        mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16),
        mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16),
        mx.array(positions), mx.array(slot_mapping),
        mx.array(kc0).astype(mx.bfloat16), mx.array(vc0).astype(mx.bfloat16),
        mx.array(w).astype(mx.bfloat16), eps=eps, gemma=gemma)
    mx.eval(kc, vc)

    kb, vb, cb, sb, wb = _bf(k), _bf(v), _bf(cos), _bf(sin), _bf(w)
    ref_k, ref_v = _bf(kc0), _bf(vc0)
    for t in range(num_tokens):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        blk, boff = slot // block_size, slot % block_size
        for h in range(H_KV):
            kn = _rmsnorm(kb[t, h], wb, eps, gemma)
            ref_k[blk, boff, h] = _rope_half(kn, cb[positions[t]], sb[positions[t]])
            ref_v[blk, boff, h] = vb[t, h]

    np.testing.assert_allclose(np.array(kc.astype(mx.float32)), ref_k, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.array(vc.astype(mx.float32)), ref_v, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("do_norm,gemma", [(False, False), (True, False), (True, True)])
def test_rope_q(dtype, D, do_norm, gemma):
    md = _MX[dtype]
    rng = np.random.default_rng(30 + D)
    H, nt, P = 4, 5, 16
    q = (0.3 * rng.normal(size=(nt, H, D))).astype(np.float32)
    w = (0.5 + 0.1 * rng.normal(size=(D,))).astype(np.float32)
    cos, sin = _cos_sin(P, D)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    out = rope_q(mx.array(q).astype(md), mx.array(cos).astype(md), mx.array(sin).astype(md),
                 mx.array(positions), norm_weight=(mx.array(w).astype(md) if do_norm else None),
                 gemma=gemma)
    mx.eval(out)
    assert out.dtype == md

    qb, cb, sb, wb = _round(q, md), _round(cos, md), _round(sin, md), _round(w, md)
    ref = qb.copy()
    for t in range(nt):
        for h in range(H):
            x = qb[t, h].copy()
            if do_norm:
                x = x / np.sqrt((x * x).mean() + 1e-6)
                x = x * ((1.0 + wb) if gemma else wb)
            ref[t, h] = _rope_half(x, cb[positions[t]], sb[positions[t]])
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    for D in (64, 128):
        for H_KV in (1, 2):
            test_rope_kv_insert("bfloat16", D, H_KV)
            print("ok", D, H_KV)
