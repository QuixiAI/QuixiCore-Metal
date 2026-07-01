"""Correctness tests for the DeepSeek MLA kernels.

P1 mla_q_norm_rope: optional RMSNorm over the full head dim (mode 0/1/2) + GPT-J interleaved
RoPE on the last rope_dim dims. Oracle mirrors vLLM's rmsnorm_no_weight + apply_rope_gptj_last_k
(test_fused_deepseek_v4_qnorm_rope_kv_insert.py), rounded through bf16.

Run from kernels/:  python -m pytest mla/correctness/test_mla.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import mla_q_norm_rope, mla_kv_insert


def make_cos_sin(P, rope_dim, base=10000.0):
    inv = base ** (-(np.arange(0, rope_dim, 2).astype(np.float32) / rope_dim))  # (rope/2,)
    ang = np.arange(P)[:, None].astype(np.float32) * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def q_norm_rope_ref(q, cos, sin, positions, nope, rope, norm_mode, eps, w):
    """(T,H,Dh) -> RMSNorm(full head) if mode + interleaved RoPE on last `rope` dims."""
    T, H, Dh = q.shape
    o = q.astype(np.float32).copy()
    if norm_mode >= 1:
        o = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps)
        if norm_mode == 2:
            o = o * w[None, None, :]
    out = o.copy()
    for t in range(T):
        p = int(positions[t])
        c, s = cos[p], sin[p]
        xe = o[t, :, nope:][..., 0::2].copy()   # copy before writing (avoid aliasing)
        xo = o[t, :, nope:][..., 1::2].copy()
        out[t, :, nope:][..., 0::2] = xe * c - xo * s
        out[t, :, nope:][..., 1::2] = xe * s + xo * c
    return out


@pytest.mark.parametrize("norm_mode", [0, 1, 2])
@pytest.mark.parametrize("T,H,nope,rope", [(3, 16, 128, 64), (2, 8, 448, 64), (2, 4, 192, 64)])
def test_mla_q_norm_rope(norm_mode, T, H, nope, rope):
    Dh = nope + rope
    rng = np.random.default_rng(T + H + nope + norm_mode)
    q = (0.5 * rng.standard_normal((T, H, Dh))).astype(np.float32)
    w = (0.5 + 0.1 * rng.standard_normal(Dh)).astype(np.float32)
    cos, sin = make_cos_sin(64, rope)
    positions = np.arange(T, dtype=np.int32)

    qm = mx.array(q).astype(mx.bfloat16)
    cm = mx.array(cos).astype(mx.bfloat16)
    sm = mx.array(sin).astype(mx.bfloat16)
    wm = mx.array(w).astype(mx.bfloat16)
    got = mla_q_norm_rope(qm, cm, sm, mx.array(positions), H, nope, rope,
                          norm_mode=norm_mode, eps=1e-6,
                          norm_weight=(wm if norm_mode == 2 else None))
    mx.eval(got)

    qb = np.array(qm.astype(mx.float32))
    wb = np.array(wm.astype(mx.float32))
    ref = q_norm_rope_ref(qb, cos, sin, positions, nope, rope, norm_mode, 1e-6, wb)
    assert np.max(np.abs(np.array(got.astype(mx.float32)) - ref)) < 3e-2


def _rope_interleaved_row(pe, cos_p, sin_p):
    xe, xo = pe[0::2].copy(), pe[1::2].copy()
    out = np.empty_like(pe)
    out[0::2] = xe * cos_p - xo * sin_p
    out[1::2] = xe * sin_p + xo * cos_p
    return out


@pytest.mark.parametrize("norm_mode", [0, 2])
@pytest.mark.parametrize("latent", [512, 128])
def test_mla_kv_insert(norm_mode, latent):
    rope = 64
    T, nb, bs = 5, 4, 4
    W = latent + rope
    rng = np.random.default_rng(latent + norm_mode)
    kv_c = (0.3 * rng.standard_normal((T, latent))).astype(np.float32)
    k_pe = (0.3 * rng.standard_normal((T, rope))).astype(np.float32)
    w = (0.5 + 0.1 * rng.standard_normal(latent)).astype(np.float32)
    cos, sin = make_cos_sin(64, rope)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)      # token 2 skipped (slot < 0)
    cache0 = (0.1 * rng.standard_normal((nb, bs, W))).astype(np.float32)

    kvm, pem = mx.array(kv_c).astype(mx.bfloat16), mx.array(k_pe).astype(mx.bfloat16)
    cm, sm = mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16)
    wm, c0 = mx.array(w).astype(mx.bfloat16), mx.array(cache0).astype(mx.bfloat16)
    got = mla_kv_insert(kvm, pem, cm, sm, mx.array(positions), mx.array(slot), c0,
                        rope_dim=rope, norm_mode=norm_mode,
                        norm_weight=(wm if norm_mode == 2 else None))
    mx.eval(got)

    ref = np.array(c0.astype(mx.float32)).copy()
    kvb, peb, wb = np.array(kvm.astype(mx.float32)), np.array(pem.astype(mx.float32)), np.array(wm.astype(mx.float32))
    for t in range(T):
        s = int(slot[t])
        if s < 0:
            continue
        blk, off = s // bs, s % bs
        lat = kvb[t].copy()
        if norm_mode >= 1:
            lat = lat / np.sqrt((lat * lat).mean() + 1e-6)
            if norm_mode == 2:
                lat = lat * wb
        ref[blk, off, :latent] = lat
        ref[blk, off, latent:] = _rope_interleaved_row(peb[t], cos[positions[t]], sin[positions[t]])
    assert np.max(np.abs(np.array(got.astype(mx.float32)) - ref)) < 3e-2


if __name__ == "__main__":
    for m in (0, 1, 2):
        test_mla_q_norm_rope(m, 2, 8, 448, 64)
        print("ok", m)
