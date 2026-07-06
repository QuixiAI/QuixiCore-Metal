"""Correctness tests for the DeepSeek MLA kernels.

P1 mla_q_norm_rope: optional RMSNorm over the full head dim (mode 0/1/2) + GPT-J interleaved
RoPE on the last rope_dim dims. Oracle mirrors vLLM's rmsnorm_no_weight + apply_rope_gptj_last_k
(test_fused_deepseek_v4_qnorm_rope_kv_insert.py), rounded through bf16.

Run from kernels/:  python -m pytest mla/correctness/test_mla.py -v
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import (mla_q_norm_rope, mla_kv_insert, mla_kv_insert_fp8, mla_decode, mla_decode_fp8,
                mla_decode_fp8_sparse)
from tk.quant import _e4m3_decode_arr


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


def test_mla_kv_insert_fp8():
    # DeepSeek-V4 packed: NoPE(448) -> e4m3 fp8 + per-64 UE8M0 scale; RoPE(64) -> interleaved bf16.
    T, nb, bs = 5, 4, 4
    rng = np.random.default_rng(3)
    kv = (0.5 * rng.standard_normal((T, 512))).astype(np.float32)
    cos, sin = make_cos_sin(64, 64)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    d0 = np.zeros((nb, bs, 576), dtype=np.uint8)
    s0 = np.zeros((nb, bs, 8), dtype=np.uint8)

    kvm = mx.array(kv).astype(mx.bfloat16)
    cm, sm = mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16)
    data, scale = mla_kv_insert_fp8(kvm, cm, sm, mx.array(positions), mx.array(slot),
                                    mx.array(d0), mx.array(s0))
    mx.eval(data, scale)
    data, scale = np.array(data), np.array(scale)
    kvb = np.array(kvm.astype(mx.float32))

    for t in range(T):
        ss = int(slot[t])
        if ss < 0:
            continue
        blk, off = ss // bs, ss % bs
        nope, rope = kvb[t, :448], kvb[t, 448:]
        # UE8M0 scale bytes are deterministic — must match exactly.
        for b in range(7):
            amax = max(np.abs(nope[b * 64:(b + 1) * 64]).max(), 1e-4)
            exp = int(np.ceil(np.log2(amax / 448.0)))
            assert scale[blk, off, b] == min(max(exp + 127, 0), 255)
        assert scale[blk, off, 7] == 0
        # NoPE dequant within e4m3 tolerance (2^-4 relative + subnormal floor).
        deq = np.zeros(448, np.float32)
        for b in range(7):
            e = int(scale[blk, off, b])
            deq[b * 64:(b + 1) * 64] = _e4m3_decode_arr(data[blk, off, b * 64:(b + 1) * 64]) * (2.0 ** (e - 127))
        assert np.all(np.abs(deq - nope) <= 0.0625 * np.abs(nope) + 2.0 * (2.0 ** -8))
        # RoPE bytes -> bf16 -> f32, vs interleaved reference.
        rope_bf = np.frombuffer(data[blk, off, 448:576].tobytes(), dtype=np.uint16)
        rope_f = (rope_bf.astype(np.uint32) << 16).view(np.float32)
        ref = _rope_interleaved_row(rope, cos[positions[t]], sin[positions[t]])
        assert np.max(np.abs(rope_f - ref)) < 3e-2


@pytest.mark.parametrize("N", [8, 16])
def test_mla_decode_end_to_end(N):
    # Full absorb pipeline (absorb W_UK -> mla_decode -> up-proj W_UV) must equal the
    # algebraically-equal MHA path (up-project the latent, then SDPA over head-dim 192/128).
    Lkv, P, R, Vh = 512, 128, 64, 128
    B, nb, bs = 2, 8, 4
    rng = np.random.default_rng(N)
    W_UK = (0.02 * rng.standard_normal((Lkv, N, P))).astype(np.float32)
    W_UV = (0.02 * rng.standard_normal((Lkv, N, Vh))).astype(np.float32)
    q_nope = (0.3 * rng.standard_normal((B, N, P))).astype(np.float32)
    q_pe = (0.3 * rng.standard_normal((B, N, R))).astype(np.float32)
    total = nb * bs
    latent = (0.3 * rng.standard_normal((total, Lkv))).astype(np.float32)
    k_pe = (0.3 * rng.standard_normal((total, R))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    scale = 1.0 / math.sqrt(P + R)

    # absorb: ql_nope = q_nope @ W_UK_T ; query = [ql_nope | q_pe] (576)
    ql = np.einsum('bnp,npl->bnl', q_nope, np.transpose(W_UK, (1, 2, 0)))
    q576 = np.concatenate([ql, q_pe], axis=-1)
    cache = np.concatenate([latent, k_pe], axis=-1).reshape(nb, bs, 576)

    o = np.array(mla_decode(mx.array(q576).astype(mx.bfloat16), mx.array(cache).astype(mx.bfloat16),
                            mx.array(block_table), mx.array(context_lens), scale=0.0).astype(mx.float32))
    v_kernel = np.einsum('bnl,nlv->bnv', o, np.transpose(W_UV, (1, 0, 2)))

    out = np.zeros((B, N, Vh), np.float32)
    for b in range(B):
        for h in range(N):
            sc, vs = [], []
            for t in range(int(context_lens[b])):
                tok = block_table[b, t // bs] * bs + (t % bs)
                s = (q_nope[b, h] @ (latent[tok] @ W_UK[:, h, :]) + q_pe[b, h] @ k_pe[tok]) * scale
                sc.append(s)
                vs.append(latent[tok] @ W_UV[:, h, :])
            p = np.exp(np.array(sc) - np.max(sc))
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert np.max(np.abs(v_kernel - out)) < 2e-2


@pytest.mark.parametrize("N", [4, 8])
def test_mla_decode_fp8(N):
    # Produce a V4 UE8M0-packed cache with mla_kv_insert_fp8, then decode it (dense, dequant on
    # read, score/value over 512) and check vs a numpy V4-dense reference.
    B, nb, bs = 2, 8, 4
    total = nb * bs
    rng = np.random.default_rng(N)
    kv = (0.3 * rng.standard_normal((total, 512))).astype(np.float32)
    cos, sin = make_cos_sin(64, 64)
    positions = np.arange(total, dtype=np.int32)
    slot = np.arange(total, dtype=np.int64)
    d0 = np.zeros((nb, bs, 576), np.uint8)
    s0 = np.zeros((nb, bs, 8), np.uint8)
    data, scale = mla_kv_insert_fp8(mx.array(kv).astype(mx.bfloat16), mx.array(cos).astype(mx.bfloat16),
                                    mx.array(sin).astype(mx.bfloat16), mx.array(positions),
                                    mx.array(slot), mx.array(d0), mx.array(s0))
    mx.eval(data, scale)
    data, scale = np.array(data), np.array(scale)

    def deq_latent(tok):
        blk, off = tok // bs, tok % bs
        lat = np.zeros(512, np.float32)
        codes = data[blk, off, :448]
        for b in range(7):
            e = int(scale[blk, off, b])
            lat[b * 64:(b + 1) * 64] = _e4m3_decode_arr(codes[b * 64:(b + 1) * 64]) * (2.0 ** (e - 127))
        rb = np.frombuffer(data[blk, off, 448:576].tobytes(), np.uint16)
        lat[448:] = (rb.astype(np.uint32) << 16).view(np.float32)
        return lat

    q = (0.3 * rng.standard_normal((B, N, 512))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    scl = 1.0 / math.sqrt(512.0)
    got = np.array(mla_decode_fp8(mx.array(q).astype(mx.bfloat16), mx.array(data), mx.array(scale),
                                  mx.array(bt), mx.array(cl)).astype(mx.float32))
    qb = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = np.zeros((B, N, 512), np.float32)
    for b in range(B):
        for h in range(N):
            sc, vs = [], []
            for t in range(int(cl[b])):
                lat = deq_latent(bt[b, t // bs] * bs + (t % bs))
                sc.append(np.dot(qb[b, h], lat) * scl)
                vs.append(lat)
            p = np.exp(np.array(sc) - np.max(sc))
            p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert np.max(np.abs(got - ref)) < 1e-2


def _build_v4_cache(total, nb, bs, seed):
    rng = np.random.default_rng(seed)
    kv = (0.3 * rng.standard_normal((total, 512))).astype(np.float32)
    cos, sin = make_cos_sin(64, 64)
    positions = np.arange(total, dtype=np.int32)
    slot = np.arange(total, dtype=np.int64)
    data, scale = mla_kv_insert_fp8(mx.array(kv).astype(mx.bfloat16), mx.array(cos).astype(mx.bfloat16),
                                    mx.array(sin).astype(mx.bfloat16), mx.array(positions),
                                    mx.array(slot), mx.array(np.zeros((nb, bs, 576), np.uint8)),
                                    mx.array(np.zeros((nb, bs, 8), np.uint8)))
    mx.eval(data, scale)
    return np.array(data), np.array(scale)


def _v4_deq_latent(data, scale, tok, bs):
    blk, off = tok // bs, tok % bs
    lat = np.zeros(512, np.float32)
    codes = data[blk, off, :448]
    for b in range(7):
        e = int(scale[blk, off, b])
        lat[b * 64:(b + 1) * 64] = _e4m3_decode_arr(codes[b * 64:(b + 1) * 64]) * (2.0 ** (e - 127))
    rb = np.frombuffer(data[blk, off, 448:576].tobytes(), np.uint16)
    lat[448:] = (rb.astype(np.uint32) << 16).view(np.float32)
    return lat


def test_mla_decode_fp8_sparse():
    B, N, nb, bs = 2, 4, 8, 4
    data, scale = _build_v4_cache(nb * bs, nb, bs, 7)
    rng = np.random.default_rng(1)
    q = (0.3 * rng.standard_normal((B, N, 512))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    idx = np.full((B, 5), -1, np.int32)
    idx[0, :4] = [0, 2, 5, 9]
    idx[1, :5] = [1, 3, 6, 10, 15]
    lens = np.array([4, 5], dtype=np.int32)
    scl = 1.0 / math.sqrt(512.0)

    got = np.array(mla_decode_fp8_sparse(mx.array(q).astype(mx.bfloat16), mx.array(data),
                                         mx.array(scale), mx.array(bt), mx.array(idx),
                                         mx.array(lens)).astype(mx.float32))
    qb = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = np.zeros((B, N, 512), np.float32)
    for b in range(B):
        sel = idx[b, :int(lens[b])]
        for h in range(N):
            sc, vs = [], []
            for t in sel:
                lat = _v4_deq_latent(data, scale, bt[b, t // bs] * bs + (t % bs), bs)
                sc.append(np.dot(qb[b, h], lat) * scl)
                vs.append(lat)
            p = np.exp(np.array(sc) - np.max(sc))
            p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert np.max(np.abs(got - ref)) < 1e-2


if __name__ == "__main__":
    for m in (0, 1, 2):
        test_mla_q_norm_rope(m, 2, 8, 448, 64)
        print("ok", m)
