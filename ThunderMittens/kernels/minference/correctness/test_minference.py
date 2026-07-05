"""Correctness tests for the MInference decode block-mask builder.

(a) exact int equality vs a numpy transcription of the marking rule; (b) end-to-end:
build a per-head mask, run paged_attention_block_sparse with it, compare against a dense
numpy attention restricted to exactly the kept blocks.

Run from kernels/:  python -m pytest minference/correctness/test_minference.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk


def _ref_mask(vert, slash, lens, max_blocks, bs, vtopk, stopk, last_n):
    B, H, _ = vert.shape
    out = np.zeros((B, H, max_blocks), np.int32)
    for b in range(B):
        ctx = int(lens[b])
        if ctx <= 0:
            continue
        q = ctx - 1
        nblocks = min((ctx + bs - 1) // bs, max_blocks)
        for h in range(H):
            for i in range(min(last_n, nblocks)):
                out[b, h, nblocks - 1 - i] = 1
            for c in vert[b, h, :vtopk]:
                if 0 <= c < ctx:
                    out[b, h, c // bs] = 1
            for o in slash[b, h, :stopk]:
                if 0 <= o <= q:
                    out[b, h, (q - o) // bs] = 1
    return out


def test_block_mask_exact():
    rng = np.random.default_rng(0)
    B, H, nnz_v, nnz_s, bs, max_blocks = 3, 4, 16, 12, 16, 32
    lens = np.array([500, 100, 17], np.int32)
    vert = rng.integers(-1, 512, (B, H, nnz_v)).astype(np.int32)
    slash = rng.integers(-1, 512, (B, H, nnz_s)).astype(np.int32)
    got = tk.minference_block_mask(mx.array(vert), mx.array(slash), mx.array(lens),
                                   max_blocks, bs, last_n_blocks=2)
    mx.eval(got)
    ref = _ref_mask(vert, slash, lens, max_blocks, bs, nnz_v, nnz_s, 2)
    np.testing.assert_array_equal(np.array(got), ref)


def test_block_mask_topk_caps():
    rng = np.random.default_rng(1)
    B, H, nnz = 2, 2, 24
    lens = np.array([640, 640], np.int32)
    vert = rng.integers(0, 640, (B, H, nnz)).astype(np.int32)
    slash = rng.integers(0, 640, (B, H, nnz)).astype(np.int32)
    got = tk.minference_block_mask(mx.array(vert), mx.array(slash), mx.array(lens),
                                   40, 16, vertical_topk=5, slash_topk=3, last_n_blocks=1)
    mx.eval(got)
    ref = _ref_mask(vert, slash, lens, 40, 16, 5, 3, 1)
    np.testing.assert_array_equal(np.array(got), ref)


def test_end_to_end_block_sparse_attention():
    """Per-head mask -> paged_attention_block_sparse == dense numpy attention over exactly
    the kept blocks (per head)."""
    rng = np.random.default_rng(2)
    B, Hq, Hkv, D, bs = 2, 4, 2, 64, 16
    ctx = 200
    max_blocks = (ctx + bs - 1) // bs + 2
    lens = np.full(B, ctx, np.int32)
    nblk = (ctx + bs - 1) // bs
    table = np.zeros((B, max_blocks), np.int32)
    ids = rng.permutation(B * max_blocks).astype(np.int32)
    for b in range(B):
        table[b] = ids[b * max_blocks:(b + 1) * max_blocks]
    kc = (0.3 * rng.standard_normal((B * max_blocks, bs, Hkv, D))).astype(np.float32)
    vc = (0.3 * rng.standard_normal((B * max_blocks, bs, Hkv, D))).astype(np.float32)
    q = (0.3 * rng.standard_normal((B, Hq, D))).astype(np.float32)
    vert = rng.integers(0, ctx, (B, Hq, 8)).astype(np.int32)
    slash = np.concatenate([np.zeros((B, Hq, 1), np.int32),          # diagonal itself
                            rng.integers(1, ctx, (B, Hq, 5)).astype(np.int32)], -1)
    mask = tk.minference_block_mask(mx.array(vert), mx.array(slash), mx.array(lens),
                                    max_blocks, bs, last_n_blocks=1)
    out = tk.paged_attention_block_sparse(mx.array(q), mx.array(kc), mx.array(vc),
                                          mx.array(table), mx.array(lens), mask)
    mx.eval(out)
    mn = np.array(mask)
    scale = 1.0 / np.sqrt(D)
    for b in range(B):
        for h in range(Hq):
            kv_h = h // (Hq // Hkv)
            keep_t = [t for t in range(ctx) if mn[b, h, t // bs] == 1]
            k_rows = np.stack([kc[table[b, t // bs], t % bs, kv_h] for t in keep_t])
            v_rows = np.stack([vc[table[b, t // bs], t % bs, kv_h] for t in keep_t])
            sc = (k_rows.astype(np.float64) @ q[b, h].astype(np.float64)) * scale
            w = np.exp(sc - sc.max())
            w /= w.sum()
            ref = w @ v_rows.astype(np.float64)
            np.testing.assert_allclose(np.array(out)[b, h], ref, atol=2e-3, rtol=2e-3)


def test_legacy_2d_mask_still_works():
    """Regression: the pre-existing per-batch 2-D mask path is unchanged."""
    rng = np.random.default_rng(3)
    B, Hq, Hkv, D, bs = 1, 2, 2, 64, 16
    ctx, max_blocks = 64, 6
    lens = np.full(B, ctx, np.int32)
    table = np.arange(B * max_blocks, dtype=np.int32).reshape(B, max_blocks)
    kc = (0.3 * rng.standard_normal((B * max_blocks, bs, Hkv, D))).astype(np.float32)
    vc = (0.3 * rng.standard_normal((B * max_blocks, bs, Hkv, D))).astype(np.float32)
    q = (0.3 * rng.standard_normal((B, Hq, D))).astype(np.float32)
    mask2d = np.array([[1, 0, 1, 1, 0, 0]], np.int32)
    out = tk.paged_attention_block_sparse(mx.array(q), mx.array(kc), mx.array(vc),
                                          mx.array(table), mx.array(lens), mx.array(mask2d))
    mx.eval(out)
    scale = 1.0 / np.sqrt(D)
    for h in range(Hq):
        keep_t = [t for t in range(ctx) if mask2d[0, t // bs] == 1]
        k_rows = np.stack([kc[table[0, t // bs], t % bs, h] for t in keep_t])
        v_rows = np.stack([vc[table[0, t // bs], t % bs, h] for t in keep_t])
        sc = (k_rows.astype(np.float64) @ q[0, h].astype(np.float64)) * scale
        w = np.exp(sc - sc.max())
        w /= w.sum()
        np.testing.assert_allclose(np.array(out)[0, h], w @ v_rows.astype(np.float64),
                                   atol=2e-3, rtol=2e-3)
