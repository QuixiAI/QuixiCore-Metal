"""KV-cache and paged-attention kernels ported from vLLM/vLLM-Metal references."""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import (
    beam_remap_block_table,
    beam_reorder_kv,
    kv_cache_copy_blocks,
    kv_cache_gather,
    kv_cache_scales,
    kv_cache_scatter,
    kv_cache_scatter_fp8,
    paged_attention,
    paged_attention_alibi,
    paged_attention_block_sparse,
    paged_attention_fp8,
    paged_attention_staged,
    paged_attention_xcache,
)
from tk.quant import _e4m3_decode_arr, _e5m2_decode_arr


def _mx_dtype(name):
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[name]


def _np(x):
    return np.array(x.astype(mx.float32))


def _cast_np(x, dtype):
    return _np(mx.array(x).astype(_mx_dtype(dtype))).astype(np.float32)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_scatter(dtype):
    rng = np.random.default_rng(0)
    T, H, D = 7, 2, 64
    block_size, num_blocks = 4, 3
    key = rng.normal(size=(T, H, D)).astype(np.float32)
    value = rng.normal(size=(T, H, D)).astype(np.float32)
    slots = np.array([0, 2, -1, 5, 8, 1, 7], dtype=np.int64)

    km = mx.array(key).astype(_mx_dtype(dtype))
    vm = mx.array(value).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_scatter(km, vm, mx.array(slots), num_blocks, block_size)
    mx.eval(got_k, got_v)

    ref_k = np.zeros((num_blocks, block_size, H, D), np.float32)
    ref_v = np.zeros_like(ref_k)
    key_r = _np(km).astype(np.float32)
    value_r = _np(vm).astype(np.float32)
    for t, slot in enumerate(slots):
        if slot < 0:
            continue
        ref_k[slot // block_size, slot % block_size] = key_r[t]
        ref_v[slot // block_size, slot % block_size] = value_r[t]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_gather(dtype):
    rng = np.random.default_rng(1)
    num_blocks, block_size, H, D = 3, 4, 2, 64
    key_cache = rng.normal(size=(num_blocks, block_size, H, D)).astype(np.float32)
    value_cache = rng.normal(size=(num_blocks, block_size, H, D)).astype(np.float32)
    block_table = np.array([[0, 1], [2, 0]], dtype=np.int32)
    cu_seq_lens = np.array([0, 5, 9], dtype=np.int32)
    num_tokens = int(cu_seq_lens[-1])

    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_gather(km, vm, mx.array(block_table), mx.array(cu_seq_lens), num_tokens)
    mx.eval(got_k, got_v)

    key_r = _np(km).astype(np.float32)
    value_r = _np(vm).astype(np.float32)
    ref_k = np.empty((num_tokens, H, D), np.float32)
    ref_v = np.empty_like(ref_k)
    for b in range(len(cu_seq_lens) - 1):
        for t in range(cu_seq_lens[b], cu_seq_lens[b + 1]):
            local = t - cu_seq_lens[b]
            block = block_table[b, local // block_size]
            slot = local % block_size
            ref_k[t] = key_r[block, slot]
            ref_v[t] = value_r[block, slot]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_copy_blocks(dtype):
    rng = np.random.default_rng(2)
    key_cache = rng.normal(size=(4, 3, 2, 64)).astype(np.float32)
    value_cache = rng.normal(size=(4, 3, 2, 64)).astype(np.float32)
    mapping = np.array([[0, 2], [1, 3]], dtype=np.int64)

    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_copy_blocks(km, vm, mx.array(mapping))
    mx.eval(got_k, got_v)

    ref_k = _np(km).astype(np.float32).copy()
    ref_v = _np(vm).astype(np.float32).copy()
    for src, dst in mapping:
        ref_k[dst] = ref_k[src]
        ref_v[dst] = ref_v[src]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_scales(dtype):
    rng = np.random.default_rng(3)
    key = rng.normal(size=(17, 2, 64)).astype(np.float32)
    value = rng.normal(size=(17, 2, 64)).astype(np.float32)

    km = mx.array(key).astype(_mx_dtype(dtype))
    vm = mx.array(value).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_scales(km, vm)
    mx.eval(got_k, got_v)

    ref_k = np.abs(_np(km).astype(np.float32)).max() / 240.0
    ref_v = np.abs(_np(vm).astype(np.float32)).max() / 240.0
    np.testing.assert_allclose(_np(got_k), np.array([ref_k], np.float32), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(_np(got_v), np.array([ref_v], np.float32), atol=1e-7, rtol=1e-6)


def _paged_ref(q, key_cache, value_cache, block_table, context_lens, scale):
    B, H, D = q.shape
    block_size = key_cache.shape[1]
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            scores = []
            vals = []
            for t in range(context_lens[b]):
                block = block_table[b, t // block_size]
                slot = t % block_size
                k = key_cache[block, slot, h]
                scores.append(float(np.dot(q[b, h], k) * scale))
                vals.append(value_cache[block, slot, h])
            if not scores:
                continue
            s = np.array(scores, np.float32)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vals, axis=0), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 2e-5), ("float16", 2e-3), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
def test_paged_attention(dtype, atol, D):
    rng = np.random.default_rng(4 + D)
    B, H = 2, 2
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention(qm, km, vm, mx.array(block_table), mx.array(context_lens), scale=0.0)
    mx.eval(got)

    ref = _paged_ref(
        _np(qm).astype(np.float32),
        _np(km).astype(np.float32),
        _np(vm).astype(np.float32),
        block_table,
        context_lens,
        scale,
    )
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=2e-3)


def _paged_ref_window(q, key_cache, value_cache, block_table, context_lens, scale, window):
    """GQA reference restricted to the `window` most recent keys [ctx-window, ctx)."""
    B, H, D = q.shape
    H_KV = key_cache.shape[2]
    group = H // H_KV
    block_size = key_cache.shape[1]
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(B):
        ctx = context_lens[b]
        t0 = max(0, ctx - window) if window > 0 else 0
        for h in range(H):
            kvh = h // group
            scores, vals = [], []
            for t in range(t0, ctx):
                block = block_table[b, t // block_size]
                slot = t % block_size
                scores.append(float(np.dot(q[b, h], key_cache[block, slot, kvh]) * scale))
                vals.append(value_cache[block, slot, kvh])
            s = np.array(scores, np.float32)
            p = np.exp(s - s.max()); p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vals, 0), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 2e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 4), (64, 8, 2), (128, 4, 4)])
@pytest.mark.parametrize("window", [1, 3, 16, 63, 64, 640])
def test_paged_attention_window(dtype, atol, D, H, H_KV, window):
    rng = np.random.default_rng(70 + D + H + window)
    B = 2
    block_size, ctx = 16, 64
    nblocks = (ctx + block_size - 1) // block_size
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H_KV, D))).astype(np.float32)
    bt = np.full((B, nblocks), -1, np.int32)
    blk = 1
    for b in range(B):
        for c in range(nblocks):
            bt[b, c] = blk; blk += 1
    cl = np.full((B,), ctx, np.int32)
    qm, km, vm = (mx.array(q).astype(_mx_dtype(dtype)), mx.array(kc).astype(_mx_dtype(dtype)),
                  mx.array(vc).astype(_mx_dtype(dtype)))
    got = paged_attention(qm, km, vm, mx.array(bt), mx.array(cl), scale=0.0, window=window)
    mx.eval(got)
    ref = _paged_ref_window(_np(qm).astype(np.float32), _np(km).astype(np.float32),
                            _np(vm).astype(np.float32), bt, cl, 1.0 / math.sqrt(D), window)
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=3e-3)
    if window >= ctx:  # window covering the whole context == the full (window=0) decode, bit-exact
        full = paged_attention(qm, km, vm, mx.array(bt), mx.array(cl), scale=0.0, window=0)
        mx.eval(full)
        np.testing.assert_array_equal(_np(got).astype(np.float32), _np(full).astype(np.float32))


def _paged_ref_gqa(q, key_cache, value_cache, block_table, context_lens, scale):
    """GQA/MQA reference: query head h reads KV head h // (H // H_KV)."""
    B, H, D = q.shape
    H_KV = key_cache.shape[2]
    group = H // H_KV
    block_size = key_cache.shape[1]
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            scores, vals = [], []
            for t in range(context_lens[b]):
                block = block_table[b, t // block_size]
                slot = t % block_size
                scores.append(float(np.dot(q[b, h], key_cache[block, slot, kvh]) * scale))
                vals.append(value_cache[block, slot, kvh])
            if not scores:
                continue
            s = np.array(scores, np.float32)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vals, axis=0), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 2e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(4, 2), (4, 1), (8, 2)])  # GQA group 2, MQA, GQA group 4
def test_paged_attention_gqa(dtype, atol, D, H, H_KV):
    rng = np.random.default_rng(11 + D + H + H_KV)
    B = 2
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention(qm, km, vm, mx.array(block_table), mx.array(context_lens), scale=0.0)
    mx.eval(got)

    ref = _paged_ref_gqa(
        _np(qm).astype(np.float32),
        _np(km).astype(np.float32),
        _np(vm).astype(np.float32),
        block_table,
        context_lens,
        scale,
    )
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=2e-3)


def _paged_ref_gqa_alibi(q, key_cache, value_cache, block_table, context_lens, scale, slopes,
                         window=0):
    B, H, D = q.shape
    H_KV = key_cache.shape[2]
    group = H // H_KV
    bs = key_cache.shape[1]
    out = np.zeros_like(q, np.float32)
    for b in range(B):
        ctx = int(context_lens[b])
        t0 = max(0, ctx - window) if window > 0 else 0
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(t0, ctx):
                blk = block_table[b, t // bs]
                slot = t % bs
                bias = slopes[h] * (t - ctx + 1)
                sc.append(float(np.dot(q[b, h], key_cache[blk, slot, kvh]) * scale + bias))
                vs.append(value_cache[blk, slot, kvh])
            if not sc:
                continue
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (8, 2), (4, 1)])  # MHA, GQA group 4, MQA
def test_paged_attention_alibi(dtype, atol, D, H, H_KV):
    rng = np.random.default_rng(80 + D + H + H_KV)
    B, num_blocks, block_size = 2, 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    slopes = (0.1 * (1.0 + np.arange(H))).astype(np.float32)   # distinct per-head slope
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention_alibi(qm, km, vm, mx.array(block_table), mx.array(context_lens),
                                mx.array(slopes), scale=0.0)
    mx.eval(got)

    ref = _paged_ref_gqa_alibi(
        _np(qm).astype(np.float32), _np(km).astype(np.float32), _np(vm).astype(np.float32),
        block_table, context_lens, scale, slopes)
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=2e-3)


def _to_xcache(dense_key, dense_value, x):
    """Dense (nb, bs, nkv, hd) K/V -> vLLM x-packed key (nb, nkv, hd/x, bs, x) + value (nb, nkv, hd, bs)."""
    nb, bs, nkv, hd = dense_key.shape
    xk = dense_key.transpose(0, 2, 3, 1).reshape(nb, nkv, hd // x, x, bs).transpose(0, 1, 2, 4, 3).copy()
    xv = dense_value.transpose(0, 2, 3, 1).copy()   # (nb, nkv, hd, bs)
    return xk, xv


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D,x", [(64, 8), (128, 8), (128, 4)])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (8, 2), (4, 1)])  # MHA, GQA group 4, MQA
def test_paged_attention_xcache(dtype, atol, D, x, H, H_KV):
    # Consuming a vLLM x-packed cache must equal the dense paged_attention on the same values.
    rng = np.random.default_rng(75 + D + H + H_KV + x)
    B, num_blocks, block_size = 2, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    dk = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    dv = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    xk, xv = _to_xcache(dk, dv, x)

    md = _mx_dtype(dtype)
    base = paged_attention(mx.array(q).astype(md), mx.array(dk).astype(md), mx.array(dv).astype(md),
                           mx.array(block_table), mx.array(context_lens))
    got = paged_attention_xcache(mx.array(q).astype(md), mx.array(xk).astype(md), mx.array(xv).astype(md),
                                 mx.array(block_table), mx.array(context_lens))
    mx.eval(base, got)
    np.testing.assert_array_equal(_np(got), _np(base))   # identical values, only memory order differs


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
def test_paged_attention_block_sparse(dtype, atol, D, H, H_KV):
    rng = np.random.default_rng(85 + D + H + H_KV)
    B, num_blocks, block_size = 2, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    block_mask = np.zeros((B, 4), dtype=np.int32)
    block_mask[:, ::2] = 1          # attend only to even logical blocks
    block_mask[:, 0] = 1            # always keep block 0 so every row has >=1 key
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention_block_sparse(qm, km, vm, mx.array(block_table), mx.array(context_lens),
                                       mx.array(block_mask), scale=0.0)
    mx.eval(got)

    # Oracle: same GQA softmax but skipping tokens in masked-out logical blocks.
    B_, H_, D_ = q.shape
    group = H_ // H_KV
    out = np.zeros_like(q, np.float32)
    for b in range(B_):
        for h in range(H_):
            kvh = h // group
            sc, vs = [], []
            for t in range(int(context_lens[b])):
                bc = t // block_size
                if block_mask[b, bc] == 0:
                    continue
                blk = block_table[b, bc]
                slot = t % block_size
                sc.append(float(np.dot(_np(qm)[b, h].astype(np.float32),
                                       _np(km)[blk, slot, kvh].astype(np.float32)) * scale))
                vs.append(_np(vm)[blk, slot, kvh].astype(np.float32))
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max()); p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    np.testing.assert_allclose(_np(got).astype(np.float32), out, atol=atol, rtol=2e-3)


@pytest.mark.parametrize("window", [3, 16, 640])
def test_paged_attention_alibi_window(window):
    # #12 composition: sliding window + per-head ALiBi bias together.
    rng = np.random.default_rng(160 + window)
    B, H, H_KV, D, bs, ctx = 2, 4, 2, 64, 16, 48
    nblocks = (ctx + bs - 1) // bs
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(B * nblocks + 1, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(B * nblocks + 1, bs, H_KV, D))).astype(np.float32)
    bt = np.full((B, nblocks), -1, np.int32)
    blk = 1
    for b in range(B):
        for c in range(nblocks):
            bt[b, c] = blk; blk += 1
    cl = np.full((B,), ctx, np.int32)
    slopes = (0.1 * (1.0 + np.arange(H))).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    got = paged_attention_alibi(mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
                                mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl),
                                mx.array(slopes), scale=0.0, window=window)
    mx.eval(got)
    ref = _paged_ref_gqa_alibi(q, kc, vc, bt, cl, scale, slopes, window)
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=2e-2, rtol=3e-3)


@pytest.mark.parametrize("window", [5, 32, 640])
def test_paged_attention_block_sparse_window(window):
    # #12 composition: sliding window + block-sparse skip together.
    rng = np.random.default_rng(170 + window)
    B, H, H_KV, D, bs, ctx = 2, 4, 2, 64, 16, 64
    nblocks = ctx // bs
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(B * nblocks, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(B * nblocks, bs, H_KV, D))).astype(np.float32)
    bt = np.arange(B * nblocks, dtype=np.int32).reshape(B, nblocks)
    cl = np.full((B,), ctx, np.int32)
    block_mask = np.ones((B, nblocks), np.int32)
    block_mask[:, 1] = 0            # skip logical block 1
    scale = 1.0 / math.sqrt(D)
    got = paged_attention_block_sparse(mx.array(q).astype(mx.bfloat16),
                                       mx.array(kc).astype(mx.bfloat16),
                                       mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl),
                                       mx.array(block_mask), scale=0.0, window=window)
    mx.eval(got)
    group = H // H_KV
    t0 = max(0, ctx - window)
    out = np.zeros_like(q, np.float32)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(t0, ctx):
                bc = t // bs
                if block_mask[b, bc] == 0:
                    continue
                blk = bt[b, bc]
                sc.append(float(np.dot(q[b, h], kc[blk, t % bs, kvh]) * scale))
                vs.append(vc[blk, t % bs, kvh])
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max()); p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    np.testing.assert_allclose(_np(got).astype(np.float32), out, atol=2e-2, rtol=3e-3)


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (8, 2), (4, 1)])  # MHA, GQA group 4, MQA
def test_paged_attention_staged(dtype, atol, D, H, H_KV):
    # KV-reuse staged decode must (a) match the softmax oracle and (b) be bit-for-bit identical
    # to the non-staged paged_attention (same math, only a different memory-traffic shape).
    rng = np.random.default_rng(60 + D + H + H_KV)
    B, num_blocks, block_size = 2, 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention_staged(qm, km, vm, mx.array(block_table), mx.array(context_lens), scale=0.0)
    base = paged_attention(qm, km, vm, mx.array(block_table), mx.array(context_lens), scale=0.0)
    mx.eval(got, base)

    # (b) exact equivalence to the reference kernel
    np.testing.assert_array_equal(_np(got), _np(base))
    # (a) matches the numpy oracle
    ref = _paged_ref_gqa(
        _np(qm).astype(np.float32), _np(km).astype(np.float32), _np(vm).astype(np.float32),
        block_table, context_lens, scale)
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=2e-3)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])  # MHA, MQA
def test_fp8_kv_roundtrip(D, H, H_KV):
    # Scatter K/V into an fp8 (uint8 e4m3) cache, then attend, dequantizing on read.
    rng = np.random.default_rng(30 + D + H + H_KV)
    B, num_blocks, block_size = 2, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    k_scale = float(np.abs(K).max() / 448.0)
    v_scale = float(np.abs(V).max() / 448.0)
    scale = 1.0 / math.sqrt(D)
    slot_mapping = np.arange(total, dtype=np.int64)

    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(slot_mapping), num_blocks, block_size, k_scale, v_scale)
    got = paged_attention_fp8(
        mx.array(q).astype(mx.bfloat16), kc, vc,
        mx.array(block_table), mx.array(context_lens), k_scale, v_scale, scale=0.0)
    mx.eval(kc, vc, got)

    # Reference: dequantize the stored codes and run the GQA softmax on the same values.
    kc_deq = _e4m3_decode_arr(np.array(kc)) * k_scale
    vc_deq = _e4m3_decode_arr(np.array(vc)) * v_scale
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _paged_ref_gqa(q_bf, kc_deq, vc_deq, block_table, context_lens, scale)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=2e-3)


@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 4), (64, 8, 2), (128, 4, 4)])
@pytest.mark.parametrize("window", [1, 16, 63, 640])
def test_paged_attention_fp8_window(D, H, H_KV, window):
    rng = np.random.default_rng(80 + D + H + window)
    B, num_blocks, block_size = 2, 8, 16
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    block_table = np.arange(num_blocks, dtype=np.int32).reshape(B, num_blocks // B)  # 4 blocks/seq
    ctx = 64
    context_lens = np.array([ctx, ctx], dtype=np.int32)
    k_scale = float(np.abs(K).max() / 448.0)
    v_scale = float(np.abs(V).max() / 448.0)
    scale = 1.0 / math.sqrt(D)
    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(np.arange(total, dtype=np.int64)), num_blocks, block_size, k_scale, v_scale)
    got = paged_attention_fp8(mx.array(q).astype(mx.bfloat16), kc, vc, mx.array(block_table),
                              mx.array(context_lens), k_scale, v_scale, scale=0.0, window=window)
    mx.eval(kc, vc, got)
    kc_deq = _e4m3_decode_arr(np.array(kc)) * k_scale
    vc_deq = _e4m3_decode_arr(np.array(vc)) * v_scale
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _paged_ref_window(q_bf, kc_deq, vc_deq, block_table, context_lens, scale, window)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=3e-3)
    if window >= ctx:
        full = paged_attention_fp8(mx.array(q).astype(mx.bfloat16), kc, vc, mx.array(block_table),
                                   mx.array(context_lens), k_scale, v_scale, scale=0.0, window=0)
        mx.eval(full)
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)),
                                      np.array(full.astype(mx.float32)))


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(4, 2), (6, 3)])
def test_fp8_kv_roundtrip_perhead(D, H, H_KV):
    # Per-head K/V scales (a distinct scale per kv-head), the production fp8-KV granularity.
    rng = np.random.default_rng(70 + D + H + H_KV)
    B, num_blocks, block_size = 2, 8, 4
    total = num_blocks * block_size
    # Give each head a different magnitude so a wrong (shared) scale would fail.
    head_gain = (1.0 + np.arange(H_KV)).astype(np.float32)[None, :, None]
    K = (0.2 * rng.normal(size=(total, H_KV, D)) * head_gain).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D)) * head_gain).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    k_scale = (np.abs(K).max(axis=(0, 2)) / 448.0).astype(np.float32)   # (H_KV,)
    v_scale = (np.abs(V).max(axis=(0, 2)) / 448.0).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    slot_mapping = np.arange(total, dtype=np.int64)

    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(slot_mapping), num_blocks, block_size, mx.array(k_scale), mx.array(v_scale))
    got = paged_attention_fp8(
        mx.array(q).astype(mx.bfloat16), kc, vc,
        mx.array(block_table), mx.array(context_lens), mx.array(k_scale), mx.array(v_scale), scale=0.0)
    mx.eval(kc, vc, got)

    kc_deq = _e4m3_decode_arr(np.array(kc)) * k_scale[None, None, :, None]
    vc_deq = _e4m3_decode_arr(np.array(vc)) * v_scale[None, None, :, None]
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _paged_ref_gqa(q_bf, kc_deq, vc_deq, block_table, context_lens, scale)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=2e-3)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
def test_fp8_kv_roundtrip_e5m2(D, H, H_KV):
    # e5m2 KV format (5-bit exponent, wider dynamic range) — scale uses QMAX=57344.
    rng = np.random.default_rng(50 + D + H + H_KV)
    B, num_blocks, block_size = 2, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    E5M2_MAX = 57344.0
    k_scale = float(np.abs(K).max() / E5M2_MAX)
    v_scale = float(np.abs(V).max() / E5M2_MAX)
    scale = 1.0 / math.sqrt(D)
    slot_mapping = np.arange(total, dtype=np.int64)

    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(slot_mapping), num_blocks, block_size, k_scale, v_scale, fmt="e5m2")
    got = paged_attention_fp8(
        mx.array(q).astype(mx.bfloat16), kc, vc,
        mx.array(block_table), mx.array(context_lens), k_scale, v_scale, scale=0.0, fmt="e5m2")
    mx.eval(kc, vc, got)

    # Reference dequantizes the stored e5m2 codes with the same per-tensor scale.
    kc_deq = _e5m2_decode_arr(np.array(kc)) * k_scale
    vc_deq = _e5m2_decode_arr(np.array(vc)) * v_scale
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _paged_ref_gqa(q_bf, kc_deq, vc_deq, block_table, context_lens, scale)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=2e-3)


@pytest.mark.parametrize("BM", [4, 8])
def test_beam_remap_block_table(BM):
    # zero-copy remap vs (a) the numpy row-gather and (b) the copy-path (beam_reorder_kv) on read.
    rng = np.random.default_rng(BM)
    B, H_KV, D, ctx = 2, 8, 128, 512
    block_size = 16
    max_blocks = ctx // block_size
    nbeams = B * BM
    num_blocks = nbeams * max_blocks
    kc = (0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks)
    parent = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
    seq = np.full(nbeams, ctx, np.int32)
    new_bt = np.array(beam_remap_block_table(mx.array(bt), mx.array(parent)))
    # (a) direct row gather
    ref = np.zeros_like(bt)
    for b in range(B):
        for k in range(BM):
            ref[b * BM + k] = bt[b * BM + int(parent[b, k])]
    np.testing.assert_array_equal(new_bt, ref)
    # (b) reading a child's blocks from the ORIGINAL cache via new_bt == reading from the
    # beam_reorder_kv'd cache via the original bt (both yield the parent's KV history).
    kc2, _ = beam_reorder_kv(mx.array(kc).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
                             mx.array(bt), mx.array(parent), mx.array(seq))
    kc2 = np.array(kc2.astype(mx.float32))
    kc_bf = np.array(mx.array(kc).astype(mx.bfloat16).astype(mx.float32))
    for row in range(nbeams):
        for c in range(max_blocks):
            b_read = kc_bf[new_bt[row, c]]          # original cache via remapped table
            a_read = kc2[bt[row, c]]                # reordered cache via original table
            np.testing.assert_allclose(a_read, b_read, atol=1e-2)


# ---------------------------------------------------------------------------
# Wave-10: fp8 KV gather + upconvert (the read path) and incremental scale update.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,decode", [(0, _e4m3_decode_arr), (1, _e5m2_decode_arr)])
@pytest.mark.parametrize("H_KV,D", [(2, 64), (4, 128)])
def test_kv_cache_gather_fp8_roundtrip(fmt, decode, H_KV, D):
    from tk import kv_cache_gather_fp8
    rng = np.random.default_rng(300 + fmt + H_KV + D)
    num_blocks, block_size = 4, 16
    total = num_blocks * block_size
    K = (0.3 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.3 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    QMAX = 448.0 if fmt == 0 else 57344.0
    ks = float(np.abs(K).max() / QMAX)
    vs = float(np.abs(V).max() / QMAX)
    # scatter into the paged fp8 cache (per-tensor scalar), token t -> slot t
    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(np.arange(total, dtype=np.int64)), num_blocks, block_size, ks, vs, fmt=fmt)
    mx.eval(kc, vc)
    # gather back: one sequence covering all tokens, identity block table, per-kv_head scales
    block_table = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
    cu = np.array([0, total], np.int32)
    ks_arr = np.full((H_KV,), ks, np.float32)
    vs_arr = np.full((H_KV,), vs, np.float32)
    ko, vo = kv_cache_gather_fp8(kc, vc, mx.array(block_table), mx.array(cu),
                                 mx.array(ks_arr), mx.array(vs_arr), total, fmt=fmt)
    mx.eval(ko, vo)
    # gather output == bf16(decode(code) * scale) — the exact value the cache holds
    k_ref = (decode(np.array(kc)) * ks).reshape(total, H_KV, D)
    v_ref = (decode(np.array(vc)) * vs).reshape(total, H_KV, D)
    np.testing.assert_allclose(np.array(ko.astype(mx.float32)), k_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.array(vo.astype(mx.float32)), v_ref, atol=2e-2, rtol=2e-2)
    # and within fp8 relative precision of the original K (e4m3 3-mantissa ~2^-3,
    # e5m2 2-mantissa ~2^-2 relative)
    rel = 2.0 ** -2 if fmt == 1 else 2.0 ** -3
    assert (np.abs(np.array(ko.astype(mx.float32)) - K) <= np.abs(K) * rel + ks + 1e-3).all()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_kv_cache_scale_update(dtype):
    from tk import kv_cache_scale_update
    md = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    rng = np.random.default_rng(301)
    k = (0.5 * rng.normal(size=(37, 4, 64))).astype(np.float32)
    v = (0.5 * rng.normal(size=(37, 4, 64))).astype(np.float32)
    old_k = np.array([0.1], np.float32)      # below new -> should raise
    old_v = np.array([10.0], np.float32)     # above new -> should hold
    nk, nv = kv_cache_scale_update(mx.array(k).astype(md), mx.array(v).astype(md),
                                   mx.array(old_k), mx.array(old_v))
    mx.eval(nk, nv)
    kb = np.array(mx.array(k).astype(md).astype(mx.float32))
    vb = np.array(mx.array(v).astype(md).astype(mx.float32))
    ref_k = max(float(old_k[0]), np.abs(kb).max() / 240.0)
    ref_v = max(float(old_v[0]), np.abs(vb).max() / 240.0)
    np.testing.assert_allclose(np.array(nk), [ref_k], rtol=1e-4)
    np.testing.assert_allclose(np.array(nv), [ref_v], rtol=1e-4)
    assert float(np.array(nv)[0]) == pytest.approx(10.0)   # old held (only raises)
