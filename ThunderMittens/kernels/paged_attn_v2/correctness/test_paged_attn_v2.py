"""Correctness test for long-context paged attention v2 (partition/reduce).

The partitioned result must equal a single full-softmax oracle for every
partition_size (which forces 1..N partitions), across MHA/GQA/MQA.

Run from kernels/:  python -m pytest paged_attn_v2/correctness/test_paged_attn_v2.py -v
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import paged_attention_v2, kv_cache_scatter_fp8, paged_attention_v2_fp8
from tk.quant import _e4m3_decode_arr, _e5m2_decode_arr

_MX = {"float32": mx.float32, "bfloat16": mx.bfloat16}


def _ref(q, kc, vc, bt, cl, scale, window=0):
    B, H, D = q.shape
    H_KV = kc.shape[2]
    group = H // H_KV
    bs = kc.shape[1]
    out = np.zeros_like(q, np.float32)
    for b in range(B):
        ctx = int(cl[b])
        t0 = max(0, ctx - window) if window > 0 else 0
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(t0, ctx):
                blk = bt[b, t // bs]
                slot = t % bs
                sc.append(float(np.dot(q[b, h], kc[blk, slot, kvh]) * scale))
                vs.append(vc[blk, slot, kvh])
            if not sc:
                continue
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-5), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize("partition_size", [4, 8, 16])  # block_size=4 -> 4,2,1 partitions
def test_paged_attention_v2(dtype, atol, D, H, H_KV, partition_size):
    rng = np.random.default_rng(20 + D + H + H_KV + partition_size)
    B, num_blocks, block_size = 2, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)  # width 4 -> max_ctx 16
    cl = np.array([10, 16], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)

    md = _MX[dtype]
    got = paged_attention_v2(
        mx.array(q).astype(md), mx.array(kc).astype(md), mx.array(vc).astype(md),
        mx.array(bt), mx.array(cl), scale=0.0, partition_size=partition_size)
    mx.eval(got)

    ref = _ref(q, kc, vc, bt, cl, scale)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=atol, rtol=2e-3)


@pytest.mark.parametrize("fmt,qmax,decode",
                         [("e4m3", 448.0, _e4m3_decode_arr), ("e5m2", 57344.0, _e5m2_decode_arr)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(4, 2), (4, 1)])
@pytest.mark.parametrize("partition_size", [4, 16])  # 4 or 1 partitions
def test_paged_attention_v2_fp8(fmt, qmax, decode, D, H, H_KV, partition_size):
    # Long-context fp8 path: scatter into a uint8 cache, then partition/reduce dequantizing on
    # read. Oracle = the same full softmax over the dequantized codes.
    rng = np.random.default_rng(90 + D + H + H_KV + partition_size + len(fmt))
    B, num_blocks, block_size = 2, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    k_scale = float(np.abs(K).max() / qmax)
    v_scale = float(np.abs(V).max() / qmax)
    scale = 1.0 / math.sqrt(D)
    slot = np.arange(total, dtype=np.int64)

    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(slot), num_blocks, block_size, k_scale, v_scale, fmt=fmt)
    got = paged_attention_v2_fp8(
        mx.array(q).astype(mx.bfloat16), kc, vc, mx.array(bt), mx.array(cl),
        k_scale, v_scale, scale=0.0, partition_size=partition_size, fmt=fmt)
    mx.eval(kc, vc, got)

    kc_deq = decode(np.array(kc)) * k_scale
    vc_deq = decode(np.array(vc)) * v_scale
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _ref(q_bf, kc_deq, vc_deq, bt, cl, scale)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=2e-3)


@pytest.mark.parametrize("window", [1, 3, 16, 63, 64, 640])
@pytest.mark.parametrize("partition_size", [16, 32])
def test_paged_attention_v2_window(window, partition_size):
    # partition_size chosen so whole partitions fall outside the window for small W.
    rng = np.random.default_rng(120 + window + partition_size)
    B, H, H_KV, D = 2, 4, 2, 64
    num_blocks, block_size, ctx = 8, 16, 64
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.arange(B * (num_blocks // B), dtype=np.int32).reshape(B, num_blocks // B)
    cl = np.array([ctx, ctx], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)
    got = paged_attention_v2(mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
                             mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl),
                             scale=0.0, partition_size=partition_size, window=window)
    mx.eval(got)
    ref = _ref(q, kc, vc, bt, cl, scale, window)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=3e-3)
    if window >= ctx:
        full = paged_attention_v2(mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
                                  mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl),
                                  scale=0.0, partition_size=partition_size, window=0)
        mx.eval(full)
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)),
                                      np.array(full.astype(mx.float32)))


@pytest.mark.parametrize("window", [1, 16, 63, 640])
def test_paged_attention_v2_fp8_window(window):
    rng = np.random.default_rng(140 + window)
    B, H, H_KV, D = 2, 4, 2, 64
    num_blocks, block_size, ctx = 8, 16, 64
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.arange(B * (num_blocks // B), dtype=np.int32).reshape(B, num_blocks // B)
    cl = np.array([ctx, ctx], dtype=np.int32)
    k_scale = float(np.abs(K).max() / 448.0)
    v_scale = float(np.abs(V).max() / 448.0)
    scale = 1.0 / math.sqrt(D)
    kc, vc = kv_cache_scatter_fp8(
        mx.array(K).astype(mx.bfloat16), mx.array(V).astype(mx.bfloat16),
        mx.array(np.arange(total, dtype=np.int64)), num_blocks, block_size, k_scale, v_scale)
    got = paged_attention_v2_fp8(mx.array(q).astype(mx.bfloat16), kc, vc, mx.array(bt), mx.array(cl),
                                 k_scale, v_scale, scale=0.0, partition_size=32, window=window)
    mx.eval(kc, vc, got)
    kc_deq = _e4m3_decode_arr(np.array(kc)) * k_scale
    vc_deq = _e4m3_decode_arr(np.array(vc)) * v_scale
    q_bf = np.array(mx.array(q).astype(mx.bfloat16).astype(mx.float32))
    ref = _ref(q_bf, kc_deq, vc_deq, bt, cl, scale, window)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=2e-2, rtol=3e-3)


if __name__ == "__main__":
    for ps in (4, 8, 16):
        test_paged_attention_v2("float32", 3e-5, 64, 4, 2, ps)
        print("ok", ps)
