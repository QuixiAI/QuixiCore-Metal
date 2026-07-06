"""Benchmark perf kernels against MLX baselines.

Run from the repo root with:
    PYTHONPATH=bindings/python python perf/harness/time_perf.py
"""

import math
import time

import mlx.core as mx

from tk import matmul_custom, gemm_staged, attn_fwd, attn_multiwarp


def bench(fn, itt=50):
    for _ in range(itt):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(itt):
        mx.eval(fn())
    return 1e3 * (time.perf_counter() - t) / itt  # ms/iter


print("=== GEMM (bf16, square N=K=M), GFLOP/s ===")
for S in [256, 512, 1024, 2048]:
    x = mx.random.uniform(shape=(S, S)).astype(mx.bfloat16)
    y = mx.random.uniform(shape=(S, S)).astype(mx.bfloat16)
    mx.metal.clear_cache()
    gf = 2.0 * S * S * S / 1e9
    naive = gf / (bench(lambda: matmul_custom(x, y)) / 1e3)
    staged = gf / (bench(lambda: gemm_staged(x, y)) / 1e3)
    mlx = gf / (bench(lambda: x @ y) / 1e3)
    print(f"  {S:5d}: matmul_custom {naive:7.0f}  gemm_staged {staged:7.0f}  mlx {mlx:8.0f}")

print("=== Attention fwd (bf16), GFLOP/s — incl. long context ===")
for (B, H, N, D) in [(8, 8, 512, 64), (8, 8, 1024, 64), (8, 8, 512, 128),
                     (2, 8, 2048, 64), (1, 8, 4096, 64)]:
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    mx.metal.clear_cache()
    gf = (4.0 * B * H * N * N * D) / 1e9  # QK^T + AV
    warp1 = gf / (bench(lambda: attn_fwd(q, k, v)) / 1e3)
    warpN = gf / (bench(lambda: attn_multiwarp(q, k, v)) / 1e3)
    sdpa = gf / (bench(lambda: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(D), mask=None)) / 1e3)
    print(f"  (B{B} H{H} N{N} D{D}): attn_fwd {warp1:7.0f}  attn_multiwarp {warpN:7.0f}  sdpa {sdpa:8.0f}")

# Perf-tuning finding: the multi-simdgroup shared-staging kernels (gemm_staged, attn_multiwarp)
# are CORRECT and competitive but do NOT beat the single-simdgroup kernels (matmul_custom,
# attn_fwd) on Apple GPUs. A bigger 4-warp BM=128 GEMM tile was -20..26% (occupancy); 2 vs 4
# warps for attention were equivalent (~5% behind attn_fwd). Root cause: Metal has no async
# global->shared copy (cp.async/TMA) to overlap staging with compute the way the H100 kernels
# do, and these shapes are compute/cache-bound, so reducing global traffic via sharing doesn't
# pay. The simpler kernels are near-optimal here.
