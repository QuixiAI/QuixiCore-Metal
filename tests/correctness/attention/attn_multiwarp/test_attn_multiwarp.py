"""Correctness test for multi-warp flash attention (non-causal).

Oracle: mx.fast.scaled_dot_product_attention with scale=1/sqrt(D) (the kernel pre-scales
q by (1/sqrt(D))*log2(e) + exp2). N must be a multiple of 32 (8 * NUM_WARPS).

Run from kernels/:  python -m pytest attn_multiwarp/correctness/test_attn_multiwarp.py -v
"""

import math

import mlx.core as mx
import pytest

from tk import attn_multiwarp

SHAPES = [(1, 2, 256, 64), (2, 4, 512, 64), (1, 2, 256, 128), (2, 2, 128, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_attn_multiwarp_matches_sdpa(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    got = attn_multiwarp(q, k, v)
    exp = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D), mask=None)
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()
    assert mx.allclose(got, exp, atol=4e-2, rtol=4e-2), f"max diff: {diff}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_attn_multiwarp_matches_sdpa(shp)
        print("ok", shp)
