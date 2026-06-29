"""Correctness test for the ThunderMittens attn_fwd Metal kernel.

attn_fwd is a warp-level flash-attention forward (non-causal), bf16, D in {64,128}.
The kernel pre-scales Q by (1/sqrt(D)) * log2(e) and uses exp2, which is exactly a
softmax with scale 1/sqrt(D). Oracle: mx.fast.scaled_dot_product_attention with that
scale (NOT scale=1, which time_attn.py uses for perf only).
Run from the kernels/ directory:

    python -m pytest attn_fwd/correctness/test_attn.py -v
"""

import math

import mlx.core as mx
import pytest

from tk import attn_fwd

# (B, H, N, D); N must be a multiple of 8.
SHAPES = [(1, 2, 256, 64), (2, 4, 512, 64), (1, 2, 256, 128), (2, 2, 128, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_attn_fwd_matches_sdpa(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)

    got = attn_fwd(q, k, v)
    exp = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(D), mask=None
    )
    mx.eval(got, exp)

    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()
    assert mx.allclose(got, exp, atol=4e-2, rtol=4e-2), f"max diff: {diff}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_attn_fwd_matches_sdpa(shp)
        print(f"ok {shp}")
