"""Correctness test for the ThunderMittens causal attention kernel.

Oracle: mx.fast.scaled_dot_product_attention with scale=1/sqrt(D) and an additive
upper-triangular -inf mask (the kernel pre-scales q by (1/sqrt(D))*log2(e) + exp2, i.e.
softmax with scale 1/sqrt(D)).

Run from kernels/:  python -m pytest attn_causal/correctness/test_attn_causal.py -v
"""

import math

import mlx.core as mx
import pytest

from tk import attn_causal

# (B, H, N, D); N a multiple of 8.
SHAPES = [(1, 2, 256, 64), (2, 4, 512, 64), (1, 2, 256, 128), (2, 2, 128, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_attn_causal_matches_sdpa(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)

    got = attn_causal(q, k, v)
    # additive causal mask: -inf where key index > query index
    rows = mx.arange(N)[:, None]
    cols = mx.arange(N)[None, :]
    mask = mx.where(cols > rows, -mx.inf, 0.0).astype(mx.float32)
    exp = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(D), mask=mask)
    mx.eval(got, exp)

    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()
    assert mx.allclose(got, exp, atol=4e-2, rtol=4e-2), f"max diff: {diff}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_attn_causal_matches_sdpa(shp)
        print("ok", shp)
