"""Correctness test for causal linear attention: out_i = sum_{j<=i} (q_i.k_j) v_j.

Reference: (tril(Q @ Kᵀ)) @ V. Validated on relative error.
Run from kernels/:  python -m pytest lin_attn_causal/correctness/test_lin_attn_causal.py -v
"""

import mlx.core as mx
import pytest

from tk import lin_attn_causal

SHAPES = [(1, 2, 64, 64), (2, 4, 128, 64), (1, 1, 256, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_lin_attn_causal(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    got = lin_attn_causal(q, k, v)
    scores = mx.matmul(q.astype(mx.float32), mx.swapaxes(k.astype(mx.float32), -1, -2))
    mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
    exp = mx.matmul(scores * mask, v.astype(mx.float32))
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_lin_attn_causal(shp)
        print("ok", shp)
