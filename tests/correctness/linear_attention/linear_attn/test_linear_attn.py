"""Correctness test for non-causal linear attention: out = Q @ (K^T @ V).

KV sums over N keys, so output magnitudes are O(N*D) and bf16 absolute error scales
with them — validate on RELATIVE error. Run from kernels/:
    python -m pytest linear_attn/correctness/test_linear_attn.py -v
"""

import mlx.core as mx
import pytest

from tk import linear_attn

SHAPES = [(1, 2, 128, 64), (2, 4, 256, 64), (1, 1, 512, 64)]


@pytest.mark.parametrize("use_kernel", [True, False], ids=["kernel", "routed"])
@pytest.mark.parametrize("shape", SHAPES)
def test_linear_attn(shape, use_kernel):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    got = linear_attn(q, k, v, use_kernel=use_kernel)
    kv = mx.matmul(mx.swapaxes(k.astype(mx.float32), -1, -2), v.astype(mx.float32))
    exp = mx.matmul(q.astype(mx.float32), kv)
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_linear_attn(shp)
        print("ok", shp)
