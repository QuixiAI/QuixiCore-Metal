"""Correctness test for Mamba-2 / SSD forward.

Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j, cumlog = cumsum(log a).
Reference: ((C@Bᵀ) ⊙ exp(cumlog_i - cumlog_j) ⊙ tril) @ X. Validated on relative error.
Run from kernels/:  python -m pytest mamba2/correctness/test_mamba2.py -v
"""

import mlx.core as mx
import pytest

from tk import mamba2

SHAPES = [(1, 2, 64, 64), (2, 2, 128, 64), (1, 1, 256, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_mamba2(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    C = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    Bm = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    X = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    a = mx.sigmoid(mx.random.normal((B, H, N))) * 0.5 + 0.5      # decay a in (0.5, 1)
    cumlog = mx.cumsum(mx.log(a), axis=-1).astype(mx.float32)
    got = mamba2(C, Bm, X, cumlog)
    scores = mx.matmul(C.astype(mx.float32), mx.swapaxes(Bm.astype(mx.float32), -1, -2))
    decay = mx.exp(cumlog[..., :, None] - cumlog[..., None, :])
    mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
    exp = mx.matmul(scores * decay * mask, X.astype(mx.float32))
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_mamba2(shp)
        print("ok", shp)
