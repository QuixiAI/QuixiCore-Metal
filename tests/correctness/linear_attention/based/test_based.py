"""Correctness test for Based 2nd-order Taylor feature-map linear attention (causal):
out_i = sum_{j<=i} (1 + x + x^2/2) * v_j, with x = (q_i . k_j)/sqrt(D_QK), D_QK=16, D_VO=64.

Reference: ((1 + x + x^2/2) ⊙ tril) @ V (unnormalized numerator, matching the TK based kernel).
Run from kernels/:  python -m pytest based/correctness/test_based.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import based

SHAPES = [(1, 2, 64), (2, 4, 128), (1, 1, 256)]   # (B, H, N); D_QK=16, D_VO=64


@pytest.mark.parametrize("shape", SHAPES)
def test_based(shape):
    B, H, N = shape
    DQK, DVO = 16, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, DVO)) * 0.5).astype(np.float32)

    got = based(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                mx.array(v).astype(mx.bfloat16))
    mx.eval(got)
    g = np.array(got.astype(mx.float32))

    x = 0.25 * (q @ np.swapaxes(k, -1, -2))            # (B,H,N,N), temp 1/sqrt(16)
    A = 1.0 + x + 0.5 * x * x                          # Taylor phi.phi
    mask = (np.arange(N)[None, :] <= np.arange(N)[:, None]).astype(np.float32)
    ref = (A * mask[None, None]) @ v                   # (B,H,N,64)

    assert got.shape == (B, H, N, DVO)
    diff = np.abs(g - ref).max()
    scale = np.abs(ref).max() + 1e-9
    assert diff / scale < 0.04, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_based(shp)
        print("ok", shp)
