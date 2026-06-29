"""Correctness test for decay/retention linear attention (RetNet / Lightning-Attention-2):
out_i = sum_{j<=i} exp(-slope_h*(i-j)) * (q_i.k_j) * v_j.

Reference: ((Q @ Kᵀ) ⊙ Λ) @ V, Λ[i,j] = exp(-slope*(i-j)) for i>=j else 0. Validated on relative error.
Run from kernels/:  python -m pytest lin_attn_decay/correctness/test_lin_attn_decay.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import lin_attn_decay

SHAPES = [(1, 2, 64, 64), (2, 4, 128, 64), (1, 1, 256, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_lin_attn_decay(shape):
    B, H, N, D = shape
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    slopes = (np.linspace(0.05, 0.5, H)).astype(np.float32)        # per-head decay rate

    got = lin_attn_decay(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                         mx.array(v).astype(mx.bfloat16), slopes)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))

    pos = np.arange(N)
    dist = pos[:, None] - pos[None, :]                              # i - j
    causal = dist >= 0
    ref = np.zeros((B, H, N, D), np.float32)
    for h in range(H):
        lam = np.where(causal, np.exp(-slopes[h] * dist), 0.0)     # Λ[i,j]
        scores = q[:, h] @ np.swapaxes(k[:, h], -1, -2)            # (B,N,N)
        ref[:, h] = (scores * lam[None]) @ v[:, h]

    assert got.shape == (B, H, N, D)
    diff = np.abs(g - ref).max()
    scale = np.abs(ref).max() + 1e-9
    assert diff / scale < 0.04, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_lin_attn_decay(shp)
        print("ok", shp)
