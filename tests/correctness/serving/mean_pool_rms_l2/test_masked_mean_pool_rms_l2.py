"""Correctness for mask-aware embedding pooling."""

import mlx.core as mx
import numpy as np
import pytest

import tk


@pytest.mark.parametrize("D", [256, 768, 1024])
def test_masked_mean_pool_rms_l2(D):
    rng = np.random.default_rng(D)
    B, T = 4, 19
    x = (0.3 * rng.standard_normal((B, T, D))).astype(np.float32)
    mask = (rng.random((B, T)) > 0.35).astype(np.int32)
    mask[0] = 0
    weight = (0.5 + 0.2 * rng.standard_normal(D)).astype(np.float32)
    xd = mx.array(x).astype(mx.bfloat16)
    wd = mx.array(weight).astype(mx.bfloat16)
    got = tk.masked_mean_pool_rms_l2(xd, mx.array(mask), wd, eps=1e-6)
    mx.eval(got)
    xb = np.array(xd.astype(mx.float32)); wb = np.array(wd.astype(mx.float32))
    ref = np.zeros((B, D), np.float32)
    for b in range(B):
        keep = mask[b] != 0
        if keep.any():
            pooled = xb[b, keep].mean(axis=0)
            normed = pooled / np.sqrt(np.mean(pooled * pooled) + 1e-6) * wb
            ref[b] = normed / np.sqrt(np.sum(normed * normed) + 1e-12)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=3e-2, rtol=3e-2)
    np.testing.assert_array_equal(np.array(got[0].astype(mx.float32)), np.zeros(D, np.float32))
