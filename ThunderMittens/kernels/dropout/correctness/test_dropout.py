"""Correctness tests for inverted dropout (fwd/bwd).

The keep-mask is a pure function of (seed, flat index) via the substrate's counter-based RNG, so we
replicate that exact hash in numpy for an EXACT oracle (not just a statistical check), and confirm
the backward recomputes the same mask.

Run from kernels/:  python -m pytest dropout/correctness/test_dropout.py -q
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import dropout, dropout_backward

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
_M32 = np.uint64(0xFFFFFFFF)


def _rng_uniform(seed, idx):
    """numpy replica of mittens::rng_uniform(seed, idx, 0) (Murmur3-style finalizer)."""
    a = idx.astype(np.uint64)
    x = (np.uint64(seed) * np.uint64(0x9E3779B9) + a * np.uint64(0x85EBCA77)) & _M32
    x = (x ^ (x >> np.uint64(16))) & _M32
    x = (x * np.uint64(0x7FEB352D)) & _M32
    x = (x ^ (x >> np.uint64(15))) & _M32
    x = (x * np.uint64(0x846CA68B)) & _M32
    x = (x ^ (x >> np.uint64(16))) & _M32
    return (x >> np.uint64(8)).astype(np.float64) * (1.0 / 16777216.0)


def _keep(seed, shape, p):
    u = _rng_uniform(seed, np.arange(int(np.prod(shape)), dtype=np.uint64)).reshape(shape)
    return u >= p


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9])
def test_dropout_fwd_bwd(dtype, p):
    rng = np.random.default_rng(int(p * 100) + 1)
    shape, seed = (64, 512), 12345
    x = rng.standard_normal(shape).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    keep = _keep(seed, shape, p)
    inv = 1.0 / (1.0 - p)
    out = np.array(dropout(mx.array(x).astype(_MX[dtype]), p, seed).astype(mx.float32))
    dx = np.array(dropout_backward(mx.array(dy).astype(_MX[dtype]), p, seed).astype(mx.float32))
    ref_out = np.where(keep, x * inv, 0.0)
    ref_dx = np.where(keep, dy * inv, 0.0)
    # kept values are scaled by inv=1/(1-p), so use a RELATIVE tol for fp16/bf16 (zeros stay exact).
    rtol = 0.0 if dtype == "float32" else 3e-2
    atol = 1e-4 if dtype == "float32" else 1e-2
    np.testing.assert_allclose(out, ref_out, rtol=rtol, atol=atol)
    np.testing.assert_allclose(dx, ref_dx, rtol=rtol, atol=atol)


def test_dropout_mean_preserving_and_reproducible():
    # E[out] ~ x over a large tensor, and the same seed reproduces the same mask.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((512, 1024)).astype(np.float32) + 3.0   # nonzero mean
    p, seed = 0.3, 999
    o1 = np.array(dropout(mx.array(x), p, seed))
    o2 = np.array(dropout(mx.array(x), p, seed))
    assert np.array_equal(o1, o2)                                   # reproducible
    assert abs(o1.mean() - x.mean()) < 0.05                         # unbiased (inverted dropout)
    frac_zero = (o1 == 0.0).mean()
    assert abs(frac_zero - p) < 0.02                                # ~p dropped


if __name__ == "__main__":
    test_dropout_fwd_bwd("float32", 0.5)
    test_dropout_mean_preserving_and_reproducible()
    print("ok")
