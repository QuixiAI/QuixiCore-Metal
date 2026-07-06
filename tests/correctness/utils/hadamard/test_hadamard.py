"""Correctness tests for the Hadamard/FWHT kernel.

Run from kernels/: python -m pytest hadamard/correctness/test_hadamard.py -q
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import hadamard


def _mx_dtype(name):
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[name]


def _np(x):
    return np.array(x.astype(mx.float32))


def _round_np(x, dtype):
    return _np(mx.array(x).astype(_mx_dtype(dtype))).astype(np.float32)


def _fwht_ref(x, scale):
    y = x.astype(np.float32).copy()
    d = y.shape[-1]
    h = 1
    while h < d:
        rows = y.reshape(-1, d)
        for i in range(0, d, h * 2):
            a = rows[:, i : i + h].copy()
            b = rows[:, i + h : i + 2 * h].copy()
            rows[:, i : i + h] = a + b
            rows[:, i + h : i + 2 * h] = a - b
        h *= 2
    return y * scale


@pytest.mark.parametrize("dtype,atol", [("float32", 1e-6), ("float16", 2e-3), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_hadamard_default_scale(dtype, atol, D):
    rng = np.random.default_rng(D)
    x = rng.normal(size=(2, 3, D)).astype(np.float32)
    xm = mx.array(x).astype(_mx_dtype(dtype))

    got = hadamard(xm)
    mx.eval(got)

    ref = _fwht_ref(_round_np(x, dtype), 1.0 / math.sqrt(D))
    ref = _round_np(ref, dtype)
    np.testing.assert_allclose(_np(got), ref, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype,atol", [("float32", 1e-6), ("float16", 2e-3), ("bfloat16", 2e-2)])
def test_hadamard_explicit_scale(dtype, atol):
    rng = np.random.default_rng(2048)
    D = 128
    scale = 0.25
    x = rng.normal(size=(5, D)).astype(np.float32)
    xm = mx.array(x).astype(_mx_dtype(dtype))

    got = hadamard(xm, scale=scale)
    mx.eval(got)

    ref = _fwht_ref(_round_np(x, dtype), scale)
    ref = _round_np(ref, dtype)
    np.testing.assert_allclose(_np(got), ref, atol=atol, rtol=1e-5)
