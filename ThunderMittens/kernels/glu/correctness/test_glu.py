"""Correctness tests for the ThunderMittens GLU-family kernels.

Run from kernels/: python -m pytest glu/correctness/test_glu.py -q
"""

import mlx.core as mx
import pytest

from tk import glu


SHAPES = [(3, 1024), (2, 17, 4096), (1, 11008)]
DTYPES = [mx.float32, mx.float16, mx.bfloat16]
MODES = ["reglu", "geglu", "swiglu", "swiglu_oai", "geglu_erf", "geglu_quick"]


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _gelu_erf(x):
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _expected(x, gate, mode, alpha=1.0, limit=1.0e20):
    xf = x.astype(mx.float32)
    gf = gate.astype(mx.float32)
    if mode == "reglu":
        out = mx.where(xf > 0, xf * gf, mx.zeros_like(xf))
    elif mode == "geglu":
        out = _gelu_tanh(xf) * gf
    elif mode == "swiglu":
        out = (xf / (1.0 + mx.exp(-xf))) * gf
    elif mode == "swiglu_oai":
        x0 = mx.minimum(xf, limit)
        x1 = mx.maximum(mx.minimum(gf, limit), -limit)
        out = (x0 / (1.0 + mx.exp(-x0 * alpha))) * (1.0 + x1)
    elif mode == "geglu_erf":
        out = _gelu_erf(xf) * gf
    elif mode == "geglu_quick":
        out = (xf / (1.0 + mx.exp(-1.702 * xf))) * gf
    else:
        raise AssertionError(mode)
    return out.astype(x.dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_glu_matches_reference(shape, dtype, mode):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(dtype)
    gate = mx.random.normal(shape).astype(dtype)
    alpha = 1.3
    limit = 2.5

    got = glu(x, gate, mode=mode, alpha=alpha, limit=limit)
    exp = _expected(x, gate, mode, alpha=alpha, limit=limit)
    mx.eval(got, exp)

    assert got.shape == x.shape
    assert got.dtype == dtype
    tol = 2e-2 if dtype == mx.bfloat16 else 5e-3
    assert mx.allclose(got.astype(mx.float32), exp.astype(mx.float32), atol=tol, rtol=tol), (
        mode,
        dtype,
        mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item(),
    )


@pytest.mark.parametrize("mode", MODES)
def test_glu_backward_finite_diff(mode):
    # tk glu_backward vs central finite-difference of the tk forward (self-consistent gradient check).
    from tk import glu_backward
    mx.random.seed(3)
    shape = (4, 512)
    x = mx.random.normal(shape).astype(mx.float32)
    gate = mx.random.normal(shape).astype(mx.float32)
    dc = mx.random.normal(shape).astype(mx.float32)
    alpha, limit = 1.0, 1.0e20   # no clamp active -> smooth (swiglu_oai clamp indicator is trivial)
    da, db = glu_backward(x, gate, dc, mode=mode, alpha=alpha, limit=limit)
    h = 1e-3
    dfx = (glu(x + h, gate, mode=mode, alpha=alpha, limit=limit)
           - glu(x - h, gate, mode=mode, alpha=alpha, limit=limit)) / (2 * h)
    dfg = (glu(x, gate + h, mode=mode, alpha=alpha, limit=limit)
           - glu(x, gate - h, mode=mode, alpha=alpha, limit=limit)) / (2 * h)
    da_fd, db_fd = dc * dfx, dc * dfg
    mask = mx.abs(x) > 0.05      # skip the reglu kink at x=0 (central-diff invalid there)
    mx.eval(da, db, da_fd, db_fd, mask)
    for got, ref in ((da, da_fd), (db, db_fd)):
        err = mx.max(mx.abs((got - ref) * mask)).item()
        assert err < 2e-2, (mode, err)
