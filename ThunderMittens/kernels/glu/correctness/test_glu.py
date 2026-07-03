"""Correctness tests for the ThunderMittens GLU-family kernels.

Run from kernels/: python -m pytest glu/correctness/test_glu.py -q
"""

import mlx.core as mx
import numpy as np
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


# --- numpy mirror of the tk A&S erf approximation and its exact analytic derivative ---
_EP, _A1, _A2, _A3, _A4, _A5 = (0.3275911, 0.254829592, -0.284496736, 1.421413741,
                                -1.453152027, 1.061405429)


def _erf_approx_np(x):
    sx = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + _EP * ax)
    poly = (((((_A5 * t + _A4) * t) + _A3) * t + _A2) * t + _A1) * t
    return sx * (1.0 - poly * np.exp(-ax * ax))


def _erf_approx_deriv_np(x):
    ax = np.abs(x)
    t = 1.0 / (1.0 + _EP * ax)
    poly = (((((_A5 * t + _A4) * t) + _A3) * t + _A2) * t + _A1) * t
    dpoly = (((5.0 * _A5 * t + 4.0 * _A4) * t + 3.0 * _A3) * t + 2.0 * _A2) * t + _A1
    return np.exp(-ax * ax) * (2.0 * ax * poly + _EP * t * t * dpoly)


def test_geglu_erf_backward_is_exact_derivative_of_forward():
    # geglu_erf forward uses the A&S erf approximation; the backward must be the EXACT derivative of
    # THAT approximation (not the ideal-erf Gaussian), so forward/backward are a bit-consistent pair.
    from tk import glu_backward
    inv_sqrt2 = 0.7071067811865476
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    gate = rng.standard_normal((4, 512)).astype(np.float32)
    dc = rng.standard_normal((4, 512)).astype(np.float32)
    da, db = glu_backward(mx.array(x), mx.array(gate), mx.array(dc), mode="geglu_erf")
    mx.eval(da, db)
    u = x * inv_sqrt2
    e = _erf_approx_np(u)
    de = _erf_approx_deriv_np(u) * inv_sqrt2            # d erf_approx(x/sqrt2) / dx
    act = 0.5 * x * (1.0 + e)                           # == glu_gelu_erf(x)
    dact = 0.5 * (1.0 + e) + 0.5 * x * de              # exact derivative of the approx forward
    db_ref = dc * act
    da_ref = dc * gate * dact
    assert np.max(np.abs(np.array(db) - db_ref)) < 3e-5, np.max(np.abs(np.array(db) - db_ref))
    assert np.max(np.abs(np.array(da) - da_ref)) < 3e-5, np.max(np.abs(np.array(da) - da_ref))


def _swiglu_oai_grad_ref(a, b, dc, alpha, limit):
    # Analytic gradient of out = (min(a,limit)*sigmoid(alpha*min(a,limit))) * (1 + clamp(b,-limit,limit)).
    # min/clamp derivatives are 0 outside the active region (the clamp KINK -- finite-diff is invalid there).
    x0 = np.minimum(a, limit)
    x1 = np.clip(b, -limit, limit)
    s0 = 1.0 / (1.0 + np.exp(-x0 * alpha))
    f = x0 * s0
    ind_a = (a < limit).astype(np.float32)                       # d min(a,limit)/da
    ind_b = ((b < limit) & (b > -limit)).astype(np.float32)      # d clamp(b)/db
    db = dc * f * ind_b
    da = dc * (1.0 + x1) * (s0 + x0 * alpha * s0 * (1.0 - s0)) * ind_a
    return da, db


def test_swiglu_oai_clamp_gradient():
    # Small limit so many inputs are on the clamped side; compare to the analytic clamped gradient
    # (NOT finite-diff -- the min/clamp kinks make central differences wrong across the boundary).
    from tk import glu_backward
    rng = np.random.default_rng(11)
    alpha, limit = 1.3, 1.5
    a = (3.0 * rng.standard_normal((8, 256))).astype(np.float32)  # spans well past +/-limit
    b = (3.0 * rng.standard_normal((8, 256))).astype(np.float32)
    dc = rng.standard_normal((8, 256)).astype(np.float32)
    # keep inputs off the exact kinks so the one-sided indicator is unambiguous vs the kernel's <
    a = np.where(np.abs(a - limit) < 1e-2, a + 0.1, a)
    b = np.where(np.abs(np.abs(b) - limit) < 1e-2, b + 0.1, b)
    da, db = glu_backward(mx.array(a), mx.array(b), mx.array(dc), mode="swiglu_oai",
                          alpha=alpha, limit=limit)
    mx.eval(da, db)
    da_ref, db_ref = _swiglu_oai_grad_ref(a, b, dc, alpha, limit)
    # assert we actually exercised both clamp branches (else the test proves nothing)
    assert (a >= limit).any() and ((b >= limit) | (b <= -limit)).any()
    assert np.max(np.abs(np.array(da) - da_ref)) < 1e-4, np.max(np.abs(np.array(da) - da_ref))
    assert np.max(np.abs(np.array(db) - db_ref)) < 1e-4, np.max(np.abs(np.array(db) - db_ref))
