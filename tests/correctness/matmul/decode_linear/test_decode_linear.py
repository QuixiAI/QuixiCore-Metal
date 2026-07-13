import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import decode_linear, decode_linear_q8, decode_linear_residual
from tk.quant import dequantize_q8_0, quantize_q8_0


def _gelu_erf_approx(x):
    """Abramowitz-Stegun erf approximation used by the Metal kernels."""
    z = x * np.float32(0.7071067811865475)
    az = np.abs(z)
    t = 1.0 / (1.0 + np.float32(0.3275911) * az)
    polynomial = (((((np.float32(1.061405429) * t - np.float32(1.453152027)) * t
                     + np.float32(1.421413741)) * t - np.float32(0.284496736)) * t
                   + np.float32(0.254829592)) * t)
    erf = np.copysign(1.0 - polynomial * np.exp(-az * az), z)
    return 0.5 * x * (1.0 + erf)


@pytest.mark.parametrize("gelu", [False, True])
def test_decode_linear_float32(gelu):
    rng = np.random.default_rng(41 + gelu)
    B, K, N = 3, 65, 37
    x = (0.15 * rng.standard_normal((B, K))).astype(np.float32)
    weight = (0.12 * rng.standard_normal((N, K))).astype(np.float32)
    bias = (0.05 * rng.standard_normal(N)).astype(np.float32)
    got = decode_linear(mx.array(x), mx.array(weight), mx.array(bias), gelu=gelu)
    mx.eval(got)
    ref = x @ weight.T + bias
    if gelu:
        ref = _gelu_erf_approx(ref)
    np.testing.assert_allclose(np.array(got), ref, rtol=3e-4, atol=3e-4)


def test_decode_linear_bfloat16_gelu():
    rng = np.random.default_rng(47)
    B, K, N = 2, 65, 37
    x, weight, bias = [
        mx.array((0.1 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
        for shape in ((B, K), (N, K), (N,))
    ]
    got = decode_linear(x, weight, bias, gelu=True)
    linear = x.astype(mx.float32) @ mx.swapaxes(weight.astype(mx.float32), 0, 1)
    linear = linear + bias.astype(mx.float32)
    ref = mx.array(_gelu_erf_approx(np.array(linear))).astype(mx.bfloat16)
    mx.eval(got, ref)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 3e-4), (mx.bfloat16, 4e-2)])
def test_decode_linear_residual_rounding(dtype, atol):
    rng = np.random.default_rng(53)
    B, K, N = 2, 69, 35
    arrays = [
        mx.array((0.15 * rng.standard_normal(shape)).astype(np.float32)).astype(dtype)
        for shape in ((B, K), (N, K), (N,), (B, N))
    ]
    x, weight, bias, residual = arrays
    got = decode_linear_residual(x, weight, bias, residual)
    linear = (x.astype(mx.float32) @ mx.swapaxes(weight.astype(mx.float32), 0, 1)
              + bias.astype(mx.float32)).astype(dtype)
    ref = (linear.astype(mx.float32) + residual.astype(mx.float32)).astype(dtype)
    mx.eval(got, ref)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=atol, atol=atol)


@pytest.mark.parametrize("gelu,use_residual", [(False, False), (True, False), (True, True)])
def test_decode_linear_q8_float32(gelu, use_residual):
    rng = np.random.default_rng(67 + gelu + use_residual)
    B, K, N = 2, 96, 39
    x = (0.12 * rng.standard_normal((B, K))).astype(np.float32)
    weight = (0.18 * rng.standard_normal((N, K))).astype(np.float32)
    packed = quantize_q8_0(weight)
    bias = (0.04 * rng.standard_normal(N)).astype(np.float32)
    residual = (0.05 * rng.standard_normal((B, N))).astype(np.float32)
    got = decode_linear_q8(
        mx.array(x), mx.array(packed), mx.array(bias),
        residual=mx.array(residual) if use_residual else None, gelu=gelu)
    mx.eval(got)

    # The q8 path rounds the linear+bias epilogue to the activation dtype before
    # applying GELU and again before an optional residual addition.
    ref = (x @ dequantize_q8_0(packed).T + bias).astype(np.float32)
    if gelu:
        ref = _gelu_erf_approx(ref).astype(np.float32)
    if use_residual:
        ref = (ref + residual).astype(np.float32)
    np.testing.assert_allclose(np.array(got), ref, rtol=4e-4, atol=4e-4)


def test_decode_linear_q8_bfloat16_epilogue_rounding():
    rng = np.random.default_rng(71)
    B, K, N = 2, 96, 39
    x = mx.array((0.1 * rng.standard_normal((B, K))).astype(np.float32)).astype(mx.bfloat16)
    packed = quantize_q8_0((0.15 * rng.standard_normal((N, K))).astype(np.float32))
    bias = mx.array((0.03 * rng.standard_normal(N)).astype(np.float32)).astype(mx.bfloat16)
    residual = mx.array((0.03 * rng.standard_normal((B, N))).astype(np.float32)).astype(
        mx.bfloat16)
    got = decode_linear_q8(x, mx.array(packed), bias, residual=residual, gelu=True)
    linear = (x.astype(mx.float32) @ mx.array(dequantize_q8_0(packed)).T +
              bias.astype(mx.float32)).astype(mx.bfloat16)
    activated = mx.array(_gelu_erf_approx(np.array(linear.astype(mx.float32)))).astype(
        mx.bfloat16)
    ref = (activated.astype(mx.float32) + residual.astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(got, ref)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=4e-2, atol=4e-2)
