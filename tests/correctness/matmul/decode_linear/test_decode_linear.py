import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import (
    decode_linear,
    decode_linear_epilogue,
    decode_linear_q8,
    decode_linear_residual,
    decode_swiglu,
)
from tk.quant import QUANT_FORMATS, dequantize_q8_0, quantize_q8_0


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


def _activation_reference(x, activation):
    if activation == "gelu":
        return _gelu_erf_approx(x)
    if activation == "silu":
        return x / (1.0 + np.exp(-x))
    return x


@pytest.mark.parametrize("activation", ["none", "gelu", "silu"])
@pytest.mark.parametrize("use_bias,use_residual", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("use_kernel", [False, True], ids=["routed", "kernel"])
def test_decode_linear_epilogue_dense(activation, use_bias, use_residual, use_kernel):
    rng = np.random.default_rng(101 + len(activation) + use_bias + use_residual)
    B, K, N = 3, 73, 41
    x = (0.12 * rng.standard_normal((B, K))).astype(np.float32)
    weight = (0.11 * rng.standard_normal((N, K))).astype(np.float32)
    bias = (0.03 * rng.standard_normal(N)).astype(np.float32)
    residual = (0.04 * rng.standard_normal((B, N))).astype(np.float32)
    got = decode_linear_epilogue(
        mx.array(x), mx.array(weight), mx.array(bias) if use_bias else None,
        mx.array(residual) if use_residual else None, activation=activation,
        use_kernel=use_kernel)
    mx.eval(got)
    ref = x @ weight.T
    if use_bias:
        ref += bias
    ref = _activation_reference(ref, activation)
    if use_residual:
        ref += residual
    np.testing.assert_allclose(np.array(got), ref, rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("fmt", ["q4_0", "q8_0", "q6_K", "nvfp4"])
@pytest.mark.parametrize("activation", ["none", "gelu", "silu"])
def test_decode_linear_epilogue_packed(fmt, activation):
    quantize, dequantize = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(131 + len(fmt) + len(activation))
    B, K, N = 2, 512, 35
    x = (0.08 * rng.standard_normal((B, K))).astype(np.float32)
    packed = quantize((0.1 * rng.standard_normal((N, K))).astype(np.float32))
    bias = (0.02 * rng.standard_normal(N)).astype(np.float32)
    residual = (0.03 * rng.standard_normal((B, N))).astype(np.float32)
    got = decode_linear_epilogue(
        mx.array(x), mx.array(packed), mx.array(bias), mx.array(residual),
        activation=activation, format=fmt)
    mx.eval(got)
    ref = _activation_reference(x @ dequantize(packed).astype(np.float32).T + bias, activation)
    ref += residual
    np.testing.assert_allclose(np.array(got), ref, rtol=7e-4, atol=7e-4)


@pytest.mark.parametrize("fmt,use_kernel", [
    (None, None), (None, False), (None, True),
    ("q4_0", True), ("q8_0", True), ("q6_K", True), ("nvfp4", True)])
def test_decode_swiglu_dense_and_packed(fmt, use_kernel):
    rng = np.random.default_rng(151 + (0 if fmt is None else len(fmt)))
    B, K, N = 2, 512, 33
    x = (0.07 * rng.standard_normal((B, K))).astype(np.float32)
    gate = (0.09 * rng.standard_normal((N, K))).astype(np.float32)
    up = (0.08 * rng.standard_normal((N, K))).astype(np.float32)
    gate_bias = (0.02 * rng.standard_normal(N)).astype(np.float32)
    up_bias = (0.02 * rng.standard_normal(N)).astype(np.float32)
    if fmt is None:
        gate_arg, up_arg = mx.array(gate), mx.array(up)
        gate_ref, up_ref = gate, up
    else:
        quantize, dequantize = QUANT_FORMATS[fmt]
        gate_packed, up_packed = quantize(gate), quantize(up)
        gate_arg, up_arg = mx.array(gate_packed), mx.array(up_packed)
        gate_ref = dequantize(gate_packed).astype(np.float32)
        up_ref = dequantize(up_packed).astype(np.float32)
    got = decode_swiglu(
        mx.array(x), gate_arg, up_arg, mx.array(gate_bias), mx.array(up_bias),
        format=fmt, use_kernel=use_kernel)
    mx.eval(got)
    gate_value = x @ gate_ref.T + gate_bias
    ref = gate_value / (1.0 + np.exp(-gate_value)) * (x @ up_ref.T + up_bias)
    np.testing.assert_allclose(np.array(got), ref, rtol=8e-4, atol=8e-4)


def test_decode_linear_epilogue_optional_output_quantization():
    rng = np.random.default_rng(177)
    x = mx.array((0.1 * rng.standard_normal((2, 64))).astype(np.float32))
    weight = mx.array((0.1 * rng.standard_normal((17, 64))).astype(np.float32))
    dense = decode_linear_epilogue(x, weight, activation="silu")
    codes, scales = decode_linear_epilogue(
        x, weight, activation="silu", output_quant="int8")
    mx.eval(dense, codes, scales)
    reconstructed = np.array(codes).astype(np.float32) * np.array(scales)[..., None]
    error = np.max(np.abs(reconstructed - np.array(dense)))
    assert codes.dtype == mx.int8 and scales.dtype == mx.float32
    assert error <= float(np.max(np.array(scales))) * 0.51 + 1e-6
