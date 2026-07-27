"""Correctness for tensor-only Whisper/Conformer convolution primitives."""

import mlx.core as mx
import numpy as np
import pytest

import tk


_DTYPES = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}


def _conv_ref(x, w, bias, stride, padding, dilation):
    B, T, C = x.shape; O, K, _ = w.shape
    OT = (T + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    out = np.empty((B, OT, O), np.float32)
    for b in range(B):
        for t in range(OT):
            for o in range(O):
                acc = 0.0 if bias is None else float(bias[o])
                for k in range(K):
                    it = t * stride + k * dilation - padding
                    if 0 <= it < T:
                        acc += float(np.dot(x[b, it].astype(np.float32), w[o, k].astype(np.float32)))
                out[b, t, o] = acc
    return out


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("geometry", [(1, 0, 1), (2, 2, 2)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_audio_conv1d_direct(dtype, geometry, with_bias):
    stride, padding, dilation = geometry
    rng = np.random.default_rng(70 + stride + padding + dilation)
    x0 = (0.2 * rng.standard_normal((2, 19, 7))).astype(np.float32)
    w0 = (0.2 * rng.standard_normal((11, 3, 7))).astype(np.float32)
    b0 = (0.1 * rng.standard_normal(11)).astype(np.float32) if with_bias else None
    xd = mx.array(x0).astype(_DTYPES[dtype]); wd = mx.array(w0).astype(_DTYPES[dtype])
    bd = mx.array(b0).astype(_DTYPES[dtype]) if with_bias else None
    got = tk.audio_conv1d_direct(xd, wd, bd, stride, padding, dilation)
    mx.eval(got)
    xb = np.array(xd.astype(mx.float32)); wb = np.array(wd.astype(mx.float32))
    bb = np.array(bd.astype(mx.float32)) if with_bias else None
    tol = 1e-5 if dtype == "f32" else 3e-2
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                               _conv_ref(xb, wb, bb, stride, padding, dilation),
                               atol=tol, rtol=tol)


@pytest.mark.parametrize("activation", ["none", "silu"])
def test_audio_depthwise_conv1d(activation):
    rng = np.random.default_rng(81)
    x0 = (0.2 * rng.standard_normal((2, 23, 13))).astype(np.float32)
    w0 = (0.2 * rng.standard_normal((13, 5))).astype(np.float32)
    b0 = (0.1 * rng.standard_normal(13)).astype(np.float32)
    xd = mx.array(x0).astype(mx.bfloat16); wd = mx.array(w0).astype(mx.bfloat16)
    bd = mx.array(b0).astype(mx.bfloat16)
    got = tk.audio_depthwise_conv1d(
        xd, wd, bd, stride=1, padding=2, activation=activation)
    mx.eval(got)
    xb = np.array(xd.astype(mx.float32)); wb = np.array(wd.astype(mx.float32)); bb = np.array(bd.astype(mx.float32))
    ref = np.empty_like(xb)
    for b in range(2):
        for t in range(23):
            for c in range(13):
                value = bb[c]
                for k in range(5):
                    it = t + k - 2
                    if 0 <= it < 23: value += xb[b, it, c] * wb[c, k]
                ref[b, t, c] = value / (1.0 + np.exp(-value)) if activation == "silu" else value
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=3e-2, rtol=3e-2)


def test_audio_causal_depthwise_conv1d():
    rng = np.random.default_rng(73)
    B, T, C, K, dilation = 2, 13, 7, 5, 2
    x = (0.2 * rng.standard_normal((B, T, C))).astype(np.float32)
    weight = (0.2 * rng.standard_normal((C, K))).astype(np.float32)
    bias = (0.1 * rng.standard_normal(C)).astype(np.float32)
    got = tk.audio_causal_depthwise_conv1d(
        mx.array(x), mx.array(weight), mx.array(bias), dilation=dilation)
    ref = np.empty_like(x); pad_left = dilation * (K - 1)
    for b in range(B):
        for t in range(T):
            for c in range(C):
                value = bias[c]
                for kk in range(K):
                    source = t + kk * dilation - pad_left
                    if 0 <= source < T:
                        value += x[b, source, c] * weight[c, kk]
                ref[b, t, c] = value
    np.testing.assert_allclose(np.array(got), ref, atol=1e-6, rtol=1e-6)
