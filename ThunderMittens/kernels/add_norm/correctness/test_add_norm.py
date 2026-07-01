"""Correctness tests for the fused residual-add + norm Metal kernels.

The kernels return two arrays: out = norm(x + residual) [* weight (+ bias)], and
res_out = x + residual (the summed residual the next block reads). The kernel
normalizes the fp32 sum and writes the bf16-rounded sum back.

Run from kernels/:  python -m pytest add_norm/correctness/test_add_norm.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import rms_norm_add, layernorm_add, rms_norm_add_fp8, layernorm_add_fp8
from tk.quant import _e4m3_decode_arr


def ref_rms_norm(sum_f32, w, eps):
    ms = (sum_f32 * sum_f32).mean(axis=-1, keepdims=True)
    return (sum_f32 * mx.rsqrt(ms + eps) * w.astype(mx.float32)).astype(mx.bfloat16)


def ref_layernorm(sum_f32, w, b, eps):
    mean = sum_f32.mean(axis=-1, keepdims=True)
    var = ((sum_f32 - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (sum_f32 - mean) * mx.rsqrt(var + eps) * w.astype(mx.float32) + b.astype(mx.float32)
    return y.astype(mx.bfloat16)


SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_add(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)

    out, added = rms_norm_add(x, r, w, eps=eps)

    sum_f32 = x.astype(mx.float32) + r.astype(mx.float32)
    added_ref = sum_f32.astype(mx.bfloat16)
    out_ref = ref_rms_norm(sum_f32, w, eps)
    mx.eval(out, added, added_ref, out_ref)

    assert out.shape == x.shape and added.shape == x.shape
    assert out.dtype == mx.bfloat16 and added.dtype == mx.bfloat16
    assert mx.allclose(added, added_ref, atol=2e-2, rtol=2e-2)
    assert mx.allclose(out, out_ref, atol=2e-2, rtol=2e-2), \
        f"max {mx.max(mx.abs(out.astype(mx.float32)-out_ref.astype(mx.float32))).item()}"


@pytest.mark.parametrize("shape", SHAPES)
def test_layernorm_add(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(1)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    b = mx.random.normal((D,)).astype(mx.bfloat16)

    out, added = layernorm_add(x, r, w, b, eps=eps)

    sum_f32 = x.astype(mx.float32) + r.astype(mx.float32)
    added_ref = sum_f32.astype(mx.bfloat16)
    out_ref = ref_layernorm(sum_f32, w, b, eps)
    mx.eval(out, added, added_ref, out_ref)

    assert mx.allclose(added, added_ref, atol=2e-2, rtol=2e-2)
    assert mx.allclose(out, out_ref, atol=2e-2, rtol=2e-2), \
        f"max {mx.max(mx.abs(out.astype(mx.float32)-out_ref.astype(mx.float32))).item()}"


FP8_SHAPES = [(8, 256), (4, 512), (16, 768), (3, 1024)]


def _rms_normed_f32(x, r, w, eps):
    s = np.array(x.astype(mx.float32)) + np.array(r.astype(mx.float32))
    ms = (s * s).mean(-1, keepdims=True)
    return s / np.sqrt(ms + eps) * np.array(w.astype(mx.float32)), s


def _ln_normed_f32(x, r, w, b, eps):
    s = np.array(x.astype(mx.float32)) + np.array(r.astype(mx.float32))
    mean = s.mean(-1, keepdims=True)
    var = ((s - mean) ** 2).mean(-1, keepdims=True)
    return (s - mean) / np.sqrt(var + eps) * np.array(w.astype(mx.float32)) + np.array(b.astype(mx.float32)), s


def _fp8_ok(codes, normed, ssafe):
    deq = _e4m3_decode_arr(np.array(codes)) * ssafe
    tol = 0.0625 * np.abs(normed) + 2.0 * ssafe
    return np.all(np.abs(deq - normed) <= tol)


@pytest.mark.parametrize("shape", FP8_SHAPES)
def test_rms_norm_add_fp8_dynamic(shape):
    eps, D = 1e-5, shape[-1]
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    r = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
    codes, added, scale = rms_norm_add_fp8(x, r, w)  # dynamic
    mx.eval(codes, added, scale)
    normed, s = _rms_normed_f32(x, r, w, eps)
    ref_scale = np.abs(normed).max(-1) / 448.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]
    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)
    assert _fp8_ok(codes, normed, ssafe)
    added_ref = np.array(mx.array(s).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(np.array(added.astype(mx.float32)), added_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", FP8_SHAPES)
def test_rms_norm_add_fp8_static(shape):
    eps, D = 1e-5, shape[-1]
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    r = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
    normed, _ = _rms_normed_f32(x, r, w, eps)
    scale = float(np.abs(normed).max() / 448.0)
    codes, added = rms_norm_add_fp8(x, r, w, scale=scale)
    mx.eval(codes, added)
    assert _fp8_ok(codes, normed, np.float32(scale))


@pytest.mark.parametrize("shape", FP8_SHAPES)
def test_layernorm_add_fp8_dynamic(shape):
    eps, D = 1e-5, shape[-1]
    rng = np.random.default_rng(2)
    x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    r = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
    codes, added, scale = layernorm_add_fp8(x, r, w, b)  # dynamic
    mx.eval(codes, added, scale)
    normed, _ = _ln_normed_f32(x, r, w, b, eps)
    ref_scale = np.abs(normed).max(-1) / 448.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]
    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)
    assert _fp8_ok(codes, normed, ssafe)


if __name__ == "__main__":
    for shp in SHAPES:
        test_rms_norm_add(shp)
        test_layernorm_add(shp)
        print("ok", shp)
