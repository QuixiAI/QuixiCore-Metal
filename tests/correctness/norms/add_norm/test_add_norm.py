"""Correctness tests for the fused residual-add + norm Metal kernels.

The kernels return two arrays: out = norm(x + residual) [* weight (+ bias)], and
res_out = x + residual (the summed residual the next block reads). The kernel
normalizes the fp32 sum and writes the bf16-rounded sum back.

Run from kernels/:  python -m pytest add_norm/correctness/test_add_norm.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (decode_layernorm_add, layernorm_add, layernorm_add_fp8,
                rms_norm_add, rms_norm_add_fp8)
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


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 4e-4), (mx.bfloat16, 3e-2)])
def test_decode_layernorm_add_materialized_rounding(dtype, atol):
    eps, shape = 1e-5, (3, 37)
    rng = np.random.default_rng(37)
    x, residual = [
        mx.array((0.4 * rng.standard_normal(shape)).astype(np.float32)).astype(dtype)
        for _ in range(2)
    ]
    weight = mx.array((0.8 + 0.1 * rng.standard_normal(shape[-1])).astype(np.float32)).astype(dtype)
    bias = mx.array((0.05 * rng.standard_normal(shape[-1])).astype(np.float32)).astype(dtype)
    got, summed = decode_layernorm_add(x, residual, weight, bias, eps)

    rounded = (x.astype(mx.float32) + residual.astype(mx.float32)).astype(dtype)
    values = rounded.astype(mx.float32)
    mean = values.mean(-1, keepdims=True)
    variance = (values * values).mean(-1, keepdims=True) - mean * mean
    ref = ((values - mean) * mx.rsqrt(mx.maximum(variance, 0) + eps) *
           weight.astype(mx.float32) + bias.astype(mx.float32)).astype(dtype)
    mx.eval(got, summed, ref, rounded)
    np.testing.assert_allclose(np.array(summed.astype(mx.float32)),
                               np.array(rounded.astype(mx.float32)), rtol=0, atol=0)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                               np.array(ref.astype(mx.float32)), rtol=atol, atol=atol)


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


@pytest.mark.parametrize("D", [256, 1024])
def test_rms_norm_add_backward(D):
    # tk.rms_norm_add_backward vs numpy autograd of the fused residual-add + RMSNorm.
    from tk import rms_norm_add_backward
    rng = np.random.default_rng(D + 3)
    N, eps = 64, 1e-5
    x = rng.standard_normal((N, D)).astype(np.float32)
    res = rng.standard_normal((N, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    dout = rng.standard_normal((N, D)).astype(np.float32)
    dres = 0.3 * rng.standard_normal((N, D)).astype(np.float32)   # grad into the residual output
    h = (x + res).astype(np.float64)
    rstd = 1.0 / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)
    # dh from the rms path: rstd*(w*dout) - (rstd^3/D)*h*sum(w*dout*h)
    wd = w.astype(np.float64) * dout.astype(np.float64)
    dh_rms = rstd * wd - (rstd ** 3 / D) * h * (wd * h).sum(-1, keepdims=True)
    dh = dh_rms + dres.astype(np.float64)        # + residual-output branch
    dw_ref = (dout.astype(np.float64) * h * rstd).sum(0)
    dx, dresidual, dweight = rms_norm_add_backward(mx.array(h.astype(np.float32)), mx.array(w),
                                                   mx.array(dout), dresidual=mx.array(dres), eps=eps)
    assert np.allclose(np.array(dx), dh, atol=2e-3)
    assert np.allclose(np.array(dresidual), dh, atol=2e-3)       # dx IS dresidual
    assert np.allclose(np.array(dweight), dw_ref, atol=2e-3)


# ---------------------------------------------------------------------------
# Wave-10: fused norm -> per-block quant + LayerNorm int8. The normed value is
# exp-free, so given the kernel's bf16-rounded (x+residual) the int8 codes are
# BIT-EXACT vs a numpy twin; fp8 uses a half-ulp reconstruction bound. Per-block
# emits (rows, D/128) group scales.
# ---------------------------------------------------------------------------
import tk as _tk
from tk.quant import _e4m3_decode_arr as _dec

_BLK_SHAPES = [(2, 64, 512), (1, 256, 768), (8, 256), (4, 128, 1024)]


# The kernel normalizes the fp32 register sum (not the bf16-rounded res_out); res_out is the
# bf16-rounded sum. Weight/bias arrive as bf16. Twin computes both in fp32.
def _normed_rms(x, r, w, eps=1e-5):
    s = (mx.array(x).astype(mx.bfloat16).astype(mx.float32)
         + mx.array(r).astype(mx.bfloat16).astype(mx.float32))
    added = np.array(s.astype(mx.bfloat16).astype(mx.float32))
    ms = (s * s).mean(-1, keepdims=True)
    y = s * mx.rsqrt(ms + eps) * mx.array(w).astype(mx.bfloat16).astype(mx.float32)
    return np.array(y), added


def _normed_ln(x, r, w, b, eps=1e-5):
    s = (mx.array(x).astype(mx.bfloat16).astype(mx.float32)
         + mx.array(r).astype(mx.bfloat16).astype(mx.float32))
    added = np.array(s.astype(mx.bfloat16).astype(mx.float32))
    mean = s.mean(-1, keepdims=True)
    var = ((s - mean) ** 2).mean(-1, keepdims=True)
    y = ((s - mean) * mx.rsqrt(var + eps) * mx.array(w).astype(mx.bfloat16).astype(mx.float32)
         + mx.array(b).astype(mx.bfloat16).astype(mx.float32))
    return np.array(y), added


@pytest.mark.parametrize("shape", [(2, 64, 512), (8, 256), (4, 128, 1024)])
def test_layernorm_add_int8_dyn(shape):
    D = shape[-1]
    mx.random.seed(3)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    b = (0.1 * mx.random.normal((D,))).astype(mx.bfloat16)
    codes, added, scale = _tk.layernorm_add_int8(x, r, w, b)
    mx.eval(codes, added, scale)
    y, added_ref = _normed_ln(x, r, w, b)
    yf = y.reshape(-1, D)
    s = np.abs(yf).max(-1) / 127.0
    inv = np.where(s > 0, 1.0 / s, 0.0)
    ref = np.clip(np.rint(yf * inv[:, None]), -128, 127).astype(np.int8)
    got = np.array(codes).reshape(-1, D).astype(np.int32)
    # off-by-one where the fp32 rsqrt/weight chain flips a borderline int8 rounding
    assert np.abs(got - ref.astype(np.int32)).max() <= 1 and (got == ref).mean() > 0.97
    np.testing.assert_allclose(np.array(scale).reshape(-1), s, rtol=2e-2)
    np.testing.assert_array_equal(np.array(added.astype(mx.float32)), added_ref.astype(np.float32))


@pytest.mark.parametrize("shape", _BLK_SHAPES)
@pytest.mark.parametrize("norm", ["rms", "ln"])
def test_per_block_int8(shape, norm):
    D = shape[-1]
    G = 128
    mx.random.seed(4)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    b = (0.1 * mx.random.normal((D,))).astype(mx.bfloat16)
    if norm == "rms":
        codes, added, scale = _tk.rms_norm_add_per_block(x, r, w, int8=True)
        y, _ = _normed_rms(x, r, w)
    else:
        codes, added, scale = _tk.layernorm_add_per_block(x, r, w, b, int8=True)
        y, _ = _normed_ln(x, r, w, b)
    mx.eval(codes, added, scale)
    rows = int(np.prod(shape[:-1]))
    yb = y.reshape(rows, D // G, G)
    s = np.abs(yb).max(-1) / 127.0                       # (rows, D/G)
    inv = np.where(s > 0, 1.0 / s, 0.0)
    ref = np.clip(np.rint(yb * inv[..., None]), -128, 127).astype(np.int8).reshape(rows, D)
    assert np.array(scale).reshape(rows, D // G).shape == (rows, D // G)
    got = np.array(codes).reshape(rows, D).astype(np.int32)
    assert np.abs(got - ref.astype(np.int32)).max() <= 1 and (got == ref).mean() > 0.97
    np.testing.assert_allclose(np.array(scale).reshape(rows, D // G), s, rtol=2e-2)


@pytest.mark.parametrize("shape", _BLK_SHAPES)
@pytest.mark.parametrize("ue8m0", [False, True])
def test_per_block_fp8(shape, ue8m0):
    D = shape[-1]
    G = 128
    mx.random.seed(5)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    codes, added, scale = _tk.rms_norm_add_per_block(x, r, w, int8=False, ue8m0=ue8m0)
    mx.eval(codes, added, scale)
    y, _ = _normed_rms(x, r, w)
    rows = int(np.prod(shape[:-1]))
    yb = y.reshape(rows, D // G, G)
    sn = np.array(scale).reshape(rows, D // G)
    assert (sn * 448.0 >= np.abs(yb).max(-1) - 1e-2).all()          # covers block amax
    if ue8m0:
        exp = np.log2(np.where(sn > 0, sn, 1.0))
        np.testing.assert_array_equal(exp, np.round(exp))           # power-of-two scales
    recon = sn[..., None] * _dec(np.array(codes).reshape(rows, D // G, G))
    assert (np.abs(recon - yb) <= np.abs(yb) * 2.0 ** -4 + sn[..., None] * 2.0 ** -6 + 5e-2).all()
