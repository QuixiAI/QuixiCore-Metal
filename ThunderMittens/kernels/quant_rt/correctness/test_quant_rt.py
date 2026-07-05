"""Correctness tests for the runtime per-token GPU quantizers (fp8 e4m3, int8).

Validates: (1) per-row scale == absmax/QMAX exactly, and (2) the round-to-nearest
reconstruction error is within half a quantization step everywhere.

Run from kernels/:  python -m pytest quant_rt/correctness/test_quant_rt.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (quantize_per_token_fp8, quantize_per_token_int8,
                quantize_per_tensor_fp8, quantize_per_tensor_int8)
from tk.quant import _e4m3_decode_arr

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
SHAPES = [(8, 256), (4, 64, 128), (3, 513)]  # last is non-multiple-of-32 hidden


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_token_fp8(dtype, shape):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_token_fp8(xq)
    mx.eval(codes, scale)

    xd = np.array(xq.astype(mx.float32)).reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref_scale = amax / 448.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]

    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)

    deq = _e4m3_decode_arr(np.array(codes).reshape(-1, D)) * ssafe
    # RNE error <= half ULP: 2^-4 relative for e4m3 normals, + 2 subnormal steps near zero.
    tol = 0.0625 * np.abs(xd) + 2.0 * ssafe
    assert np.all(np.abs(deq - xd) <= tol), \
        f"max excess {(np.abs(deq - xd) - tol).max()}"


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_token_int8(dtype, shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_token_int8(xq)
    mx.eval(codes, scale)

    xd = np.array(xq.astype(mx.float32)).reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref_scale = amax / 127.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]

    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)

    c = np.array(codes).astype(np.int32)
    assert c.min() >= -127 and c.max() <= 127
    deq = c.reshape(-1, D).astype(np.float32) * ssafe
    # round-to-nearest int: error <= half a step (= half the scale).
    assert np.all(np.abs(deq - xd) <= 0.5 * ssafe + 1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_tensor_fp8(dtype, shape):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_tensor_fp8(xq)
    mx.eval(codes, scale)
    xd = np.array(xq.astype(mx.float32))
    ref_scale = np.abs(xd).max() / 448.0
    np.testing.assert_allclose(float(np.array(scale).reshape(-1)[0]), ref_scale, rtol=1e-3, atol=1e-8)
    ssafe = max(ref_scale, 1e-30)
    deq = _e4m3_decode_arr(np.array(codes)) * ssafe
    assert np.all(np.abs(deq - xd) <= 0.0625 * np.abs(xd) + 2.0 * ssafe)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_tensor_int8(dtype, shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_tensor_int8(xq)
    mx.eval(codes, scale)
    xd = np.array(xq.astype(mx.float32))
    ref_scale = np.abs(xd).max() / 127.0
    np.testing.assert_allclose(float(np.array(scale).reshape(-1)[0]), ref_scale, rtol=1e-3, atol=1e-8)
    ssafe = max(ref_scale, 1e-30)
    c = np.array(codes).astype(np.int32)
    assert c.min() >= -127 and c.max() <= 127
    deq = c.astype(np.float32) * ssafe
    assert np.all(np.abs(deq - xd) <= 0.5 * ssafe + 1e-6)


if __name__ == "__main__":
    for shp in SHAPES:
        test_quantize_per_token_fp8("float32", shp)
        test_quantize_per_tensor_fp8("float32", shp)
        print("ok", shp)


# ---------------------------------------------------------------------------
# Per-group + asymmetric (azp) quantizers — codes bit-exact vs a numpy twin that
# replicates the kernel's fp32 arithmetic order exactly (no transcendentals in
# the int8 paths; the fp8 path shares tk's e4m3 nearest-decoded-value grid).
# ---------------------------------------------------------------------------

def _np_e4m3_encode_arr(v):
    from tk.quant import _E4M3_CODES, _E4M3_VALS, _nearest
    return _nearest(v.astype(np.float32), _E4M3_CODES, _E4M3_VALS)


def test_quantize_per_group_int8_exact():
    from tk import quantize_per_group_int8
    rng = np.random.default_rng(40)
    T, D, G = 33, 384, 128
    x = rng.standard_normal((T, D)).astype(np.float32)
    codes, scale = quantize_per_group_int8(mx.array(x), group_size=G)
    mx.eval(codes, scale)
    xg = x.reshape(T, D // G, G)
    amax = np.abs(xg).max(-1)
    s = (amax / 127.0).astype(np.float32)
    inv = np.where(s > 0, 1.0 / s, 0.0).astype(np.float32)
    ref = np.clip(np.rint(xg * inv[..., None]), -128, 127).astype(np.int8).reshape(T, D)
    np.testing.assert_array_equal(np.array(codes), ref)
    np.testing.assert_array_equal(np.array(scale), s)


def test_quantize_per_group_fp8_ue8m0():
    from tk import quantize_per_group_fp8
    rng = np.random.default_rng(41)
    T, D, G = 16, 256, 128
    x = rng.standard_normal((T, D)).astype(np.float32)
    codes, scale = quantize_per_group_fp8(mx.array(x), group_size=G, ue8m0=True)
    mx.eval(codes, scale)
    sn = np.array(scale)
    # every scale is a power of two and >= amax/448
    amax = np.abs(x.reshape(T, D // G, G)).max(-1)
    exp = np.log2(sn)
    np.testing.assert_array_equal(exp, np.round(exp))
    assert (sn * 448.0 >= amax - 1e-5).all()
    # reconstruction within one fp8 step of the group scale
    from tk.quant import _e4m3_decode_arr
    recon = sn[..., None].repeat(G, -1).reshape(T, D) * _e4m3_decode_arr(np.array(codes))
    # e4m3 half-ulp relative error is 2^-4 (3 mantissa bits) + a subnormal floor
    assert (np.abs(recon - x) <= np.abs(x) * (2.0 ** -4) + sn.max() * 2.0 ** -6 + 1e-6).all()


def test_quantize_per_token_int8_azp_exact():
    from tk import quantize_per_token_int8_azp
    rng = np.random.default_rng(42)
    T, D = 65, 192
    x = (rng.standard_normal((T, D)) + 0.7).astype(np.float32)   # asymmetric distribution
    codes, scale, azp = quantize_per_token_int8_azp(mx.array(x))
    mx.eval(codes, scale, azp)
    mn, mxv = x.min(1), x.max(1)
    s = ((mxv - mn) / 255.0).astype(np.float32)
    inv = (1.0 / s).astype(np.float32)
    a = np.rint(-128.0 - mn * inv).astype(np.int32)
    ref = np.clip(np.rint(x * inv[:, None]).astype(np.int64) + a[:, None], -128, 127)
    np.testing.assert_array_equal(np.array(codes), ref.astype(np.int8))
    np.testing.assert_array_equal(np.array(scale), s)
    np.testing.assert_array_equal(np.array(azp), a)
    # reconstruction error bounded by scale/2 per element
    recon = s[:, None] * (np.array(codes, np.float32) - a[:, None])
    assert np.abs(recon - x).max() <= (s.max() / 2) * 1.01 + 1e-6


def test_qgemm_w8a8_azp_int_exact():
    """Integer path exact: kernel result equals the fp math on int32-exact accumulators."""
    from tk import quantize_per_token_int8_azp, qgemm_w8a8_azp
    rng = np.random.default_rng(43)
    N, K, M = 64, 256, 16
    Wint = rng.integers(-127, 128, (N, K)).astype(np.int8)
    x = (rng.standard_normal((M, K)) + 0.5).astype(np.float32)
    xq, s_a, azp = quantize_per_token_int8_azp(mx.array(x))
    mx.eval(xq, s_a, azp)
    w_scale = (0.01 * (1.0 + rng.random(N))).astype(np.float32)
    w_rowsum = Wint.astype(np.int32).sum(1)
    y = qgemm_w8a8_azp(mx.array(Wint), xq, mx.array(w_scale).astype(mx.float16), s_a,
                       mx.array(w_rowsum), azp)
    mx.eval(y)
    acc = Wint.astype(np.int64) @ np.array(xq).astype(np.int64).T          # (N, M)
    corr = acc - np.array(azp)[None, :].astype(np.int64) * w_rowsum[:, None]
    ref = (corr.astype(np.float64)
           * np.array(mx.array(w_scale).astype(mx.float16).astype(mx.float32))[:, None]
           * np.array(s_a)[None, :])
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), ref, atol=1e-2, rtol=1e-3)
