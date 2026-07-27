"""Correctness for the reusable F16-adapter LoRA operation."""

import mlx.core as mx
import numpy as np
import pytest

import tk


_DTYPES = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}


def _ref(x, A, B, base, scale):
    low = (x.astype(np.float32) @ A.astype(np.float32).T).astype(np.float16)
    delta = (low.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
    out = delta.astype(np.float32) * np.float32(scale)
    if base is not None:
        out += base.astype(np.float32)
    return out


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 257, 193, 4), (4, 1024, 513, 16)])
@pytest.mark.parametrize("with_base", [False, True])
def test_lora_apply_direct(dtype, shape, with_base):
    M, K, N, R = shape
    rng = np.random.default_rng(M + K + N + R)
    x0 = (0.15 * rng.standard_normal((M, K))).astype(np.float32)
    A = (0.10 * rng.standard_normal((R, K))).astype(np.float16)
    B = (0.10 * rng.standard_normal((N, R))).astype(np.float16)
    base0 = (0.20 * rng.standard_normal((M, N))).astype(np.float32) if with_base else None
    xd = mx.array(x0).astype(_DTYPES[dtype])
    base = mx.array(base0).astype(_DTYPES[dtype]) if with_base else None
    got = tk.lora_apply_direct(xd, mx.array(A), mx.array(B), base=base, scale=0.75)
    mx.eval(got)
    x = np.array(xd.astype(mx.float32))
    base_ref = np.array(base.astype(mx.float32)) if with_base else None
    ref = _ref(x, A, B, base_ref, 0.75)
    tol = 8e-3 if dtype == "f32" else 3e-2
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=tol, rtol=tol)


def test_lora_apply_prefill_route_matches_explicit_f16_composition():
    rng = np.random.default_rng(90)
    M, K, N, R = 8, 384, 320, 12
    x = mx.array((0.2 * rng.standard_normal((M, K))).astype(np.float32)).astype(mx.bfloat16)
    A = mx.array((0.1 * rng.standard_normal((R, K))).astype(np.float16))
    B = mx.array((0.1 * rng.standard_normal((N, R))).astype(np.float16))
    base = mx.array((0.2 * rng.standard_normal((M, N))).astype(np.float32)).astype(mx.bfloat16)
    got = tk.lora_apply(x, A, B, base=base, scale=1.25)
    ref = (base.astype(mx.float32) +
           (x.astype(mx.float16) @ mx.transpose(A) @ mx.transpose(B)).astype(mx.float32) *
           1.25).astype(mx.bfloat16)
    mx.eval(got, ref)
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)))


def test_lora_apply_high_rank_decode_routes_to_framework_composition():
    rng = np.random.default_rng(91)
    M, K, N, R = 1, 384, 320, 32
    x = mx.array((0.2 * rng.standard_normal((M, K))).astype(np.float32)).astype(mx.bfloat16)
    A = mx.array((0.1 * rng.standard_normal((R, K))).astype(np.float16))
    B = mx.array((0.1 * rng.standard_normal((N, R))).astype(np.float16))
    got = tk.lora_apply(x, A, B, scale=0.5)
    ref = ((x.astype(mx.float16) @ mx.transpose(A) @ mx.transpose(B)).astype(mx.float32) *
           0.5).astype(mx.bfloat16)
    mx.eval(got, ref)
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)))


def test_lora_apply_rejects_non_f16_adapters():
    with pytest.raises(ValueError):
        tk.lora_apply_direct(
            mx.zeros((1, 32), dtype=mx.float32),
            mx.zeros((4, 32), dtype=mx.float32),
            mx.zeros((16, 4), dtype=mx.float16))
