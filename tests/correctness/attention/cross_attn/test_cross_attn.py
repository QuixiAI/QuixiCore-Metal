"""Correctness for independent-length GQA cross attention."""

import mlx.core as mx
import numpy as np
import pytest

import tk


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cross_attention(dtype, D, with_bias):
    rng = np.random.default_rng(100 + D + with_bias)
    B, Hq, Hkv, Tq, Tk = 2, 4, 2, 3, 11
    q0 = (0.15 * rng.standard_normal((B, Hq, Tq, D))).astype(np.float32)
    k0 = (0.15 * rng.standard_normal((B, Hkv, Tk, D))).astype(np.float32)
    v0 = (0.20 * rng.standard_normal((B, Hkv, Tk, D))).astype(np.float32)
    lengths = np.array([11, 7], np.int32)
    bias0 = (0.05 * rng.standard_normal((B, Hq, Tq, Tk))).astype(np.float32) if with_bias else None
    dt = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}[dtype]
    q = mx.array(q0).astype(dt); k = mx.array(k0).astype(dt); v = mx.array(v0).astype(dt)
    bias = mx.array(bias0) if with_bias else None
    got = tk.cross_attention(q, k, v, mx.array(lengths), bias=bias, softcap=3.0)
    mx.eval(got)
    qb, kb, vb = (np.array(z.astype(mx.float32)) for z in (q, k, v))
    ref = np.zeros_like(qb)
    scale = D ** -0.5
    for b in range(B):
        for h in range(Hq):
            hk = h // (Hq // Hkv)
            scores = qb[b, h] @ kb[b, hk, :lengths[b]].T * scale
            if with_bias: scores += bias0[b, h, :, :lengths[b]]
            scores = 3.0 * np.tanh(scores / 3.0)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores); probs /= probs.sum(axis=-1, keepdims=True)
            ref[b, h] = probs @ vb[b, hk, :lengths[b]]
    tol = 2e-5 if dtype == "f32" else 3e-2
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=tol, rtol=tol)


def test_cross_attention_empty_keys_emit_zero():
    q = mx.ones((1, 2, 1, 64), dtype=mx.bfloat16)
    k = mx.ones((1, 1, 5, 64), dtype=mx.bfloat16)
    v = mx.ones((1, 1, 5, 64), dtype=mx.bfloat16)
    got = tk.cross_attention(q, k, v, mx.array([0], dtype=mx.int32))
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), np.zeros((1, 2, 1, 64)))


def test_cross_attention_long_memory_framework_route():
    """The measured public route remains correct above the direct-kernel threshold."""
    rng = np.random.default_rng(301)
    B, Hq, Hkv, Tq, Tk, D = 2, 4, 2, 3, 257, 64
    q = mx.array((0.15 * rng.standard_normal((B, Hq, Tq, D))).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array((0.15 * rng.standard_normal((B, Hkv, Tk, D))).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array((0.20 * rng.standard_normal((B, Hkv, Tk, D))).astype(np.float32)).astype(mx.bfloat16)
    lengths = mx.array([Tk, 211], dtype=mx.int32)

    got = tk.cross_attention(q, k, v, lengths)
    forced_framework = tk.cross_attention(q, k, v, lengths, use_kernel=False)
    mx.eval(got, forced_framework)
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)),
        np.array(forced_framework.astype(mx.float32)))
