"""Correctness tests for GDN / GatedDeltaNet linear attention (gdn_recur).

Oracle: fp64 numpy transcription of the delta rule, per (request, hv):
  S = g[t]*S;  kv = K[t] @ S;  delta = (V[t] - kv) * beta[t]  (per dv);
  S += outer(delta, K[t]);  y[t] = S @ Q[t]        with S laid out (Dv, Dk).

Run from kernels/:  python -m pytest gdn/correctness/test_gdn.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (
    gdn_gate_beta,
    gdn_gated_rmsnorm,
    gdn_qkv_prepare,
    gdn_recur,
    gdn_short_conv,
)

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
_ATOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}


def _ref(q, k, v, g, beta, pool, cu, slots, load_initial):
    """q/k (T, Hk, Dk); v (T, Hv, Dv); g/beta (T, Hv); pool (S, Hv, Dv, Dk)."""
    T, Hk, Dk = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    grp = Hv // Hk
    y = np.zeros((T, Hv, Dv))
    new_pool = pool.astype(np.float64).copy()
    for r in range(len(slots)):
        s0, e0 = int(cu[r]), int(cu[r + 1])
        slot = int(slots[r])
        for hv in range(Hv):
            hk = hv // grp
            S = new_pool[slot, hv].copy() if load_initial else np.zeros((Dv, Dk))
            for t in range(s0, e0):
                S = S * np.float64(g[t, hv])
                kv = S @ k[t, hk].astype(np.float64)               # (Dv,)
                delta = (v[t, hv].astype(np.float64) - kv) * np.float64(beta[t, hv])
                S = S + np.outer(delta, k[t, hk].astype(np.float64))
                y[t, hv] = S @ q[t, hk].astype(np.float64)
            new_pool[slot, hv] = S
    return y, new_pool


def _mk(rng, lens, Hk, Hv, Dk, Dv, nslots):
    T = sum(lens)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    q = (0.3 * rng.standard_normal((T, Hk, Dk))).astype(np.float32)
    k = (0.3 * rng.standard_normal((T, Hk, Dk))).astype(np.float32)
    v = (0.3 * rng.standard_normal((T, Hv, Dv))).astype(np.float32)
    g = rng.uniform(0.85, 1.0, (T, Hv)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, (T, Hv)).astype(np.float32)
    pool = (0.2 * rng.standard_normal((nslots, Hv, Dv, Dk))).astype(np.float32)
    slots = rng.permutation(nslots)[:len(lens)].astype(np.int32)
    return q, k, v, g, beta, pool, cu, slots


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("Dk,Dv,Hk,Hv", [(64, 64, 2, 4), (128, 128, 2, 8)])
def test_gdn_varlen_gqa(dtype, Dk, Dv, Hk, Hv):
    rng = np.random.default_rng(Dk + Hv)
    q, k, v, g, beta, pool, cu, slots = _mk(rng, [5, 8, 3], Hk, Hv, Dk, Dv, 6)
    md = _MX[dtype]
    to = lambda a: mx.array(a).astype(md)
    y, np_pool = gdn_recur(to(q), to(k), to(v), to(g), to(beta), mx.array(pool),
                           mx.array(cu), mx.array(slots), load_initial=True)
    mx.eval(y, np_pool)
    r = lambda a: np.array(mx.array(a).astype(md).astype(mx.float32))
    ref_y, ref_pool = _ref(r(q), r(k), r(v), r(g), r(beta), pool, cu, slots, True)
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), ref_y,
                               atol=_ATOL[dtype], rtol=_ATOL[dtype])
    np.testing.assert_allclose(np.array(np_pool), ref_pool, atol=1e-2, rtol=1e-2)
    # untouched pool slots preserved exactly
    untouched = [i for i in range(6) if i not in slots.tolist()]
    np.testing.assert_array_equal(np.array(np_pool)[untouched], pool[untouched])


def test_gdn_decode_step():
    """R single-token requests continuing from the pool (the decode shape)."""
    rng = np.random.default_rng(3)
    R, Hk, Hv, Dk, Dv = 16, 2, 8, 128, 128
    q, k, v, g, beta, pool, cu, slots = _mk(rng, [1] * R, Hk, Hv, Dk, Dv, R)
    y, np_pool = gdn_recur(mx.array(q), mx.array(k), mx.array(v), mx.array(g),
                           mx.array(beta), mx.array(pool), mx.array(cu), mx.array(slots),
                           load_initial=True)
    mx.eval(y, np_pool)
    ref_y, ref_pool = _ref(q, k, v, g, beta, pool, cu, slots, True)
    np.testing.assert_allclose(np.array(y), ref_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(np_pool), ref_pool, atol=1e-5, rtol=1e-5)


def test_gdn_fresh_prefill_ignores_pool():
    rng = np.random.default_rng(4)
    q, k, v, g, beta, pool, cu, slots = _mk(rng, [6, 4], 2, 4, 64, 64, 4)
    y, np_pool = gdn_recur(mx.array(q), mx.array(k), mx.array(v), mx.array(g),
                           mx.array(beta), mx.array(pool), mx.array(cu), mx.array(slots),
                           load_initial=False)
    mx.eval(y, np_pool)
    ref_y, ref_pool = _ref(q, k, v, g, beta, pool, cu, slots, False)
    np.testing.assert_allclose(np.array(y), ref_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(np_pool), ref_pool, atol=1e-5, rtol=1e-5)


def _short_conv_ref(x, weight, pool, cu, slots, load_initial, apply_silu):
    kernel_size = weight.shape[1]
    out = np.zeros_like(x, dtype=np.float32)
    new_pool = pool.copy()
    for request, slot in enumerate(slots):
        history = (pool[slot].copy() if load_initial
                   else np.zeros_like(pool[slot]))
        for token in range(int(cu[request]), int(cu[request + 1])):
            values = np.concatenate([history, x[token, :, None]], axis=1)
            y = np.sum(values * weight, axis=1, dtype=np.float32)
            if apply_silu:
                y = y / (1.0 + np.exp(-y))
            out[token] = y
            history = values[:, 1:]
        new_pool[slot] = history
    return out, new_pool


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("apply_silu", [False, True])
def test_gdn_short_conv_varlen_functional(dtype, apply_silu):
    rng = np.random.default_rng(101)
    lens, channels, kernel_size, slots_n = [1, 5, 2], 67, 4, 5
    total = sum(lens)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    slots = np.array([3, 0, 2], np.int32)
    x0 = (0.4 * rng.standard_normal((total, channels))).astype(np.float32)
    w0 = (0.3 * rng.standard_normal((channels, kernel_size))).astype(np.float32)
    pool = (0.2 * rng.standard_normal(
        (slots_n, channels, kernel_size - 1))).astype(np.float32)
    md = _MX[dtype]
    x = mx.array(x0).astype(md)
    weight = mx.array(w0).astype(md)
    out, new_pool = gdn_short_conv(
        x, weight, mx.array(pool), mx.array(cu), mx.array(slots),
        load_initial=True, apply_silu=apply_silu)
    mx.eval(out, new_pool)
    xr = np.array(x.astype(mx.float32))
    wr = np.array(weight.astype(mx.float32))
    ref_out, ref_pool = _short_conv_ref(
        xr, wr, pool, cu, slots, True, apply_silu)
    atol = 1e-5 if dtype == "float32" else 1.6e-2
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref_out,
                               atol=atol, rtol=atol)
    np.testing.assert_allclose(np.array(new_pool), ref_pool, atol=1e-6, rtol=0)
    untouched = [slot for slot in range(slots_n) if slot not in slots]
    np.testing.assert_array_equal(np.array(new_pool)[untouched], pool[untouched])
    np.testing.assert_array_equal(np.array(mx.array(pool)), pool)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("Dk,Dv", [(64, 64), (128, 128)])
def test_gdn_qkv_prepare(dtype, Dk, Dv):
    rng = np.random.default_rng(109 + Dk)
    tokens, Hk, Hv = 5, 2, 4
    channels = 2 * Hk * Dk + Hv * Dv
    mixed0 = (0.7 * rng.standard_normal((tokens, channels))).astype(np.float32)
    md = _MX[dtype]
    mixed = mx.array(mixed0).astype(md)
    q, k, v = gdn_qkv_prepare(mixed, Hk, Hv, Dk, Dv)
    mx.eval(q, k, v)
    m = np.array(mixed.astype(mx.float32))
    q0 = m[:, :Hk * Dk].reshape(tokens, Hk, Dk)
    k0 = m[:, Hk * Dk:2 * Hk * Dk].reshape(tokens, Hk, Dk)
    v0 = m[:, 2 * Hk * Dk:].reshape(tokens, Hv, Dv)
    qref = q0 / np.sqrt(np.mean(q0 * q0, axis=-1, keepdims=True) + 1e-6) / Dk
    kref = k0 / np.sqrt(np.mean(k0 * k0, axis=-1, keepdims=True) + 1e-6) / np.sqrt(Dk)
    atol = 2e-5 if dtype == "float32" else 5e-3
    np.testing.assert_allclose(np.array(q.astype(mx.float32)), qref, atol=atol, rtol=atol)
    np.testing.assert_allclose(np.array(k.astype(mx.float32)), kref, atol=atol, rtol=atol)
    np.testing.assert_array_equal(np.array(v.astype(mx.float32)), v0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_gdn_gate_beta(dtype):
    rng = np.random.default_rng(113)
    tokens, heads = 11, 8
    md = _MX[dtype]
    a = mx.array(rng.standard_normal((tokens, heads)).astype(np.float32)).astype(md)
    b = mx.array(rng.standard_normal((tokens, heads)).astype(np.float32)).astype(md)
    A_log = rng.uniform(-2.0, 1.5, heads).astype(np.float32)
    dt_bias = rng.uniform(-1.0, 1.0, heads).astype(np.float32)
    decay, beta = gdn_gate_beta(a, b, mx.array(A_log), mx.array(dt_bias))
    mx.eval(decay, beta)
    ar = np.array(a.astype(mx.float32))
    br = np.array(b.astype(mx.float32))
    softplus = np.logaddexp(0.0, ar + dt_bias)
    decay_ref = np.exp(-np.exp(A_log) * softplus)
    beta_ref = 1.0 / (1.0 + np.exp(-br))
    np.testing.assert_allclose(np.array(decay), decay_ref, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.array(beta), beta_ref, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("dim", [64, 128])
def test_gdn_gated_rmsnorm(dtype, dim):
    rng = np.random.default_rng(127 + dim)
    md = _MX[dtype]
    y = mx.array((0.6 * rng.standard_normal((7, 3, dim))).astype(np.float32)).astype(md)
    z = mx.array((0.5 * rng.standard_normal((7, 3, dim))).astype(np.float32)).astype(md)
    weight = mx.array(rng.uniform(0.7, 1.3, dim).astype(np.float32)).astype(md)
    out = gdn_gated_rmsnorm(y, z, weight)
    mx.eval(out)
    yr = np.array(y.astype(mx.float32))
    zr = np.array(z.astype(mx.float32))
    wr = np.array(weight.astype(mx.float32))
    ref = (yr / np.sqrt(np.mean(yr * yr, axis=-1, keepdims=True) + 1e-6))
    ref *= wr * (zr / (1.0 + np.exp(-zr)))
    atol = 2e-5 if dtype == "float32" else 1.6e-2
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref,
                               atol=atol, rtol=atol)
