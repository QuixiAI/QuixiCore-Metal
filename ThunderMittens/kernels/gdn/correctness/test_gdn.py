"""Correctness tests for GDN / GatedDeltaNet linear attention (gdn_recur).

Oracle: fp64 numpy transcription of the delta rule, per (request, hv):
  S = g[t]*S;  kv = K[t] @ S;  delta = (V[t] - kv) * beta[t]  (per dv);
  S += outer(delta, K[t]);  y[t] = S @ Q[t]        with S laid out (Dv, Dk).

Run from kernels/:  python -m pytest gdn/correctness/test_gdn.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import gdn_recur

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
