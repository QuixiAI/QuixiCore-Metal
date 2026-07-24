"""Correctness tests for the sampler-zoo logit/prob transforms.

Strategy per the port plan: (a) EXACT-SET tests on margin-safe inputs — logits quantized to
a coarse 1/64 grid so no token sits within float-epsilon of a threshold; the float64 numpy
oracle then pins the kept/masked SET exactly; (b) property tests on continuous inputs;
(c) DRY / no-repeat-ngram compared against direct python transcriptions of the reference
loops (deterministic, exact token sets).

Run from kernels/:  python -m pytest sampling/correctness/test_transforms.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk

NEG = -1e30  # anything below this counts as masked


def _grid_logits(rng, T, V):
    return (np.round(rng.standard_normal((T, V)) * 3 * 64) / 64).astype(np.float32)


def _masked(a):
    return np.array(a) < NEG


def test_quadratic_transform():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 333)).astype(np.float32)
    got = tk.quadratic_transform(mx.array(x), factor=0.3, curve=1.5)
    mx.eval(got)
    mxv = x.max(1, keepdims=True)
    k = 0.3 * (3 - 1.5) / 2
    s = 0.3 * (1.5 - 1) / 2
    diff = (x - mxv).astype(np.float64)
    diff = diff - diff * diff * (s * diff - k)
    ref = mxv + diff
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4, rtol=1e-5)
    # factor 0 = identity
    got0 = tk.quadratic_transform(mx.array(x), factor=0.0)
    mx.eval(got0)
    np.testing.assert_allclose(np.array(got0), x, atol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_logits_softcap(dtype):
    rng = np.random.default_rng(41)
    x = (8.0 * rng.standard_normal((7, 513))).astype(np.float32)
    xd = mx.array(x).astype(dtype)
    got = tk.logits_softcap(xd, 30.0)
    mx.eval(got)
    rounded = np.array(xd.astype(mx.float32))
    ref = (30.0 * np.tanh(rounded / 30.0)).astype(
        np.float32 if dtype == mx.float32 else np.float16)
    tol = 2e-2 if dtype == mx.bfloat16 else 2e-3
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref.astype(np.float32),
                               atol=tol, rtol=tol)


def test_logits_softcap_rejects_invalid_cap():
    with pytest.raises(ValueError):
        tk.logits_softcap(mx.zeros((2, 4)), 0.0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_value_clip_arbitrary_shape_and_dtype(dtype):
    x = np.array([[[-4.0, -1.25, 0.0], [0.75, 2.5, 9.0]]], dtype=np.float32)
    xd = mx.array(x).astype(dtype)
    got = tk.value_clip(xd, -1.25, 2.5)
    mx.eval(got)
    assert got.shape == xd.shape
    assert got.dtype == dtype
    rounded = np.array(xd.astype(mx.float32))
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.clip(rounded, -1.25, 2.5))


def test_value_clip_infinite_bounds_and_validation():
    x = mx.array([[-2.0, 1.0]], dtype=mx.float32)
    np.testing.assert_array_equal(np.array(tk.value_clip(x, -np.inf, np.inf)), np.array(x))
    with pytest.raises(ValueError):
        tk.value_clip(x, 2.0, -2.0)
    with pytest.raises(ValueError):
        tk.value_clip(x, np.nan, 2.0)


def test_top_nsigma_exact_set():
    rng = np.random.default_rng(1)
    x = _grid_logits(rng, 16, 400)
    got = tk.top_nsigma_mask(mx.array(x), nsigma=1.5)
    mx.eval(got)
    thr = x.max(1) - 1.5 * x.std(1, ddof=1)
    ref_masked = x < thr[:, None] - 1e-6
    np.testing.assert_array_equal(_masked(got), ref_masked)
    kept = ~_masked(got)
    np.testing.assert_allclose(np.array(got)[kept], x[kept], atol=1e-6)


def test_top_a_exact_set():
    rng = np.random.default_rng(2)
    x = _grid_logits(rng, 16, 400)
    a = 0.2
    got = tk.top_a_mask(mx.array(x), top_a=a)
    mx.eval(got)
    p = np.exp(x - x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref_masked = (p < a * (p.max(1, keepdims=True) ** 2)) & (x < x.max(1, keepdims=True))
    np.testing.assert_array_equal(_masked(got), ref_masked)


def test_epsilon_cutoff_exact_set():
    rng = np.random.default_rng(3)
    x = _grid_logits(rng, 16, 400)
    eps = 3e-3
    got = tk.epsilon_cutoff_mask(mx.array(x), epsilon=eps)
    mx.eval(got)
    p = np.exp(x - x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref_masked = (p < eps) & (x < x.max(1, keepdims=True))
    np.testing.assert_array_equal(_masked(got), ref_masked)


def test_eta_cutoff_exact_set():
    rng = np.random.default_rng(4)
    x = _grid_logits(rng, 16, 400)
    eta = 2e-3
    got = tk.eta_cutoff_mask(mx.array(x), eta=eta)
    mx.eval(got)
    p = np.exp((x - x.max(1, keepdims=True)).astype(np.float64))
    p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(p)).sum(1)
    eps_eff = np.minimum(eta, np.sqrt(eta) * np.exp(-ent))
    ref_masked = (p < eps_eff[:, None]) & (x < x.max(1, keepdims=True))
    np.testing.assert_array_equal(_masked(got), ref_masked)


def test_xtc_semantics_and_coin():
    rng = np.random.default_rng(5)
    x = _grid_logits(rng, 32, 200)
    thr = 0.10
    # probability=1 -> always applied
    got = tk.xtc_mask(mx.array(x), threshold=thr, probability=1.0, seed=7)
    mx.eval(got)
    p = np.exp((x - x.max(1, keepdims=True)).astype(np.float64))
    p /= p.sum(1, keepdims=True)
    for t in range(x.shape[0]):
        elig = np.where(p[t] >= thr - 1e-12)[0]
        m = _masked(got)[t]
        if len(elig) <= 1:
            assert not m.any()
        else:
            pmin = p[t][elig].min()
            expect = set(int(i) for i in elig if p[t][i] > pmin + 1e-12)
            assert set(np.where(m)[0]) == expect
    # probability=0 -> identity (tempered copy)
    got0 = tk.xtc_mask(mx.array(x), threshold=thr, probability=0.0, seed=7)
    mx.eval(got0)
    np.testing.assert_allclose(np.array(got0), x, atol=1e-6)


def test_skew_transform():
    rng = np.random.default_rng(6)
    T, V = 8, 257
    p = rng.random((T, V)).astype(np.float32)
    p /= p.sum(1, keepdims=True)
    skew = 0.7
    got = tk.skew_transform(mx.array(p), skew=skew)
    mx.eval(got)
    e = np.exp(skew)
    cdf = np.cumsum(p.astype(np.float64), 1)
    ref = np.power(cdf, e) - np.power(np.concatenate([np.zeros((T, 1)), cdf[:, :-1]], 1), e)
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4)
    np.testing.assert_allclose(np.array(got).sum(1), np.power(cdf[:, -1], e), atol=1e-4)


def test_top_k_renorm():
    rng = np.random.default_rng(7)
    T, V, K = 16, 500, 12
    p = rng.random((T, V)).astype(np.float32)
    p = np.round(p * 64) / 64 + 1e-4          # margin-safe values, all distinct enough
    p /= p.sum(1, keepdims=True)
    got = tk.top_k_renorm(mx.array(p), K)
    mx.eval(got)
    gn = np.array(got, np.float64)
    for t in range(T):
        order = np.lexsort((np.arange(V), -p[t]))
        kept = set(order[:K].tolist())
        nz = set(np.where(gn[t] > 0)[0].tolist())
        assert nz == kept
    np.testing.assert_allclose(gn.sum(1), 1.0, atol=1e-5)


def test_top_p_renorm():
    rng = np.random.default_rng(8)
    T, V = 16, 500
    p = rng.random((T, V)).astype(np.float32) ** 4      # peaked
    p /= p.sum(1, keepdims=True)
    top_p = 0.8
    got = tk.top_p_renorm(mx.array(p), top_p)
    mx.eval(got)
    gn = np.array(got, np.float64)
    np.testing.assert_allclose(gn.sum(1), 1.0, atol=1e-5)
    for t in range(T):
        kept = gn[t] > 0
        # kept mass >= top_p and removing the smallest kept element drops below top_p
        assert p[t][kept].sum() >= top_p - 1e-4
        if kept.sum() > 1:
            smallest = p[t][kept].min()
            assert p[t][kept].sum() - smallest < top_p + 1e-4
    # top_p = 1 keeps everything
    got1 = tk.top_p_renorm(mx.array(p), 1.0)
    mx.eval(got1)
    assert (np.array(got1) > 0).all()


def test_no_repeat_ngram():
    rng = np.random.default_rng(9)
    T, V, L, n = 8, 64, 40, 3
    x = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(0, 16, (T, L)).astype(np.int32)   # small alphabet -> many repeats
    lens = rng.integers(n, L + 1, T).astype(np.int32)
    got = tk.no_repeat_ngram_mask(mx.array(x), mx.array(prev), mx.array(lens), n)
    mx.eval(got)
    for t in range(T):
        ln = int(lens[t])
        h = prev[t, :ln]
        banned = set()
        suffix = h[ln - (n - 1):ln].tolist()
        for s0 in range(0, ln - n + 1):
            if h[s0:s0 + n - 1].tolist() == suffix:
                banned.add(int(h[s0 + n - 1]))
        assert set(np.where(_masked(got)[t])[0]) == banned


def _dry_ref(x, prev, lens, breakers, mult, base, allowed, rng_, max_ngram, max_occ, early):
    """Direct transcription of the metal-forge dry_penalty loop."""
    T, V = x.shape
    out = x.astype(np.float64).copy()
    brk = set(int(b) for b in breakers if b >= 0)
    for t in range(T):
        ln = int(lens[t])
        if mult == 0.0 or ln < 2:
            continue
        h = prev[t, :ln]
        last = int(h[-1])
        if last in brk:
            continue
        start = max(0, ln - rng_) if rng_ > 0 else 0
        cmg = -1
        for gi in range(min(ln - start, max_ngram + 1)):
            if int(h[ln - gi - 1]) in brk:
                break
            cmg = gi
        if cmg <= allowed:
            continue
        seen = 0
        for idx in range(ln - 2, start - 1, -1):
            if int(h[idx]) != last:
                continue
            if seen >= max_occ:
                break
            seen += 1
            match_len = 0
            for unwind in range(1, min(idx - start, cmg) + 1):
                cand = int(h[idx - unwind])
                if cand in brk or cand != int(h[ln - unwind - 1]):
                    break
                match_len = unwind
            if match_len <= 0:
                continue
            nxt = int(h[idx + 1])
            if 0 <= nxt < V:
                new_len = match_len + 1
                pen = mult * base ** (new_len - allowed)
                out[t, nxt] = min(out[t, nxt], x[t, nxt] - pen)
                if new_len >= early:
                    break
    return out


def test_dry_penalty():
    rng = np.random.default_rng(10)
    T, V, L = 8, 64, 48
    x = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(0, 12, (T, L)).astype(np.int32)   # tiny alphabet -> repeats
    lens = rng.integers(8, L + 1, T).astype(np.int32)
    breakers = np.array([3, -1, -1, -1], np.int32)
    got = tk.dry_penalty(mx.array(x), mx.array(prev), mx.array(lens), mx.array(breakers),
                         multiplier=1.2, base=1.75, allowed_length=1, range=0,
                         max_ngram=16, max_occurrences=8, early_exit_match_len=16)
    mx.eval(got)
    ref = _dry_ref(x, prev, lens, breakers, 1.2, 1.75, 1, 0, 16, 8, 16)
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4, rtol=1e-5)
