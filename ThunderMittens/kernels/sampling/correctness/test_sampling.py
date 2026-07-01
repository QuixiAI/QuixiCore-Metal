"""Correctness tests for the sampling kernels.

argmax (greedy): token index of the max logit over the vocab axis; ties resolve
to the smallest index (== numpy argmax first-occurrence).

Run from kernels/:  python -m pytest sampling/correctness/test_sampling.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import argmax_sample, sample_categorical, top_k_sample, top_p_sample, apply_penalty


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _nucleus(logits, p):
    sm = _softmax(logits)
    order = np.argsort(-sm)
    csum = np.cumsum(sm[order])
    n = int(np.searchsorted(csum, p)) + 1
    return order[:n], sm

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(4, 1000), (8, 32000), (2, 3, 257)])
def test_argmax_sample(dtype, shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    got = argmax_sample(xq)
    mx.eval(got)
    xd = np.array(xq.astype(mx.float32))
    ref = np.argmax(xd, axis=-1).astype(np.int32)
    assert np.array_equal(np.array(got).reshape(ref.shape), ref)


def test_sample_categorical_distribution():
    # Each row shares the same logits but a distinct RNG stream (row index), so the
    # empirical token frequencies must converge to softmax(logits).
    V = 8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = sample_categorical(mx.array(x), temperature=1.0, seed=1234)
    mx.eval(got)
    idx = np.array(got).reshape(-1)
    freq = np.bincount(idx, minlength=V).astype(np.float64) / N
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert np.max(np.abs(freq - p)) < 0.02, f"freq {freq} vs p {p}"


def test_sample_categorical_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 100)).astype(np.float32))
    a = sample_categorical(x, temperature=0.8, seed=7)
    b = sample_categorical(x, temperature=0.8, seed=7)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_sample_categorical_temperature_flattens():
    # High temperature -> closer to uniform than low temperature.
    V = 16
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal(V) * 2).astype(np.float32)
    N = 20000
    x = np.broadcast_to(logits, (N, V)).copy()
    hot = np.bincount(np.array(sample_categorical(mx.array(x), temperature=5.0, seed=3)).reshape(-1),
                      minlength=V) / N
    cold = np.bincount(np.array(sample_categorical(mx.array(x), temperature=0.5, seed=3)).reshape(-1),
                       minlength=V) / N
    # entropy(hot) > entropy(cold)
    ent = lambda q: -np.sum(np.where(q > 0, q * np.log(q + 1e-12), 0.0))
    assert ent(hot) > ent(cold)


@pytest.mark.parametrize("K", [1, 5, 40])
def test_top_k_sample_in_topk(K):
    rng = np.random.default_rng(0)
    T, V = 200, 1000
    x = rng.standard_normal((T, V)).astype(np.float32)
    got = np.array(top_k_sample(mx.array(x), K, temperature=1.0, seed=42)).reshape(-1)
    topk_ids = np.argsort(-x, axis=1)[:, :K]
    for t in range(T):
        assert got[t] in topk_ids[t]


def test_top_k_sample_k1_is_argmax():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 500)).astype(np.float32)
    got = np.array(top_k_sample(mx.array(x), 1, seed=99)).reshape(-1)
    assert np.array_equal(got, np.argmax(x, axis=1))


def test_top_k_sample_distribution():
    V, K = 50, 5
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = np.array(top_k_sample(mx.array(x), K, temperature=1.0, seed=7)).reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    order = np.argsort(-logits)[:K]
    p = np.zeros(V)
    ex = np.exp(logits[order] - logits[order].max())
    p[order] = ex / ex.sum()
    assert np.max(np.abs(freq - p)) < 0.02


def test_top_k_sample_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 200)).astype(np.float32))
    a = top_k_sample(x, 8, seed=3)
    b = top_k_sample(x, 8, seed=3)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("p", [0.5, 0.9, 0.99])
def test_top_p_sample_in_nucleus(p):
    rng = np.random.default_rng(0)
    T, V = 200, 500
    x = rng.standard_normal((T, V)).astype(np.float32)
    got = np.array(top_p_sample(mx.array(x), p, temperature=1.0, seed=42)).reshape(-1)
    for t in range(T):
        nuc, _ = _nucleus(x[t], p)
        assert got[t] in set(nuc.tolist())


def test_top_p_sample_small_p_is_argmax():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 500)).astype(np.float32)
    got = np.array(top_p_sample(mx.array(x), 0.001, seed=99)).reshape(-1)
    assert np.array_equal(got, np.argmax(x, axis=1))


def test_top_p_sample_distribution():
    V, p = 40, 0.8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = np.array(top_p_sample(mx.array(x), p, temperature=1.0, seed=7)).reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    nuc, sm = _nucleus(logits, p)
    pn = np.zeros(V)
    pn[nuc] = sm[nuc] / sm[nuc].sum()
    assert np.max(np.abs(freq - pn)) < 0.02


def test_top_p_sample_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 200)).astype(np.float32))
    a = top_p_sample(x, 0.9, seed=3)
    b = top_p_sample(x, 0.9, seed=3)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def _ref_penalty(ld, prev, temp, rep, presence, freq,
                 bias=None, eos_id=-1, min_length=0, gen_len=0):
    T, V = ld.shape
    ref = ld / temp
    for t in range(T):
        c = np.zeros(V)
        for tok in prev[t]:
            if 0 <= tok < V:
                c[int(tok)] += 1
        for v in range(V):
            if c[v] > 0:
                l = ref[t, v]
                l = l * rep if l < 0 else l / rep
                l -= presence
                l -= freq * c[v]
                ref[t, v] = l
    if bias is not None:
        ref = ref + bias[None, :]
    if eos_id >= 0 and gen_len < min_length:
        ref[:, eos_id] = -np.inf
    return ref


def test_apply_penalty_bias_minlen():
    rng = np.random.default_rng(3)
    T, V, L = 8, 500, 40
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)
    bias = rng.standard_normal(V).astype(np.float32)
    kw = dict(temperature=0.8, repetition_penalty=1.3, presence_penalty=0.1, frequency_penalty=0.05)
    eos_id = 7
    # gen_len < min_length -> EOS forbidden
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev), bias=mx.array(bias),
                                 eos_id=eos_id, min_length=10, gen_len=5, **kw))
    ref = _ref_penalty(logits, prev, kw["temperature"], kw["repetition_penalty"],
                       kw["presence_penalty"], kw["frequency_penalty"],
                       bias=bias, eos_id=eos_id, min_length=10, gen_len=5)
    m = np.arange(V) != eos_id
    np.testing.assert_allclose(got[:, m], ref[:, m], atol=1e-4, rtol=2e-3)
    assert np.all(got[:, eos_id] < -1e30)                 # EOS masked
    # gen_len >= min_length -> EOS not masked
    got2 = np.array(apply_penalty(mx.array(logits), mx.array(prev), bias=mx.array(bias),
                                  eos_id=eos_id, min_length=10, gen_len=15, **kw))
    assert got2[0, eos_id] > -1e30


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_apply_penalty(dtype):
    rng = np.random.default_rng(0)
    T, V, L = 8, 500, 40
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)  # -1 = padding (ignored)
    temp, rep, presence, freq = 0.8, 1.3, 0.1, 0.05
    got = np.array(apply_penalty(
        mx.array(logits).astype(_MX[dtype]), mx.array(prev),
        temperature=temp, repetition_penalty=rep,
        presence_penalty=presence, frequency_penalty=freq).astype(mx.float32))
    ld = np.array(mx.array(logits).astype(_MX[dtype]).astype(mx.float32))
    ref = _ref_penalty(ld, prev, temp, rep, presence, freq)
    atol = 1e-4 if dtype == "float32" else 3e-2
    np.testing.assert_allclose(got, ref, atol=atol, rtol=2e-3)


def test_apply_penalty_identity():
    # temperature=1, rep=1, presence=freq=0 -> logits unchanged.
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((4, 300)).astype(np.float32)
    prev = rng.integers(0, 300, size=(4, 20)).astype(np.int32)
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev)))
    np.testing.assert_allclose(got, logits, atol=1e-5)


def test_apply_penalty_beam_parent():
    # Beam search: each row's occurrence history is gathered from its parent beam (parent_ids).
    rng = np.random.default_rng(4)
    T, V, L = 6, 200, 12
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(0, V, size=(T, L)).astype(np.int32)
    parent = np.array([0, 1, 0, 1, 2, 3], dtype=np.int32)   # rows inherit from earlier rows
    temp, rep, presence, freq = 0.8, 1.3, 0.1, 0.05
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev), temperature=temp,
                                 repetition_penalty=rep, presence_penalty=presence,
                                 frequency_penalty=freq, parent_ids=mx.array(parent)))
    # Reference: each row t uses its OWN logits but the parent beam's history prev[parent[t]].
    ref = _ref_penalty(logits, prev[parent], temp, rep, presence, freq)
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=2e-3)
    # Sanity: differs from the identity (non-beam) result on the redirected rows.
    ident = np.array(apply_penalty(mx.array(logits), mx.array(prev), temperature=temp,
                                   repetition_penalty=rep, presence_penalty=presence,
                                   frequency_penalty=freq))
    assert np.max(np.abs(got - ident)) > 1e-2


if __name__ == "__main__":
    for shp in [(4, 1000), (8, 32000), (2, 3, 257)]:
        test_argmax_sample("float32", shp)
        print("ok", shp)
    test_sample_categorical_distribution()
    test_top_k_sample_distribution()
    test_top_p_sample_distribution()
    test_apply_penalty("float32")
    print("ok sampling")
