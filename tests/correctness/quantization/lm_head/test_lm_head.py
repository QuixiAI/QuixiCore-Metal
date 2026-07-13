"""Correctness tests for the fused LM-head + sampling kernels.

tk.lm_head_sample(h, W, mode, ...) selects a decode token per row of h WITHOUT materializing the
(T, V) logits. The oracle computes the full logits (h @ W.T) from the SAME rounded inputs the kernel
sees and runs the reference sampler; the Gumbel noise is indexed by the global vocab id so the fused
draw matches the unfused sampler. Because the fused serial dot differs from a numpy dot by ULPs, the
selection is validated with a tie tolerance (a fused pick whose logit is within eps of the winner is
a valid draw).

Run from kernels/:  python -m pytest lm_head/correctness/test_lm_head.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _rng_uniform(seed, a, b):
    x = (np.uint32(seed) * np.uint32(0x9E3779B9) + np.uint32(a) * np.uint32(0x85EBCA77)
         + np.uint32(b) * np.uint32(0xC2B2AE3D)).astype(np.uint32)
    x ^= x >> np.uint32(16); x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15); x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return np.float32(x >> np.uint32(8)) * np.float32(1.0 / 16777216.0)


def _gumbel(seed, a, b):
    u = max(float(_rng_uniform(seed, a, b)), 1e-20)
    return -np.log(-np.log(u))


def _logits(hm, Wm):
    hb = np.array(hm.astype(mx.float32)).astype(np.float64)
    Wb = np.array(Wm.astype(mx.float32)).astype(np.float64)
    return hb @ Wb.T


def _mk(T, V, K, dtype, seed=0):
    rng = np.random.default_rng(seed)
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    W = (0.5 * rng.standard_normal((V, K))).astype(np.float32)
    return mx.array(h).astype(_MX[dtype]), mx.array(W).astype(_MX[dtype])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("T,V,K", [(1, 32000, 2048), (8, 1000, 512), (4, 128256, 1024)])
def test_argmax(dtype, T, V, K):
    hm, Wm = _mk(T, V, K, dtype)
    tok = np.array(tk.lm_head_sample(hm, Wm, mode="argmax"))
    L = _logits(hm, Wm)
    assert tok.shape == (T,)
    for t in range(T):
        assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-3


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("T,V,K", [(4, 32000, 2048), (8, 1000, 512)])
def test_categorical(dtype, T, V, K):
    hm, Wm = _mk(T, V, K, dtype)
    temp, seed = 0.8, 123
    tok = np.array(tk.lm_head_sample(hm, Wm, mode="categorical", temperature=temp, seed=seed))
    L = _logits(hm, Wm)
    for t in range(T):
        P = L[t] / temp + np.array([_gumbel(seed, t, v) for v in range(V)])
        assert tok[t] == P.argmax() or (P.max() - P[tok[t]]) < 2e-3


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("k", [1, 8, 40])
def test_topk(dtype, k):
    T, V, K = 4, 32000, 1024
    hm, Wm = _mk(T, V, K, dtype)
    temp, seed = 0.7, 7
    tok = np.array(tk.lm_head_sample(hm, Wm, mode="topk", k=k, temperature=temp, seed=seed))
    L = _logits(hm, Wm)
    for t in range(T):
        top = set(int(v) for v in np.argsort(-L[t], kind="stable")[:k])
        # the picked token is one of the top-k (boundary ties tolerated), and if k==1 it's the argmax
        assert tok[t] in top or (L[t].max() - L[t, tok[t]]) < 1e-3
        if k == 1:
            assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-3


def test_bias():
    T, V, K = 2, 500, 256
    hm, Wm = _mk(T, V, K, "float32")
    rng = np.random.default_rng(1)
    bias = rng.standard_normal(V).astype(np.float32)
    tok = np.array(tk.lm_head_sample(hm, Wm, mode="argmax", bias=mx.array(bias)))
    L = _logits(hm, Wm) + bias[None]
    for t in range(T):
        assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-3


def test_matches_argmax_sample():
    # Fused argmax == materialize-logits + tk.argmax_sample (same rounded logits path).
    T, V, K = 4, 2000, 512
    hm, Wm = _mk(T, V, K, "float32", seed=3)
    fused = np.array(tk.lm_head_sample(hm, Wm, mode="argmax"))
    L = mx.matmul(hm, Wm.T)
    unfused = np.array(tk.argmax_sample(L))
    Ln = np.array(L)
    for t in range(T):
        assert fused[t] == unfused[t] or abs(Ln[t, fused[t]] - Ln[t, unfused[t]]) < 1e-3


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode,k", [("argmax", 0), ("categorical", 0), ("topk", 8)])
@pytest.mark.parametrize("T,V,K", [(1, 32000, 2048), (8, 32000, 4096), (4, 32000, 2049)])
def test_fused_kernel_vs_oracle(dtype, mode, k, T, V, K):
    # The no-materialization fused=True kernel (vec4 path + a K%4!=0 scalar-tail case) must match the
    # logits oracle just like the default matmul path (tie-tolerant; global-vocab Gumbel).
    hm, Wm = _mk(T, V, K, dtype)
    tok = np.array(tk.lm_head_sample(hm, Wm, mode=mode, k=k, temperature=0.8, seed=5, fused=True))
    L = _logits(hm, Wm)
    for t in range(T):
        if mode == "argmax":
            assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-3
        elif mode == "categorical":
            P = L[t] / 0.8 + np.array([_gumbel(5, t, v) for v in range(V)])
            assert tok[t] == P.argmax() or (P.max() - P[tok[t]]) < 3e-3
        else:
            top = set(int(v) for v in np.argsort(-L[t], kind="stable")[:k])
            assert tok[t] in top or (L[t].max() - L[t, tok[t]]) < 1e-3


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["argmax", "categorical"])
def test_quant(fmt, dtype, mode):
    # Fused LM-head over quantized weights vs the dequantize(Wq) @ h oracle (tie-tolerant).
    from tk.quant import QUANT_FORMATS
    quant, dequant = QUANT_FORMATS[fmt]
    T, V, K = 4, 4000, 512
    rng = np.random.default_rng(11)
    W = (0.3 * rng.standard_normal((V, K))).astype(np.float32)
    Wq = quant(W)
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    hm = mx.array(h).astype(_MX[dtype])
    tok = np.array(tk.lm_head_sample(hm, mx.array(Wq), mode=mode, temperature=0.8, seed=3,
                                     format=fmt))
    Wdq = dequant(Wq).astype(np.float64)
    L = np.array(hm.astype(mx.float32)).astype(np.float64) @ Wdq.T   # dequant-then-matmul oracle
    for t in range(T):
        if mode == "argmax":
            assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-2
        else:
            P = L[t] / 0.8 + np.array([_gumbel(3, t, v) for v in range(V)])
            assert tok[t] == P.argmax() or (P.max() - P[tok[t]]) < 1e-2


@pytest.mark.parametrize("mode", ["argmax", "categorical"])
@pytest.mark.parametrize("fused", [False, True])
def test_q6_k_float32(mode, fused):
    """The Q6_K LM-head path supports fp32 argmax and categorical decode."""
    from tk.quant import quantize_q6_K, dequantize_q6_K
    T, V, K = 3, 1025, 512
    rng = np.random.default_rng(57)
    weight = (0.2 * rng.standard_normal((V, K))).astype(np.float32)
    packed = quantize_q6_K(weight)
    h = (0.25 * rng.standard_normal((T, K))).astype(np.float32)
    temperature, seed = 0.8, 19
    tok = np.array(tk.lm_head_sample(
        mx.array(h), mx.array(packed), mode=mode, temperature=temperature,
        seed=seed, format="q6_K", fused=fused))
    logits = h.astype(np.float64) @ dequantize_q6_K(packed).astype(np.float64).T
    for t in range(T):
        scores = logits[t]
        if mode == "categorical":
            scores = scores / temperature + np.array([_gumbel(seed, t, v) for v in range(V)])
        assert tok[t] == scores.argmax() or (scores.max() - scores[tok[t]]) < 2e-3


@pytest.mark.parametrize("forbid_eos", [False, True])
def test_lm_head_constrained_matches_post_logsoftmax_mask(forbid_eos):
    rng = np.random.default_rng(83 + forbid_eos)
    T, V, K, eos_id = 4, 67, 31, 2
    h = (0.15 * rng.standard_normal((T, K))).astype(np.float32)
    weight = (0.2 * rng.standard_normal((V, K))).astype(np.float32)
    bias = (0.04 * rng.standard_normal(V)).astype(np.float32)
    forbidden = (rng.random((V, V)) < 0.18).astype(np.uint8)
    previous = np.array([0, 7, 31, 66], np.int32)
    # Keep at least one legal token per exercised grammar row.
    forbidden[previous, 0] = 0
    got_token, got_logprob = tk.lm_head_constrained(
        mx.array(h), mx.array(weight), mx.array(forbidden), mx.array(previous),
        bias=mx.array(bias), eos_id=eos_id, forbid_eos=forbid_eos)
    mx.eval(got_token, got_logprob)

    logits = (h @ weight.T).astype(np.float32)
    logits = (logits + bias).astype(np.float32)
    maximum = logits.max(-1, keepdims=True)
    log_z = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(-1))
    expected = []
    for t in range(T):
        allowed = forbidden[previous[t]] == 0
        if forbid_eos:
            allowed[eos_id] = False
        masked = np.where(allowed, logits[t], -np.inf)
        expected.append(int(masked.argmax()))
    expected = np.array(expected, np.int32)
    expected_logprob = logits[np.arange(T), expected] - log_z
    np.testing.assert_array_equal(np.array(got_token), expected)
    np.testing.assert_allclose(np.array(got_logprob), expected_logprob, rtol=3e-4, atol=3e-4)


def test_lm_head_constrained_no_legal_token_or_invalid_previous():
    rng = np.random.default_rng(89)
    T, V, K = 3, 17, 13
    h = (0.1 * rng.standard_normal((T, K))).astype(np.float32)
    weight = (0.1 * rng.standard_normal((V, K))).astype(np.float32)
    forbidden = np.ones((V, V), np.uint8)
    previous = np.array([-1, V, 0], np.int32)
    token, logprob = tk.lm_head_constrained(
        mx.array(h), mx.array(weight), mx.array(forbidden), mx.array(previous))
    mx.eval(token, logprob)
    np.testing.assert_array_equal(np.array(token), np.full(T, -1, np.int32))
    np.testing.assert_array_equal(np.array(logprob), np.full(T, -np.inf, np.float32))


def test_lm_head_constrained_bfloat16_path():
    rng = np.random.default_rng(97)
    T, V, K = 3, 29, 21
    h = mx.array((0.12 * rng.standard_normal((T, K))).astype(np.float32)).astype(mx.bfloat16)
    weight = mx.array((0.15 * rng.standard_normal((V, K))).astype(np.float32)).astype(
        mx.bfloat16)
    bias = mx.array((0.03 * rng.standard_normal(V)).astype(np.float32)).astype(mx.bfloat16)
    forbidden = (rng.random((V, V)) < 0.2).astype(np.uint8)
    previous = np.array([0, 7, 17], np.int32)
    forbidden[previous, 0] = 0
    token, logprob = tk.lm_head_constrained(
        h, weight, mx.array(forbidden), mx.array(previous), bias=bias)
    mx.eval(token, logprob)

    # Mirror the kernel's two bf16 epilogue roundings: projection first, then
    # bias addition. The log-softmax reduction itself remains fp32.
    projected = np.array(h.astype(mx.float32)) @ np.array(weight.astype(mx.float32)).T
    rounded = mx.array(projected).astype(mx.bfloat16)
    logits_m = (rounded + bias).astype(mx.bfloat16)
    mx.eval(logits_m)
    logits = np.array(logits_m.astype(mx.float32))
    allowed = forbidden[previous] == 0
    expected = np.where(allowed, logits, -np.inf).argmax(-1).astype(np.int32)
    maximum = logits.max(-1)
    log_z = maximum + np.log(np.exp(logits - maximum[:, None]).sum(-1))
    expected_logprob = logits[np.arange(T), expected] - log_z
    np.testing.assert_array_equal(np.array(token), expected)
    np.testing.assert_allclose(np.array(logprob), expected_logprob, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
@pytest.mark.parametrize("k", [1, 8, 32])
def test_quant_topk(fmt, k):
    # Fused quantized-weight top-k (no logits materialization) vs the dequantize(Wq)@h top-k oracle.
    from tk.quant import QUANT_FORMATS
    quant, dequant = QUANT_FORMATS[fmt]
    T, V, K = 4, 4000, 512
    rng = np.random.default_rng(21 + k)
    W = (0.3 * rng.standard_normal((V, K))).astype(np.float32)
    Wq = quant(W)
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    tok = np.array(tk.lm_head_sample(mx.array(h), mx.array(Wq), mode="topk", k=k, temperature=0.8,
                                     seed=3, format=fmt))
    Wdq = dequant(Wq).astype(np.float64)
    L = h.astype(np.float64) @ Wdq.T
    for t in range(T):
        top = set(int(v) for v in np.argsort(-L[t], kind="stable")[:k])
        assert tok[t] in top or (L[t].max() - L[t, tok[t]]) < 1e-2
        if k == 1:
            assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-2


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
@pytest.mark.parametrize("p", [0.5, 0.9])
def test_quant_topp(fmt, p):
    # Fused quant top-p (nucleus over the over-selected top-k' pool) vs the dequant-logits nucleus.
    # The pool nucleus is a subset of the full-vocab nucleus, so the sampled token must lie in it.
    from tk.quant import QUANT_FORMATS
    quant, dequant = QUANT_FORMATS[fmt]
    T, V, K, temp = 2, 4000, 512, 0.8
    rng = np.random.default_rng(41 + int(p * 10))
    W = (0.3 * rng.standard_normal((V, K))).astype(np.float32)
    Wq = quant(W)
    h = (0.6 * rng.standard_normal((T, K))).astype(np.float32)
    Wdq = dequant(Wq).astype(np.float64)
    L = (h.astype(np.float64) @ Wdq.T) / temp     # tempered logits
    nucleus = []
    for t in range(T):
        ls = L[t]
        mxv = ls.max(); pex = np.exp(ls - mxv); Z = pex.sum()
        order = np.argsort(-ls, kind="stable")
        cum, keep = 0.0, set()
        for v in order:
            cum += pex[v] / Z
            keep.add(int(v))
            if cum >= p:
                break
        nucleus.append(keep)
    for seed in range(40):
        tok = np.array(tk.lm_head_sample(mx.array(h), mx.array(Wq), mode="topp",
                                         k=32, temperature=temp, seed=seed, format=fmt, top_p=p))
        for t in range(T):
            assert int(tok[t]) in nucleus[t], (fmt, p, seed, t, int(tok[t]))


def test_quant_topp_true_normalizer():
    # The fused quant top-p uses the TRUE full-vocab normalizer (per-tile logsumexps), so the nucleus
    # is the exact full-vocab nucleus (not the pool-only approximation, which would use a too-small Z
    # and drop the low-prob tail of the nucleus). Construct a moderately-tailed distribution where the
    # true nucleus is strictly LARGER than the pool-Z nucleus but still fits inside the top-k' pool,
    # then assert (a) every sample lies in the true nucleus and (b) the kernel samples tokens the
    # pool-Z approximation would have excluded -- i.e. the wider (correct) nucleus.
    from tk.quant import QUANT_FORMATS
    quant, dequant = QUANT_FORMATS["q8_0"]
    V, K, temp, k, TILE_V, p, scale = 1024, 256, 1.0, 32, 256, 0.7, 0.35
    rng = np.random.default_rng(0)
    W = (scale * rng.standard_normal((V, K))).astype(np.float32); Wq = quant(W)
    Wdq = dequant(Wq).astype(np.float64)
    h1 = (scale * rng.standard_normal((1, K))).astype(np.float32)
    L = (h1.astype(np.float64) @ Wdq.T / temp)[0]
    # reconstruct the kernel's candidate pool: top-k within each TILE_V-sized vocab tile
    pool = []
    for tv in range(V // TILE_V):
        seg = L[tv * TILE_V:(tv + 1) * TILE_V]
        pool += list(np.argsort(-seg)[:k] + tv * TILE_V)
    pool = np.array(sorted(pool)); mx_l = L.max()
    Z_full = np.exp(L - mx_l).sum(); Z_pool = np.exp(L[pool] - mx_l).sum()
    order = pool[np.argsort(-L[pool])]

    def nucleus_with(Z):
        cum, s = 0.0, []
        for v in order:
            cum += np.exp(L[v] - mx_l) / Z; s.append(int(v))
            if cum >= p:
                break
        return set(s), cum

    nuc_true, cum_true = nucleus_with(Z_full)
    nuc_pool, _ = nucleus_with(Z_pool)
    assert cum_true >= p                         # true nucleus fits inside the pool
    assert nuc_true > nuc_pool                   # strictly larger (this is what true-Z fixes)
    extra = nuc_true - nuc_pool                  # tokens only the true normalizer keeps

    # many independent samples in one launch: identical rows, Gumbel varies with the row index
    Tn = 800
    h = np.repeat(h1, Tn, axis=0)
    tok = np.array(tk.lm_head_sample(mx.array(h), mx.array(Wq), mode="topp",
                                     k=k, temperature=temp, seed=0, format="q8_0", top_p=p))
    sampled = set(int(x) for x in tok)
    assert sampled <= nuc_true, sampled - nuc_true          # never sample outside the true nucleus
    n_extra = sum(int(x) in extra for x in tok)
    assert n_extra > 0, (len(extra), "kernel must sample the true-normalizer tail of the nucleus")


if __name__ == "__main__":
    test_argmax("bfloat16", 1, 32000, 2048)
    test_categorical("float32", 4, 32000, 2048)
    test_topk("float32", 8)
    print("ok")
