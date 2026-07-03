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


if __name__ == "__main__":
    test_argmax("bfloat16", 1, 32000, 2048)
    test_categorical("float32", 4, 32000, 2048)
    test_topk("float32", 8)
    print("ok")
