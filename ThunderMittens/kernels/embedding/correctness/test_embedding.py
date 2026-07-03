"""Correctness tests for token embedding lookup + multimodal span merge.

Run from kernels/:  python -m pytest embedding/correctness/test_embedding.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import embedding_backward, embedding_lookup, merge_multimodal_spans

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("D", [64, 128, 4096])
def test_embedding_lookup(dtype, D):
    rng = np.random.default_rng(D)
    vocab, T = 200, 12
    table = (0.3 * rng.standard_normal((vocab, D))).astype(np.float32)
    tok = rng.integers(0, vocab, size=T).astype(np.int32)
    tok[4] = -1                                    # padding token -> zeros
    scale = 1.5
    o = np.array(embedding_lookup(mx.array(tok), mx.array(table).astype(_MX[dtype]),
                                  scale=scale).astype(mx.float32))
    ref = np.where(tok[:, None] >= 0, table[np.clip(tok, 0, vocab - 1)] * scale, 0.0)
    atol = 1e-4 if dtype == "float32" else 3e-2
    np.testing.assert_allclose(o, ref, atol=atol)


def test_embedding_lookup_pos():
    rng = np.random.default_rng(1)
    vocab, D, T = 100, 128, 8
    table = (0.3 * rng.standard_normal((vocab, D))).astype(np.float32)
    pos = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    tok = rng.integers(0, vocab, size=T).astype(np.int32)
    o = np.array(embedding_lookup(mx.array(tok), mx.array(table), pos_table=mx.array(pos), scale=1.0))
    ref = table[tok] + pos
    np.testing.assert_allclose(o, ref, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_embedding_backward(dtype, D):
    # scatter-add grad by token id (with duplicates + a padding id) vs the numpy reference.
    rng = np.random.default_rng(D + 7)
    vocab, T = 50, 40
    tok = rng.integers(0, vocab, size=T).astype(np.int32)
    tok[3] = -1                                    # padding id -> no contribution
    tok[7] = tok[11] = tok[19] = 5                 # duplicate id -> atomic accumulation
    dY = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
    scale = 1.5
    dtab = np.array(embedding_backward(mx.array(tok), mx.array(dY).astype(_MX[dtype]),
                                       vocab=vocab, scale=scale).astype(mx.float32))
    ref = np.zeros((vocab, D), np.float64)
    for t in range(T):
        if tok[t] >= 0:
            ref[tok[t]] += dY[t].astype(np.float64) * scale
    atol = 1e-4 if dtype == "float32" else 5e-2
    np.testing.assert_allclose(dtab, ref, atol=atol)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_merge_multimodal_spans(dtype):
    rng = np.random.default_rng(2)
    T, M, D = 16, 6, 128
    text = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    modal = (0.3 * rng.standard_normal((M, D))).astype(np.float32)
    src = np.full(T, -1, np.int32)
    # two image spans: text[2:5] <- modal[0:3], text[9:11] <- modal[3:5]
    src[2:5] = np.arange(0, 3)
    src[9:11] = np.arange(3, 5)
    o = np.array(merge_multimodal_spans(mx.array(text).astype(_MX[dtype]),
                                        mx.array(modal).astype(_MX[dtype]),
                                        mx.array(src)).astype(mx.float32))
    ref = np.where(src[:, None] >= 0, modal[np.clip(src, 0, M - 1)], text)
    atol = 1e-4 if dtype == "float32" else 3e-2
    np.testing.assert_allclose(o, ref, atol=atol)


if __name__ == "__main__":
    test_embedding_lookup("float32", 64)
    test_merge_multimodal_spans("float32")
    print("ok")
