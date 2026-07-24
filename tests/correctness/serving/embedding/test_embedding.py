"""Correctness tests for token embedding lookup + multimodal span merge.

Run from kernels/:  python -m pytest embedding/correctness/test_embedding.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (embedding_backward, embedding_lookup, embedding_lookup_types, merge_multimodal_spans,
                build_multimodal_src)

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
def test_embedding_lookup_types(dtype):
    rng = np.random.default_rng(17)
    token_table = (0.2 * rng.standard_normal((101, 96))).astype(np.float32)
    type_table = (0.2 * rng.standard_normal((3, 96))).astype(np.float32)
    token_ids = np.array([4, -1, 100, 7, 101], np.int32)
    type_ids = np.array([0, 1, -1, 2, 9], np.int32)
    got = embedding_lookup_types(
        mx.array(token_ids), mx.array(type_ids),
        mx.array(token_table).astype(_MX[dtype]),
        mx.array(type_table).astype(_MX[dtype]), token_scale=1.25)
    got = np.array(got.astype(mx.float32))
    ref = np.zeros_like(got)
    for t in range(token_ids.size):
        if 0 <= token_ids[t] < token_table.shape[0]:
            ref[t] += 1.25 * token_table[token_ids[t]]
        if 0 <= type_ids[t] < type_table.shape[0]:
            ref[t] += type_table[type_ids[t]]
    np.testing.assert_allclose(got, ref, atol=1e-4 if dtype == "float32" else 3e-2)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("method", ["atomic", "sorted"])
def test_embedding_backward(dtype, D, method):
    # scatter-add grad by token id (with duplicates + a padding id) vs the numpy reference.
    rng = np.random.default_rng(D + 7)
    vocab, T = 50, 40
    tok = rng.integers(0, vocab, size=T).astype(np.int32)
    tok[3] = -1                                    # padding id -> no contribution
    tok[7] = tok[11] = tok[19] = 5                 # duplicate id -> accumulation
    dY = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
    scale = 1.5
    dtab = np.array(embedding_backward(mx.array(tok), mx.array(dY).astype(_MX[dtype]),
                                       vocab=vocab, scale=scale, method=method).astype(mx.float32))
    ref = np.zeros((vocab, D), np.float64)
    for t in range(T):
        if tok[t] >= 0:
            ref[tok[t]] += dY[t].astype(np.float64) * scale
    atol = 1e-4 if dtype == "float32" else 5e-2
    np.testing.assert_allclose(dtab, ref, atol=atol)


@pytest.mark.parametrize("method", ["atomic", "sorted"])
def test_embedding_backward_high_dup(method):
    # heavy id duplication (only 4 distinct ids over 4096 tokens) + all-negative rows; both methods
    # must equal the scatter-add reference. This is the regime the sorted (atomic-free) path targets.
    rng = np.random.default_rng(123)
    vocab, T, D = 8, 4096, 128
    tok = rng.integers(0, 4, size=T).astype(np.int32)     # ids 0..3 only -> huge segments
    tok[rng.integers(0, T, size=64)] = -1                 # scattered padding rows
    dY = (0.2 * rng.standard_normal((T, D))).astype(np.float32)
    dtab = np.array(embedding_backward(mx.array(tok), mx.array(dY), vocab=vocab, scale=1.0,
                                       method=method))
    ref = np.zeros((vocab, D), np.float64)
    for t in range(T):
        if tok[t] >= 0:
            ref[tok[t]] += dY[t].astype(np.float64)
    np.testing.assert_allclose(dtab, ref, atol=2e-3)


def test_embedding_backward_methods_agree():
    rng = np.random.default_rng(9)
    vocab, T, D = 100, 500, 64
    tok = rng.integers(-2, vocab, size=T).astype(np.int32)   # includes negatives + oob-free dups
    dY = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    a = np.array(embedding_backward(mx.array(tok), mx.array(dY), vocab=vocab, scale=0.7,
                                    method="atomic"))
    b = np.array(embedding_backward(mx.array(tok), mx.array(dY), vocab=vocab, scale=0.7,
                                    method="sorted"))
    np.testing.assert_allclose(a, b, atol=1e-5)


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


def test_build_multimodal_src():
    # device src builder vs the host span loop, then the full build->merge pipeline.
    rng = np.random.default_rng(9)
    T, M, D = 40, 20, 64
    # 3 non-overlapping modal spans
    span_off = np.array([5, 15, 30], np.int32)
    span_len = np.array([4, 6, 5], np.int32)
    modal_start = np.array([0, 4, 10], np.int32)
    src_ref = np.full(T, -1, np.int32)
    for k in range(len(span_off)):
        for o in range(int(span_len[k])):
            src_ref[span_off[k] + o] = modal_start[k] + o
    src = np.array(build_multimodal_src(mx.array(span_off), mx.array(span_len),
                                        mx.array(modal_start), T))
    np.testing.assert_array_equal(src, src_ref)
    # full pipeline: build src on device -> merge
    text = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    modal = (0.3 * rng.standard_normal((M, D))).astype(np.float32)
    o = np.array(merge_multimodal_spans(mx.array(text), mx.array(modal),
                                        build_multimodal_src(mx.array(span_off), mx.array(span_len),
                                                             mx.array(modal_start), T)))
    ref = np.where(src_ref[:, None] >= 0, modal[np.clip(src_ref, 0, M - 1)], text)
    np.testing.assert_allclose(o, ref, atol=1e-5)


if __name__ == "__main__":
    test_embedding_lookup("float32", 64)
    test_merge_multimodal_spans("float32")
    print("ok")
