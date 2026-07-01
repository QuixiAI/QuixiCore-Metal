"""Correctness tests for MoE routing.

moe_route_topk: per token, select top-k experts by router logit (descending) and
return softmax weights over the k selected logits (Mixtral renormalized top-k).

Run from kernels/:  python -m pytest moe/correctness/test_moe.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import moe_route_topk, moe_permute, moe_finalize, moe_grouped_gemm

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("E,K", [(8, 2), (64, 4), (16, 1), (128, 8)])
def test_moe_route_topk(dtype, E, K):
    rng = np.random.default_rng(0)
    T = 100
    x = rng.standard_normal((T, E)).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    ids, w = moe_route_topk(xq, K)
    mx.eval(ids, w)
    ids = np.array(ids)
    w = np.array(w)
    xd = np.array(xq.astype(mx.float32))

    gathered = np.take_along_axis(xd, ids, axis=1)         # logits at chosen experts
    true_top = -np.sort(-xd, axis=1)[:, :K]                # true top-k values (descending)

    # the chosen set of logits matches the true top-k set
    np.testing.assert_allclose(np.sort(gathered, axis=1), np.sort(true_top, axis=1), atol=2e-2)
    # returned in descending order
    assert np.all(np.diff(gathered, axis=1) <= 1e-3)
    # weights == softmax over the chosen logits
    m = gathered.max(axis=1, keepdims=True)
    ex = np.exp(gathered - m)
    ref_w = ex / ex.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(w, ref_w, atol=2e-3, rtol=2e-3)
    # ids are a valid permutation subset (no duplicates per row)
    assert all(len(set(row)) == K for row in ids)
    # exact id order for float32 (no down-cast ties)
    if dtype == "float32":
        order = np.argsort(-xd, axis=1, kind="stable")[:, :K].astype(np.int32)
        np.testing.assert_array_equal(ids, order)


@pytest.mark.parametrize("E,K", [(8, 2), (4, 1), (16, 4), (300, 2)])  # E=300 spans >1 scan tile
def test_moe_permute(E, K):
    rng = np.random.default_rng(0)
    T = 50
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    sorted_idx, offsets, inv = moe_permute(mx.array(ids), E)
    mx.eval(sorted_idx, offsets, inv)
    sorted_idx, offsets, inv = np.array(sorted_idx), np.array(offsets), np.array(inv)
    flat = ids.reshape(-1)
    TK = T * K

    counts = np.bincount(flat, minlength=E)
    ref_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    np.testing.assert_array_equal(offsets, ref_off)
    # rows are grouped by expert in the permuted order
    assert np.all(np.diff(flat[sorted_idx]) >= 0)
    for e in range(E):
        s, en = offsets[e], offsets[e + 1]
        assert set(sorted_idx[s:en].tolist()) == set(np.where(flat == e)[0].tolist())
    # inv is the inverse permutation
    assert np.array_equal(sorted_idx[inv], np.arange(TK))


@pytest.mark.parametrize("K,H", [(2, 64), (4, 128), (1, 256)])
def test_moe_finalize(K, H):
    rng = np.random.default_rng(1)
    T, E = 20, 4
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    w = rng.random((T, K)).astype(np.float32)
    _, _, inv = moe_permute(mx.array(ids), E)
    mx.eval(inv)
    inv = np.array(inv)
    expert_out = rng.standard_normal((T * K, H)).astype(np.float32)
    got = moe_finalize(mx.array(expert_out), mx.array(inv), mx.array(w), K)
    mx.eval(got)
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for k in range(K):
            ref[t] += w[t, k] * expert_out[inv[t * K + k]]
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4)


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_forward_end_to_end(E, K):
    # route -> permute -> per-expert GEMM (host loop) -> finalize, vs a dense reference.
    rng = np.random.default_rng(2)
    T, H = 32, 64
    x = rng.standard_normal((T, H)).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W = (rng.standard_normal((E, H, H)) * 0.1).astype(np.float32)

    ids, w = moe_route_topk(mx.array(rl), K)
    mx.eval(ids, w)
    ids, w = np.array(ids), np.array(w)
    sidx, off, inv = moe_permute(mx.array(ids.astype(np.int32)), E)
    mx.eval(sidx, off, inv)
    sidx, off, inv = np.array(sidx), np.array(off), np.array(inv)

    permuted_x = x[sidx // K]
    out_perm = np.zeros((T * K, H), np.float32)
    for e in range(E):
        s, en = off[e], off[e + 1]
        if en > s:
            out_perm[s:en] = permuted_x[s:en] @ W[e]
    y = np.array(moe_finalize(mx.array(out_perm), mx.array(inv), mx.array(w), K))

    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(K):
            ref[t] += w[t, j] * (x[t] @ W[ids[t, j]])
    np.testing.assert_allclose(y, ref, atol=1e-3, rtol=1e-3)


def _padded_schedule(counts):
    padded = [((int(c) + 31) // 32) * 32 for c in counts]
    off_pad = np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)
    total = int(off_pad[-1])
    tile_base = (off_pad // 32).astype(np.int64)
    eot = np.zeros(total // 32, np.int32)
    for e in range(len(counts)):
        eot[tile_base[e]:tile_base[e + 1]] = e
    return off_pad, total, eot


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-4), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("H", [64, 128])
def test_moe_grouped_gemm(dtype, atol, H):
    rng = np.random.default_rng(5)
    E = 4
    counts = [40, 5, 70, 20]  # per-expert token counts -> padded [64,32,96,32], total 224
    off_pad, total, eot = _padded_schedule(counts)
    pi = (0.1 * rng.standard_normal((total, H))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, H, H))).astype(np.float32)
    md = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    pim, Wm = mx.array(pi).astype(md), mx.array(W).astype(md)
    out = moe_grouped_gemm(pim, Wm, mx.array(eot))
    mx.eval(out)
    pir = np.array(pim.astype(mx.float32))
    Wr = np.array(Wm.astype(mx.float32))
    ref = np.zeros((total, H), np.float32)
    for e in range(E):
        s, en = int(off_pad[e]), int(off_pad[e + 1])
        ref[s:en] = pir[s:en] @ Wr[e]
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, atol=atol, rtol=2e-2)


if __name__ == "__main__":
    for E, K in [(8, 2), (64, 4), (16, 1), (128, 8)]:
        test_moe_route_topk("float32", E, K)
        print("ok", E, K)
    test_moe_permute(8, 2)
    test_moe_forward_end_to_end(8, 2)
    print("ok moe pipeline")
