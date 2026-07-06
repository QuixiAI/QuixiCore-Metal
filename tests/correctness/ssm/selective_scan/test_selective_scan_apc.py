"""Correctness tests for selective_scan_varlen_apc (automatic prefix caching).

Oracle: the same fp64 S6 recurrence as the varlen test, but transcribing the APC block
indexing — initial state read from initial_state_idx's block, running state checkpointed
into the paged pool at each chunk boundary (last chunk -> block_idx_last_scheduled_token).

Run from kernels/:  python -m pytest selective_scan/correctness/test_selective_scan_apc.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import selective_scan_varlen_apc

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _apc_ref(u, delta, A, B, C, qsl, cache_indices, his, state, bif, bil, isi, block_size,
             cache_indices_stride, D=None, delta_bias=None, z=None, softplus=True,
             null_block_id=-1):
    """fp64 transcription of the APC kernel (use_chunk_metadata=False path)."""
    dim, total = u.shape
    n_groups, dstate, _ = B.shape
    out = np.zeros((dim, total))
    new_state = state.astype(np.float64).copy()
    batch = len(qsl) - 1
    group_ratio = dim // n_groups
    for b in range(batch):
        s0, e0 = int(qsl[b]), int(qsl[b + 1])
        seqlen = e0 - s0
        if seqlen <= 0:
            continue
        init_blk = int(isi[b])
        init_slot = int(cache_indices[b * cache_indices_stride + init_blk])
        if init_slot == null_block_id:
            continue
        for d in range(dim):
            g = d // group_ratio
            a_row = A[d].astype(np.float64)
            load = his[b] != 0
            running = new_state[init_slot, d].astype(np.float64).copy() if load \
                else np.zeros(dstate)
            bias = float(delta_bias[d]) if delta_bias is not None else 0.0
            dv = float(D[d]) if D is not None else 0.0
            n_chunks = (seqlen + block_size - 1) // block_size
            current_position = 0
            tokens_processed = 0
            for chunk in range(n_chunks):
                if tokens_processed >= seqlen:
                    break
                chunk_tokens = min(block_size, seqlen - tokens_processed)
                cstart = s0 + tokens_processed
                for off in range(chunk_tokens):
                    t = cstart + off
                    uu = float(u[d, t])
                    dd = float(delta[d, t]) + bias
                    if softplus:
                        dd = _softplus(dd)
                    bt = B[g, :, t].astype(np.float64)
                    ct = C[g, :, t].astype(np.float64)
                    running = np.exp(dd * a_row) * running + bt * dd * uu
                    val = dv * uu + float(running @ ct)
                    if z is not None:
                        zv = float(z[d, t])
                        val *= zv / (1.0 + np.exp(-zv))
                    out[d, t] = val
                # checkpoint at the chunk boundary
                if chunk == n_chunks - 1:
                    sblk = int(bil[b])
                else:
                    sblk = (current_position + chunk_tokens - 1) // block_size
                sslot = int(cache_indices[b * cache_indices_stride + sblk])
                if sslot != null_block_id:
                    new_state[sslot, d] = running
                tokens_processed += chunk_tokens
                current_position += chunk_tokens
    return out, new_state


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_apc_no_chunk_metadata(dtype):
    """use_chunk_metadata=False: uniform block_size chunking, single-block requests."""
    rng = np.random.default_rng(7)
    dim, dstate, n_groups = 16, 16, 2
    lens = [5, 12, 3]
    total = sum(lens)
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    batch = len(lens)
    block_size = 8
    stride = 4                                          # blocks per request
    nslots = 12
    u = (0.4 * rng.standard_normal((dim, total))).astype(np.float32)
    delta = (0.3 * rng.standard_normal((dim, total))).astype(np.float32)
    A = (-0.5 - rng.random((dim, dstate))).astype(np.float32)   # A < 0
    B = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    C = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    D = rng.standard_normal(dim).astype(np.float32)
    state = (0.2 * rng.standard_normal((nslots, dim, dstate))).astype(np.float32)
    # per request: distinct blocks; initial block 0, last block = ceil(len/bs)-1
    cache_indices = rng.permutation(nslots)[:batch * stride].astype(np.int32).reshape(batch, stride)
    his = np.array([1, 0, 1], np.uint8)
    bif = np.zeros(batch, np.int32)
    bil = np.array([(l + block_size - 1) // block_size - 1 for l in lens], np.int32)
    isi = np.zeros(batch, np.int32)
    ccs = np.zeros(1, np.int32)      # unused when use_chunk_metadata=False
    lci = np.zeros(batch, np.int32)
    md = _MX[dtype]
    to = lambda a: mx.array(a).astype(md)
    out, ns = selective_scan_varlen_apc(
        to(u), to(delta), mx.array(A), to(B), to(C), mx.array(qsl),
        mx.array(cache_indices.reshape(-1)), mx.array(his), mx.array(state),
        mx.array(bif), mx.array(bil), mx.array(isi), mx.array(ccs), mx.array(lci),
        block_size, stride, False, D=mx.array(D), delta_softplus=True)
    mx.eval(out, ns)
    r = lambda a: np.array(mx.array(a).astype(md).astype(mx.float32))
    ref_out, ref_state = _apc_ref(r(u), r(delta), A, r(B), r(C), qsl,
                                  cache_indices.reshape(-1), his, state, bif, bil, isi,
                                  block_size, stride, D=D, softplus=True)
    atol = 2e-2 if dtype != "float32" else 1e-4
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref_out, atol=atol, rtol=atol)
    np.testing.assert_allclose(np.array(ns), ref_state, atol=atol, rtol=atol)


def test_apc_prefix_cache_initial_state():
    """load_initial=1 reads the initial state from initial_state_idx's block, not block 0."""
    rng = np.random.default_rng(8)
    dim, dstate, n_groups = 8, 16, 1
    lens = [6]
    total = sum(lens)
    qsl = np.array([0, total], np.int32)
    block_size, stride, nslots = 8, 3, 6
    u = (0.4 * rng.standard_normal((dim, total))).astype(np.float32)
    delta = (0.3 * rng.standard_normal((dim, total))).astype(np.float32)
    A = (-0.5 - rng.random((dim, dstate))).astype(np.float32)
    B = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    C = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    state = (0.5 * rng.standard_normal((nslots, dim, dstate))).astype(np.float32)
    cache_indices = np.array([[4, 1, 2]], np.int32)   # initial block 1 -> slot 1
    his = np.array([1], np.uint8)
    bif = np.zeros(1, np.int32)
    bil = np.array([0], np.int32)                      # single chunk -> last block slot
    isi = np.array([1], np.int32)                      # <-- prefix-cache block
    out, ns = selective_scan_varlen_apc(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C), mx.array(qsl),
        mx.array(cache_indices.reshape(-1)), mx.array(his), mx.array(state),
        mx.array(bif), mx.array(bil), mx.array(isi), mx.array(np.zeros(1, np.int32)),
        mx.array(np.zeros(1, np.int32)), block_size, stride, False, delta_softplus=True)
    mx.eval(out, ns)
    ref_out, ref_state = _apc_ref(u, delta, A, B, C, qsl, cache_indices.reshape(-1), his,
                                  state, bif, bil, isi, block_size, stride, softplus=True)
    np.testing.assert_allclose(np.array(out), ref_out, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(ns), ref_state, atol=1e-4, rtol=1e-4)
    # untouched slots preserved
    touched = {int(cache_indices[0, 0])}   # last-block slot 4 (single chunk)
    for sl in range(nslots):
        if sl not in touched:
            np.testing.assert_allclose(np.array(ns)[sl], state[sl], atol=1e-6)


def test_apc_multichunk_checkpoints():
    """A request longer than block_size checkpoints intermediate chunk state into blocks."""
    rng = np.random.default_rng(9)
    dim, dstate, n_groups = 8, 16, 1
    lens = [20]                                         # 3 chunks at block_size 8
    total = sum(lens)
    qsl = np.array([0, total], np.int32)
    block_size, stride, nslots = 8, 4, 8
    u = (0.3 * rng.standard_normal((dim, total))).astype(np.float32)
    delta = (0.2 * rng.standard_normal((dim, total))).astype(np.float32)
    A = (-0.5 - rng.random((dim, dstate))).astype(np.float32)
    B = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    C = (0.3 * rng.standard_normal((n_groups, dstate, total))).astype(np.float32)
    state = np.zeros((nslots, dim, dstate), np.float32)
    cache_indices = np.array([[0, 1, 2, 3]], np.int32)
    his = np.array([0], np.uint8)                       # fresh
    bif = np.zeros(1, np.int32)
    bil = np.array([2], np.int32)                       # last chunk -> block 2
    isi = np.zeros(1, np.int32)
    out, ns = selective_scan_varlen_apc(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C), mx.array(qsl),
        mx.array(cache_indices.reshape(-1)), mx.array(his), mx.array(state),
        mx.array(bif), mx.array(bil), mx.array(isi), mx.array(np.zeros(1, np.int32)),
        mx.array(np.zeros(1, np.int32)), block_size, stride, False, delta_softplus=True)
    mx.eval(out, ns)
    ref_out, ref_state = _apc_ref(u, delta, A, B, C, qsl, cache_indices.reshape(-1), his,
                                  state, bif, bil, isi, block_size, stride, softplus=True)
    np.testing.assert_allclose(np.array(out), ref_out, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(ns), ref_state, atol=1e-4, rtol=1e-4)
    # blocks 0,1,2 all got checkpoints (intermediate + final); block 3 untouched
    np.testing.assert_allclose(np.array(ns)[3], state[3], atol=1e-6)
    assert np.abs(np.array(ns)[0]).max() > 0 and np.abs(np.array(ns)[2]).max() > 0
