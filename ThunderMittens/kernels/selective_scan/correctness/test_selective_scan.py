"""Correctness tests for the Mamba-1 (S6) selective scan (dense + varlen).

Oracle: fp64 numpy transcription of the recurrence (mamba_ssm.selective_scan_ref semantics,
channel-major layouts): dA = exp(delta*A); h = dA*h + delta*B*u; y = sum(h*C) + D*u;
y *= silu(z); delta = softplus(delta + delta_bias).

Run from kernels/:  python -m pytest selective_scan/correctness/test_selective_scan.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import selective_scan, selective_scan_varlen

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
_ATOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}


def _ref_scan(u, delta, A, B, C, D, bias, z, h0, softplus):
    """u/delta/z (batch, dim, L); B/C (batch, G, N, L); A (dim, N); h0 (batch, dim, N)."""
    b, d, L = u.shape
    G, N = B.shape[1], B.shape[2]
    g_of = np.arange(d) // (d // G)
    h = h0.astype(np.float64).copy()
    out = np.zeros((b, d, L))
    dl = delta.astype(np.float64) + (bias[None, :, None] if bias is not None else 0.0)
    if softplus:
        dl = np.where(dl <= 20.0, np.log1p(np.exp(np.minimum(dl, 20.0))), dl)
    for t in range(L):
        for bi in range(b):
            dA = np.exp(dl[bi, :, t, None] * A)                     # (dim, N)
            Bu = B[bi, g_of, :, t] * dl[bi, :, t, None] * u[bi, :, t, None]
            h[bi] = dA * h[bi] + Bu
            y = (h[bi] * C[bi, g_of, :, t]).sum(-1)
            if D is not None:
                y = y + D * u[bi, :, t]
            if z is not None:
                zv = z[bi, :, t].astype(np.float64)
                y = y * (zv / (1.0 + np.exp(-zv)))
            out[bi, :, t] = y
    return out, h


def _mk_inputs(rng, b, d, L, G, N):
    u = (0.5 * rng.standard_normal((b, d, L))).astype(np.float32)
    delta = (0.3 * rng.standard_normal((b, d, L))).astype(np.float32)
    A = (-np.exp(rng.standard_normal((d, N)) * 0.5)).astype(np.float32)   # A < 0
    B = (0.5 * rng.standard_normal((b, G, N, L))).astype(np.float32)
    C = (0.5 * rng.standard_normal((b, G, N, L))).astype(np.float32)
    D = rng.standard_normal(d).astype(np.float32)
    bias = (0.1 * rng.standard_normal(d)).astype(np.float32)
    z = (0.5 * rng.standard_normal((b, d, L))).astype(np.float32)
    h0 = (0.2 * rng.standard_normal((b, d, N))).astype(np.float32)
    return u, delta, A, B, C, D, bias, z, h0


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("b,d,L,G,N", [(2, 64, 16, 1, 16), (2, 64, 24, 4, 32), (1, 32, 8, 1, 160)])
def test_dense(dtype, b, d, L, G, N):
    rng = np.random.default_rng(b + d + N)
    u, delta, A, B, C, D, bias, z, h0 = _mk_inputs(rng, b, d, L, G, N)
    md = _MX[dtype]
    to = lambda a: mx.array(a).astype(md)
    out, st = selective_scan(to(u), to(delta), mx.array(A), to(B), to(C), mx.array(h0),
                             D=mx.array(D), delta_bias=mx.array(bias), z=to(z))
    mx.eval(out, st)
    # reference over dtype-rounded inputs
    r = lambda a: np.array(mx.array(a).astype(md).astype(mx.float32))
    ref, href = _ref_scan(r(u), r(delta), A, r(B), r(C), D, bias, r(z), h0, True)
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref,
                               atol=_ATOL[dtype], rtol=_ATOL[dtype])
    np.testing.assert_allclose(np.array(st), href, atol=1e-2, rtol=1e-2)


def test_dense_no_optionals():
    rng = np.random.default_rng(9)
    u, delta, A, B, C, _, _, _, h0 = _mk_inputs(rng, 2, 32, 12, 2, 16)
    out, st = selective_scan(mx.array(u), mx.array(delta), mx.array(A), mx.array(B),
                             mx.array(C), mx.array(h0), delta_softplus=False)
    mx.eval(out, st)
    ref, href = _ref_scan(u, delta, A, B, C, None, None, None, h0, False)
    np.testing.assert_allclose(np.array(out), ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(st), href, atol=1e-5, rtol=1e-5)


def test_varlen_matches_dense_per_seq():
    """Varlen over ragged [5, 9, 3] must equal per-sequence dense scans, with the paged
    state pool updated at cache_indices slots and untouched slots preserved."""
    rng = np.random.default_rng(11)
    d, G, N = 32, 2, 16
    lens = [5, 9, 3]
    total = sum(lens)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    u = (0.5 * rng.standard_normal((d, total))).astype(np.float32)
    delta = (0.3 * rng.standard_normal((d, total))).astype(np.float32)
    A = (-np.exp(rng.standard_normal((d, N)) * 0.5)).astype(np.float32)
    B = (0.5 * rng.standard_normal((G, N, total))).astype(np.float32)
    C = (0.5 * rng.standard_normal((G, N, total))).astype(np.float32)
    D = rng.standard_normal(d).astype(np.float32)
    pool = (0.2 * rng.standard_normal((6, d, N))).astype(np.float32)
    cidx = np.array([4, 1, 5], np.int32)              # scattered slots
    his = np.array([1, 0, 1], np.uint8)               # request 1 starts fresh

    out, new_pool = selective_scan_varlen(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C),
        mx.array(cu), mx.array(pool), D=mx.array(D),
        cache_indices=mx.array(cidx), has_initial_state=mx.array(his))
    mx.eval(out, new_pool)
    on, pn = np.array(out), np.array(new_pool)

    for r, (s0, e0) in enumerate(zip(cu[:-1], cu[1:])):
        h0 = pool[cidx[r]] if his[r] else np.zeros((d, N), np.float32)
        ref, href = _ref_scan(u[None, :, s0:e0], delta[None, :, s0:e0], A,
                              B[None, :, :, s0:e0], C[None, :, :, s0:e0], D, None, None,
                              h0[None], True)
        np.testing.assert_allclose(on[:, s0:e0], ref[0], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(pn[cidx[r]], href[0], atol=1e-5, rtol=1e-5)
    # untouched pool slots preserved exactly
    untouched = [i for i in range(6) if i not in cidx.tolist()]
    np.testing.assert_array_equal(pn[untouched], pool[untouched])


def test_varlen_null_block_skipped():
    rng = np.random.default_rng(12)
    d, G, N, total = 32, 1, 16, 8
    cu = np.array([0, 4, 8], np.int32)
    u = rng.standard_normal((d, total)).astype(np.float32)
    delta = rng.standard_normal((d, total)).astype(np.float32)
    A = (-np.ones((d, N))).astype(np.float32)
    B = rng.standard_normal((G, N, total)).astype(np.float32)
    C = rng.standard_normal((G, N, total)).astype(np.float32)
    pool = rng.standard_normal((3, d, N)).astype(np.float32)
    cidx = np.array([0, -1], np.int32)                # request 1 -> null block, skipped
    out, new_pool = selective_scan_varlen(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C),
        mx.array(cu), mx.array(pool), cache_indices=mx.array(cidx), null_block_id=-1)
    mx.eval(out, new_pool)
    np.testing.assert_array_equal(np.array(new_pool)[1:], pool[1:])   # slots 1,2 untouched
