"""Correctness tests for the quantized grouped expert GEMMs (moe_grouped_gemm_rect_q /
moe_grouped_gemm_swiglu_q).

Oracle strategy: the kernel and the reference consume the SAME dequantized weight values
(tk.quant.dequantize_expert_stack of the same packed bytes), so quantization error cancels
and the comparison isolates the bf16 MMA path. Both sides round A and dequant(W) to bf16
before the fp32-accumulated matmul.

Run from kernels/:  python -m pytest moe/correctness/test_moe_q.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (moe_grouped_gemm_rect, moe_grouped_gemm_rect_q, moe_grouped_gemm_swiglu_q,
                moe_mlp)
from tk.quant import MOE_Q_FORMATS, quantize_expert_stack, dequantize_expert_stack

E, K_DIM, N_OUT = 4, 256, 64          # K=256 satisfies every format's block_k (16..256)
TILES = 6                             # one padding tile (expert -1) in the schedule
ROWS = 32 * TILES
EOT = np.array([0, 2, 1, 3, -1, 2], np.int32)


def _bf16(x):
    return np.array(mx.array(np.ascontiguousarray(x, np.float32)).astype(mx.bfloat16)
                    .astype(mx.float32))


def _ref_grouped(A, Wd, eot, bias=None):
    """Segmented per-tile matmul in fp32 over bf16-rounded operands. Returns (out, valid_mask)."""
    rows = A.shape[0]
    out = np.zeros((rows, Wd.shape[2]), np.float32)
    mask = np.zeros(rows, bool)
    Ab, Wb = _bf16(A).astype(np.float64), _bf16(Wd).astype(np.float64)
    for t, e in enumerate(eot):
        if e < 0:
            continue
        sl = slice(32 * t, 32 * (t + 1))
        with np.errstate(all="ignore"):   # numpy-on-Accelerate emits spurious FP warnings
            out[sl] = Ab[sl] @ Wb[e]      # for strided small matmuls; values are correct
        if bias is not None:
            out[sl] += _bf16(bias[e])
        mask[sl] = True
    return out, mask


def _inputs(rng, k_dim=K_DIM, n_out=N_OUT):
    A = rng.standard_normal((ROWS, k_dim)).astype(np.float32)
    W = (rng.standard_normal((E, k_dim, n_out)) / np.sqrt(k_dim)).astype(np.float32)
    return A, W


@pytest.mark.parametrize("format", list(MOE_Q_FORMATS))
@pytest.mark.parametrize("with_bias", [False, True])
def test_rect_q(format, with_bias):
    rng = np.random.default_rng(0)
    A, W = _inputs(rng)
    Wq = quantize_expert_stack(W, format)
    Wd = dequantize_expert_stack(Wq, K_DIM, format)          # shared dequant values
    bias = rng.standard_normal((E, N_OUT)).astype(np.float32) if with_bias else None

    got = moe_grouped_gemm_rect_q(
        mx.array(A).astype(mx.bfloat16), mx.array(Wq), mx.array(EOT), format=format,
        bias=None if bias is None else mx.array(bias).astype(mx.bfloat16))
    mx.eval(got)
    got = np.array(got.astype(mx.float32))

    ref, mask = _ref_grouped(A, Wd, EOT, bias)
    np.testing.assert_allclose(got[mask], ref[mask], atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("format", ["mxfp4", "q8_0"])
@pytest.mark.parametrize("act", ["swiglu", "swiglu_oai"])
def test_swiglu_q(format, act):
    rng = np.random.default_rng(1)
    H, inter = 256, 64
    alpha, limit = 1.702, 7.0
    A = rng.standard_normal((ROWS, H)).astype(np.float32)
    W1 = (rng.standard_normal((E, H, 2 * inter)) / np.sqrt(H)).astype(np.float32)
    bias = (0.1 * rng.standard_normal((E, 2 * inter))).astype(np.float32)
    W1q = quantize_expert_stack(W1, format)
    W1d = dequantize_expert_stack(W1q, H, format)

    got = moe_grouped_gemm_swiglu_q(
        mx.array(A).astype(mx.bfloat16), mx.array(W1q), mx.array(EOT), format=format,
        bias=mx.array(bias).astype(mx.bfloat16), act=act, alpha=alpha, limit=limit)
    mx.eval(got)
    got = np.array(got.astype(mx.float32))

    pre, mask = _ref_grouped(A, W1d, EOT, bias)
    g, u = pre[:, :inter], pre[:, inter:]
    if act == "swiglu_oai":
        g = np.minimum(g, limit)
        u = np.clip(u, -limit, limit)
        ref = (g / (1.0 + np.exp(-alpha * g))) * (1.0 + u)
    else:
        ref = (g / (1.0 + np.exp(-g))) * u
    np.testing.assert_allclose(got[mask], ref[mask], atol=6e-2, rtol=6e-2)


def test_q8_0_exact_values_match_dense():
    """W chosen exactly representable in q8_0 (integer grid x power-of-two scale): the packed
    path must agree with the dense bf16 grouped GEMM on identical weight values. The two
    kernels differ only in bf16 simdgroup-MMA rounding (K-step 32 A@Wq^T vs K-step 16 A@W),
    whose measured noise floor is ~1% relative — same tolerance the dense kernel's own
    tests use (6e-2 at unit output magnitude)."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((ROWS, K_DIM)).astype(np.float32)
    Wint = rng.integers(-127, 128, size=(E, K_DIM, N_OUT)).astype(np.float32)
    Wint[:, ::32, :] = 127.0     # every 32-block (along K) hits absmax 127 -> scale exactly 2^-11
    W = Wint * 2.0 ** -11
    Wq = quantize_expert_stack(W, "q8_0")
    # the load-bearing exactness assertion: packed bytes decode to W bit-exactly
    np.testing.assert_allclose(dequantize_expert_stack(Wq, K_DIM, "q8_0"), W, atol=0)

    Ab = mx.array(A).astype(mx.bfloat16)
    got_q = moe_grouped_gemm_rect_q(Ab, mx.array(Wq), mx.array(EOT), format="q8_0")
    got_d = moe_grouped_gemm_rect(Ab, mx.array(W).astype(mx.bfloat16), mx.array(EOT))
    mx.eval(got_q, got_d)
    gq = np.array(got_q.astype(mx.float32))
    gd = np.array(got_d.astype(mx.float32))
    mask = np.repeat(EOT >= 0, 32)
    np.testing.assert_allclose(gq[mask], gd[mask], atol=6e-2, rtol=6e-2)


def test_moe_mlp_quant_end_to_end():
    """Full quant moe_mlp vs the dense moe_mlp run on the dequantized weights."""
    rng = np.random.default_rng(3)
    T, H, inter, k = 64, 256, 64, 2
    x = rng.standard_normal((T, H)).astype(np.float32)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    W1 = (rng.standard_normal((E, H, 2 * inter)) / np.sqrt(H)).astype(np.float32)
    W2 = (rng.standard_normal((E, inter, H)) / np.sqrt(inter)).astype(np.float32)
    W1q = quantize_expert_stack(W1, "mxfp4")
    W2q = quantize_expert_stack(W2, "mxfp4")
    W1d = dequantize_expert_stack(W1q, H, "mxfp4")
    W2d = dequantize_expert_stack(W2q, inter, "mxfp4")

    xb = mx.array(x).astype(mx.bfloat16)
    lg = mx.array(logits)
    got = moe_mlp(xb, lg, mx.array(W1q), mx.array(W2q), k, quant_format="mxfp4")
    ref = moe_mlp(xb, lg, mx.array(W1d).astype(mx.bfloat16),
                  mx.array(W2d).astype(mx.bfloat16), k)
    mx.eval(got, ref)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                               np.array(ref.astype(mx.float32)), atol=6e-2, rtol=6e-2)
