"""Quantized-KV attention: softmax(QK^T)·V with K,V dequantized from blocks. Validated against a
reference attention computed on the *dequantized* K/V (so quant error is isolated from the kernel),
across q8_0 / q4_0 / fp8_e4m3 and D in {64,128}. Run from kernels/:
    python -m pytest attn_q/correctness/test_attn_q.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import attn_q
from tk.quant import quantize_kv, dequantize_kv


def _ref_attn(q, k, v):                                       # (B,H,N,D) fp32, non-causal
    s = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s); p = p / p.sum(-1, keepdims=True)
    return p @ v


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "fp8_e4m3"])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_q(D, fmt):
    B, H, N = 1, 2, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq, Vq = quantize_kv(k, fmt), quantize_kv(v, fmt)
    dk, dv = dequantize_kv(Kq, fmt), dequantize_kv(Vq, fmt)    # the K/V the kernel actually sees
    got = attn_q(mx.array(q).astype(mx.bfloat16), mx.array(Kq), mx.array(Vq), format=fmt)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = _ref_attn(q, dk, dv)
    assert got.shape == (B, H, N, D)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 0.1, f"{fmt} D{D} rel {rel}"


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "fp8_e4m3"])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_q_causal(D, fmt):
    B, H, N = 1, 2, 64
    rng = np.random.default_rng(1)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq, Vq = quantize_kv(k, fmt), quantize_kv(v, fmt)
    dk, dv = dequantize_kv(Kq, fmt), dequantize_kv(Vq, fmt)
    got = attn_q(mx.array(q).astype(mx.bfloat16), mx.array(Kq), mx.array(Vq), format=fmt, causal=True)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    s = (q @ np.swapaxes(dk, -1, -2)) / np.sqrt(D)
    mask = np.triu(np.ones((N, N), bool), 1)
    s = np.where(mask, -1e30, s); s = s - s.max(-1, keepdims=True)
    p = np.exp(s); p = p / p.sum(-1, keepdims=True); ref = p @ dv
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 0.1, f"{fmt} D{D} causal rel {rel}"


@pytest.mark.parametrize("fmt", ["q8_0", "fp8_e4m3"])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_q_multiwarp(D, fmt):
    B, H, N = 1, 4, 128                                       # N % (8*4) == 0 for multiwarp
    rng = np.random.default_rng(2)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq, Vq = quantize_kv(k, fmt), quantize_kv(v, fmt)
    dk, dv = dequantize_kv(Kq, fmt), dequantize_kv(Vq, fmt)
    got = attn_q(mx.array(q).astype(mx.bfloat16), mx.array(Kq), mx.array(Vq), format=fmt, multiwarp=True)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = _ref_attn(q, dk, dv)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 0.1, f"{fmt} D{D} mw rel {rel}"


if __name__ == "__main__":
    for f in ["q8_0", "q4_0", "fp8_e4m3"]:
        test_attn_q(64, f); test_attn_q(128, f)
    print("ok")
