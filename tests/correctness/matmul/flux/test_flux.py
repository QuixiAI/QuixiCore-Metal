"""Correctness tests for the fused Flux kernels.

  flux_gelu: gelu(x @ w + bias)
  flux_gate: (x @ w + bias) * gate + residual

Per-column bias/gate vary across columns so a broadcast/alignment bug surfaces.
Run from kernels/:  python -m pytest flux/correctness/test_flux.py -v
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tk import flux_gelu, flux_gate

# (N, K, M) — tile multiples: N%32, M%32, K%16.
SHAPES = [(32, 16, 32), (64, 32, 64), (128, 64, 128)]


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 1e-2), (mx.bfloat16, 0.5)])
@pytest.mark.parametrize("shape", SHAPES)
def test_flux_gelu(shape, dtype, atol):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(dtype)
    w = mx.random.uniform(shape=(K, M)).astype(dtype)
    bias = mx.random.normal((M,)).astype(dtype)
    got = flux_gelu(x, w, bias)
    ref = nn.gelu_approx(
        x.astype(mx.float32) @ w.astype(mx.float32) + bias.astype(mx.float32)).astype(dtype)
    mx.eval(got, ref)
    assert got.shape == (N, M)
    assert mx.allclose(got, ref, atol=atol, rtol=atol), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32)-ref.astype(mx.float32))).item()}"


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 1e-2), (mx.bfloat16, 0.5)])
@pytest.mark.parametrize("shape", SHAPES)
def test_flux_gate(shape, dtype, atol):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(dtype)
    w = mx.random.uniform(shape=(K, M)).astype(dtype)
    bias = mx.random.normal((M,)).astype(dtype)
    gate = mx.random.normal((M,)).astype(dtype)
    residual = mx.random.normal((N, M)).astype(dtype)
    got = flux_gate(x, w, bias, gate, residual)
    xf, wf = x.astype(mx.float32), w.astype(mx.float32)
    ref = ((xf @ wf + bias.astype(mx.float32)) * gate.astype(mx.float32)
           + residual.astype(mx.float32)).astype(dtype)
    mx.eval(got, ref)
    assert got.shape == (N, M)
    assert mx.allclose(got, ref, atol=atol, rtol=atol), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32)-ref.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_flux_gelu(shp, mx.float32, 1e-2)
        test_flux_gate(shp, mx.float32, 1e-2)
        print("ok", shp)
