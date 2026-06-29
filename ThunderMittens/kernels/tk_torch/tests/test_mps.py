"""Correctness tests for the ThunderMittens PyTorch MPS backend.

Run from the kernels/ directory:

    python -m pytest tk_torch/tests/test_mps.py -v
"""

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk_torch  # noqa: E402


def _maxdiff(a, b):
    torch.mps.synchronize()
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_layernorm(shape):
    D = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    b = torch.randn(D, dtype=torch.bfloat16, device="mps")
    got = tk_torch.layernorm(x, w, b, 1e-5)
    exp = F.layer_norm(x, (D,), w, b, 1e-5)
    assert _maxdiff(got, exp) < 0.06


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 8), (64, 128), (128, 64)])
def test_add_rt(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="mps")
    y = torch.randn(shape, dtype=dtype, device="mps")
    assert _maxdiff(tk_torch.add_rt(x, y), x + y) < 0.02


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-2), (torch.bfloat16, 0.4)])
@pytest.mark.parametrize("nkm", [(32, 16, 32), (128, 64, 128), (256, 128, 256)])
def test_matmul_custom(nkm, dtype, atol):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=dtype, device="mps")
    y = torch.rand(K, M, dtype=dtype, device="mps")
    got = tk_torch.matmul_custom(x, y)
    exp = (x.float() @ y.float()).to(dtype)
    assert got.shape == (N, M)
    assert _maxdiff(got, exp) < atol


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 4, 512, 64), (2, 2, 128, 128)])
def test_attn_fwd(shape):
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.attn_fwd(q, k, v)
    # both use the default scale 1/sqrt(D); non-causal, no mask.
    exp = F.scaled_dot_product_attention(q, k, v)
    assert _maxdiff(got, exp) < 0.06


def test_dispatch_routes_torch_to_mps():
    """tk.<kernel>(torch.Tensor) routes to the MPS backend (no MLX needed)."""
    import tk

    D = 512
    torch.manual_seed(0)
    x = torch.randn(4, D, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    b = torch.randn(D, dtype=torch.bfloat16, device="mps")
    got = tk.layernorm(x, w, b)
    exp = F.layer_norm(x, (D,), w, b, 1e-5)
    assert _maxdiff(got, exp) < 0.06
