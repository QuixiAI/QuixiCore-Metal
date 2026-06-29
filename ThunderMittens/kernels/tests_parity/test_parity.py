"""Cross-backend parity tests: the MLX and PyTorch-MPS backends run the SAME
compiled metallib kernel, so for identical inputs they must produce (near) identical
output. This is the strongest guarantee for the dual-backend design and catches any
host-ABI drift between <kernel>.cpp (MLX) and torch_kernels.mm (Torch).

Requires both mlx and torch; skips cleanly if either is missing. Run from kernels/:

    python -m pytest tests_parity/test_parity.py -v
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk  # the type-dispatching API  # noqa: E402


def _mk(arr, fw, dtype="bf16"):
    """Build a matched input on each framework from one numpy fp32 array."""
    if fw == "torch":
        t = torch.from_numpy(arr)
        t = t.to(torch.bfloat16) if dtype == "bf16" else t.to(torch.float32)
        return t.to("mps")
    a = mx.array(arr)
    return a.astype(mx.bfloat16) if dtype == "bf16" else a.astype(mx.float32)


def _np(x):
    """Bring an mlx array or torch tensor back to fp32 numpy."""
    if type(x).__module__.split(".")[0] == "torch":
        return x.detach().float().cpu().numpy()
    mx.eval(x)
    return np.array(x.astype(mx.float32))


def _assert_parity(o_mlx, o_torch, atol):
    mx.eval(o_mlx)
    torch.mps.synchronize()
    a, b = _np(o_mlx), _np(o_torch)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = float(np.max(np.abs(a - b)))
    assert d <= atol, f"MLX vs MPS max|diff|={d} (atol={atol})"


@pytest.mark.parametrize("shape", [(2, 128, 1024), (1, 256, 768), (8, 256)])
def test_layernorm_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    b = rng.standard_normal((D,)).astype(np.float32)
    om = tk.layernorm(_mk(x, "mlx"), _mk(w, "mlx"), _mk(b, "mlx"))
    ot = tk.layernorm(_mk(x, "torch"), _mk(w, "torch"), _mk(b, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
def test_add_rt_parity(shape, dtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    y = rng.standard_normal(shape).astype(np.float32)
    om = tk.add_rt(_mk(x, "mlx", dtype), _mk(y, "mlx", dtype))
    ot = tk.add_rt(_mk(x, "torch", dtype), _mk(y, "torch", dtype))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("dtype,atol", [("f32", 1e-3), ("bf16", 1e-2)])
@pytest.mark.parametrize("nkm", [(32, 16, 32), (128, 64, 128)])
def test_matmul_parity(nkm, dtype, atol):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    y = rng.random((K, M), dtype=np.float32)
    om = tk.matmul_custom(_mk(x, "mlx", dtype), _mk(y, "mlx", dtype))
    ot = tk.matmul_custom(_mk(x, "torch", dtype), _mk(y, "torch", dtype))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 2, 128, 128)])
def test_attn_fwd_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.attn_fwd(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.attn_fwd(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1e-2)
