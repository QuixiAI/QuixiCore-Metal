# Copyright © 2023 Apple Inc.
"""ThunderMittens kernels — unified Python API.

`tk.<kernel>(x, ...)` auto-routes by the input type:
  - mlx.core.array   -> the MLX backend (tk._ext, built via setup.py build_ext)
  - torch.Tensor     -> the PyTorch MPS backend (tk_torch)

Backends are imported lazily, so you only need the framework whose tensors you pass
(e.g. a PyTorch-only user never triggers the MLX import).
"""

# --- lazy backend loaders ---
_mlx_ext = None
_torch_backend = None


def _mlx():
    global _mlx_ext
    if _mlx_ext is None:
        from . import _ext as e  # compiled MLX extension
        _mlx_ext = e
    return _mlx_ext


def _torch():
    global _torch_backend
    if _torch_backend is None:
        import tk_torch  # standalone PyTorch MPS backend
        _torch_backend = tk_torch
    return _torch_backend


def _is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


# --- dispatching kernels ---
def layernorm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().layernorm(x, weight, bias, eps)
    return _mlx().layernorm(x, weight, bias, eps=eps)


def add_rt(x, y):
    """Elementwise x + y. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().add_rt(x, y)
    return _mlx().add_rt(x, y)


def matmul_custom(x, y):
    """(N,K) @ (K,M) GEMM. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().matmul_custom(x, y)
    return _mlx().matmul_custom(x, y)


def attn_fwd(q, k, v):
    """Non-causal attention forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_fwd(q, k, v)
    return _mlx().attn_fwd(q, k, v)
