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


def _ceil(a, m):
    return ((a + m - 1) // m) * m


def matmul_custom(x, y):
    """(N,K) @ (K,M) GEMM, arbitrary shapes. Accepts mlx.array or torch.Tensor (MPS).

    The kernel is tile-blocked (needs N%32, M%32, K%16); arbitrary shapes are handled by
    zero-padding to the next tile multiple and slicing the result (shared-tile staging /
    a truly general kernel is a perf follow-up)."""
    if _is_torch(x):
        return _torch().matmul_custom(x, y)  # tk_torch pads/slices
    import mlx.core as mx

    N, K = x.shape[-2], x.shape[-1]
    M = y.shape[-1]
    Np, Kp, Mp = _ceil(N, 32), _ceil(K, 16), _ceil(M, 32)
    xp = mx.pad(x, [(0, Np - N), (0, Kp - K)]) if (Np != N or Kp != K) else x
    yp = mx.pad(y, [(0, Kp - K), (0, Mp - M)]) if (Kp != K or Mp != M) else y
    out = _mlx().matmul_custom(xp, yp)
    return out[:N, :M]


def attn_fwd(q, k, v):
    """Non-causal attention forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_fwd(q, k, v)
    return _mlx().attn_fwd(q, k, v)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rms_norm(x, weight, eps)
    return _mlx().rms_norm(x, weight, eps=eps)


def softmax(x):
    """Softmax over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().softmax(x)
    return _mlx().softmax(x)


def rotary(x, cos, sin):
    """RoPE (split-half). x is (B,H,N,D), cos/sin (N,D/2). mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rotary(x, cos, sin)
    return _mlx().rotary(x, cos, sin)


def gelu(x):
    """GELU (tanh approx) over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().gelu(x)
    return _mlx().gelu(x)


def attn_causal(q, k, v):
    """Causal attention forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_causal(q, k, v)
    return _mlx().attn_causal(q, k, v)
