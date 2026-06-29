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


def flux_gelu(x, w, bias):
    """Fused gelu(x @ w + bias). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().flux_gelu(x, w, bias)
    return _mlx().flux_gelu(x, w, bias)


def flux_gate(x, w, bias, gate, residual):
    """Fused (x @ w + bias) * gate + residual. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().flux_gate(x, w, bias, gate, residual)
    return _mlx().flux_gate(x, w, bias, gate, residual)


def gemm_staged(x, y):
    """Multi-simdgroup threadgroup-staged GEMM (x @ y), tile-multiple shapes.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().gemm_staged(x, y)
    return _mlx().gemm_staged(x, y)


def attn_multiwarp(q, k, v):
    """Multi-warp flash attention forward (shared K/V). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_multiwarp(q, k, v)
    return _mlx().attn_multiwarp(q, k, v)


def linear_attn(q, k, v):
    """Non-causal linear attention Q@(K^T@V). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().linear_attn(q, k, v)
    return _mlx().linear_attn(q, k, v)


def hedgehog(q, k, v):
    """Hedgehog feature-map linear attention. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().hedgehog(q, k, v)
    return _mlx().hedgehog(q, k, v)


def lin_attn_causal(q, k, v):
    """Causal linear attention (chunked scan). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().lin_attn_causal(q, k, v)
    return _mlx().lin_attn_causal(q, k, v)


def mamba2(C, B, X, cumlog):
    """Mamba-2 / SSD forward. cumlog = cumsum(log a). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(C):
        return _torch().mamba2(C, B, X, cumlog)
    return _mlx().mamba2(C, B, X, cumlog)


def lin_attn_decay(q, k, v, slopes):
    """Decay / retention linear attention (RetNet / Lightning-Attention-2):
    out_i = sum_{j<=i} exp(-slope_h*(i-j)) * (q_i.k_j) * v_j. q,k,v (B,H,N,D) bf16, D=64; `slopes`
    is the per-head decay rate (H,). Builds the decay-log ramp cl=-slope*position internally and runs
    the retention kernel. Accepts mlx.array or torch.Tensor (MPS)."""
    import numpy as np
    B, H, N, _ = q.shape
    pos = np.arange(int(N), dtype=np.float32)
    sl = np.asarray(slopes, np.float32).reshape(int(H))
    cl = np.ascontiguousarray(
        np.broadcast_to(-(sl[:, None] * pos[None, :])[None], (int(B), int(H), int(N))), np.float32)
    if _is_torch(q):
        import torch
        return _torch().lin_attn_decay(q, k, v, torch.from_numpy(cl).to(q.device))
    import mlx.core as mx
    return _mlx().lin_attn_decay(q, k, v, mx.array(cl))


def cmplx_matmul(a, b):
    """Complex GEMM D=A@B; operands carry a leading size-2 (real,imag) axis: a (2,N,K),
    b (2,K,M) -> (2,N,M). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(a):
        return _torch().cmplx_matmul(a, b)
    return _mlx().cmplx_matmul(a, b)


def fftconv(x, fmat, twf, finv, twi, kf):
    """Monarch FFT convolution (N=S*S). Complex inputs with a leading size-2 (real,imag) axis:
    x (2,B,H,S,S), fmat/twf/finv/twi (2,S,S), kf (2,H,S,S) -> real (B,H,S,S).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().fftconv(x, fmat, twf, finv, twi, kf)
    return _mlx().fftconv(x, fmat, twf, finv, twi, kf)


def qgemm(wq, x, format="q8_0"):
    """Quantized GEMM (Marlin's method): out = dequantize(wq) @ x. wq is packed weight blocks
    (N, K//block_k, block_bytes) uint8; x is (K, M) float16 -> (N, M) float16.
    Routes batch-1 (M==1) to the qgemv decode path. Accepts mlx.array or torch.Tensor (MPS)."""
    if x.shape[-1] == 1:                       # batch-1 decode -> GEMV
        return qgemv(wq, x, format)
    if _is_torch(wq):
        return _torch().qgemm(wq, x, format)
    return _mlx().qgemm(wq, x, format=format)


def qgemm_direct(wq, x, format="q8_0"):
    """qgemm with dequant-direct-to-fragment (Marlin zero-shuffle, no threadgroup staging). MLX
    only (experimental perf variant of qgemm; same result). Falls back to qgemm on torch."""
    if _is_torch(wq):
        return _torch().qgemm(wq, x, format)
    return _mlx().qgemm_direct(wq, x, format=format)


def attn_q(q, kq, vq, format="q8_0", causal=False, multiwarp=False):
    """Quantized-KV flash attention: softmax(QK^T)·V with K,V given as quantized blocks (format).
    q bf16 (B,H,N,D); kq/vq uint8 (B,H,N,D/block_k,block_bytes) -> bf16 (B,H,N,D). D in {64,128}.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_q(q, kq, vq, format, causal, multiwarp)
    return _mlx().attn_q(q, kq, vq, format=format, causal=causal, multiwarp=multiwarp)


def qgemm_actorder(wq, x, perm, w_format="kU4B8", fused=False):
    """GPTQ act-order (desc_act): the weight is quantized in g_idx-permuted column (K) order so its
    groups are contiguous; recover W@X by gathering the activation rows by the same permutation, then
    running the standard qgemm. `perm` is a length-K index array (= argsort(g_idx)). A load-time
    reordering layer, not a new format. `fused=True` (MLX/torch) instead gathers the X K-rows inside
    the kernel (qgemm_actorder_k) — no materialized permuted-X copy; needs M%32==0, N%32==0, x fp16.
    Accepts mlx.array or torch.Tensor (MPS)."""
    import numpy as np
    if fused:
        if _is_torch(x):
            import torch
            p = torch.as_tensor(np.asarray(perm), dtype=torch.int32, device=x.device)
            return _torch().qgemm_actorder_k(wq, x.to(torch.float16), p, w_format)
        import mlx.core as mx
        return _mlx().qgemm_actorder_k(wq, x.astype(mx.float16),
                                       mx.array(np.asarray(perm, np.int32)), format=w_format)
    if _is_torch(x):
        import torch
        idx = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=x.device)
        return qgemm(wq, x.index_select(0, idx), w_format)
    import mlx.core as mx
    return qgemm(wq, mx.take(x, mx.array(np.asarray(perm, np.int32)), axis=0), w_format)


def qgemm_w8a8(wq, xq, w_scale, a_scale):
    """W8A8 prefill GEMM (M>1, bit-exact int32): out[n,m]=w_scale[n]*a_scale[m]*sum_k Wq[n,k]*Xq[m,k].
    wq int8 (N,K); xq int8 (M,K) token-major; w_scale (N,) half; a_scale (M,) half -> (N,M) half.
    NOTE: int prefill is perf-negative on Apple (no int matmul); use for exact int32 numerics."""
    if _is_torch(wq):
        return _torch().qgemm_w8a8(wq, xq, w_scale, a_scale)
    return _mlx().qgemm_w8a8(wq, xq, w_scale, a_scale)


def qgemm_w2a8(wq, xq, a_scale):
    """BitNet W2A8 prefill GEMM (M>1): ternary 2-bit weight x int8 act (M,K), per-group absmean scale
    * a_scale[m] -> (N,M) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_w2a8(wq, xq, a_scale)
    return _mlx().qgemm_w2a8(wq, xq, a_scale)


def qgemm_fp8_block2d(wq, x, scale2d):
    """fp8_block2d GEMM: codes-only fp8 weights (N,K/128,128) + a separate (N/128,K/128) tile scale
    (storage-optimal fp8_block). x (K,M) f16 -> (N,M) f16. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_blockscale(wq, x, scale2d)
    return _mlx().qgemm_blockscale(wq, x, scale2d)


def qgemm_fp8_scaled(wq, xq, w_scale, a_scale):
    """fp8 rank-1 scaled GEMM: BOTH operands fp8 e4m3 codes (wq (N,K), xq (K,M)), per-channel w_scale (N,)
    and per-token a_scale (M,) f16 -> (N,M) f16. out[n,m]=w_scale[n]*a_scale[m]*sum_k dequant·dequant.
    The fp8 analog of W8A8/SmoothQuant. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_fp8_scaled(wq, xq, w_scale, a_scale)
    return _mlx().qgemm_fp8_scaled(wq, xq, w_scale, a_scale)


def qgemv_w8a8(wq, xq, w_scale, a_scale):
    """W8A8/SmoothQuant decode GEMV: int8 weight (N,K) x int8 act (K,1) -> int32, *w_scale[n]*a_scale.
    w_scale (N,) half, a_scale (1,) half -> (N,1) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv_w8a8(wq, xq, w_scale, a_scale)
    return _mlx().qgemv_w8a8(wq, xq, w_scale, a_scale)


def qgemv_w2a8(wq, xq, a_scale):
    """BitNet W2A8 decode GEMV: ternary 2-bit weight (bitnet blocks) x int8 act (K,1) -> int32,
    per-group absmean scale * a_scale -> (N,1) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv_w2a8(wq, xq, a_scale)
    return _mlx().qgemv_w2a8(wq, xq, a_scale)


def qgemv(wq, x, format="q8_0"):
    """Quantized GEMV (batch-1 decode): out = dequantize(wq) @ x. wq packed weight blocks
    (N, K//block_k, block_bytes) uint8; x is (K, 1) float16 -> (N, 1) float16.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv(wq, x, format)
    return _mlx().qgemv(wq, x, format=format)


def qflux_gelu(wq, x, bias, format="q8_0"):
    """Quantized fused GEMM+GELU: gelu(dequantize(wq) @ x + bias). wq packed weight blocks;
    x (K,M) float16; bias (M,) float16 -> (N,M) float16. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qflux_gelu(wq, x, bias, format)
    return _mlx().qflux_gelu(wq, x, bias, format=format)


def _round_activation(x, act):
    """Snap activations x (K,M) to the 8-bit grid (int8/fp8), returning a fp16 array of the same
    framework. On Apple there's no int8/fp8 matmul, so W·A8 = round activations then the half GEMM
    (parity numerics). Rounding is done in numpy (a parity tool, not a perf path)."""
    import numpy as np
    from .quant import ACT_FORMATS
    if act not in ACT_FORMATS:
        raise ValueError(f"act must be one of {list(ACT_FORMATS)} or None, got {act!r}")
    if _is_torch(x):
        import torch
        xr = ACT_FORMATS[act](x.detach().float().cpu().numpy())[0]
        return torch.from_numpy(xr).to(x.device, torch.float16)
    import mlx.core as mx
    xr = ACT_FORMATS[act](np.array(x.astype(mx.float32)))[0]
    return mx.array(xr).astype(mx.float16)


def qmm(wq, x, w_format="q8_0", act=None):
    """Quantized matmul = dequantize(wq) @ x. Weight quantized via `w_format`; if `act` is
    "int8"/"fp8" the activations are also quantized (W·A8 parity: fp8 W8A8, int8 W8A8, int8 W4A8),
    else they stay fp16 (W·A16). Routes batch-1 (M==1) to the GEMV decode path. wq (N,K/bk,bytes)
    uint8; x (K,M) -> (N,M) float16. Accepts mlx.array or torch.Tensor (MPS)."""
    if act is not None:
        xq = _round_activation(x, act)
    elif _is_torch(x):
        import torch
        xq = x.to(torch.float16)
    else:
        import mlx.core as mx
        xq = x.astype(mx.float16)
    return qgemm(wq, xq, w_format)
