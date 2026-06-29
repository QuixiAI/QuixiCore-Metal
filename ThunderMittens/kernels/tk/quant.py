"""Host-side weight quantization / packing for the ThunderMittens quantized kernels.

Block layouts mirror llama.cpp's GGUF formats (ggml-common.h). Each `quantize_<fmt>` returns a
packed uint8 array shaped (N, K//block_k, block_bytes); `dequantize_<fmt>` is the exact inverse,
defining the kernel's fp32 oracle:  out = dequantize(Wq) @ X.

All numpy so tests can feed either MLX or PyTorch.
"""

import numpy as np

# ---- q8_0 : { float16 d; int8 qs[32]; } = 34 bytes, 32 weights/block, value = d * q ----
Q8_0_BLOCK_K = 32
Q8_0_BLOCK_BYTES = 34


def quantize_q8_0(W: np.ndarray) -> np.ndarray:
    """W: (N, K) float, K % 32 == 0 -> packed uint8 (N, K//32, 34)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    assert K % Q8_0_BLOCK_K == 0, "K must be a multiple of 32"
    nb = K // Q8_0_BLOCK_K
    Wb = W.reshape(N, nb, Q8_0_BLOCK_K)
    amax = np.abs(Wb).max(axis=2)                              # (N, nb)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    qs = np.clip(np.rint(Wb / d_safe[..., None]), -127, 127).astype(np.int8)  # (N, nb, 32)
    out = np.zeros((N, nb, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:Q8_0_BLOCK_BYTES] = qs.view(np.uint8)
    return out


def dequantize_q8_0(packed: np.ndarray) -> np.ndarray:
    """packed uint8 (N, nb, 34) -> W (N, nb*32) float32."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nb, nbytes = packed.shape
    assert nbytes == Q8_0_BLOCK_BYTES
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16)
    d = d.astype(np.float32).reshape(N, nb, 1)                 # (N, nb, 1)
    qs = np.ascontiguousarray(packed[:, :, 2:Q8_0_BLOCK_BYTES]).view(np.int8).astype(np.float32)
    return (qs * d).reshape(N, nb * Q8_0_BLOCK_K)
