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


# ---- q4_0 : { float16 d; uint8 qs[16]; } = 18 bytes, 32 weights/block, value = d*(nibble-8).
# Nibble packing (ggml): weight i (i<16) in low nibble of qs[i]; weight i+16 in high nibble. ----
Q4_0_BLOCK_K = 32
Q4_0_BLOCK_BYTES = 18


def quantize_q4_0(W: np.ndarray) -> np.ndarray:
    """W: (N, K) float, K % 32 == 0 -> packed uint8 (N, K//32, 18)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    assert K % Q4_0_BLOCK_K == 0, "K must be a multiple of 32"
    nb = K // Q4_0_BLOCK_K
    Wb = W.reshape(N, nb, Q4_0_BLOCK_K)
    amax = np.abs(Wb).max(axis=2)                              # (N, nb)
    d = (amax / 7.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    q = np.clip(np.rint(Wb / d_safe[..., None]) + 8, 0, 15).astype(np.uint8)  # 0..15
    out = np.zeros((N, nb, Q4_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    lo, hi = q[:, :, 0:16], q[:, :, 16:32]
    out[:, :, 2:Q4_0_BLOCK_BYTES] = (lo | (hi << 4)).astype(np.uint8)
    return out


def dequantize_q4_0(packed: np.ndarray) -> np.ndarray:
    """packed uint8 (N, nb, 18) -> W (N, nb*32) float32."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nb, nbytes = packed.shape
    assert nbytes == Q4_0_BLOCK_BYTES
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16)
    d = d.astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:Q4_0_BLOCK_BYTES].astype(np.int32)     # (N, nb, 16)
    lo = qs & 0x0F
    hi = qs >> 4
    nib = np.concatenate([lo, hi], axis=2).astype(np.float32)  # cols 0..15 lo, 16..31 hi
    return ((nib - 8.0) * d).reshape(N, nb * Q4_0_BLOCK_K)


# ---- q4_K : { float16 d; float16 dmin; uint8 scales[12]; uint8 qs[128]; } = 144 bytes,
# 256-weight super-block = 8 sub-blocks of 32; per sub-block 6-bit scale `sc` + 6-bit min `m`
# packed GGUF-style; value = (d*sc)*nibble - (dmin*m). (Simplified affine sub-block quantizer;
# the oracle is dequantize(Wq), so quant quality doesn't affect the kernel-vs-oracle test.) ----
Q4_K_BLOCK_K = 256
Q4_K_BLOCK_BYTES = 144


def quantize_q4_K(W: np.ndarray) -> np.ndarray:
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    assert K % Q4_K_BLOCK_K == 0, "K must be a multiple of 256"
    nb = K // Q4_K_BLOCK_K
    sub = W.reshape(N, nb, 8, 32)
    mn = sub.min(axis=3)                                   # (N, nb, 8)
    mx = sub.max(axis=3)
    scale_sub = (mx - mn) / 15.0
    eff_min = np.maximum(-mn, 0.0)                          # value = scale*q - eff_min
    d = (scale_sub.max(axis=2) / 63.0).astype(np.float32)  # (N, nb)
    dmin = (eff_min.max(axis=2) / 63.0).astype(np.float32)
    ds, dms = np.where(d == 0, 1.0, d), np.where(dmin == 0, 1.0, dmin)
    sc = np.clip(np.rint(scale_sub / ds[..., None]), 0, 63).astype(np.int32)  # (N, nb, 8)
    m = np.clip(np.rint(eff_min / dms[..., None]), 0, 63).astype(np.int32)
    ssafe = np.where(scale_sub == 0, 1.0, scale_sub)
    q = np.clip(np.rint((sub - mn[..., None]) / ssafe[..., None]), 0, 15).astype(np.int32)  # (N,nb,8,32)

    out = np.zeros((N, nb, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:4] = dmin.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    for j in range(4):  # pack 8 sc + 8 m (6-bit each) into scales[12] (inverse of get_scale_min_k4)
        out[:, :, 4 + j] = ((sc[:, :, j] & 63) | (((sc[:, :, j + 4] >> 4) & 3) << 6)).astype(np.uint8)
        out[:, :, 8 + j] = ((m[:, :, j] & 63) | (((m[:, :, j + 4] >> 4) & 3) << 6)).astype(np.uint8)
        out[:, :, 12 + j] = ((sc[:, :, j + 4] & 0x0F) | ((m[:, :, j + 4] & 0x0F) << 4)).astype(np.uint8)
    for chunk in range(4):  # pack qs[128]: byte chunk*32+r = sub[2*chunk][r] | sub[2*chunk+1][r]<<4
        lo, hi = q[:, :, 2 * chunk, :], q[:, :, 2 * chunk + 1, :]
        out[:, :, 16 + chunk * 32: 16 + chunk * 32 + 32] = (lo | (hi << 4)).astype(np.uint8)
    return out


def dequantize_q4_K(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nb, nbytes = packed.shape
    assert nbytes == Q4_K_BLOCK_BYTES
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    dmin = np.ascontiguousarray(packed[:, :, 2:4]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    s = packed[:, :, 4:16].astype(np.int32)
    sc = np.zeros((N, nb, 8), np.int32); m = np.zeros((N, nb, 8), np.int32)
    for j in range(4):
        sc[:, :, j] = s[:, :, j] & 63
        m[:, :, j] = s[:, :, j + 4] & 63
    for j in range(4, 8):
        sc[:, :, j] = (s[:, :, j + 4] & 0x0F) | ((s[:, :, j - 4] >> 6) << 4)
        m[:, :, j] = (s[:, :, j + 4] >> 4) | ((s[:, :, j] >> 6) << 4)
    qs = packed[:, :, 16:144].astype(np.int32)
    q = np.zeros((N, nb, 256), np.int32)
    for chunk in range(4):
        b = qs[:, :, chunk * 32: chunk * 32 + 32]
        q[:, :, chunk * 64: chunk * 64 + 32] = b & 0x0F
        q[:, :, chunk * 64 + 32: chunk * 64 + 64] = b >> 4
    sub_of_col = np.arange(256) // 32
    val = d[..., None] * sc[:, :, sub_of_col] * q - dmin[..., None] * m[:, :, sub_of_col]
    return val.astype(np.float32).reshape(N, nb * 256)


# ---- kU4B8 : GPTQ/Marlin grouped int4, group=128. { float16 scale; uint8 qs[64]; } = 66 bytes.
# value = scale*(nibble-8). Nibble packing like q4_0 (col<64 low of qs[col]; col>=64 high of qs[col-64]). ----
KU4B8_BLOCK_K = 128
KU4B8_BLOCK_BYTES = 66


def quantize_kU4B8(W: np.ndarray) -> np.ndarray:
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    assert K % KU4B8_BLOCK_K == 0, "K must be a multiple of 128"
    nb = K // KU4B8_BLOCK_K
    Wb = W.reshape(N, nb, KU4B8_BLOCK_K)
    amax = np.abs(Wb).max(axis=2)
    d = (amax / 7.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    q = np.clip(np.rint(Wb / d_safe[..., None]) + 8, 0, 15).astype(np.uint8)   # (N, nb, 128)
    out = np.zeros((N, nb, KU4B8_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    lo, hi = q[:, :, 0:64], q[:, :, 64:128]
    out[:, :, 2:KU4B8_BLOCK_BYTES] = (lo | (hi << 4)).astype(np.uint8)
    return out


def dequantize_kU4B8(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nb, nbytes = packed.shape
    assert nbytes == KU4B8_BLOCK_BYTES
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16)
    d = d.astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:KU4B8_BLOCK_BYTES].astype(np.int32)   # (N, nb, 64)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2).astype(np.float32)  # cols 0..63 lo, 64..127 hi
    return ((nib - 8.0) * d).reshape(N, nb * KU4B8_BLOCK_K)


# ---- kU4 : AWQ grouped int4, group=128, per-group zero-point.
# { float16 scale; float16 zp; uint8 qs[64]; } = 68 bytes. value = scale*(nibble - zp). ----
KU4_BLOCK_K = 128
KU4_BLOCK_BYTES = 68


def quantize_kU4(W: np.ndarray) -> np.ndarray:
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    assert K % KU4_BLOCK_K == 0, "K must be a multiple of 128"
    nb = K // KU4_BLOCK_K
    Wb = W.reshape(N, nb, KU4_BLOCK_K)
    mn, mx = Wb.min(axis=2), Wb.max(axis=2)
    scale = ((mx - mn) / 15.0).astype(np.float32)
    ssafe = np.where(scale == 0, 1.0, scale)
    zp = np.clip(np.rint(-mn / ssafe), 0, 15).astype(np.float32)            # (N, nb)
    q = np.clip(np.rint(Wb / ssafe[..., None] + zp[..., None]), 0, 15).astype(np.uint8)
    out = np.zeros((N, nb, KU4_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:4] = zp.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    lo, hi = q[:, :, 0:64], q[:, :, 64:128]
    out[:, :, 4:KU4_BLOCK_BYTES] = (lo | (hi << 4)).astype(np.uint8)
    return out


def dequantize_kU4(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nb, _ = packed.shape
    scale = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    zp = np.ascontiguousarray(packed[:, :, 2:4]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 4:KU4_BLOCK_BYTES].astype(np.int32)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2).astype(np.float32)
    return (scale * (nib - zp)).reshape(N, nb * KU4_BLOCK_K)


# ---- float-code codebooks (host encode = nearest decoded value, so host decode == kernel decode) ----
def _e4m3_decode_arr(b):
    b = b.astype(np.int32); s = (b >> 7) & 1; e = (b >> 3) & 0xF; m = b & 0x7
    val = np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7))
    return np.where(s == 1, -val, val).astype(np.float32)


def _e2m1_decode_arr(n):
    n = n.astype(np.int32); s = (n >> 3) & 1; e = (n >> 1) & 3; m = n & 1
    val = np.where(e == 0, np.where(m == 1, 0.5, 0.0), (1.0 + m * 0.5) * 2.0 ** (e - 1))
    return np.where(s == 1, -val, val).astype(np.float32)


_E4M3_CODES = np.array([b for b in range(256) if not (((b >> 3) & 0xF) == 0xF and (b & 7) == 7)], np.uint8)
_E4M3_VALS = _e4m3_decode_arr(_E4M3_CODES)
_E2M1_CODES = np.arange(16, dtype=np.uint8)
_E2M1_VALS = _e2m1_decode_arr(_E2M1_CODES)


def _nearest(x, codes, vals):
    idx = np.abs(x[..., None].astype(np.float32) - vals).argmin(axis=-1)
    return codes[idx]


# ---- fp8_e4m3 : per-group (32) half-scaled fp8. { half scale; uint8 qs[32]; } = 34 bytes. ----
def quantize_fp8_e4m3(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    scale = (np.abs(Wb).max(axis=2) / 448.0).astype(np.float32)
    ssafe = np.where(scale == 0, 1.0, scale)
    codes = _nearest(Wb / ssafe[..., None], _E4M3_CODES, _E4M3_VALS)        # (N,nb,32) uint8
    out = np.zeros((N, nb, 34), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:34] = codes
    return out


def dequantize_fp8_e4m3(packed):
    N, nb, _ = packed.shape
    scale = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    return (scale * _e4m3_decode_arr(packed[:, :, 2:34])).reshape(N, nb * 32)


def _pack_nibbles(codes, half_n):
    """codes (..., 2*half_n) -> bytes (..., half_n): byte r = codes[r] | codes[r+half_n]<<4."""
    lo, hi = codes[..., :half_n], codes[..., half_n:]
    return (lo | (hi << 4)).astype(np.uint8)


# ---- fp4_e2m1 : per-group (32) half-scaled fp4 (nibbles). { half scale; uint8 qs[16]; } = 18 bytes. ----
def quantize_fp4_e2m1(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    scale = (np.abs(Wb).max(axis=2) / 6.0).astype(np.float32)
    ssafe = np.where(scale == 0, 1.0, scale)
    codes = _nearest(Wb / ssafe[..., None], _E2M1_CODES, _E2M1_VALS)        # (N,nb,32)
    out = np.zeros((N, nb, 18), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:18] = _pack_nibbles(codes, 16)
    return out


def dequantize_fp4_e2m1(packed):
    N, nb, _ = packed.shape
    scale = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:18].astype(np.int32)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
    return (scale * _e2m1_decode_arr(nib)).reshape(N, nb * 32)


# ---- mxfp8 : 32-block, e8m0 power-of-two scale + fp8 e4m3. { uint8 e8m0; uint8 qs[32]; } = 33 bytes. ----
def quantize_mxfp8(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    amax = np.abs(Wb).max(axis=2)
    exp = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30) / 448.0)), 0.0)
    e8m0 = np.clip(exp + 127, 0, 254).astype(np.int32)                      # (N,nb)
    scale = (2.0 ** (e8m0 - 127)).astype(np.float32)
    codes = _nearest(Wb / scale[..., None], _E4M3_CODES, _E4M3_VALS)
    out = np.zeros((N, nb, 33), np.uint8)
    out[:, :, 0] = e8m0.astype(np.uint8)
    out[:, :, 1:33] = codes
    return out


def dequantize_mxfp8(packed):
    N, nb, _ = packed.shape
    scale = (2.0 ** (packed[:, :, 0].astype(np.int32) - 127)).astype(np.float32)[..., None]
    return (scale * _e4m3_decode_arr(packed[:, :, 1:33])).reshape(N, nb * 32)


# ---- nvfp4 : 16-block, fp8 e4m3 block scale + fp4 e2m1 codes. { uint8 e4m3; uint8 qs[8]; } = 9 bytes. ----
def quantize_nvfp4(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 16
    Wb = W.reshape(N, nb, 16)
    target = (np.abs(Wb).max(axis=2) / 6.0).astype(np.float32)
    scale_byte = _nearest(target, _E4M3_CODES, _E4M3_VALS)                  # (N,nb) uint8
    scale = _e4m3_decode_arr(scale_byte)
    ssafe = np.where(scale == 0, 1.0, scale)
    codes = _nearest(Wb / ssafe[..., None], _E2M1_CODES, _E2M1_VALS)        # (N,nb,16)
    out = np.zeros((N, nb, 9), np.uint8)
    out[:, :, 0] = scale_byte
    out[:, :, 1:9] = _pack_nibbles(codes, 8)
    return out


def dequantize_nvfp4(packed):
    N, nb, _ = packed.shape
    scale = _e4m3_decode_arr(packed[:, :, 0])[..., None]
    qs = packed[:, :, 1:9].astype(np.int32)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
    return (scale * _e2m1_decode_arr(nib)).reshape(N, nb * 16)


# ---- mxfp4 : 32-block, e8m0 power-of-two scale + fp4 e2m1 codes. { uint8 e8m0; uint8 qs[16]; } = 17 bytes. ----
def quantize_mxfp4(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    amax = np.abs(Wb).max(axis=2)
    exp = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0.0)
    e8m0 = np.clip(exp + 127, 0, 254).astype(np.int32)
    scale = (2.0 ** (e8m0 - 127)).astype(np.float32)
    codes = _nearest(Wb / scale[..., None], _E2M1_CODES, _E2M1_VALS)        # (N,nb,32)
    out = np.zeros((N, nb, 17), np.uint8)
    out[:, :, 0] = e8m0.astype(np.uint8)
    out[:, :, 1:17] = _pack_nibbles(codes, 16)
    return out


def dequantize_mxfp4(packed):
    N, nb, _ = packed.shape
    scale = (2.0 ** (packed[:, :, 0].astype(np.int32) - 127)).astype(np.float32)[..., None]
    qs = packed[:, :, 1:17].astype(np.int32)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
    return (scale * _e2m1_decode_arr(nib)).reshape(N, nb * 32)


# Format registry: name -> (quantize, dequantize). Drives the parametrized tests.
QUANT_FORMATS = {
    "q8_0": (quantize_q8_0, dequantize_q8_0),
    "q4_0": (quantize_q4_0, dequantize_q4_0),
    "q4_K": (quantize_q4_K, dequantize_q4_K),
    "kU4B8": (quantize_kU4B8, dequantize_kU4B8),
    "kU4": (quantize_kU4, dequantize_kU4),
    "fp8_e4m3": (quantize_fp8_e4m3, dequantize_fp8_e4m3),
    "fp4_e2m1": (quantize_fp4_e2m1, dequantize_fp4_e2m1),
    "mxfp8": (quantize_mxfp8, dequantize_mxfp8),
    "nvfp4": (quantize_nvfp4, dequantize_nvfp4),
    "mxfp4": (quantize_mxfp4, dequantize_mxfp4),
}
