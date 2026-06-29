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


# ---- bitnet : BitNet b1.58 ternary {-1,0,+1}, group 32, per-group absmean scale.
# 2-bit codes (code in {0,1,2} -> value = scale*(code-1)), 4/byte. { half scale; uint8 qs[8]; } = 10 bytes. ----
def quantize_bitnet(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    scale = np.abs(Wb).mean(axis=2).astype(np.float32)                      # absmean (BitNet b1.58)
    ssafe = np.where(scale == 0, 1.0, scale)
    wq = np.clip(np.rint(Wb / ssafe[..., None]), -1, 1).astype(np.int32)    # ternary
    code = (wq + 1).astype(np.uint32).reshape(N, nb, 8, 4)                  # 0,1,2
    out = np.zeros((N, nb, 10), np.uint8)
    out[:, :, 0:2] = scale.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:10] = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4) | (code[..., 3] << 6)).astype(np.uint8)
    return out


def dequantize_bitnet(packed):
    packed = np.ascontiguousarray(packed, np.uint8)
    N, nb, _ = packed.shape
    scale = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:10].astype(np.int32)                               # (N, nb, 8)
    codes = np.stack([(qs >> (j * 2)) & 0x3 for j in range(4)], axis=-1).reshape(N, nb, 32)
    return (scale * (codes.astype(np.float32) - 1.0)).reshape(N, nb * 32)


# ---- Activation quantization (for W·A8 schemes: fp8 W8A8, int8 W8A8, int8 W4A8).
# On Apple there is no int8/fp8 matmul, so "A8" = snap activations to the 8-bit grid then run the
# existing dequant-to-half GEMM. This reproduces the W·A8 fake-quant numerics (parity), not speed.
# Per-token (per output column of X (K,M)) symmetric scales. Each returns:
#   (x_rounded float32  [= activation snapped to the grid, feed this as the fp16 GEMM input],
#    codes, scale). ----
def quantize_act_int8(X):
    X = np.ascontiguousarray(X, np.float32)
    s = (np.abs(X).max(axis=0, keepdims=True) / 127.0).astype(np.float32)   # (1, M) per-token
    ssafe = np.where(s == 0, 1.0, s)
    Xq = np.clip(np.rint(X / ssafe), -127, 127).astype(np.int8)
    return (Xq.astype(np.float32) * s), Xq, s


def quantize_act_fp8(X):
    X = np.ascontiguousarray(X, np.float32)
    s = (np.abs(X).max(axis=0, keepdims=True) / 448.0).astype(np.float32)
    ssafe = np.where(s == 0, 1.0, s)
    codes = _nearest(X / ssafe, _E4M3_CODES, _E4M3_VALS)
    return (_e4m3_decode_arr(codes) * s), codes, s


ACT_FORMATS = {"int8": quantize_act_int8, "fp8": quantize_act_fp8}


# ---- W8A8 / SmoothQuant weight: int8 (N,K) + per-channel (per-row) symmetric scale (N,).
# For the integer GEMV path (int8 weight x int8 activation -> int32). ----
def quantize_w8a8(W):
    W = np.ascontiguousarray(W, np.float32)
    s = (np.abs(W).max(axis=1) / 127.0).astype(np.float32)            # per-channel (N,)
    ssafe = np.where(s == 0, 1.0, s)
    Wq = np.clip(np.rint(W / ssafe[:, None]), -127, 127).astype(np.int8)
    return Wq, s


# ---- iq4_nl : GGUF non-linear int4 codebook. { half d; uint8 qs[16]; } = 18 bytes, 32 weights.
# A nibble indexes the 16-entry non-linear table; value = d * kvalues_iq4nl[idx]. q4_0 nibble layout. ----
_IQ4NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                         dtype=np.float32)


def _nearest_index(x, table):
    """Index of the nearest table entry for each element of x (table 1-D)."""
    return np.abs(x[..., None] - table).argmin(axis=-1).astype(np.uint8)


def quantize_iq4_nl(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32)
    d = (np.abs(Wb).max(axis=2) / 127.0).astype(np.float32)           # 127 = max |codebook|
    dsafe = np.where(d == 0, 1.0, d)
    idx = _nearest_index(Wb / dsafe[..., None], _IQ4NL_VALUES)        # (N,nb,32) in 0..15
    out = np.zeros((N, nb, 18), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:18] = (idx[:, :, :16] | (idx[:, :, 16:] << 4)).astype(np.uint8)   # lo | hi<<4
    return out


def dequantize_iq4_nl(packed):
    packed = np.ascontiguousarray(packed, np.uint8)
    N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = packed[:, :, 2:18].astype(np.int32)                         # (N,nb,16)
    idx = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)              # (N,nb,32): lo then hi
    return (d * _IQ4NL_VALUES[idx]).reshape(N, nb * 32)


# ---- iq4_xs : 256-superblock IQ4_NL. { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }
# = 136 bytes. 8 sub-blocks of 32; 6-bit sub-scale ls in [1,31] (stored ls+32 in [33,63], split as
# 4 low bits in scales_l + 2 high bits in scales_h). value = d*ls * kvalues_iq4nl[nibble]. ----
def quantize_iq4_xs(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)                                    # 8 sub-blocks of 32
    sub_scale = (np.abs(Wsb).max(axis=3) / 127.0).astype(np.float32)  # (N,nb,8) ideal sub scale
    d = (sub_scale.max(axis=2) / 31.0).astype(np.float32)            # (N,nb) super scale
    dsafe = np.where(d == 0, 1.0, d)
    ls = np.clip(np.rint(sub_scale / dsafe[..., None]), 1, 31).astype(np.int32)   # (N,nb,8) in [1,31]
    dl = d[..., None] * ls.astype(np.float32)                        # (N,nb,8) effective sub scale
    dlsafe = np.where(dl == 0, 1.0, dl)
    idx = _nearest_index(Wsb / dlsafe[..., None], _IQ4NL_VALUES)     # (N,nb,8,32) in 0..15
    ls_raw = (ls + 32).astype(np.uint32)                            # [33,63], 6-bit
    out = np.zeros((N, nb, 136), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    sh = (ls_raw >> 4) & 0x3                                        # (N,nb,8) high 2 bits
    scales_h = np.zeros((N, nb), np.uint32)
    for ib in range(8):
        scales_h |= (sh[:, :, ib] << (2 * ib))
    out[:, :, 2:4] = scales_h.astype(np.uint16).view(np.uint8).reshape(N, nb, 2)
    sl = (ls_raw & 0x0F).astype(np.uint8)                          # (N,nb,8) low 4 bits
    out[:, :, 4:8] = (sl[:, :, 0::2] | (sl[:, :, 1::2] << 4))       # 2 sub-blocks/byte
    lo = idx[:, :, :, :16]; hi = idx[:, :, :, 16:]                  # per sub-block nibbles
    out[:, :, 8:136] = (lo | (hi << 4)).reshape(N, nb, 128).astype(np.uint8)
    return out


def dequantize_iq4_xs(packed):
    packed = np.ascontiguousarray(packed, np.uint8)
    N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    scales_h = np.ascontiguousarray(packed[:, :, 2:4]).reshape(N, nb * 2).view(np.uint16).astype(np.int32).reshape(N, nb)
    scales_l = packed[:, :, 4:8].astype(np.int32)                  # (N,nb,4)
    qs = packed[:, :, 8:136].astype(np.int32).reshape(N, nb, 8, 16)  # 8 sub * 16 bytes
    out = np.zeros((N, nb, 8, 32), np.float32)
    for ib in range(8):
        sl = (scales_l[:, :, ib >> 1] >> (4 * (ib & 1))) & 0x0F
        sh = (scales_h >> (2 * ib)) & 0x3
        ls = (sl | (sh << 4)) - 32                                 # (N,nb)
        dl = (d * ls.astype(np.float32))[..., None]                # (N,nb,1)
        b = qs[:, :, ib, :]                                        # (N,nb,16)
        idx = np.concatenate([b & 0x0F, b >> 4], axis=-1)          # (N,nb,32)
        out[:, :, ib, :] = dl * _IQ4NL_VALUES[idx]
    return out.reshape(N, nb * 256)


# ---- iq2_xxs : E8-lattice 2.0625 bpw. { half d; uint16 qs[32]; } = 66 bytes, 256 weights.
# Decode mirrors ggml dequantize_iq2_xxs (grid lookup + ksigns + 4-bit sub-scale). The encoder
# produces a valid packed block (nearest grid entry per group of 8, signs in the low 7 bits, scale
# from the magnitude); kernel-vs-oracle only requires kernel decode == this dequantize. ----
from .quant_tables import IQ2XXS_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS

_IQ2XXS_GMAG = np.stack([((IQ2XXS_GRID >> (8 * e)) & 0xFF).astype(np.float32) for e in range(8)], axis=1)  # (256,8)


def dequantize_iq2_xxs(packed):
    packed = np.ascontiguousarray(packed, np.uint8)
    N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    qs = np.ascontiguousarray(packed[:, :, 2:66]).reshape(N, nb * 64).view(np.uint16).reshape(N, nb, 32).astype(np.uint32)
    out = np.zeros((N, nb, 256), np.float32)
    for ib in range(8):
        q2 = qs[:, :, 4 * ib:4 * ib + 4]
        aux_g = q2[:, :, 0] | (q2[:, :, 1] << 16)
        aux_s = q2[:, :, 2] | (q2[:, :, 3] << 16)
        dl = d * (0.5 + (aux_s >> 28).astype(np.float32)) * 0.25                  # (N,nb)
        for sub in range(4):
            g = (aux_g >> (8 * sub)) & 0xFF                                       # (N,nb)
            ge = IQ2XXS_GRID[g]                                                   # (N,nb) uint64
            signs = KSIGNS_IQ2XS[(aux_s >> (7 * sub)) & 127].astype(np.int32)     # (N,nb)
            for e in range(8):
                gv = ((ge >> np.uint64(8 * e)) & np.uint64(0xFF)).astype(np.float32)
                sgn = np.where(signs & int(KMASK_IQ2XS[e]), -1.0, 1.0)
                out[:, :, ib * 32 + sub * 8 + e] = dl * gv * sgn
    return out.reshape(N, nb * 256)


def quantize_iq2_xxs(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)
    dl_target = (np.abs(Wsb).mean(axis=3) / 8.0).astype(np.float32)               # ~ mean grid mag 8
    d = (4.0 * dl_target).max(axis=2) / 15.5                                      # (N,nb) super scale
    dsafe = np.where(d == 0, 1.0, d).astype(np.float32)
    s4 = np.clip(np.rint(4.0 * dl_target / dsafe[..., None] - 0.5), 0, 15).astype(np.int32)
    dl = dsafe[..., None] * (0.5 + s4) * 0.25
    dlsafe = np.where(dl == 0, 1.0, dl)
    qs = np.zeros((N, nb, 32), np.uint16)
    for ib in range(8):
        aux_s = (s4[:, :, ib].astype(np.uint32) << 28)
        aux_g = np.zeros((N, nb), np.uint32)
        for sub in range(4):
            seg = Wsb[:, :, ib, sub * 8:sub * 8 + 8]                              # (N,nb,8)
            target = np.abs(seg) / dlsafe[:, :, ib, None]
            dist = ((target[..., None, :] - _IQ2XXS_GMAG[None, None, :, :]) ** 2).sum(-1)  # (N,nb,256)
            g = dist.argmin(-1).astype(np.uint32)
            aux_g |= (g << (8 * sub))
            patt = np.zeros((N, nb), np.uint32)
            for i in range(8):
                patt |= ((seg[..., i] < 0).astype(np.uint32) << i)
            aux_s |= ((patt & 0x7F) << (7 * sub))
        qs[:, :, 4 * ib + 0] = (aux_g & 0xFFFF).astype(np.uint16)
        qs[:, :, 4 * ib + 1] = (aux_g >> 16).astype(np.uint16)
        qs[:, :, 4 * ib + 2] = (aux_s & 0xFFFF).astype(np.uint16)
        qs[:, :, 4 * ib + 3] = (aux_s >> 16).astype(np.uint16)
    out = np.zeros((N, nb, 66), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:66] = qs.view(np.uint8).reshape(N, nb, 64)
    return out


# ---- iq2_xs / iq3_xxs / iq1_s : the rest of the GGUF E8-lattice family. Decoders mirror the
# ggml-metal kernels exactly; encoders produce valid packed blocks (nearest grid entry per group),
# which is all the kernel-vs-oracle test needs. ----
from .quant_tables import (IQ2XS_GRID, IQ3XXS_GRID, IQ1S_GRID_GPU, IQ1S_DELTA)

_IQ2XS_GMAG = np.stack([((IQ2XS_GRID >> (8 * e)) & 0xFF).astype(np.float32) for e in range(8)], axis=1)   # (512,8)
_IQ3XXS_GMAG = np.stack([((IQ3XXS_GRID >> np.uint32(8 * e)) & np.uint32(0xFF)).astype(np.float32) for e in range(4)], axis=1)  # (256,4)
# iq1s grid: 8 nibbles per entry in weight order [b0&f,b1&f,b2&f,b3&f,b0>>4,b1>>4,b2>>4,b3>>4]
_IQ1S_NIB = np.zeros((len(IQ1S_GRID_GPU), 8), np.float32)
for _b in range(4):
    _byte = ((IQ1S_GRID_GPU >> np.uint32(8 * _b)) & np.uint32(0xF)).astype(np.float32)
    _IQ1S_NIB[:, _b] = ((IQ1S_GRID_GPU >> np.uint32(8 * _b)) & np.uint32(0xF)).astype(np.float32)
    _IQ1S_NIB[:, _b + 4] = ((IQ1S_GRID_GPU >> np.uint32(8 * _b + 4)) & np.uint32(0xF)).astype(np.float32)


def dequantize_iq2_xs(packed):
    packed = np.ascontiguousarray(packed, np.uint8); N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    qs = np.ascontiguousarray(packed[:, :, 2:66]).reshape(N, nb * 64).view(np.uint16).reshape(N, nb, 32).astype(np.uint32)
    scales = packed[:, :, 66:74].astype(np.int32)
    out = np.zeros((N, nb, 256), np.float32)
    for ib in range(8):
        for il in range(2):
            sc = (scales[:, :, ib] >> (4 * il)) & 0xF
            dl = d * (0.5 + sc.astype(np.float32)) * 0.25
            for sub2 in range(2):
                idx16 = qs[:, :, 4 * ib + 2 * il + sub2]
                ge = IQ2XS_GRID[idx16 & 511]
                signs = KSIGNS_IQ2XS[(idx16 >> 9) & 127].astype(np.int32)
                for e in range(8):
                    gv = ((ge >> np.uint64(8 * e)) & np.uint64(0xFF)).astype(np.float32)
                    sgn = np.where(signs & int(KMASK_IQ2XS[e]), -1.0, 1.0)
                    out[:, :, ib * 32 + il * 16 + sub2 * 8 + e] = dl * gv * sgn
    return out.reshape(N, nb * 256)


def quantize_iq2_xs(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)
    Whalf = Wsb.reshape(N, nb, 8, 2, 16)
    dl_t = (np.abs(Whalf).mean(axis=4) / 8.0).astype(np.float32)                   # (N,nb,8,2)
    d = (4.0 * dl_t).reshape(N, nb, 16).max(axis=2) / 15.5
    dsafe = np.where(d == 0, 1.0, d).astype(np.float32)
    sc4 = np.clip(np.rint(4.0 * dl_t / dsafe[..., None, None] - 0.5), 0, 15).astype(np.int32)
    dl = dsafe[..., None, None] * (0.5 + sc4) * 0.25
    qs = np.zeros((N, nb, 32), np.uint16); scales = np.zeros((N, nb, 8), np.uint8)
    for ib in range(8):
        for il in range(2):
            scales[:, :, ib] |= (sc4[:, :, ib, il].astype(np.uint8) << (4 * il))
            for sub2 in range(2):
                seg = Wsb[:, :, ib, il * 16 + sub2 * 8: il * 16 + sub2 * 8 + 8]
                dlh = dl[:, :, ib, il][..., None]
                target = np.abs(seg) / np.where(dlh == 0, 1.0, dlh)
                g = (((target[..., None, :] - _IQ2XS_GMAG[None, None]) ** 2).sum(-1)).argmin(-1).astype(np.uint32)
                patt = np.zeros((N, nb), np.uint32)
                for i in range(8):
                    patt |= ((seg[..., i] < 0).astype(np.uint32) << i)
                qs[:, :, 4 * ib + 2 * il + sub2] = ((g & 511) | ((patt & 0x7F) << 9)).astype(np.uint16)
    out = np.zeros((N, nb, 74), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:66] = qs.view(np.uint8).reshape(N, nb, 64)
    out[:, :, 66:74] = scales
    return out


def dequantize_iq3_xxs(packed):
    packed = np.ascontiguousarray(packed, np.uint8); N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    q3 = packed[:, :, 2:66].astype(np.uint32)                                      # grid indices
    gas = np.ascontiguousarray(packed[:, :, 66:98]).reshape(N, nb * 32).view(np.uint16).reshape(N, nb, 16).astype(np.uint32)
    out = np.zeros((N, nb, 256), np.float32)
    for ib in range(8):
        aux32 = gas[:, :, 2 * ib] | (gas[:, :, 2 * ib + 1] << 16)
        dl = d * (0.5 + (aux32 >> 28).astype(np.float32)) * 0.5
        for il in range(2):
            for r in range(4):
                ge = IQ3XXS_GRID[q3[:, :, 8 * ib + 4 * il + r]]
                signs = KSIGNS_IQ2XS[(aux32 >> (14 * il + 7 * (r >> 1))) & 127].astype(np.int32)
                for i in range(4):
                    gv = ((ge >> np.uint32(8 * i)) & np.uint32(0xFF)).astype(np.float32)
                    sgn = np.where(signs & int(KMASK_IQ2XS[i + 4 * (r & 1)]), -1.0, 1.0)
                    out[:, :, ib * 32 + il * 16 + r * 4 + i] = dl * gv * sgn
    return out.reshape(N, nb * 256)


def quantize_iq3_xxs(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)
    mg = float(_IQ3XXS_GMAG.mean())
    dl_t = (np.abs(Wsb).mean(axis=3) / mg).astype(np.float32)                       # (N,nb,8)
    d = (2.0 * dl_t).max(axis=2) / 15.5
    dsafe = np.where(d == 0, 1.0, d).astype(np.float32)
    s4 = np.clip(np.rint(2.0 * dl_t / dsafe[..., None] - 0.5), 0, 15).astype(np.int32)
    dl = dsafe[..., None] * (0.5 + s4) * 0.5
    q3 = np.zeros((N, nb, 64), np.uint8); gas = np.zeros((N, nb, 16), np.uint16)
    for ib in range(8):
        aux32 = (s4[:, :, ib].astype(np.uint32) << 28)
        dlsafe = np.where(dl[:, :, ib] == 0, 1.0, dl[:, :, ib])[..., None]
        for il in range(2):
            for r in range(4):
                seg = Wsb[:, :, ib, il * 16 + r * 4: il * 16 + r * 4 + 4]
                target = np.abs(seg) / dlsafe
                g = (((target[..., None, :] - _IQ3XXS_GMAG[None, None]) ** 2).sum(-1)).argmin(-1)
                q3[:, :, 8 * ib + 4 * il + r] = g.astype(np.uint8)
            for h in range(2):
                patt = np.zeros((N, nb), np.uint32)
                for sub_r in range(2):
                    r = 2 * h + sub_r
                    seg = Wsb[:, :, ib, il * 16 + r * 4: il * 16 + r * 4 + 4]
                    for i in range(4):
                        patt |= ((seg[..., i] < 0).astype(np.uint32) << (i + 4 * sub_r))
                aux32 |= ((patt & 0x7F) << (14 * il + 7 * h))
        gas[:, :, 2 * ib] = (aux32 & 0xFFFF).astype(np.uint16)
        gas[:, :, 2 * ib + 1] = (aux32 >> 16).astype(np.uint16)
    out = np.zeros((N, nb, 98), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:66] = q3
    out[:, :, 66:98] = gas.view(np.uint8).reshape(N, nb, 32)
    return out


def dequantize_iq1_s(packed):
    packed = np.ascontiguousarray(packed, np.uint8); N, nb, _ = packed.shape
    d = np.ascontiguousarray(packed[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    qs = packed[:, :, 2:34].astype(np.uint32)
    qh = np.ascontiguousarray(packed[:, :, 34:50]).reshape(N, nb * 16).view(np.uint16).reshape(N, nb, 8).astype(np.uint32)
    out = np.zeros((N, nb, 256), np.float32)
    for ib in range(8):
        qhv = qh[:, :, ib]
        dl = d * (2 * ((qhv >> 12) & 7) + 1).astype(np.float32)
        ml = dl * np.where(qhv & 0x8000, -1.0 - IQ1S_DELTA, -1.0 + IQ1S_DELTA)
        for il in range(2):
            h = qhv >> (6 * il)
            gi1 = qs[:, :, 4 * ib + 2 * il + 0] | ((h << 8) & 0x700)
            gi2 = qs[:, :, 4 * ib + 2 * il + 1] | ((h << 5) & 0x700)
            for which in range(4):
                ge = IQ1S_GRID_GPU[gi1] if which < 2 else IQ1S_GRID_GPU[gi2]
                for i in range(4):
                    b = (ge >> np.uint32(8 * i)) & np.uint32(0xFF)
                    nib = ((b >> np.uint32(4)) if (which & 1) else (b & np.uint32(0xF))).astype(np.float32)
                    out[:, :, ib * 32 + il * 16 + which * 4 + i] = dl * nib + ml
    return out.reshape(N, nb * 256)


def quantize_iq1_s(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)
    # 3-bit scale: dl = d*(2*sc+1), sc in 0..7. ml ~ -dl. target nibble ~ (w - ml)/dl ~ w/dl + 1.
    dl_t = (np.abs(Wsb).max(axis=3) / 7.0).astype(np.float32)                       # rough
    d = dl_t.max(axis=2) / 15.0
    dsafe = np.where(d == 0, 1.0, d).astype(np.float32)
    sc = np.clip(np.rint((dl_t / dsafe[..., None] - 1) / 2), 0, 7).astype(np.int32)  # (N,nb,8)
    dl = dsafe[..., None] * (2 * sc + 1)
    qs = np.zeros((N, nb, 32), np.uint8); qh = np.zeros((N, nb, 8), np.uint16)
    for ib in range(8):
        dlsafe = np.where(dl[:, :, ib] == 0, 1.0, dl[:, :, ib])[..., None]
        ml = -dl[:, :, ib][..., None] * (1.0 - IQ1S_DELTA)                          # sign bit = 0
        qhv = (sc[:, :, ib].astype(np.uint32) << 12)                                # scale bits
        for il in range(2):
            for grid_sel in range(2):                                              # two grid entries
                cols = (il * 16 + grid_sel * 8)
                seg = Wsb[:, :, ib, cols: cols + 8]                                # 8 weights
                target = (seg - ml) / dlsafe                                       # ~ nibble
                g = (((target[..., None, :] - _IQ1S_NIB[None, None]) ** 2).sum(-1)).argmin(-1).astype(np.uint32)
                qs[:, :, 4 * ib + 2 * il + grid_sel] = (g & 0xFF).astype(np.uint8)
                gh = (g >> 8) & 0x7                                                # 3 high bits
                shift = (0 if grid_sel == 0 else 3) + 6 * il
                qhv |= (gh << shift)
        qh[:, :, ib] = qhv.astype(np.uint16)
    out = np.zeros((N, nb, 50), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:34] = qs
    out[:, :, 34:50] = qh.view(np.uint8).reshape(N, nb, 16)
    return out


# ================= Phase 3: GGUF k-quant + legacy fan-out (byte layouts per ggml-common.h) =======
def _f16le(x, N, nb):  # pack float32 -> 2 little-endian uint8 bytes (N,nb,2)
    return x.astype(np.float16).view(np.uint8).reshape(N, nb, 2)


def quantize_q4_1(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 32
    Wb = W.reshape(N, nb, 32); mn = Wb.min(2); d = ((Wb.max(2) - mn) / 15.0).astype(np.float32)
    ds = np.where(d == 0, 1.0, d)
    q = np.clip(np.rint((Wb - mn[..., None]) / ds[..., None]), 0, 15).astype(np.uint8)
    out = np.zeros((N, nb, 20), np.uint8)
    out[:, :, 0:2] = _f16le(d, N, nb); out[:, :, 2:4] = _f16le(mn.astype(np.float32), N, nb)
    out[:, :, 4:20] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return out


def dequantize_q4_1(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    d = np.ascontiguousarray(p[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    m = np.ascontiguousarray(p[:, :, 2:4]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb, 1)
    qs = p[:, :, 4:20].astype(np.int32); nib = np.concatenate([qs & 0xF, qs >> 4], axis=2).astype(np.float32)
    return (d * nib + m).reshape(N, nb * 32)


def _pack_q5(W, asym):  # shared q5_0/q5_1 packer; asym=True -> d*q+m, else d*(q-16)
    N, K = W.shape; nb = K // 32; Wb = W.reshape(N, nb, 32)
    if asym:
        mn = Wb.min(2); d = ((Wb.max(2) - mn) / 31.0).astype(np.float32); ds = np.where(d == 0, 1.0, d)
        q = np.clip(np.rint((Wb - mn[..., None]) / ds[..., None]), 0, 31).astype(np.uint32)
    else:
        mn = None; d = (np.abs(Wb).max(2) / 15.0).astype(np.float32); ds = np.where(d == 0, 1.0, d)
        q = np.clip(np.rint(Wb / ds[..., None]) + 16, 0, 31).astype(np.uint32)
    nib = (q & 0xF).astype(np.uint8); hb = (q >> 4) & 1
    qh = np.zeros((N, nb), np.uint32)
    for c in range(32):
        qh |= (hb[:, :, c] << c)
    qs = (nib[:, :, :16] | (nib[:, :, 16:] << 4)).astype(np.uint8)
    return N, nb, d, mn, qh, qs


def quantize_q5_0(W):
    W = np.ascontiguousarray(W, np.float32); N, nb, d, _, qh, qs = _pack_q5(W, False)
    out = np.zeros((N, nb, 22), np.uint8)
    out[:, :, 0:2] = _f16le(d, N, nb)
    out[:, :, 2:6] = qh[..., None].astype(np.uint32).view(np.uint8).reshape(N, nb, 4)
    out[:, :, 6:22] = qs
    return out


def dequantize_q5_0(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    d = np.ascontiguousarray(p[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    qh = (p[:, :, 2].astype(np.uint32) | (p[:, :, 3].astype(np.uint32) << 8)
          | (p[:, :, 4].astype(np.uint32) << 16) | (p[:, :, 5].astype(np.uint32) << 24))
    qs = p[:, :, 6:22].astype(np.int32); out = np.zeros((N, nb, 32), np.float32)
    for col in range(32):
        nib = (qs[:, :, col] & 0xF) if col < 16 else (qs[:, :, col - 16] >> 4)
        q = nib | (((qh >> col) & 1) << 4)
        out[:, :, col] = d * (q.astype(np.float32) - 16)
    return out.reshape(N, nb * 32)


def quantize_q5_1(W):
    W = np.ascontiguousarray(W, np.float32); N, nb, d, mn, qh, qs = _pack_q5(W, True)
    out = np.zeros((N, nb, 24), np.uint8)
    out[:, :, 0:2] = _f16le(d, N, nb); out[:, :, 2:4] = _f16le(mn.astype(np.float32), N, nb)
    out[:, :, 4:8] = qh[..., None].astype(np.uint32).view(np.uint8).reshape(N, nb, 4)
    out[:, :, 8:24] = qs
    return out


def dequantize_q5_1(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    d = np.ascontiguousarray(p[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    m = np.ascontiguousarray(p[:, :, 2:4]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    qh = (p[:, :, 4].astype(np.uint32) | (p[:, :, 5].astype(np.uint32) << 8)
          | (p[:, :, 6].astype(np.uint32) << 16) | (p[:, :, 7].astype(np.uint32) << 24))
    qs = p[:, :, 8:24].astype(np.int32); out = np.zeros((N, nb, 32), np.float32)
    for col in range(32):
        nib = (qs[:, :, col] & 0xF) if col < 16 else (qs[:, :, col - 16] >> 4)
        q = nib | (((qh >> col) & 1) << 4)
        out[:, :, col] = d * q.astype(np.float32) + m
    return out.reshape(N, nb * 32)


def quantize_q2_K(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 16, 16)                          # sub-block g == kernel index `is`
    mn = Wsb.min(3); sc_t = (Wsb.max(3) - mn) / 3.0; min_t = np.maximum(-mn, 0.0)
    d = (sc_t.max(2) / 15.0).astype(np.float32); dmin = (min_t.max(2) / 15.0).astype(np.float32)
    ds, dms = np.where(d == 0, 1.0, d), np.where(dmin == 0, 1.0, dmin)
    sc = np.clip(np.rint(sc_t / ds[..., None]), 0, 15).astype(np.int32)
    m = np.clip(np.rint(min_t / dms[..., None]), 0, 15).astype(np.int32)
    dl = d[..., None] * sc; ml = dmin[..., None] * m; dls = np.where(dl == 0, 1.0, dl)
    qel = np.clip(np.rint((Wsb + ml[..., None]) / dls[..., None]), 0, 3).astype(np.uint8)
    out = np.zeros((N, nb, 84), np.uint8)
    out[:, :, 0:16] = (sc | (m << 4)).astype(np.uint8)
    for g in range(16):
        chunk, sidx, sub = g // 8, (g % 8) // 2, g % 2
        for l in range(16):
            out[:, :, 16 + chunk * 32 + sub * 16 + l] |= (qel[:, :, g, l] << (2 * sidx))
    out[:, :, 80:82] = _f16le(d, N, nb); out[:, :, 82:84] = _f16le(dmin, N, nb)
    return out


def dequantize_q2_K(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    scales = p[:, :, 0:16].astype(np.int32); qs = p[:, :, 16:80].astype(np.int32)
    d = np.ascontiguousarray(p[:, :, 80:82]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    dmin = np.ascontiguousarray(p[:, :, 82:84]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    out = np.zeros((N, nb, 256), np.float32)
    for col in range(256):
        chunk, pos = col >> 7, col & 127; sidx, sub, l = pos >> 5, (pos >> 4) & 1, pos & 15
        is_ = chunk * 8 + sidx * 2 + sub
        q = (qs[:, :, chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3
        out[:, :, col] = d * (scales[:, :, is_] & 0xF) * q - dmin * (scales[:, :, is_] >> 4)
    return out.reshape(N, nb * 256)


def quantize_q3_K(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 16, 16)
    scl = (np.abs(Wsb).max(3) / 4.0).astype(np.float32)     # q3 in -4..3
    d = (scl.max(2) / 31.0).astype(np.float32); ds = np.where(d == 0, 1.0, d)
    s6 = np.clip(np.rint(scl / ds[..., None]) + 32, 1, 63).astype(np.int32)   # 6-bit, used as s-32
    dl = d[..., None] * (s6 - 32); dls = np.where(dl == 0, 1.0, dl)
    code = np.clip(np.rint(Wsb / dls[..., None]) + 4, 0, 7).astype(np.int32)  # q3v+4 in 0..7
    out = np.zeros((N, nb, 110), np.uint8)
    for g in range(16):
        chunk, sidx, sub = g // 8, (g % 8) // 2, g % 2
        for l in range(16):
            low2 = code[:, :, g, l] & 3; hb = (code[:, :, g, l] >> 2) & 1
            out[:, :, 32 + chunk * 32 + sub * 16 + l] |= (low2.astype(np.uint8) << (2 * sidx))
            out[:, :, sub * 16 + l] |= (hb.astype(np.uint8) << (chunk * 4 + sidx))
    sca = np.zeros((N, nb, 12), np.uint8)                   # pack 16 6-bit scales (inverse of kernel unpack)
    for g in range(16):
        w, b = g >> 2, g & 3; v = s6[:, :, g].astype(np.uint8)
        if w == 0:   sca[:, :, b] |= (v & 0xF); sca[:, :, 8 + b] |= ((v >> 4) & 3)
        elif w == 1: sca[:, :, 4 + b] |= (v & 0xF); sca[:, :, 8 + b] |= (((v >> 4) & 3) << 2)
        elif w == 2: sca[:, :, b] |= ((v & 0xF) << 4); sca[:, :, 8 + b] |= (((v >> 4) & 3) << 4)
        else:        sca[:, :, 4 + b] |= ((v & 0xF) << 4); sca[:, :, 8 + b] |= (((v >> 4) & 3) << 6)
    out[:, :, 96:108] = sca; out[:, :, 108:110] = _f16le(d, N, nb)
    return out


def dequantize_q3_K(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    hmask = p[:, :, 0:32].astype(np.int32); qs = p[:, :, 32:96].astype(np.int32); sca = p[:, :, 96:108].astype(np.int32)
    d = np.ascontiguousarray(p[:, :, 108:110]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    out = np.zeros((N, nb, 256), np.float32)
    for col in range(256):
        chunk, pos = col >> 7, col & 127; sidx, sub, l = pos >> 5, (pos >> 4) & 1, pos & 15
        is_ = chunk * 8 + sidx * 2 + sub
        low2 = (qs[:, :, chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3
        hb = ((hmask[:, :, sub * 16 + l] >> (chunk * 4 + sidx)) & 1)
        q3v = (low2 | (hb << 2)) - 4
        w, b = is_ >> 2, is_ & 3
        if w == 0:   s = (sca[:, :, b] & 0xF) | ((sca[:, :, 8 + b] & 3) << 4)
        elif w == 1: s = (sca[:, :, 4 + b] & 0xF) | (((sca[:, :, 8 + b] >> 2) & 3) << 4)
        elif w == 2: s = ((sca[:, :, b] >> 4) & 0xF) | (((sca[:, :, 8 + b] >> 4) & 3) << 4)
        else:        s = ((sca[:, :, 4 + b] >> 4) & 0xF) | (((sca[:, :, 8 + b] >> 6) & 3) << 4)
        out[:, :, col] = d * (s - 32) * q3v
    return out.reshape(N, nb * 256)


def quantize_q5_K(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    Wsb = W.reshape(N, nb, 8, 32)                           # sub-block g32 == kernel index `is`
    mn = Wsb.min(3); sc_t = (Wsb.max(3) - mn) / 31.0; min_t = np.maximum(-mn, 0.0)
    d = (sc_t.max(2) / 63.0).astype(np.float32); dmin = (min_t.max(2) / 63.0).astype(np.float32)
    ds, dms = np.where(d == 0, 1.0, d), np.where(dmin == 0, 1.0, dmin)
    sc = np.clip(np.rint(sc_t / ds[..., None]), 0, 63).astype(np.int32)
    m = np.clip(np.rint(min_t / dms[..., None]), 0, 63).astype(np.int32)
    dl = d[..., None] * sc; ml = dmin[..., None] * m; dls = np.where(dl == 0, 1.0, dl)
    code = np.clip(np.rint((Wsb + ml[..., None]) / dls[..., None]), 0, 31).astype(np.int32)
    out = np.zeros((N, nb, 176), np.uint8)
    out[:, :, 0:2] = _f16le(d, N, nb); out[:, :, 2:4] = _f16le(dmin, N, nb)
    sca = np.zeros((N, nb, 12), np.uint8)                   # get_scale_min_k4 inverse (as q4_K)
    for j in range(4):
        sca[:, :, j] = (sc[:, :, j] & 63) | (((sc[:, :, j + 4] >> 4) & 3) << 6)
        sca[:, :, 8 + j] = (sc[:, :, j + 4] & 0xF) | ((m[:, :, j + 4] & 0xF) << 4)
        sca[:, :, 4 + j] = (m[:, :, j] & 63) | (((m[:, :, j + 4] >> 4) & 3) << 6)
    out[:, :, 4:16] = sca
    for g in range(8):
        chunk, sub = g // 2, g % 2
        for l in range(32):
            byte = chunk * 32 + l
            out[:, :, 48 + byte] |= ((code[:, :, g, l] & 0xF) << (4 * sub)).astype(np.uint8)
            out[:, :, 16 + l] |= (((code[:, :, g, l] >> 4) & 1) << (2 * chunk + sub)).astype(np.uint8)
    return out


def dequantize_q5_K(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    d = np.ascontiguousarray(p[:, :, 0:2]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    dmin = np.ascontiguousarray(p[:, :, 2:4]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    sca = p[:, :, 4:16].astype(np.int32); qh = p[:, :, 16:48].astype(np.int32); qs = p[:, :, 48:176].astype(np.int32)
    out = np.zeros((N, nb, 256), np.float32)
    for col in range(256):
        chunk, pos = col >> 6, col & 63; sub, l = pos >> 5, pos & 31; is_ = 2 * chunk + sub
        nib = (qs[:, :, chunk * 32 + l] >> 4) if sub else (qs[:, :, chunk * 32 + l] & 0xF)
        hb = (qh[:, :, l] >> (2 * chunk + sub)) & 1
        q = nib + hb * 16
        if is_ < 4:
            sc = sca[:, :, is_] & 63; mn = sca[:, :, is_ + 4] & 63
        else:
            sc = (sca[:, :, is_ + 4] & 0xF) | ((sca[:, :, is_ - 4] >> 6) << 4)
            mn = (sca[:, :, is_ + 4] >> 4) | ((sca[:, :, is_] >> 6) << 4)
        out[:, :, col] = d * sc * q - dmin * mn
    return out.reshape(N, nb * 256)


def quantize_q6_K(W):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape; nb = K // 256
    sc_of_col = np.array([(c >> 7) * 8 + ((c & 31) >> 4) + ((c & 127) >> 5) * 2 for c in range(256)])
    Wf = W.reshape(N, nb, 256)
    scl = np.zeros((N, nb, 16), np.float32)
    for s in range(16):
        cols = np.where(sc_of_col == s)[0]
        scl[:, :, s] = np.abs(Wf[:, :, cols]).max(2) / 32.0
    d = (scl.max(2) / 127.0).astype(np.float32); dsf = np.where(d == 0, 1.0, d)
    sc8 = np.clip(np.rint(scl / dsf[..., None]), -127, 127).astype(np.int32)
    out = np.zeros((N, nb, 210), np.uint8)
    for col in range(256):
        chunk, pos = col >> 7, col & 127; group, l = pos >> 5, pos & 31; s = sc_of_col[col]
        dl = d * sc8[:, :, s]; dls = np.where(dl == 0, 1.0, dl)
        code = np.clip(np.rint(Wf[:, :, col] / dls) + 32, 0, 63).astype(np.int32)
        ql_byte = chunk * 64 + l + 32 * (group & 1)
        if group & 2:
            out[:, :, ql_byte] |= ((code & 0xF) << 4).astype(np.uint8)
        else:
            out[:, :, ql_byte] |= (code & 0xF).astype(np.uint8)
        out[:, :, 128 + chunk * 32 + l] |= (((code >> 4) & 3) << (2 * group)).astype(np.uint8)
    out[:, :, 192:208] = sc8.astype(np.int8).view(np.uint8)
    out[:, :, 208:210] = _f16le(d, N, nb)
    return out


def dequantize_q6_K(packed):
    p = np.ascontiguousarray(packed, np.uint8); N, nb, _ = p.shape
    ql = p[:, :, 0:128].astype(np.int32); qh = p[:, :, 128:192].astype(np.int32)
    sca = np.ascontiguousarray(p[:, :, 192:208]).view(np.int8).astype(np.int32)
    d = np.ascontiguousarray(p[:, :, 208:210]).reshape(N, nb * 2).view(np.float16).astype(np.float32).reshape(N, nb)
    out = np.zeros((N, nb, 256), np.float32)
    for col in range(256):
        chunk, pos = col >> 7, col & 127; group, l = pos >> 5, pos & 31
        ql_byte = ql[:, :, chunk * 64 + l + 32 * (group & 1)]
        nib = (ql_byte >> 4) if (group & 2) else (ql_byte & 0xF)
        hbits = (qh[:, :, chunk * 32 + l] >> (2 * group)) & 3
        q = (nib | (hbits << 4)) - 32
        sc_idx = chunk * 8 + (l >> 4) + group * 2
        out[:, :, col] = d * sca[:, :, sc_idx] * q
    return out.reshape(N, nb * 256)


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
    "bitnet": (quantize_bitnet, dequantize_bitnet),
    "iq4_nl": (quantize_iq4_nl, dequantize_iq4_nl),
    "iq4_xs": (quantize_iq4_xs, dequantize_iq4_xs),
    "iq2_xxs": (quantize_iq2_xxs, dequantize_iq2_xxs),
    "q4_1": (quantize_q4_1, dequantize_q4_1),
    "q5_0": (quantize_q5_0, dequantize_q5_0),
    "q5_1": (quantize_q5_1, dequantize_q5_1),
    "q2_K": (quantize_q2_K, dequantize_q2_K),
    "q3_K": (quantize_q3_K, dequantize_q3_K),
    "q5_K": (quantize_q5_K, dequantize_q5_K),
    "q6_K": (quantize_q6_K, dequantize_q6_K),
    "iq2_xs": (quantize_iq2_xs, dequantize_iq2_xs),
    "iq3_xxs": (quantize_iq3_xxs, dequantize_iq3_xxs),
    "iq1_s": (quantize_iq1_s, dequantize_iq1_s),
}
