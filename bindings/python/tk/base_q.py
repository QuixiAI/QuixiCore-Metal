"""Framework-independent helpers for the canonical BaseQN storage contract.

These routines are deliberately NumPy-only. They define the bit-layout oracle
used by both backend test suites and are suitable for small fixture generation;
runtime model conversion belongs in a converter, not in the kernel binding.
"""

from __future__ import annotations

import numpy as np


BASE_Q_BITS = (2, 3, 4, 5, 6, 8)
BASE_Q_GROUP_SIZES = (32, 64, 128)


def _check_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in BASE_Q_BITS:
        raise ValueError("BaseQN bits must be one of 2, 3, 4, 5, 6, or 8")
    return bits


def pack_base_q_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned code lanes along the last axis into Metal-affine bytes."""
    bits = _check_bits(bits)
    values = np.asarray(codes)
    if values.ndim == 0:
        raise ValueError("BaseQN codes must have at least one dimension")
    columns = values.shape[-1]
    if columns * bits % 8:
        raise ValueError("BaseQN last dimension * bits must be byte aligned")
    limit = 1 << bits
    if np.any(values < 0) or np.any(values >= limit):
        raise ValueError(f"BaseQN q{bits} codes must be in [0, {limit})")

    rows = values.reshape(-1, columns).astype(np.uint32, copy=False)
    packed = np.zeros((rows.shape[0], columns * bits // 8), dtype=np.uint8)
    for column in range(columns):
        bit_index = column * bits
        byte_index, shift = divmod(bit_index, 8)
        lane = rows[:, column]
        packed[:, byte_index] |= ((lane << shift) & 0xFF).astype(np.uint8)
        if shift + bits > 8:
            packed[:, byte_index + 1] |= (lane >> (8 - shift)).astype(np.uint8)
    return packed.reshape(values.shape[:-1] + (columns * bits // 8,))


def unpack_base_q_codes(
    packed: np.ndarray, bits: int, columns: int | None = None
) -> np.ndarray:
    """Inverse of :func:`pack_base_q_codes` for the Metal-affine layout."""
    bits = _check_bits(bits)
    data = np.asarray(packed, dtype=np.uint8)
    if data.ndim == 0:
        raise ValueError("BaseQN packed codes must have at least one dimension")
    packed_bytes = data.shape[-1]
    if columns is None:
        if packed_bytes * 8 % bits:
            raise ValueError("packed byte count does not determine an integral lane count")
        columns = packed_bytes * 8 // bits
    columns = int(columns)
    if columns <= 0 or columns * bits != packed_bytes * 8:
        raise ValueError("columns * bits must equal packed_bytes * 8")

    rows = data.reshape(-1, packed_bytes).astype(np.uint32)
    output = np.empty((rows.shape[0], columns), dtype=np.uint8)
    mask = (1 << bits) - 1
    for column in range(columns):
        bit_index = column * bits
        byte_index, shift = divmod(bit_index, 8)
        lane = rows[:, byte_index]
        if shift + bits > 8:
            lane |= rows[:, byte_index + 1] << 8
        output[:, column] = ((lane >> shift) & mask).astype(np.uint8)
    return output.reshape(data.shape[:-1] + (columns,))


def decode_e8m0(values: np.ndarray) -> np.ndarray:
    """Decode OCP E8M0 exponent bytes, including code 0 = 2**-127."""
    codes = np.asarray(values, dtype=np.uint8)
    bits = codes.astype(np.uint32) << np.uint32(23)
    bits = np.where(codes == 0, np.uint32(0x00400000), bits).astype(np.uint32)
    return bits.view(np.float32)


def decode_e4m3(values: np.ndarray) -> np.ndarray:
    """Decode finite OCP E4M3 values using the same arithmetic as Metal."""
    codes = np.asarray(values, dtype=np.uint8)
    sign = np.where(codes & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = (codes >> 3) & 0x0F
    mantissa = codes & 0x07
    normal = np.ldexp(
        1.0 + mantissa.astype(np.float32) / 8.0,
        exponent.astype(np.int32) - 7,
    )
    subnormal = mantissa.astype(np.float32) * np.float32(2.0**-9)
    return sign * np.where(exponent == 0, subnormal, normal).astype(np.float32)


def _decode_bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype=np.uint16)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def decode_base_q_scale(values: np.ndarray, scale_dtype: str) -> np.ndarray:
    """Decode a scale plane; uint16 BF16/F16 inputs are treated as raw bits."""
    name = str(scale_dtype).lower()
    data = np.asarray(values)
    if name in ("bf16", "bfloat16"):
        return _decode_bf16_bits(data) if data.dtype == np.uint16 else data.astype(np.float32)
    if name in ("f16", "float16"):
        return data.view(np.float16).astype(np.float32) if data.dtype == np.uint16 else data.astype(np.float16).astype(np.float32)
    if name == "e8m0":
        return decode_e8m0(data)
    if name == "e4m3":
        return decode_e4m3(data)
    raise ValueError("BaseQN scale_dtype must be bf16, f16, e8m0, or e4m3")


def dequantize_base_q(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray | None,
    bits: int,
    group_size: int,
    scale_dtype: str = "bf16",
    symmetric: bool = False,
) -> np.ndarray:
    """CPU oracle for the separate-plane canonical BaseQN tensor contract."""
    bits = _check_bits(bits)
    group_size = int(group_size)
    if group_size not in BASE_Q_GROUP_SIZES:
        raise ValueError("BaseQN group_size must be 32, 64, or 128")
    scale_values = decode_base_q_scale(scales, scale_dtype)
    if scale_values.ndim < 2:
        raise ValueError("BaseQN scales must have at least two dimensions")
    leading_shape = scale_values.shape[:-1]
    groups = scale_values.shape[-1]
    columns = groups * group_size
    codes = unpack_base_q_codes(packed, bits, columns)
    if codes.shape[:-1] != leading_shape:
        raise ValueError("BaseQN packed codes must match the scale leading dimensions")
    expanded_scale = np.repeat(scale_values, group_size, axis=-1)
    if symmetric:
        return (codes.astype(np.int32) - (1 << (bits - 1))).astype(np.float32) * expanded_scale
    if biases is None:
        raise ValueError("BaseQN biases are required for asymmetric tensors")
    bias_values = decode_base_q_scale(biases, scale_dtype)
    if bias_values.shape != scale_values.shape:
        raise ValueError("BaseQN biases must match scales")
    return codes.astype(np.float32) * expanded_scale + np.repeat(
        bias_values, group_size, axis=-1
    )
