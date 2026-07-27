import numpy as np
import pytest

from tk.base_q import (
    decode_e4m3,
    decode_e8m0,
    pack_base_q_codes,
    unpack_base_q_codes,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_all_lane_and_cross_byte_boundaries(bits):
    columns = 128
    lanes = (np.arange(3 * columns, dtype=np.uint16) % (1 << bits)).reshape(3, columns)
    packed = pack_base_q_codes(lanes, bits)
    restored = unpack_base_q_codes(packed, bits, columns)
    np.testing.assert_array_equal(restored, lanes.astype(np.uint8))
    assert packed.shape == (3, columns * bits // 8)


@pytest.mark.parametrize(
    "bits,lanes,expected",
    [
        (2, [0, 1, 2, 3] * 4, [0xE4, 0xE4, 0xE4, 0xE4]),
        (3, list(range(8)), [0x88, 0xC6, 0xFA]),
        (4, list(range(8)), [0x10, 0x32, 0x54, 0x76]),
        (5, list(range(8)), [0x20, 0x88, 0x41, 0x8A, 0x39]),
        (6, list(range(4)), [0x40, 0x20, 0x0C]),
        (8, list(range(8)), list(range(8))),
    ],
)
def test_canonical_chunk_bytes(bits, lanes, expected):
    np.testing.assert_array_equal(
        pack_base_q_codes(np.array([lanes], dtype=np.uint8), bits),
        np.array([expected], dtype=np.uint8),
    )


def test_q4_canonical_little_endian_bytes():
    lanes = np.array([[0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x1, 0x2]], dtype=np.uint8)
    np.testing.assert_array_equal(
        pack_base_q_codes(lanes, 4),
        np.array([[0xBA, 0xDC, 0xFE, 0x21]], dtype=np.uint8),
    )


def test_q3_bit_spread_bytes():
    lanes = np.arange(8, dtype=np.uint8)[None, :]
    accumulator = sum(int(value) << (index * 3) for index, value in enumerate(lanes[0]))
    expected = np.array(
        [[accumulator & 0xFF, (accumulator >> 8) & 0xFF, (accumulator >> 16) & 0xFF]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(pack_base_q_codes(lanes, 3), expected)


def test_declared_float_code_edges():
    e8 = decode_e8m0(np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8))
    assert e8[0] == np.float32(2.0**-127)
    assert e8[2] == 1.0 and e8[3] == 2.0
    assert np.isinf(e8[-1])
    e4 = decode_e4m3(np.array([0x00, 0x01, 0x38, 0x40, 0xB8], dtype=np.uint8))
    np.testing.assert_array_equal(e4, np.array([0.0, 2.0**-9, 1.0, 2.0, -1.0], np.float32))
