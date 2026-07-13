import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import dequant_gather
from tk.quant import QUANT_FORMATS


@pytest.mark.parametrize("fmt,columns", [("q4_0", 96), ("q8_0", 96), ("q6_K", 512)])
def test_dequant_gather_valid_and_invalid_ids(fmt, columns):
    quantize, dequantize = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(columns)
    rows = 7
    source = (0.3 * rng.standard_normal((rows, columns))).astype(np.float32)
    packed = quantize(source)
    ids = np.array([[3, -1], [6, rows]], dtype=np.int64)
    scale = np.float32(np.sqrt(1536.0))
    got = dequant_gather(mx.array(packed), mx.array(ids), fmt, scale=scale)
    mx.eval(got)

    table = dequantize(packed)
    ref = np.zeros((*ids.shape, columns), dtype=np.float16)
    for index in np.ndindex(ids.shape):
        row = ids[index]
        if 0 <= row < rows:
            ref[index] = (table[row] * np.float32(scale)).astype(np.float16)
    assert got.shape == ref.shape and got.dtype == mx.float16
    np.testing.assert_array_equal(np.array(got), ref)
