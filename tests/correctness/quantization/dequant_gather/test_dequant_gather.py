import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import dequant_gather, quantized_embedding, quantized_embedding_bag
from tk.quant import QUANT_FORMATS


EMBEDDING_FORMATS = {
    "q4_0": 32,
    "q8_0": 32,
    "q4_K": 256,
    "q5_K": 256,
    "q6_K": 256,
    "q2_K": 256,
    "q3_K": 256,
    "iq4_nl": 32,
    "iq4_xs": 256,
    "kU4B8": 128,
    "kU4": 128,
    "hqq": 64,
    "fp8_e4m3": 32,
    "mxfp8": 32,
    "nvfp4": 16,
    "mxfp4": 32,
}


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


@pytest.mark.parametrize("fmt,block_k", EMBEDDING_FORMATS.items())
def test_quantized_embedding_formats_add_and_invalid_ids(fmt, block_k):
    quantize, dequantize = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(100 + block_k + len(fmt))
    rows, columns = 5, 2 * block_k
    source = (0.25 * rng.standard_normal((rows, columns))).astype(np.float32)
    packed = quantize(source)
    ids = np.array([[3, -1], [1, rows]], dtype=np.int32)
    add = (0.03 * rng.standard_normal((*ids.shape, columns))).astype(np.float32)
    scale = np.float32(1.375)

    got = quantized_embedding(
        mx.array(packed), mx.array(ids), fmt, scale=scale, add=mx.array(add),
        output_dtype="float32")
    mx.eval(got)

    table = dequantize(packed).astype(np.float32)
    ref = np.zeros((*ids.shape, columns), dtype=np.float32)
    for index in np.ndindex(ids.shape):
        row = ids[index]
        if 0 <= row < rows:
            ref[index] = table[row] * scale + add[index]
    assert got.shape == ref.shape and got.dtype == mx.float32
    np.testing.assert_allclose(np.array(got), ref, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("fmt,columns", [
    ("q4_0", 96), ("q8_0", 96), ("q6_K", 512), ("mxfp8", 96),
])
@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_quantized_embedding_bag_weights_repeats_and_invalid(fmt, columns, mode):
    quantize, dequantize = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(columns + (mode == "mean"))
    rows = 6
    packed = quantize((0.2 * rng.standard_normal((rows, columns))).astype(np.float32))
    ids = np.array([1, 1, -1, 4, rows, 2], dtype=np.int32)
    offsets = np.array([0, 3, 5, 6, 6], dtype=np.int32)
    sample_weights = np.array([0.5, 1.25, 8.0, -0.75, 4.0, 2.0], dtype=np.float32)
    scale = np.float32(0.625)

    got = quantized_embedding_bag(
        mx.array(packed), mx.array(ids), mx.array(offsets), fmt,
        sample_weights=mx.array(sample_weights), mode=mode, scale=scale,
        output_dtype="float32")
    mx.eval(got)

    table = dequantize(packed).astype(np.float32)
    ref = np.zeros((len(offsets) - 1, columns), dtype=np.float32)
    for bag, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        valid = 0
        for index in range(begin, end):
            row = ids[index]
            if 0 <= row < rows:
                ref[bag] += table[row] * scale * sample_weights[index]
                valid += 1
        if mode == "mean" and valid:
            ref[bag] /= valid
    np.testing.assert_allclose(np.array(got), ref, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("output_dtype,mx_dtype", [
    ("float16", mx.float16), ("bfloat16", mx.bfloat16), ("float32", mx.float32)
])
def test_quantized_embedding_output_dtype(output_dtype, mx_dtype):
    source = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(2, 32)
    packed = QUANT_FORMATS["q8_0"][0](source)
    got = quantized_embedding(
        mx.array(packed), mx.array(np.array([0, 1], np.int32)), "q8_0",
        output_dtype=output_dtype)
    mx.eval(got)
    assert got.dtype == mx_dtype
