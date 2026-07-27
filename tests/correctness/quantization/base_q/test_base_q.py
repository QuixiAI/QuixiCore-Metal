import mlx.core as mx
import numpy as np
import pytest

from tk import (
    base_qdequant,
    base_qembedding,
    base_qgemm,
    base_qgemv,
    base_qgemv_qkv,
    base_qgemv_swiglu,
    base_qlm_head_argmax,
    base_qmoe_gemm,
    base_qmoe_swiglu,
)
from tk.base_q import dequantize_base_q, pack_base_q_codes


def _planes(bits, group_size=32, scale_dtype="bf16", symmetric=False, rows=3):
    groups = 2
    columns = groups * group_size
    lane = np.arange(rows * columns, dtype=np.uint16).reshape(rows, columns)
    lane = (lane * 5 + 3) % (1 << bits)
    packed = pack_base_q_codes(lane, bits)
    base_scale = np.array([[0.125, 0.25], [0.5, 0.0625], [0.03125, 1.0]], np.float32)[:rows]
    base_bias = np.array([[0.5, 1.0], [0.25, 0.125], [2.0, 0.0625]], np.float32)[:rows]
    if scale_dtype == "bf16":
        scales = mx.array(base_scale).astype(mx.bfloat16)
        biases = None if symmetric else mx.array(base_bias).astype(mx.bfloat16)
        mx.eval(scales, *([] if biases is None else [biases]))
        scale_oracle = np.array(scales.astype(mx.float32))
        bias_oracle = None if biases is None else np.array(biases.astype(mx.float32))
    elif scale_dtype == "f16":
        scale_oracle = base_scale.astype(np.float16)
        bias_oracle = None if symmetric else base_bias.astype(np.float16)
        scales = mx.array(scale_oracle)
        biases = None if symmetric else mx.array(bias_oracle)
    elif scale_dtype == "e8m0":
        scale_oracle = np.array([[124, 125], [126, 123], [122, 127]], np.uint8)[:rows]
        bias_oracle = None if symmetric else np.array([[126, 127], [125, 124], [128, 123]], np.uint8)[:rows]
        scales = mx.array(scale_oracle)
        biases = None if symmetric else mx.array(bias_oracle)
    elif scale_dtype == "e4m3":
        scale_oracle = np.array([[0x30, 0x38], [0x40, 0x28], [0x20, 0x48]], np.uint8)[:rows]
        bias_oracle = None if symmetric else np.array([[0xB0, 0x38], [0x30, 0xA8], [0x40, 0x20]], np.uint8)[:rows]
        scales = mx.array(scale_oracle)
        biases = None if symmetric else mx.array(bias_oracle)
    else:
        raise AssertionError(scale_dtype)
    oracle = dequantize_base_q(
        packed, scale_oracle, bias_oracle, bits, group_size,
        scale_dtype=scale_dtype, symmetric=symmetric,
    )
    return mx.array(packed), scales, biases, oracle


def _random_f16_planes(bits, rows, seed, group_size=32, symmetric=False):
    rng = np.random.default_rng(seed)
    groups = 2
    columns = groups * group_size
    lanes = rng.integers(0, 1 << bits, size=(rows, columns), dtype=np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    scales = (0.01 + 0.03 * rng.random((rows, groups))).astype(np.float16)
    biases = None if symmetric else (
        -0.05 + 0.1 * rng.random((rows, groups))).astype(np.float16)
    oracle = dequantize_base_q(
        packed, scales, biases, bits, group_size, "f16", symmetric,
    )
    return (
        mx.array(packed), mx.array(scales),
        None if biases is None else mx.array(biases), oracle,
    )


def _random_f16_expert_planes(bits, experts, rows, seed, group_size=32,
                              symmetric=False):
    rng = np.random.default_rng(seed)
    columns = 64
    lanes = rng.integers(
        0, 1 << bits, size=(experts, rows, columns), dtype=np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    scales = (
        0.002 + 0.004 * rng.random((experts, rows, columns // group_size))
    ).astype(np.float16)
    biases = None if symmetric else (
        -0.02 + 0.04 * rng.random((experts, rows, columns // group_size))
    ).astype(np.float16)
    oracle = dequantize_base_q(
        packed, scales, biases, bits, group_size, "f16", symmetric,
    )
    return (
        mx.array(packed), mx.array(scales),
        None if biases is None else mx.array(biases), oracle,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_dequant_all_widths_and_affine_modes(bits, symmetric):
    codes, scales, biases, oracle = _planes(bits, symmetric=symmetric)
    got = base_qdequant(
        codes, scales, biases, bits, 32, symmetric=symmetric,
        output_dtype="float32",
    )
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), oracle, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "bits,scale_dtype", [(4, "bf16"), (4, "f16"), (4, "e8m0"), (8, "e4m3")]
)
def test_scale_storage_types(bits, scale_dtype):
    codes, scales, biases, oracle = _planes(bits, scale_dtype=scale_dtype)
    got = base_qdequant(
        codes, scales, biases, bits, 32, scale_dtype=scale_dtype,
        output_dtype="float32",
    )
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), oracle, rtol=0, atol=1e-6)


@pytest.mark.parametrize("bits,group_size", [(2, 32), (4, 64), (8, 128)])
def test_canonical_group_sizes(bits, group_size):
    codes, scales, biases, oracle = _planes(bits, group_size=group_size)
    got = base_qdequant(
        codes, scales, biases, bits, group_size, output_dtype="float32",
        layout=f"metal_lane_strided_q{bits}",
    )
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), oracle, rtol=0, atol=1e-6)


@pytest.mark.parametrize("output_dtype", ["float16", "bfloat16", "float32"])
def test_dequant_output_dtypes(output_dtype):
    codes, scales, biases, oracle = _planes(6)
    got = base_qdequant(
        codes, scales, biases, 6, 32, output_dtype=output_dtype,
    )
    mx.eval(got)
    expected = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[output_dtype]
    assert got.dtype == expected
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), oracle, rtol=5e-3, atol=2e-2)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_gemv_and_gemm(bits):
    codes, scales, biases, weights = _planes(bits)
    rng = np.random.default_rng(bits)
    vector = rng.standard_normal((weights.shape[1], 1)).astype(np.float16)
    matrix = rng.standard_normal((weights.shape[1], 7)).astype(np.float16)
    got_v = base_qgemv(codes, scales, biases, mx.array(vector), bits, 32)
    got_m = base_qgemm(codes, scales, biases, mx.array(matrix), bits, 32)
    routed_v = base_qgemm(codes, scales, biases, mx.array(vector), bits, 32)
    mx.eval(got_v, got_m, routed_v)
    np.testing.assert_allclose(np.array(got_v), weights @ vector, rtol=5e-3, atol=0.1)
    np.testing.assert_allclose(np.array(got_m), weights @ matrix, rtol=5e-3, atol=0.1)
    np.testing.assert_array_equal(np.array(got_v), np.array(routed_v))


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_fused_qkv_and_swiglu(bits, symmetric):
    q = _random_f16_planes(bits, 5, bits + 10, symmetric=symmetric)
    k = _random_f16_planes(bits, 3, bits + 20, symmetric=symmetric)
    v = _random_f16_planes(bits, 4, bits + 30, symmetric=symmetric)
    gate = _random_f16_planes(bits, 7, bits + 40, symmetric=symmetric)
    up = _random_f16_planes(bits, 7, bits + 50, symmetric=symmetric)
    rng = np.random.default_rng(bits + 60)
    x = rng.standard_normal((q[3].shape[1], 1)).astype(np.float16)
    q_out, k_out, v_out = base_qgemv_qkv(
        *q[:3], *k[:3], *v[:3], mx.array(x), bits, 32, "f16",
        symmetric=symmetric,
    )
    swiglu_out = base_qgemv_swiglu(
        *gate[:3], *up[:3], mx.array(x), bits, 32, "f16",
        symmetric=symmetric,
    )
    mx.eval(q_out, k_out, v_out, swiglu_out)
    np.testing.assert_allclose(np.array(q_out), q[3] @ x, rtol=5e-3, atol=0.1)
    np.testing.assert_allclose(np.array(k_out), k[3] @ x, rtol=5e-3, atol=0.1)
    np.testing.assert_allclose(np.array(v_out), v[3] @ x, rtol=5e-3, atol=0.1)
    gate_ref = gate[3] @ x
    up_ref = up[3] @ x
    swiglu_ref = (gate_ref / (1.0 + np.exp(-gate_ref))) * up_ref
    np.testing.assert_allclose(
        np.array(swiglu_out), swiglu_ref, rtol=8e-3, atol=0.2,
    )


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_lm_head_argmax_batched(bits, symmetric, batch):
    weights = _random_f16_planes(
        bits, 19, 700 + bits, symmetric=symmetric,
    )
    rng = np.random.default_rng(800 + bits)
    x = (0.2 * rng.standard_normal((weights[3].shape[1], batch))).astype(np.float16)
    got = base_qlm_head_argmax(
        *weights[:3], mx.array(x), bits, 32, "f16", symmetric=symmetric,
    )
    mx.eval(got)
    expected = np.argmax(
        (weights[3].astype(np.float32) @ x.astype(np.float32)).astype(np.float16),
        axis=0,
    ).astype(np.int32)
    np.testing.assert_array_equal(np.array(got), expected)


def test_lm_head_argmax_ties_choose_lower_token():
    vocab, inner = 17, 32
    codes = pack_base_q_codes(np.zeros((vocab, inner), np.uint8), 4)
    scales = np.zeros((vocab, 1), np.float16)
    biases = np.zeros((vocab, 1), np.float16)
    x = np.ones((inner, 2), np.float16)
    got = base_qlm_head_argmax(
        mx.array(codes), mx.array(scales), mx.array(biases), mx.array(x),
        4, 32, "f16",
    )
    mx.eval(got)
    np.testing.assert_array_equal(np.array(got), np.array([0, 0], np.int32))


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_grouped_expert_gemm_and_swiglu(bits, symmetric):
    weights = _random_f16_expert_planes(
        bits, experts=2, rows=64, seed=1100 + bits, symmetric=symmetric,
    )
    rng = np.random.default_rng(1200 + bits)
    x = (0.2 * rng.standard_normal((64, 64))).astype(np.float16)
    expert_of_tile = np.array([0, 1], np.int32)
    got_rect = base_qmoe_gemm(
        *weights[:3], mx.array(x), mx.array(expert_of_tile), bits, 32, "f16",
        symmetric=symmetric,
    )
    got_swiglu = base_qmoe_swiglu(
        *weights[:3], mx.array(x), mx.array(expert_of_tile), bits, 32, "f16",
        symmetric=symmetric,
    )
    mx.eval(got_rect, got_swiglu)

    rect_ref = np.empty((64, 64), np.float32)
    swiglu_ref = np.empty((64, 32), np.float32)
    for tile, expert in enumerate(expert_of_tile):
        row_slice = slice(tile * 32, (tile + 1) * 32)
        logits = x[row_slice].astype(np.float32) @ weights[3][expert].T
        rect_ref[row_slice] = logits
        gate, up = np.split(logits, 2, axis=1)
        swiglu_ref[row_slice] = (gate / (1.0 + np.exp(-gate))) * up
    np.testing.assert_allclose(
        np.array(got_rect), rect_ref, rtol=8e-3, atol=0.08,
    )
    np.testing.assert_allclose(
        np.array(got_swiglu), swiglu_ref, rtol=1e-2, atol=0.08,
    )


@pytest.mark.parametrize("activation_dtype", ["float16", "bfloat16", "float32"])
def test_grouped_expert_activation_dtypes(activation_dtype):
    weights = _random_f16_expert_planes(4, 1, 64, 1600)
    rng = np.random.default_rng(1700)
    host = (0.1 * rng.standard_normal((32, 64))).astype(np.float32)
    dtype = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[activation_dtype]
    x = mx.array(host).astype(dtype)
    expert_of_tile = mx.array(np.array([0], np.int32))
    got = base_qmoe_gemm(
        *weights[:3], x, expert_of_tile, 4, 32, "f16",
    )
    mx.eval(got, x)
    ref = np.array(x.astype(mx.float32)) @ weights[3][0].T
    assert got.dtype == dtype
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), ref, rtol=1e-2, atol=0.08,
    )


@pytest.mark.parametrize(
    "bits,scale_dtype", [(4, "bf16"), (4, "e8m0"), (8, "e4m3")])
def test_grouped_expert_scale_storage_types(bits, scale_dtype):
    lanes = (
        (np.arange(32 * 32).reshape(1, 32, 32) * 3 + 1) % (1 << bits)
    ).astype(np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    if scale_dtype == "bf16":
        scales_host = np.full((1, 32, 1), 0.125, np.float32)
        biases_host = np.full((1, 32, 1), -0.25, np.float32)
        scales = mx.array(scales_host).astype(mx.bfloat16)
        biases = mx.array(biases_host).astype(mx.bfloat16)
        mx.eval(scales, biases)
        scales_oracle = np.array(scales.astype(mx.float32))
        biases_oracle = np.array(biases.astype(mx.float32))
    elif scale_dtype == "e8m0":
        scales_oracle = np.full((1, 32, 1), 124, np.uint8)
        biases_oracle = np.full((1, 32, 1), 123, np.uint8)
        scales, biases = mx.array(scales_oracle), mx.array(biases_oracle)
    else:
        scales_oracle = np.full((1, 32, 1), 0x30, np.uint8)
        biases_oracle = np.full((1, 32, 1), 0xA8, np.uint8)
        scales, biases = mx.array(scales_oracle), mx.array(biases_oracle)
    weights = dequantize_base_q(
        packed, scales_oracle, biases_oracle, bits, 32, scale_dtype,
    )
    x = np.linspace(-0.1, 0.1, 32 * 32, dtype=np.float32).reshape(32, 32)
    got = base_qmoe_gemm(
        mx.array(packed), scales, biases, mx.array(x).astype(mx.float16),
        mx.array(np.array([0], np.int32)), bits, 32, scale_dtype,
    )
    mx.eval(got)
    np.testing.assert_allclose(
        np.array(got), x.astype(np.float16).astype(np.float32) @ weights[0].T,
        rtol=1e-2, atol=0.08,
    )


def test_embedding_shape_invalid_ids_and_values():
    codes, scales, biases, weights = _planes(5)
    ids = np.array([[2, -1], [99, 0]], np.int32)
    got = base_qembedding(
        codes, scales, biases, mx.array(ids), 5, 32, output_dtype="float32",
    )
    mx.eval(got)
    expected = np.stack([weights[2], np.zeros(weights.shape[1]),
                         np.zeros(weights.shape[1]), weights[0]]).reshape(2, 2, -1)
    np.testing.assert_allclose(np.array(got), expected, rtol=0, atol=1e-6)


def test_validation_rejects_mismatched_planes():
    codes, scales, biases, _ = _planes(4)
    with pytest.raises(ValueError, match="codes.shape"):
        base_qdequant(codes[:, :-1], scales, biases, 4, 32)
    with pytest.raises(ValueError, match="biases are required"):
        base_qdequant(codes, scales, None, 4, 32)
    with pytest.raises(ValueError, match="only for q8"):
        base_qdequant(codes, mx.zeros(scales.shape, mx.uint8),
                      mx.zeros(scales.shape, mx.uint8), 4, 32,
                      scale_dtype="e4m3")
