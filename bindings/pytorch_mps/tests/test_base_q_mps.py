"""PyTorch-MPS correctness coverage for the canonical BaseQN contract."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk_torch
from tk.base_q import dequantize_base_q, pack_base_q_codes


def _fixture(bits, symmetric=False):
    rows, group_size, groups = 4, 32, 2
    columns = group_size * groups
    lanes = ((np.arange(rows * columns).reshape(rows, columns) * 7 + 1) % (1 << bits)).astype(np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    scales = np.array([[0.125, 0.25], [0.5, 0.0625],
                       [0.03125, 1.0], [0.75, 0.375]], np.float16)
    biases = None if symmetric else np.array(
        [[-0.5, 0.25], [1.0, -0.125], [0.5, 2.0], [-1.0, 0.75]], np.float16
    )
    oracle = dequantize_base_q(
        packed, scales, biases, bits, group_size, "f16", symmetric,
    )
    return (
        torch.from_numpy(packed).to("mps"),
        torch.from_numpy(scales).to("mps"),
        None if biases is None else torch.from_numpy(biases).to("mps"),
        oracle,
    )


def _random_fixture(bits, rows, seed, symmetric=False):
    """Small variable-row fixture for fused consumer kernels."""
    rng = np.random.default_rng(seed)
    group_size, columns = 32, 64
    lanes = rng.integers(0, 1 << bits, size=(rows, columns), dtype=np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    scales = (0.002 + 0.004 * rng.random((rows, columns // group_size))).astype(np.float16)
    biases = None if symmetric else (
        -0.02 + 0.04 * rng.random((rows, columns // group_size))
    ).astype(np.float16)
    oracle = dequantize_base_q(
        packed, scales, biases, bits, group_size, "f16", symmetric,
    )
    return (
        torch.from_numpy(packed).to("mps"),
        torch.from_numpy(scales).to("mps"),
        None if biases is None else torch.from_numpy(biases).to("mps"),
        oracle,
    )


def _random_expert_fixture(bits, experts, rows, seed, symmetric=False):
    rng = np.random.default_rng(seed)
    group_size, columns = 32, 64
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
        torch.from_numpy(packed).to("mps"),
        torch.from_numpy(scales).to("mps"),
        None if biases is None else torch.from_numpy(biases).to("mps"),
        oracle,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_base_qdequant_mps(bits, symmetric):
    codes, scales, biases, oracle = _fixture(bits, symmetric)
    got = tk_torch.base_qdequant(
        codes, scales, biases, bits, 32, scale_dtype="f16",
        symmetric=symmetric, output_dtype="float32",
    )
    torch.mps.synchronize()
    np.testing.assert_allclose(got.cpu().numpy(), oracle, rtol=0, atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
def test_base_qgemv_gemm_and_embedding_mps(bits):
    codes, scales, biases, weights = _fixture(bits)
    rng = np.random.default_rng(bits)
    vector = rng.standard_normal((weights.shape[1], 1)).astype(np.float16)
    matrix = rng.standard_normal((weights.shape[1], 5)).astype(np.float16)
    tv = torch.from_numpy(vector).to("mps")
    tm = torch.from_numpy(matrix).to("mps")
    got_v = tk_torch.base_qgemv(
        codes, scales, biases, tv, bits, 32, scale_dtype="f16")
    got_m = tk_torch.base_qgemm(
        codes, scales, biases, tm, bits, 32, scale_dtype="f16")
    ids = torch.tensor([3, -1, 99, 0], dtype=torch.int32, device="mps")
    got_e = tk_torch.base_qembedding(
        codes, scales, biases, ids, bits, 32, scale_dtype="f16",
        output_dtype="float32")
    torch.mps.synchronize()
    np.testing.assert_allclose(got_v.float().cpu().numpy(), weights @ vector, rtol=5e-3, atol=0.1)
    np.testing.assert_allclose(got_m.float().cpu().numpy(), weights @ matrix, rtol=5e-3, atol=0.1)
    expected_e = np.stack([weights[3], np.zeros(weights.shape[1]),
                           np.zeros(weights.shape[1]), weights[0]])
    np.testing.assert_allclose(got_e.cpu().numpy(), expected_e, rtol=0, atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_base_q_fused_consumers_mps(bits, symmetric):
    q = _random_fixture(bits, 5, 100 + bits, symmetric)
    k = _random_fixture(bits, 3, 200 + bits, symmetric)
    v = _random_fixture(bits, 4, 300 + bits, symmetric)
    gate = _random_fixture(bits, 7, 400 + bits, symmetric)
    up = _random_fixture(bits, 7, 500 + bits, symmetric)
    rng = np.random.default_rng(600 + bits)
    x = (0.2 * rng.standard_normal((64, 1))).astype(np.float16)
    tx = torch.from_numpy(x).to("mps")

    qkv = tk_torch.base_qgemv_qkv(
        q[0], q[1], q[2], k[0], k[1], k[2], v[0], v[1], v[2],
        tx, bits, 32, scale_dtype="f16", symmetric=symmetric,
    )
    swiglu = tk_torch.base_qgemv_swiglu(
        gate[0], gate[1], gate[2], up[0], up[1], up[2], tx, bits, 32,
        scale_dtype="f16", symmetric=symmetric,
    )
    torch.mps.synchronize()

    for got, fixture in zip(qkv, (q, k, v)):
        np.testing.assert_allclose(
            got.float().cpu().numpy(), fixture[3] @ x, rtol=5e-3, atol=0.02,
        )
    gate_dot = gate[3].astype(np.float32) @ x.astype(np.float32)
    up_dot = up[3].astype(np.float32) @ x.astype(np.float32)
    expected = (gate_dot / (1.0 + np.exp(-gate_dot))) * up_dot
    np.testing.assert_allclose(
        swiglu.float().cpu().numpy(), expected, rtol=5e-3, atol=0.02,
    )


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_base_q_lm_head_argmax_mps(bits, symmetric, batch):
    weights = _random_fixture(bits, 19, 900 + bits, symmetric)
    rng = np.random.default_rng(1000 + bits)
    x = (0.2 * rng.standard_normal((64, batch))).astype(np.float16)
    tx = torch.from_numpy(x).to("mps")
    got = tk_torch.base_qlm_head_argmax(
        weights[0], weights[1], weights[2], tx, bits, 32,
        scale_dtype="f16", symmetric=symmetric,
    )
    torch.mps.synchronize()
    expected = np.argmax(
        (weights[3].astype(np.float32) @ x.astype(np.float32)).astype(np.float16),
        axis=0,
    ).astype(np.int32)
    np.testing.assert_array_equal(got.cpu().numpy(), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_base_q_grouped_expert_gemm_and_swiglu_mps(bits, symmetric):
    weights = _random_expert_fixture(
        bits, experts=2, rows=64, seed=1300 + bits, symmetric=symmetric,
    )
    rng = np.random.default_rng(1400 + bits)
    x = (0.2 * rng.standard_normal((64, 64))).astype(np.float16)
    expert_of_tile = np.array([0, 1], np.int32)
    tx = torch.from_numpy(x).to("mps")
    te = torch.from_numpy(expert_of_tile).to("mps")
    rect = tk_torch.base_qmoe_gemm(
        *weights[:3], tx, te, bits, 32, "f16", symmetric=symmetric,
    )
    swiglu = tk_torch.base_qmoe_swiglu(
        *weights[:3], tx, te, bits, 32, "f16", symmetric=symmetric,
    )
    torch.mps.synchronize()

    rect_ref = np.empty((64, 64), np.float32)
    swiglu_ref = np.empty((64, 32), np.float32)
    for tile, expert in enumerate(expert_of_tile):
        row_slice = slice(tile * 32, (tile + 1) * 32)
        logits = x[row_slice].astype(np.float32) @ weights[3][expert].T
        rect_ref[row_slice] = logits
        gate, up = np.split(logits, 2, axis=1)
        swiglu_ref[row_slice] = (gate / (1.0 + np.exp(-gate))) * up
    np.testing.assert_allclose(
        rect.float().cpu().numpy(), rect_ref, rtol=8e-3, atol=0.08,
    )
    np.testing.assert_allclose(
        swiglu.float().cpu().numpy(), swiglu_ref, rtol=1e-2, atol=0.08,
    )


@pytest.mark.parametrize(
    "activation_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_base_q_grouped_expert_activation_dtypes_mps(activation_dtype):
    weights = _random_expert_fixture(4, 1, 64, 1800)
    rng = np.random.default_rng(1900)
    host = (0.1 * rng.standard_normal((32, 64))).astype(np.float32)
    x = torch.from_numpy(host).to(dtype=activation_dtype, device="mps")
    expert_of_tile = torch.tensor([0], dtype=torch.int32, device="mps")
    got = tk_torch.base_qmoe_gemm(
        *weights[:3], x, expert_of_tile, 4, 32, "f16",
    )
    torch.mps.synchronize()
    ref = x.float().cpu().numpy() @ weights[3][0].T
    assert got.dtype == activation_dtype
    np.testing.assert_allclose(
        got.float().cpu().numpy(), ref, rtol=1e-2, atol=0.08,
    )
