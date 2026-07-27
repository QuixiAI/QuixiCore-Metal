import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk
from tk.base_q import pack_base_q_codes


def _numpy(value):
    if type(value).__module__.split(".")[0] == "torch":
        return value.detach().float().cpu().numpy()
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def _assert_same(mlx_value, torch_value, atol=0):
    mx.eval(mlx_value)
    torch.mps.synchronize()
    np.testing.assert_allclose(
        _numpy(mlx_value), _numpy(torch_value), rtol=0, atol=atol,
    )


@pytest.mark.parametrize("bits,scale_dtype", [(4, "bf16"), (6, "f16"), (8, "e4m3")])
def test_base_q_operation_parity(bits, scale_dtype):
    rows, group_size, groups = 4, 32, 2
    columns = group_size * groups
    lanes = ((np.arange(rows * columns).reshape(rows, columns) * 3 + 2) % (1 << bits)).astype(np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    if scale_dtype == "bf16":
        scale_host = np.array([[0.125, 0.25], [0.5, 0.0625],
                               [0.03125, 1.0], [0.75, 0.375]], np.float32)
        bias_host = np.array([[-0.5, 0.25], [1.0, -0.125],
                              [0.5, 2.0], [-1.0, 0.75]], np.float32)
        ms, mb = mx.array(scale_host).astype(mx.bfloat16), mx.array(bias_host).astype(mx.bfloat16)
        ts = torch.from_numpy(scale_host).to(torch.bfloat16).to("mps")
        tb = torch.from_numpy(bias_host).to(torch.bfloat16).to("mps")
    elif scale_dtype == "f16":
        scale_host = np.array([[0.125, 0.25], [0.5, 0.0625],
                               [0.03125, 1.0], [0.75, 0.375]], np.float16)
        bias_host = np.array([[-0.5, 0.25], [1.0, -0.125],
                              [0.5, 2.0], [-1.0, 0.75]], np.float16)
        ms, mb = mx.array(scale_host), mx.array(bias_host)
        ts, tb = torch.from_numpy(scale_host).to("mps"), torch.from_numpy(bias_host).to("mps")
    else:
        scale_host = np.array([[0x30, 0x38], [0x40, 0x28],
                               [0x20, 0x48], [0x34, 0x3C]], np.uint8)
        bias_host = np.array([[0xB0, 0x38], [0x30, 0xA8],
                              [0x40, 0x20], [0xB4, 0x3C]], np.uint8)
        ms, mb = mx.array(scale_host), mx.array(bias_host)
        ts, tb = torch.from_numpy(scale_host).to("mps"), torch.from_numpy(bias_host).to("mps")
    mc, tc = mx.array(packed), torch.from_numpy(packed).to("mps")
    x_host = np.linspace(-0.5, 0.75, columns * 5, dtype=np.float32).reshape(columns, 5).astype(np.float16)
    mx_input, torch_input = mx.array(x_host), torch.from_numpy(x_host).to("mps")
    ids_host = np.array([3, -1, 99, 0], np.int32)
    mi, ti = mx.array(ids_host), torch.from_numpy(ids_host).to("mps")

    _assert_same(
        tk.base_qdequant(mc, ms, mb, bits, group_size, scale_dtype, output_dtype="float32"),
        tk.base_qdequant(tc, ts, tb, bits, group_size, scale_dtype, output_dtype="float32"),
    )
    _assert_same(
        tk.base_qgemm(mc, ms, mb, mx_input, bits, group_size, scale_dtype),
        tk.base_qgemm(tc, ts, tb, torch_input, bits, group_size, scale_dtype),
        atol=2e-2,
    )
    _assert_same(
        tk.base_qembedding(mc, ms, mb, mi, bits, group_size, scale_dtype, output_dtype="float32"),
        tk.base_qembedding(tc, ts, tb, ti, bits, group_size, scale_dtype, output_dtype="float32"),
    )
    _assert_same(
        tk.base_qlm_head_argmax(
            mc, ms, mb, mx_input, bits, group_size, scale_dtype,
        ),
        tk.base_qlm_head_argmax(
            tc, ts, tb, torch_input, bits, group_size, scale_dtype,
        ),
    )

    mlx_vector, torch_vector = mx_input[:, :1], torch_input[:, :1]
    mlx_qkv = tk.base_qgemv_qkv(
        mc, ms, mb, mc, ms, mb, mc, ms, mb, mlx_vector,
        bits, group_size, scale_dtype,
    )
    torch_qkv = tk.base_qgemv_qkv(
        tc, ts, tb, tc, ts, tb, tc, ts, tb, torch_vector,
        bits, group_size, scale_dtype,
    )
    for mlx_projection, torch_projection in zip(mlx_qkv, torch_qkv):
        _assert_same(mlx_projection, torch_projection, atol=2e-2)

    _assert_same(
        tk.base_qgemv_swiglu(
            mc, ms, mb, mc, ms, mb, mlx_vector,
            bits, group_size, scale_dtype,
        ),
        tk.base_qgemv_swiglu(
            tc, ts, tb, tc, ts, tb, torch_vector,
            bits, group_size, scale_dtype,
        ),
        atol=2e-2,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_base_q_grouped_expert_parity(bits):
    experts, output_rows, inner = 2, 64, 64
    rng = np.random.default_rng(1500 + bits)
    lanes = rng.integers(
        0, 1 << bits, size=(experts, output_rows, inner), dtype=np.uint8)
    packed = pack_base_q_codes(lanes, bits)
    scales = (
        0.002 + 0.004 * rng.random((experts, output_rows, 2))
    ).astype(np.float16)
    biases = (
        -0.02 + 0.04 * rng.random((experts, output_rows, 2))
    ).astype(np.float16)
    activations = (0.2 * rng.standard_normal((64, inner))).astype(np.float16)
    expert_of_tile = np.array([0, 1], np.int32)

    mc, ms, mb = mx.array(packed), mx.array(scales), mx.array(biases)
    ma, me = mx.array(activations), mx.array(expert_of_tile)
    tc = torch.from_numpy(packed).to("mps")
    ts = torch.from_numpy(scales).to("mps")
    tb = torch.from_numpy(biases).to("mps")
    ta = torch.from_numpy(activations).to("mps")
    te = torch.from_numpy(expert_of_tile).to("mps")

    _assert_same(
        tk.base_qmoe_gemm(mc, ms, mb, ma, me, bits, 32, "f16"),
        tk.base_qmoe_gemm(tc, ts, tb, ta, te, bits, 32, "f16"),
        atol=2e-2,
    )
    _assert_same(
        tk.base_qmoe_swiglu(mc, ms, mb, ma, me, bits, 32, "f16"),
        tk.base_qmoe_swiglu(tc, ts, tb, ta, te, bits, 32, "f16"),
        atol=2e-2,
    )
