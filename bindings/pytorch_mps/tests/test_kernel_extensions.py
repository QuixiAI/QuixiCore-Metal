"""Focused MPS correctness coverage for fused and specialized kernels."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk_torch  # noqa: E402


def _np(tensor):
    torch.mps.synchronize()
    return tensor.float().cpu().numpy()


def test_decode_norm_and_dynamic_layernorm_mps():
    import tk

    torch.manual_seed(101)
    B, D, eps = 3, 320, 1e-5
    x = torch.randn(B, D, dtype=torch.bfloat16, device="mps")
    residual = torch.randn_like(x)
    weight = torch.randn(D, dtype=torch.bfloat16, device="mps")
    bias = torch.randn(D, dtype=torch.bfloat16, device="mps")

    dynamic = tk_torch.layernorm(x, weight, bias, eps)
    assert torch.allclose(dynamic, F.layer_norm(x, (D,), weight, bias, eps), atol=0.04, rtol=0.04)
    routed = tk.layernorm(x, weight, bias, eps)
    assert torch.allclose(routed, F.layer_norm(x, (D,), weight, bias, eps), atol=0.04, rtol=0.04)

    normalized, summed = tk_torch.decode_layernorm_add(x, residual, weight, bias, eps)
    rounded = (x.float() + residual.float()).to(torch.bfloat16)
    reference = F.layer_norm(rounded, (D,), weight, bias, eps)
    assert torch.equal(summed, rounded)
    assert torch.allclose(normalized, reference, atol=0.04, rtol=0.04)


def test_head_major_decode_attention_mps():
    import tk

    torch.manual_seed(103)
    B, Hq, Hkv, cache_t, context, D = 2, 4, 2, 19, 13, 32
    q = torch.randn(B, Hq, D, device="mps") * 0.2
    k = torch.randn(B, Hkv, cache_t, D, device="mps") * 0.2
    v = torch.randn(B, Hkv, cache_t, D, device="mps") * 0.2
    got = tk_torch.attn_decode_bh(q, k, v, context)
    group = Hq // Hkv
    reference = torch.empty_like(q)
    for head in range(Hq):
        key = k[:, head // group, :context]
        value = v[:, head // group, :context]
        score = torch.einsum("bd,btd->bt", q[:, head], key) / math.sqrt(D)
        reference[:, head] = torch.einsum("bt,btd->bd", torch.softmax(score, -1), value)
    assert torch.allclose(got, reference, atol=2e-5, rtol=2e-5)
    assert torch.allclose(tk.attn_decode_bh(q, k, v, context), reference, atol=2e-5, rtol=2e-5)


def test_swin_attention_mps():
    import tk

    torch.manual_seed(107)
    BW, N, H, D, windows = 4, 9, 3, 32, 2
    qkv = torch.randn(BW, N, 3, H, D, device="mps") * 0.15
    relative_bias = torch.randn(H, N, N, device="mps") * 0.05
    mask = torch.zeros(windows, N, N, device="mps")
    mask[1, :, N // 2:] = -10.0
    got = tk_torch.swin_attn_d32(qkv, relative_bias, mask, windows)
    reference = torch.empty(BW, N, H, D, device="mps")
    for window in range(BW):
        for head in range(H):
            q = qkv[window, :, 0, head]
            k = qkv[window, :, 1, head]
            v = qkv[window, :, 2, head]
            score = q @ k.T / math.sqrt(D) + relative_bias[head] + mask[window % windows]
            reference[window, :, head] = torch.softmax(score, -1) @ v
    assert torch.allclose(got, reference, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        tk.swin_attn_d32(qkv, relative_bias, mask, windows), reference,
        atol=3e-5, rtol=3e-5)

    singleton = torch.randn(1, 1, 3, 1, D, device="mps") * 0.15
    singleton_out = tk_torch.swin_attn_d32(
        singleton, torch.zeros(1, 1, 1, device="mps"),
        torch.full((1, 1, 1), -7.0, device="mps"), 1)
    assert torch.equal(singleton_out, singleton[:, :, 2])


@pytest.mark.parametrize("height,width", [(4, 6), (3, 5)])
def test_patch_merge_mps(height, width):
    torch.manual_seed(height * 10 + width)
    B, C, eps = 2, 12, 1e-5
    x = torch.randn(B, height * width, C, dtype=torch.bfloat16, device="mps") * 0.3
    weight = torch.randn(4 * C, dtype=torch.bfloat16, device="mps") * 0.2 + 0.5
    bias = torch.randn(4 * C, dtype=torch.bfloat16, device="mps") * 0.1
    got = tk_torch.patch_merge_layernorm(x, weight, bias, height, width, eps)

    image = x.reshape(B, height, width, C)
    padded = F.pad(image, (0, 0, 0, width % 2, 0, height % 2))
    merged = torch.cat((padded[:, 0::2, 0::2], padded[:, 1::2, 0::2],
                        padded[:, 0::2, 1::2], padded[:, 1::2, 1::2]), -1)
    reference = F.layer_norm(merged.reshape(B, -1, 4 * C), (4 * C,), weight, bias, eps)
    assert torch.allclose(got, reference, atol=0.04, rtol=0.04)


def test_edge_mlp_mps():
    import tk

    torch.manual_seed(109)
    B, L = 1, 3
    hidden = torch.randn(B, L, 256, device="mps") * 0.08
    first_weight = torch.randn(256, 512, device="mps") * 0.06
    first_bias = torch.randn(256, device="mps") * 0.04
    second_weight = torch.randn(7, 256, device="mps") * 0.05
    second_bias = torch.randn(7, device="mps") * 0.03
    got = tk_torch.edge_mlp_256x7(
        hidden, first_weight, first_bias, second_weight, second_bias)
    left = hidden @ first_weight[:, :256].T
    right = hidden @ first_weight[:, 256:].T + first_bias
    activation = F.gelu(left[:, :, None, :] + right[:, None, :, :], approximate="none")
    reference = torch.einsum("bijh,ch->bcij", activation, second_weight)
    reference += second_bias[None, :, None, None]
    assert torch.allclose(got, reference, atol=3e-4, rtol=3e-4)
    assert torch.allclose(
        tk.edge_mlp_256x7(hidden, first_weight, first_bias, second_weight, second_bias),
        reference, atol=3e-4, rtol=3e-4)


def test_decode_linear_and_flux_erf_mps():
    torch.manual_seed(113)
    B, K, N = 2, 65, 37
    x = torch.randn(B, K, device="mps") * 0.15
    weight = torch.randn(N, K, device="mps") * 0.12
    bias = torch.randn(N, device="mps") * 0.04
    residual = torch.randn(B, N, device="mps") * 0.05
    got = tk_torch.decode_linear(x, weight, bias, True)
    reference = F.gelu(x @ weight.T + bias, approximate="none")
    assert torch.allclose(got, reference, atol=3e-4, rtol=3e-4)
    got_residual = tk_torch.decode_linear_residual(x, weight, bias, residual)
    assert torch.allclose(got_residual, x @ weight.T + bias + residual, atol=3e-4, rtol=3e-4)

    # Flux has a tiled weight orientation (K, M).
    flux_x = torch.randn(32, 16, device="mps") * 0.1
    flux_weight = torch.randn(16, 32, device="mps") * 0.1
    flux_bias = torch.randn(32, device="mps") * 0.04
    flux = tk_torch.flux_gelu_erf(flux_x, flux_weight, flux_bias)
    flux_ref = F.gelu(flux_x @ flux_weight + flux_bias, approximate="none")
    assert torch.allclose(flux, flux_ref, atol=3e-4, rtol=3e-4)


def test_q8_decode_linear_and_dequant_gather_mps():
    from tk.quant import QUANT_FORMATS, dequantize_q8_0, quantize_q8_0

    rng = np.random.default_rng(127)
    B, K, N = 2, 96, 39
    x = (0.12 * rng.standard_normal((B, K))).astype(np.float32)
    weight = (0.18 * rng.standard_normal((N, K))).astype(np.float32)
    packed = quantize_q8_0(weight)
    bias = (0.04 * rng.standard_normal(N)).astype(np.float32)
    residual = (0.05 * rng.standard_normal((B, N))).astype(np.float32)
    got = tk_torch.decode_linear_q8(
        torch.from_numpy(x).to("mps"), torch.from_numpy(packed).to("mps"),
        torch.from_numpy(bias).to("mps"), torch.from_numpy(residual).to("mps"), True, True)
    reference = F.gelu(
        torch.from_numpy(x @ dequantize_q8_0(packed).T + bias).to("mps"), approximate="none")
    reference = reference + torch.from_numpy(residual).to("mps")
    assert torch.allclose(got, reference, atol=5e-4, rtol=5e-4)

    ids = np.array([[2, -1], [N - 1, N]], np.int32)
    gather_scale = np.float32(np.sqrt(1536.0))
    for fmt in ("q4_0", "q8_0", "q6_K"):
        quantize, dequantize = QUANT_FORMATS[fmt]
        columns = 512 if fmt == "q6_K" else 96
        table = (0.2 * rng.standard_normal((N, columns))).astype(np.float32)
        table_q = quantize(table)
        gathered = tk_torch.dequant_gather(
            torch.from_numpy(table_q).to("mps"), torch.from_numpy(ids).to("mps"), fmt,
            float(gather_scale))
        reference_np = np.zeros((*ids.shape, columns), np.float16)
        dequantized = dequantize(table_q)
        reference_np[0, 0] = (dequantized[2] * gather_scale).astype(np.float16)
        reference_np[1, 0] = (dequantized[N - 1] * gather_scale).astype(np.float16)
        np.testing.assert_array_equal(gathered.cpu().numpy(), reference_np)


def test_q6_gather_preserves_single_rounding_under_mps_fast_math():
    """Pin a half-way case that fast reassociation otherwise rounds down one ULP."""
    from tk.quant import dequantize_q6_K

    packed = np.zeros((1, 1, 210), dtype=np.uint8)
    packed[0, 0, 105] = 10       # q6 value -22 at column 169
    packed[0, 0, 192 + 10] = 205  # signed subscale -51
    packed[0, 0, 208:210] = (0x27, 0x8C)  # fp16 scale -0.00025343895
    scale = np.float32(np.sqrt(1536.0))
    reference = (dequantize_q6_K(packed) * scale).astype(np.float16)
    got = tk_torch.dequant_gather(
        torch.from_numpy(packed).to("mps"),
        torch.zeros(1, dtype=torch.int32, device="mps"),
        "q6_K", float(scale))

    np.testing.assert_array_equal(got.cpu().numpy(), reference)
    assert got.cpu().numpy()[0, 169].view(np.uint16) == np.uint16(0xC993)


def test_q6_decode_and_constrained_lm_head_mps():
    import tk
    from tk.quant import dequantize_q6_K, quantize_q6_K

    rng = np.random.default_rng(131)
    T, V, K = 3, 513, 512
    h = (0.2 * rng.standard_normal((T, K))).astype(np.float32)
    weight = (0.18 * rng.standard_normal((V, K))).astype(np.float32)
    packed = quantize_q6_K(weight)
    ht = torch.from_numpy(h).to("mps")
    token = tk_torch.lm_head_sample_q(
        ht, torch.from_numpy(packed).to("mps"), torch.zeros(1, device="mps"),
        V, K, "q6_K", 0, 0, 1.0, 0, 0.9)
    reference_logits = h.astype(np.float64) @ dequantize_q6_K(packed).astype(np.float64).T
    np.testing.assert_array_equal(token.cpu().numpy(), reference_logits.argmax(-1).astype(np.int32))
    routed_token = tk.lm_head_sample(
        ht, torch.from_numpy(packed).to("mps"), mode="argmax", format="q6_K")
    np.testing.assert_array_equal(
        routed_token.cpu().numpy(), reference_logits.argmax(-1).astype(np.int32))

    dense_v, dense_k = 67, 31
    dense_h = (0.15 * rng.standard_normal((T, dense_k))).astype(np.float32)
    dense_w = (0.2 * rng.standard_normal((dense_v, dense_k))).astype(np.float32)
    bias = (0.04 * rng.standard_normal(dense_v)).astype(np.float32)
    forbidden = (rng.random((dense_v, dense_v)) < 0.2).astype(np.uint8)
    previous = np.array([0, 7, 31], np.int32)
    forbidden[previous, 0] = 0
    got_token, got_logprob = tk_torch.lm_head_constrained(
        torch.from_numpy(dense_h).to("mps"), torch.from_numpy(dense_w).to("mps"),
        torch.from_numpy(bias).to("mps"), torch.from_numpy(forbidden).to("mps"),
        torch.from_numpy(previous).to("mps"))
    logits = dense_h @ dense_w.T + bias
    allowed_logits = np.where(forbidden[previous] == 0, logits, -np.inf)
    expected = allowed_logits.argmax(-1).astype(np.int32)
    maximum = logits.max(-1)
    log_z = maximum + np.log(np.exp(logits - maximum[:, None]).sum(-1))
    expected_logprob = logits[np.arange(T), expected] - log_z
    np.testing.assert_array_equal(got_token.cpu().numpy(), expected)
    np.testing.assert_allclose(got_logprob.cpu().numpy(), expected_logprob, rtol=3e-4, atol=3e-4)

    # Invalid grammar rows and rows with no legal continuation must not index
    # outside the mask. They report the explicit no-token sentinel.
    blocked = np.ones((dense_v, dense_v), np.uint8)
    invalid_previous = np.array([-1, dense_v, 0], np.int32)
    blocked_token, blocked_logprob = tk_torch.lm_head_constrained(
        torch.from_numpy(dense_h).to("mps"), torch.from_numpy(dense_w).to("mps"),
        torch.from_numpy(bias).to("mps"), torch.from_numpy(blocked).to("mps"),
        torch.from_numpy(invalid_previous).to("mps"))
    np.testing.assert_array_equal(blocked_token.cpu().numpy(), np.full(T, -1, np.int32))
    np.testing.assert_array_equal(blocked_logprob.cpu().numpy(), np.full(T, -np.inf, np.float32))


def test_bfloat16_kernel_variants_mps():
    import tk

    torch.manual_seed(137)

    # Exercise bf16 variants even though the benchmark uses f32 for tighter
    # timing comparisons.
    B, Hq, Hkv, T, D = 1, 4, 2, 11, 32
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="mps") * 0.1
    k = torch.randn(B, Hkv, T, D, dtype=torch.bfloat16, device="mps") * 0.1
    v = torch.randn(B, Hkv, T, D, dtype=torch.bfloat16, device="mps") * 0.1
    attention = tk_torch.attn_decode_bh(q, k, v, T)
    attention_ref = torch.empty_like(q)
    for head in range(Hq):
        score = torch.einsum("bd,btd->bt", q[:, head].float(),
                             k[:, head // (Hq // Hkv)].float()) / math.sqrt(D)
        attention_ref[:, head] = torch.einsum(
            "bt,btd->bd", torch.softmax(score, -1),
            v[:, head // (Hq // Hkv)].float()).to(torch.bfloat16)
    assert torch.allclose(attention, attention_ref, atol=0.04, rtol=0.04)

    BW, N, H = 2, 5, 2
    qkv = torch.randn(BW, N, 3, H, D, dtype=torch.bfloat16, device="mps") * 0.1
    relative = torch.randn(H, N, N, dtype=torch.bfloat16, device="mps") * 0.03
    mask = torch.zeros(1, N, N, device="mps")
    swin = tk_torch.swin_attn_d32(qkv, relative, mask, 1)
    swin_ref = tk.swin_attn_d32(qkv, relative, mask, 1, use_kernel=False)
    assert torch.allclose(swin, swin_ref, atol=0.04, rtol=0.04)

    hidden = torch.randn(1, 2, 256, dtype=torch.bfloat16, device="mps") * 0.04
    first_weight = torch.randn(256, 512, dtype=torch.bfloat16, device="mps") * 0.03
    first_bias = torch.randn(256, dtype=torch.bfloat16, device="mps") * 0.02
    second_weight = torch.randn(7, 256, dtype=torch.bfloat16, device="mps") * 0.03
    second_bias = torch.randn(7, dtype=torch.bfloat16, device="mps") * 0.02
    edge = tk_torch.edge_mlp_256x7(
        hidden, first_weight, first_bias, second_weight, second_bias)
    edge_ref = tk.edge_mlp_256x7(
        hidden, first_weight, first_bias, second_weight, second_bias, use_kernel=False)
    assert torch.allclose(edge, edge_ref, atol=0.06, rtol=0.06)

    x = torch.randn(2, 65, dtype=torch.bfloat16, device="mps") * 0.1
    weight = torch.randn(37, 65, dtype=torch.bfloat16, device="mps") * 0.1
    bias = torch.randn(37, dtype=torch.bfloat16, device="mps") * 0.03
    linear = tk_torch.decode_linear(x, weight, bias, True)
    linear_ref = F.gelu(x @ weight.T + bias, approximate="none")
    assert torch.allclose(linear, linear_ref, atol=0.04, rtol=0.04)

    flux_x = torch.randn(32, 16, dtype=torch.bfloat16, device="mps") * 0.08
    flux_weight = torch.randn(16, 32, dtype=torch.bfloat16, device="mps") * 0.08
    flux_bias = torch.randn(32, dtype=torch.bfloat16, device="mps") * 0.03
    flux = tk_torch.flux_gelu_erf(flux_x, flux_weight, flux_bias)
    flux_ref = F.gelu(flux_x @ flux_weight + flux_bias, approximate="none")
    assert torch.allclose(flux, flux_ref, atol=0.06, rtol=0.06)


def test_quantized_embedding_lookup_and_bag_mps():
    import tk
    from tk.quant import QUANT_FORMATS

    rng = np.random.default_rng(901)
    rows, columns = 7, 96
    source = (0.2 * rng.standard_normal((rows, columns))).astype(np.float32)
    packed = QUANT_FORMATS["q4_0"][0](source)
    table = QUANT_FORMATS["q4_0"][1](packed).astype(np.float32)
    ids = np.array([[3, -1], [1, rows]], np.int32)
    add = (0.03 * rng.standard_normal((*ids.shape, columns))).astype(np.float32)
    got = tk.quantized_embedding(
        torch.from_numpy(packed).to("mps"), torch.from_numpy(ids).to("mps"),
        "q4_0", scale=1.25, add=torch.from_numpy(add).to("mps"),
        output_dtype="float32")
    reference = np.zeros((*ids.shape, columns), np.float32)
    reference[0, 0] = table[3] * 1.25 + add[0, 0]
    reference[1, 0] = table[1] * 1.25 + add[1, 0]
    np.testing.assert_allclose(got.cpu().numpy(), reference, rtol=3e-6, atol=3e-6)

    bag_ids = np.array([1, 1, -1, 4, 2], np.int32)
    offsets = np.array([0, 3, 5, 5], np.int32)
    weights = np.array([0.5, 1.25, 9.0, -0.75, 2.0], np.float32)
    bag = tk_torch.quantized_embedding_bag(
        torch.from_numpy(packed).to("mps"), torch.from_numpy(bag_ids).to("mps"),
        torch.from_numpy(offsets).to("mps"), torch.from_numpy(weights).to("mps"),
        "q4_0", 0.75, True, True, "float32")
    bag_ref = np.zeros((3, columns), np.float32)
    bag_ref[0] = (table[1] * 0.5 + table[1] * 1.25) * 0.75 / 2
    bag_ref[1] = (table[4] * -0.75 + table[2] * 2.0) * 0.75 / 2
    np.testing.assert_allclose(bag.cpu().numpy(), bag_ref, rtol=3e-6, atol=3e-6)


def test_decode_epilogue_and_swiglu_mps():
    import tk
    from tk.quant import QUANT_FORMATS

    rng = np.random.default_rng(907)
    batch, hidden, output = 2, 256, 37
    x = (0.08 * rng.standard_normal((batch, hidden))).astype(np.float32)
    gate = (0.1 * rng.standard_normal((output, hidden))).astype(np.float32)
    up = (0.09 * rng.standard_normal((output, hidden))).astype(np.float32)
    bias = (0.02 * rng.standard_normal(output)).astype(np.float32)
    residual = (0.03 * rng.standard_normal((batch, output))).astype(np.float32)
    xt, gt, ut, bt, rt = [torch.from_numpy(value).to("mps")
                           for value in (x, gate, up, bias, residual)]
    got = tk.decode_linear_epilogue(xt, gt, bt, rt, activation="silu")
    linear = x @ gate.T + bias
    reference = linear / (1.0 + np.exp(-linear)) + residual
    np.testing.assert_allclose(got.cpu().numpy(), reference, rtol=5e-4, atol=5e-4)

    dense_swiglu = tk.decode_swiglu(xt, gt, ut, bt, bt)
    dense_gate = x @ gate.T + bias
    dense_reference = dense_gate / (1.0 + np.exp(-dense_gate)) * (x @ up.T + bias)
    np.testing.assert_allclose(
        dense_swiglu.cpu().numpy(), dense_reference, rtol=5e-4, atol=5e-4)

    quantize, dequantize = QUANT_FORMATS["q4_0"]
    gate_q, up_q = quantize(gate), quantize(up)
    swiglu = tk_torch.decode_swiglu(
        xt, torch.from_numpy(gate_q).to("mps"), torch.from_numpy(up_q).to("mps"),
        bt, bt, "q4_0", True)
    gate_value = x @ dequantize(gate_q).T + bias
    up_value = x @ dequantize(up_q).T + bias
    swiglu_ref = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    np.testing.assert_allclose(swiglu.cpu().numpy(), swiglu_ref, rtol=7e-4, atol=7e-4)


def test_masked_and_candidate_lm_head_mps():
    rng = np.random.default_rng(911)
    tokens, vocab, hidden, topk = 2, 73, 61, 3
    h = (0.1 * rng.standard_normal((tokens, hidden))).astype(np.float32)
    weight = (0.1 * rng.standard_normal((vocab, hidden))).astype(np.float32)
    bias = (0.02 * rng.standard_normal(vocab)).astype(np.float32)
    rows = [[2, 7, 36, 65], [0, 11, 42, 70]]
    mask = np.zeros((tokens, (vocab + 31) // 32), np.int32)
    for row, values in enumerate(rows):
        for value in values:
            mask[row, value // 32] |= np.int32(1 << (value % 32))
    args = [torch.from_numpy(value).to("mps") for value in (h, weight, bias, mask)]
    ids, logprobs = tk_torch.lm_head_masked(*args, "", topk, True)
    flat = np.array(rows[0] + rows[1], np.int32)
    offsets = np.array([0, len(rows[0]), len(flat)], np.int32)
    cids, clogprobs = tk_torch.lm_head_candidates(
        args[0], args[1], args[2], torch.from_numpy(flat).to("mps"),
        torch.from_numpy(offsets).to("mps"), "", topk)

    logits = h @ weight.T + bias
    ref_ids, ref_lp = [], []
    for row, candidates in enumerate(rows):
        candidates = np.array(candidates, np.int32)
        order = np.lexsort((candidates, -logits[row, candidates]))[:topk]
        selected = candidates[order]
        values = logits[row, candidates]
        maximum = values.max()
        log_z = maximum + np.log(np.exp(values - maximum).sum())
        ref_ids.append(selected)
        ref_lp.append(logits[row, selected] - log_z)
    np.testing.assert_array_equal(ids.cpu().numpy(), np.stack(ref_ids))
    np.testing.assert_array_equal(cids.cpu().numpy(), np.stack(ref_ids))
    np.testing.assert_allclose(logprobs.cpu().numpy(), np.stack(ref_lp), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(clogprobs.cpu().numpy(), np.stack(ref_lp), rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("use_kernel", [None, False, True], ids=["auto", "routed", "kernel"])
def test_space_to_depth_norm_linear_mps(use_kernel):
    import tk

    rng = np.random.default_rng(919)
    batch, height, width, channels, output, block = 2, 5, 7, 11, 29, 4
    dimension = block * block * channels
    x = (0.15 * rng.standard_normal((batch, height * width, channels))).astype(np.float32)
    norm_weight = (0.8 + 0.1 * rng.standard_normal(dimension)).astype(np.float32)
    norm_bias = (0.02 * rng.standard_normal(dimension)).astype(np.float32)
    projection = (0.06 * rng.standard_normal((output, dimension))).astype(np.float32)
    projection_bias = (0.02 * rng.standard_normal(output)).astype(np.float32)
    tensors = [torch.from_numpy(value).to("mps") for value in
               (x, norm_weight, norm_bias, projection, projection_bias)]
    got = tk.space_to_depth_norm_linear(
        tensors[0], tensors[1], tensors[3], height, width,
        norm_bias=tensors[2], projection_bias=tensors[4], block_size=block,
        use_kernel=use_kernel)

    image = x.reshape(batch, height, width, channels)
    merged = []
    for oy in range((height + block - 1) // block):
        for ox in range((width + block - 1) // block):
            values = []
            for dy in range(block):
                for dx in range(block):
                    sy, sx = oy * block + dy, ox * block + dx
                    values.append(image[:, sy, sx] if sy < height and sx < width
                                  else np.zeros((batch, channels), np.float32))
            merged.append(np.concatenate(values, -1))
    merged = np.stack(merged, 1)
    mean = merged.mean(-1, keepdims=True)
    variance = (merged * merged).mean(-1, keepdims=True) - mean * mean
    normalized = (merged - mean) / np.sqrt(np.maximum(variance, 0) + 1e-5)
    reference = (normalized * norm_weight + norm_bias) @ projection.T + projection_bias
    np.testing.assert_allclose(got.cpu().numpy(), reference, rtol=7e-4, atol=7e-4)


@pytest.mark.parametrize("use_kernel", [None, False, True], ids=["auto", "routed", "kernel"])
def test_decode_cache_attention_mps(use_kernel):
    import tk

    rng = np.random.default_rng(929)
    batch, heads_q, heads_kv, dimension, cache_length = 2, 4, 2, 64, 8
    contexts = np.array([0, 5], np.int32)
    positions = np.array([2, 6], np.int32)
    shapes = ((batch, heads_q, dimension), (batch, heads_kv, dimension),
              (batch, heads_kv, dimension),
              (batch, heads_kv, cache_length, dimension),
              (batch, heads_kv, cache_length, dimension))
    q, new_k, new_v, key_cache, value_cache = [
        (0.12 * rng.standard_normal(shape)).astype(np.float32) for shape in shapes]
    angles = (np.arange(9)[:, None] + 1) * (np.arange(dimension // 2)[None, :] + 1) * 0.002
    cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    arrays = [torch.from_numpy(value).to("mps") for value in
              (q, new_k, new_v, cos, sin, positions, contexts, key_cache, value_cache)]
    output, next_k, next_v = tk.decode_cache_attention(
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[6],
        arrays[7], arrays[8], use_kernel=use_kernel)

    half = dimension // 2
    q_rotated, k_rotated = np.empty_like(q), np.empty_like(new_k)
    for row in range(batch):
        c, s = cos[positions[row]], sin[positions[row]]
        q_rotated[row, :, :half] = q[row, :, :half] * c - q[row, :, half:] * s
        q_rotated[row, :, half:] = q[row, :, half:] * c + q[row, :, :half] * s
        k_rotated[row, :, :half] = new_k[row, :, :half] * c - new_k[row, :, half:] * s
        k_rotated[row, :, half:] = new_k[row, :, half:] * c + new_k[row, :, :half] * s
    ref_k, ref_v = key_cache.copy(), value_cache.copy()
    reference = np.empty_like(q)
    for row in range(batch):
        ref_k[row, :, contexts[row]] = k_rotated[row]
        ref_v[row, :, contexts[row]] = new_v[row]
        for head in range(heads_q):
            kv_head = head // (heads_q // heads_kv)
            score = (ref_k[row, kv_head, :contexts[row] + 1] @ q_rotated[row, head]
                     / math.sqrt(dimension))
            probability = np.exp(score - score.max()); probability /= probability.sum()
            reference[row, head] = probability @ ref_v[row, kv_head, :contexts[row] + 1]
    np.testing.assert_allclose(output.cpu().numpy(), reference, rtol=7e-5, atol=7e-5)
    np.testing.assert_allclose(next_k.cpu().numpy(), ref_k, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(next_v.cpu().numpy(), ref_v)
