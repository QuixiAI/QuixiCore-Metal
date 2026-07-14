import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import attn_decode, attn_decode_bh, decode_cache_attention


@pytest.mark.parametrize("Hq,Hkv,D,Tk", [(4, 2, 64, 37), (3, 1, 128, 65)])
def test_attn_decode_matches_numpy(Hq, Hkv, D, Tk):
    rng = np.random.default_rng(97 + Hq + D)
    q_np = (0.2 * rng.standard_normal((Hq, D))).astype(np.float32)
    k_np = (0.2 * rng.standard_normal((Tk, Hkv, D))).astype(np.float32)
    v_np = (0.2 * rng.standard_normal((Tk, Hkv, D))).astype(np.float32)
    got = attn_decode(mx.array(q_np).astype(mx.float32),
                      mx.array(k_np).astype(mx.float32),
                      mx.array(v_np).astype(mx.float32))
    mx.eval(got)
    ref = np.zeros((Hq, D), dtype=np.float32)
    group = Hq // Hkv
    scale = 1.0 / np.sqrt(D)
    for h in range(Hq):
        kvh = h // group
        scores = k_np[:, kvh, :] @ q_np[h] * scale
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        ref[h] = probs @ v_np[:, kvh, :]
    np.testing.assert_allclose(np.array(got), ref, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("use_kernel", [False, True])
@pytest.mark.parametrize("dtype,atol", [(mx.float32, 4e-5), (mx.bfloat16, 4e-2)])
def test_attn_decode_bh_matches_preallocated_cache_layout(use_kernel, dtype, atol):
    rng = np.random.default_rng(311)
    B, Hq, Hkv, D, cache_T, Tk = 2, 4, 2, 64, 19, 13
    q = (0.2 * rng.standard_normal((B, Hq, D))).astype(np.float32)
    k = (0.2 * rng.standard_normal((B, Hkv, cache_T, D))).astype(np.float32)
    v = (0.2 * rng.standard_normal((B, Hkv, cache_T, D))).astype(np.float32)
    qd, kd, vd = (mx.array(value).astype(dtype) for value in (q, k, v))
    got = attn_decode_bh(qd, kd, vd, Tk, use_kernel=use_kernel)
    mx.eval(got)
    q, k, v = (np.array(value.astype(mx.float32)) for value in (qd, kd, vd))
    ref = np.empty_like(q)
    for batch in range(B):
        for head in range(Hq):
            kv_head = head // (Hq // Hkv)
            scores = k[batch, kv_head, :Tk] @ q[batch, head] / np.sqrt(D)
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            ref[batch, head] = probs @ v[batch, kv_head, :Tk]
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), ref, rtol=atol, atol=atol)


def _rms_norm(x, weight, eps, gemma):
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * (weight + (1.0 if gemma else 0.0))


def _split_rope(x, cos, sin):
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return np.concatenate((first * cos - second * sin,
                           second * cos + first * sin), axis=-1)


@pytest.mark.parametrize("dimension", [64, 128])
@pytest.mark.parametrize("use_norm,gemma", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("use_kernel", [None, False, True], ids=["auto", "routed", "kernel"])
def test_decode_cache_attention_fused_step_matches_reference(
        dimension, use_norm, gemma, use_kernel):
    rng = np.random.default_rng(701 + dimension + use_norm + gemma)
    batch, heads_q, heads_kv, cache_length = 2, 4, 2, 9
    eps, scale = 2e-6, 0.37
    contexts = np.array([0, 5], dtype=np.int32)
    positions = np.array([2, 7], dtype=np.int32)
    q = (0.17 * rng.standard_normal((batch, heads_q, dimension))).astype(np.float32)
    new_k = (0.16 * rng.standard_normal((batch, heads_kv, dimension))).astype(np.float32)
    new_v = (0.15 * rng.standard_normal((batch, heads_kv, dimension))).astype(np.float32)
    key_cache = (
        0.14 * rng.standard_normal((batch, heads_kv, cache_length, dimension))).astype(np.float32)
    value_cache = (
        0.13 * rng.standard_normal((batch, heads_kv, cache_length, dimension))).astype(np.float32)
    q_weight = (0.8 + 0.2 * rng.standard_normal(dimension)).astype(np.float32)
    k_weight = (0.7 + 0.2 * rng.standard_normal(dimension)).astype(np.float32)
    angles = (np.arange(12, dtype=np.float32)[:, None] + 1.0) * (
        np.arange(dimension // 2, dtype=np.float32)[None, :] + 0.5) * 0.003
    cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    key_input, value_input = mx.array(key_cache), mx.array(value_cache)
    output, next_k, next_v = decode_cache_attention(
        mx.array(q), mx.array(new_k), mx.array(new_v), mx.array(cos), mx.array(sin),
        mx.array(positions), mx.array(contexts), key_input, value_input,
        q_weight=mx.array(q_weight) if use_norm else None,
        k_weight=mx.array(k_weight) if use_norm else None,
        eps=eps, gemma=gemma, softmax_scale=scale, use_kernel=use_kernel)
    mx.eval(output, next_k, next_v)

    q_used = _rms_norm(q, q_weight, eps, gemma) if use_norm else q
    k_used = _rms_norm(new_k, k_weight, eps, gemma) if use_norm else new_k
    q_rotated = np.empty_like(q_used)
    k_rotated = np.empty_like(k_used)
    for row in range(batch):
        q_rotated[row] = _split_rope(q_used[row], cos[positions[row]], sin[positions[row]])
        k_rotated[row] = _split_rope(k_used[row], cos[positions[row]], sin[positions[row]])
    ref_k, ref_v = key_cache.copy(), value_cache.copy()
    ref_output = np.empty_like(q)
    for row in range(batch):
        ref_k[row, :, contexts[row]] = k_rotated[row]
        ref_v[row, :, contexts[row]] = new_v[row]
        for head in range(heads_q):
            kv_head = head // (heads_q // heads_kv)
            scores = (ref_k[row, kv_head, :contexts[row] + 1] @
                      q_rotated[row, head]) * scale
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            ref_output[row, head] = probabilities @ ref_v[
                row, kv_head, :contexts[row] + 1]

    np.testing.assert_allclose(np.array(output), ref_output, rtol=7e-5, atol=7e-5)
    np.testing.assert_allclose(np.array(next_k), ref_k, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(np.array(next_v), ref_v)
    # The public operation is functional: caller-owned cache arrays remain unchanged.
    np.testing.assert_array_equal(np.array(key_input), key_cache)
    np.testing.assert_array_equal(np.array(value_input), value_cache)


@pytest.mark.parametrize("use_kernel", [None, False, True], ids=["auto", "routed", "kernel"])
def test_decode_cache_attention_bfloat16_dtype_and_default_scale(use_kernel):
    rng = np.random.default_rng(809)
    batch, heads_q, heads_kv, dimension, cache_length = 1, 2, 1, 64, 4
    arrays = [
        mx.array((0.1 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
        for shape in ((batch, heads_q, dimension), (batch, heads_kv, dimension),
                      (batch, heads_kv, dimension), (6, dimension // 2),
                      (6, dimension // 2),
                      (batch, heads_kv, cache_length, dimension),
                      (batch, heads_kv, cache_length, dimension))
    ]
    q, new_k, new_v, cos, sin, key_cache, value_cache = arrays
    output, next_k, next_v = decode_cache_attention(
        q, new_k, new_v, cos, sin, mx.array([3]), mx.array([2]),
        key_cache, value_cache, use_kernel=use_kernel)
    mx.eval(output, next_k, next_v)
    assert output.dtype == next_k.dtype == next_v.dtype == mx.bfloat16
    assert output.shape == q.shape and next_k.shape == key_cache.shape


def test_decode_cache_attention_auto_route_uses_kernel(monkeypatch):
    import tk

    original = tk._mlx()

    class TrackingExtension:
        def __init__(self):
            self.calls = 0

        def decode_cache_attention(self, *args, **kwargs):
            self.calls += 1
            return original.decode_cache_attention(*args, **kwargs)

    tracking = TrackingExtension()
    monkeypatch.setattr(tk, "_mlx_ext", tracking)
    dimension = 64
    q = mx.zeros((1, 2, dimension), dtype=mx.float32)
    new_k = mx.zeros((1, 1, dimension), dtype=mx.float32)
    new_v = mx.zeros((1, 1, dimension), dtype=mx.float32)
    cos = mx.ones((1, dimension // 2), dtype=mx.float32)
    sin = mx.zeros((1, dimension // 2), dtype=mx.float32)
    positions = mx.array([0], dtype=mx.int32)
    contexts = mx.array([0], dtype=mx.int32)

    for cache_length in (511, 512):
        calls_before = tracking.calls
        cache = mx.zeros((1, 1, cache_length, dimension), dtype=mx.float32)
        outputs = tk.decode_cache_attention(
            q, new_k, new_v, cos, sin, positions, contexts, cache, cache)
        mx.eval(*outputs)
        assert tracking.calls == calls_before + 1

    calls_before = tracking.calls
    cache = mx.zeros((1, 1, 512, dimension), dtype=mx.float32)
    outputs = tk.decode_cache_attention(
        q, new_k, new_v, cos, sin, positions, contexts, cache, cache,
        use_kernel=False)
    mx.eval(*outputs)
    assert tracking.calls == calls_before
