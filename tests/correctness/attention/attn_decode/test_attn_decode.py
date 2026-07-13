import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import attn_decode, attn_decode_bh


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
