import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import attn_decode


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
