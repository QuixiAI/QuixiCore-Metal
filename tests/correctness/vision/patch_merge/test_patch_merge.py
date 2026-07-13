import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import patch_merge_layernorm


@pytest.mark.parametrize("height,width", [(4, 6), (3, 5)])
def test_patch_merge_layernorm_order_and_padding(height, width):
    rng = np.random.default_rng(height * 10 + width)
    B, C, eps = 2, 12, 1e-5
    source = (0.4 * rng.standard_normal((B, height * width, C))).astype(np.float32)
    weight = (0.5 + 0.2 * rng.standard_normal(4 * C)).astype(np.float32)
    bias = (0.1 * rng.standard_normal(4 * C)).astype(np.float32)
    x = mx.array(source).astype(mx.bfloat16)
    w = mx.array(weight).astype(mx.bfloat16)
    b = mx.array(bias).astype(mx.bfloat16)
    got = patch_merge_layernorm(x, w, b, height, width, eps)
    mx.eval(got)

    rounded = np.array(x.astype(mx.float32)).reshape(B, height, width, C)
    rows = []
    for oy in range((height + 1) // 2):
        for ox in range((width + 1) // 2):
            chunks = []
            for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
                sy, sx = 2 * oy + dy, 2 * ox + dx
                chunks.append(rounded[:, sy, sx] if sy < height and sx < width
                              else np.zeros((B, C), np.float32))
            rows.append(np.concatenate(chunks, axis=-1))
    merged = np.stack(rows, axis=1)
    mean = merged.mean(-1, keepdims=True)
    var = (merged * merged).mean(-1, keepdims=True) - mean * mean
    ref = ((merged - mean) / np.sqrt(np.maximum(var, 0) + eps) *
           np.array(w.astype(mx.float32)) + np.array(b.astype(mx.float32)))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, rtol=3e-2, atol=3e-2)
