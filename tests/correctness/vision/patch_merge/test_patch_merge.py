import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import patch_merge_layernorm, space_to_depth_norm_linear


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


def _space_to_depth_reference(x, height, width, block_size):
    batch, _, channels = x.shape
    image = x.reshape(batch, height, width, channels)
    patches = []
    for output_y in range((height + block_size - 1) // block_size):
        for output_x in range((width + block_size - 1) // block_size):
            values = []
            for dy in range(block_size):
                for dx in range(block_size):
                    source_y = output_y * block_size + dy
                    source_x = output_x * block_size + dx
                    if source_y < height and source_x < width:
                        values.append(image[:, source_y, source_x])
                    else:
                        values.append(np.zeros((batch, channels), dtype=np.float32))
            patches.append(np.concatenate(values, axis=-1))
    return np.stack(patches, axis=1)


@pytest.mark.parametrize(
    "height,width,channels,out_channels,block_size",
    [(4, 6, 13, 29, 2), (3, 5, 17, 31, 2), (5, 7, 20, 43, 4)],
)
@pytest.mark.parametrize("use_biases", [False, True])
@pytest.mark.parametrize("use_kernel", [False, True], ids=["routed", "kernel"])
def test_space_to_depth_norm_linear_matches_reference(
        height, width, channels, out_channels, block_size, use_biases, use_kernel):
    rng = np.random.default_rng(height * 1000 + width * 100 + channels)
    batch, eps = 2, 2e-5
    dimension = block_size * block_size * channels
    x = (0.25 * rng.standard_normal((batch, height * width, channels))).astype(np.float32)
    norm_weight = (0.8 + 0.2 * rng.standard_normal(dimension)).astype(np.float32)
    norm_bias = (0.03 * rng.standard_normal(dimension)).astype(np.float32)
    projection_weight = (
        0.08 * rng.standard_normal((out_channels, dimension))).astype(np.float32)
    projection_bias = (0.02 * rng.standard_normal(out_channels)).astype(np.float32)
    got = space_to_depth_norm_linear(
        mx.array(x), mx.array(norm_weight), mx.array(projection_weight),
        height, width,
        norm_bias=mx.array(norm_bias) if use_biases else None,
        projection_bias=mx.array(projection_bias) if use_biases else None,
        block_size=block_size, eps=eps, use_kernel=use_kernel)
    mx.eval(got)

    merged = _space_to_depth_reference(x, height, width, block_size)
    mean = merged.mean(axis=-1, keepdims=True)
    variance = (merged * merged).mean(axis=-1, keepdims=True) - mean * mean
    normalized = (merged - mean) / np.sqrt(np.maximum(variance, 0.0) + eps)
    normalized *= norm_weight
    if use_biases:
        normalized += norm_bias
    ref = normalized @ projection_weight.T
    if use_biases:
        ref += projection_bias
    np.testing.assert_allclose(np.array(got), ref, rtol=6e-4, atol=6e-4)


@pytest.mark.parametrize("use_kernel", [False, True], ids=["routed", "kernel"])
def test_space_to_depth_norm_linear_bfloat16_rounding(use_kernel):
    rng = np.random.default_rng(599)
    batch, height, width, channels, out_channels = 1, 3, 5, 12, 19
    dimension = 4 * channels
    arrays = [
        mx.array((0.1 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
        for shape in (
            (batch, height * width, channels), (dimension,), (dimension,),
            (out_channels, dimension), (out_channels,))
    ]
    x, norm_weight, norm_bias, projection_weight, projection_bias = arrays
    got = space_to_depth_norm_linear(
        x, norm_weight, projection_weight, height, width,
        norm_bias=norm_bias, projection_bias=projection_bias, use_kernel=use_kernel)
    mx.eval(got)
    assert got.dtype == mx.bfloat16 and got.shape == (1, 6, out_channels)
