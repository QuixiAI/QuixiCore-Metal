import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import swin_attn_d32


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("use_kernel", [False, True])
def test_swin_attn_d32_matches_numpy(with_mask, use_kernel):
    rng = np.random.default_rng(101 + with_mask)
    BW, N, H, D, windows_per_image = 4, 9, 3, 32, 2
    qkv = (0.2 * rng.standard_normal((BW, N, 3, H, D))).astype(np.float32)
    relative_bias = (0.1 * rng.standard_normal((H, N, N))).astype(np.float32)
    mask = np.zeros((windows_per_image, N, N), np.float32)
    if with_mask:
        mask[1, :, N // 2:] = -10.0

    got = swin_attn_d32(
        mx.array(qkv), mx.array(relative_bias),
        mx.array(mask) if with_mask else None, windows_per_image, use_kernel=use_kernel)
    mx.eval(got)

    ref = np.empty((BW, N, H, D), np.float32)
    for window in range(BW):
        for head in range(H):
            q = qkv[window, :, 0, head]
            k = qkv[window, :, 1, head]
            v = qkv[window, :, 2, head]
            scores = q @ k.T / np.sqrt(D) + relative_bias[head]
            if with_mask:
                scores += mask[window % windows_per_image]
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            ref[window, :, head] = probs @ v
    np.testing.assert_allclose(np.array(got), ref, rtol=4e-5, atol=4e-5)


def test_swin_attn_singleton_mask_is_not_treated_as_placeholder():
    rng = np.random.default_rng(109)
    qkv = (0.2 * rng.standard_normal((1, 1, 3, 1, 32))).astype(np.float32)
    got = swin_attn_d32(
        mx.array(qkv), mx.zeros((1, 1, 1)), mx.full((1, 1, 1), -7.0),
        windows_per_image=1, use_kernel=True)
    mx.eval(got)
    np.testing.assert_array_equal(np.array(got), qkv[:, :, 2])


def test_swin_attn_no_mask_allows_zero_window_count():
    rng = np.random.default_rng(113)
    qkv = (0.2 * rng.standard_normal((1, 2, 3, 1, 32))).astype(np.float32)
    got = swin_attn_d32(
        mx.array(qkv), mx.zeros((1, 2, 2)), use_kernel=True)
    mx.eval(got)
    assert got.shape == (1, 2, 1, 32)


def test_swin_attn_bfloat16_kernel_matches_framework_route():
    rng = np.random.default_rng(127)
    BW, N, H = 2, 5, 2
    qkv = mx.array((0.2 * rng.standard_normal((BW, N, 3, H, 32))).astype(np.float32)).astype(
        mx.bfloat16)
    bias = mx.array((0.05 * rng.standard_normal((H, N, N))).astype(np.float32)).astype(
        mx.bfloat16)
    mask = mx.zeros((1, N, N), dtype=mx.float32)
    got = swin_attn_d32(qkv, bias, mask, 1, use_kernel=True)
    ref = swin_attn_d32(qkv, bias, mask, 1, use_kernel=False)
    mx.eval(got, ref)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=4e-2, atol=4e-2)
