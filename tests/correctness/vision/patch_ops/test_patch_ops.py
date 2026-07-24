"""Correctness for reusable vision/audio 2-D tensor preparation."""

import mlx.core as mx
import numpy as np
import pytest

import tk


_DTYPES = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}


def _patch_ref(x, kh, kw, sh, sw, ph, pw):
    B, H, W, C = x.shape
    OH, OW = (H + 2 * ph - kh) // sh + 1, (W + 2 * pw - kw) // sw + 1
    out = np.zeros((B, OH * OW, kh * kw * C), np.float32)
    for b in range(B):
        for oy in range(OH):
            for ox in range(OW):
                patch = []
                for ky in range(kh):
                    for kx in range(kw):
                        iy, ix = oy * sh + ky - ph, ox * sw + kx - pw
                        patch.append(x[b, iy, ix] if 0 <= iy < H and 0 <= ix < W else np.zeros(C))
                out[b, oy * OW + ox] = np.concatenate(patch)
    return out


def _patch3d_ref(x, kt, kh, kw, st, sh, sw, pt, ph, pw):
    B, T, H, W, C = x.shape
    OT = (T + 2 * pt - kt) // st + 1
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1
    out = np.zeros((B, OT * OH * OW, kt * kh * kw * C), np.float32)
    for b in range(B):
        for ot in range(OT):
            for oy in range(OH):
                for ox in range(OW):
                    patch = []
                    for tt in range(kt):
                        for yy in range(kh):
                            for xx in range(kw):
                                it, iy, ix = ot * st + tt - pt, oy * sh + yy - ph, ox * sw + xx - pw
                                patch.append(x[b, it, iy, ix] if 0 <= it < T and 0 <= iy < H and 0 <= ix < W else np.zeros(C))
                    out[b, (ot * OH + oy) * OW + ox] = np.concatenate(patch)
    return out


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(2, 7, 9, 3, 3, 2, 2, 1, 1, 0),
                                    (1, 8, 8, 4, 2, 2, 2, 2, 0, 0)])
def test_extract_patches_2d(dtype, shape):
    B, H, W, C, kh, kw, sh, sw, ph, pw = shape
    rng = np.random.default_rng(sum(shape))
    x0 = (0.2 * rng.standard_normal((B, H, W, C))).astype(np.float32)
    xd = mx.array(x0).astype(_DTYPES[dtype])
    got = tk.extract_patches_2d(xd, kh, kw, sh, sw, ph, pw)
    mx.eval(got)
    xb = np.array(xd.astype(mx.float32))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                               _patch_ref(xb, kh, kw, sh, sw, ph, pw), atol=0, rtol=0)


def test_extract_patches_3d_general_geometry():
    rng = np.random.default_rng(27)
    x = (0.2 * rng.standard_normal((2, 5, 7, 8, 3))).astype(np.float32)
    got = tk.extract_patches_3d(mx.array(x), 2, 3, 2, 2, 2, 3, 1, 1, 0,
                                use_kernel=True)
    ref = _patch3d_ref(x, 2, 3, 2, 2, 2, 3, 1, 1, 0)
    np.testing.assert_allclose(np.array(got), ref, atol=0, rtol=0)


def test_vision_patch_projection_3d():
    rng = np.random.default_rng(28)
    x = (0.2 * rng.standard_normal((1, 4, 8, 8, 3))).astype(np.float32)
    w = (0.2 * rng.standard_normal((16, 2, 4, 4, 3))).astype(np.float32)
    b = (0.1 * rng.standard_normal(16)).astype(np.float32)
    got = tk.vision_patch_projection_3d(mx.array(x), mx.array(w), mx.array(b))
    ref = _patch3d_ref(x, 2, 4, 4, 2, 4, 4, 0, 0, 0) @ w.reshape(16, -1).T + b
    np.testing.assert_allclose(np.array(got), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("align", [False, True])
def test_interpolate_position_2d(align):
    rng = np.random.default_rng(31 + align)
    x = (0.2 * rng.standard_normal((3, 5, 17))).astype(np.float32)
    got = tk.interpolate_position_2d(mx.array(x), 7, 4, align_corners=align)
    ref = np.empty((7, 4, 17), np.float32)
    for oy in range(7):
        fy = oy * 2 / 6 if align else (oy + 0.5) * 3 / 7 - 0.5
        cy = np.clip(fy, 0, 2); y0 = int(np.floor(cy)); y1 = min(y0 + 1, 2); wy = cy - y0
        for ox in range(4):
            fx = ox * 4 / 3 if align else (ox + 0.5) * 5 / 4 - 0.5
            cx = np.clip(fx, 0, 4); x0 = int(np.floor(cx)); x1 = min(x0 + 1, 4); wx = cx - x0
            ref[oy, ox] = ((1 - wy) * ((1 - wx) * x[y0, x0] + wx * x[y0, x1]) +
                               wy * ((1 - wx) * x[y1, x0] + wx * x[y1, x1]))
    np.testing.assert_allclose(np.array(got), ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("ceil", [False, True])
def test_avg_pool2d_tokens(ceil):
    rng = np.random.default_rng(40 + ceil)
    x = (0.2 * rng.standard_normal((2, 7, 9, 33))).astype(np.float32)
    got = tk.avg_pool2d_tokens(mx.array(x), 3, 2, 2, 2, ceil_mode=ceil)
    oh = ((7 - 3 + 1) // 2 + 1) if ceil else ((7 - 3) // 2 + 1)
    ow = ((9 - 2 + 1) // 2 + 1) if ceil else ((9 - 2) // 2 + 1)
    ref = np.empty((2, oh, ow, 33), np.float32)
    for b in range(2):
        for y in range(oh):
            for xx in range(ow):
                ref[b, y, xx] = x[b, y * 2:min(y * 2 + 3, 7), xx * 2:min(xx * 2 + 2, 9)].mean((0, 1))
    np.testing.assert_allclose(np.array(got), ref, atol=1e-6, rtol=1e-6)


def test_vision_patch_projection():
    rng = np.random.default_rng(44)
    x = (0.2 * rng.standard_normal((1, 8, 8, 3))).astype(np.float32)
    w = (0.2 * rng.standard_normal((16, 4, 4, 3))).astype(np.float32)
    b = (0.1 * rng.standard_normal(16)).astype(np.float32)
    got = tk.vision_patch_projection(mx.array(x), mx.array(w), mx.array(b))
    ref = _patch_ref(x, 4, 4, 4, 4, 0, 0) @ w.reshape(16, -1).T + b
    np.testing.assert_allclose(np.array(got), ref, atol=1e-5, rtol=1e-5)


def test_factorized_position_2d_with_padding():
    rng = np.random.default_rng(52)
    table = (0.2 * rng.standard_normal((2, 7, 33))).astype(np.float32)
    ids = np.array([[[0, 1], [4, 2], [6, 5], [3, 0]],
                    [[2, 6], [1, 3], [5, 4], [0, 0]]], dtype=np.int32)
    valid = np.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=np.int32)
    td = mx.array(table).astype(mx.bfloat16)
    got = tk.factorized_position_2d(mx.array(ids), td, mx.array(valid))
    mx.eval(got)
    tb = np.array(td.astype(mx.float32)); ref = np.zeros((2, 4, 33), np.float32)
    for b in range(2):
        for n in range(4):
            if valid[b, n]:
                ref[b, n] = tb[0, ids[b, n, 0]] + tb[1, ids[b, n, 1]]
    ref = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=0, rtol=0)


def test_pool_tokens_by_position_shuffled_and_padded():
    rng = np.random.default_rng(53)
    B, N, D = 2, 16, 17
    coords = np.array([(x, y) for y in range(4) for x in range(4)], np.int32)
    ids = np.stack([coords[rng.permutation(N)], coords[rng.permutation(N)]])
    valid = np.ones((B, N), np.int32)
    valid[0, np.all(ids[0] < 2, axis=1)] = 0
    valid[1, 3] = 0
    x = (0.1 * rng.standard_normal((B, N, D))).astype(np.float32)
    got, got_mask = tk.pool_tokens_by_position(
        mx.array(x), mx.array(ids), mx.array(valid), 4, 2, 4)
    mx.eval(got, got_mask)
    ref = np.zeros((B, 4, D), np.float32); ref_mask = np.zeros((B, 4), np.int32)
    scale = np.sqrt(D) / 4.0
    for b in range(B):
        for n in range(N):
            if valid[b, n]:
                px, py = ids[b, n]; bucket = px // 2 + 2 * (py // 2)
                ref[b, bucket] += x[b, n] * scale; ref_mask[b, bucket] = 1
    np.testing.assert_allclose(np.array(got), ref, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(np.array(got_mask), ref_mask)
