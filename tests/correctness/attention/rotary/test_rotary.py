"""Correctness test for the ThunderMittens rotary (RoPE) Metal kernel.

Split-half / GPT-NeoX convention. Validated two ways:
  1. vs mx.fast.rope(traditional=False, freqs=inv_freq) — same inverse frequencies, so this
     checks the kernel matches the standard oracle convention.
  2. vs an explicit fp32 split-half reference using the same cos/sin tables — checks the kernel
     math independent of the oracle.

Run from kernels/:  python -m pytest rotary/correctness/test_rotary.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import mrope, qwen_vision_rope_2d, rotary, rotary_positioned, vision_rope_2d


def make_cos_sin(N, D, base=10000.0):
    inv_freq = base ** (-(np.arange(0, D, 2).astype(np.float32) / D))  # (D/2,)
    pos = np.arange(N).astype(np.float32)[:, None]                     # (N,1)
    ang = pos * inv_freq[None, :]                                      # (N, D/2)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), inv_freq


# (B, H, N, D)
SHAPES = [(1, 2, 256, 64), (2, 4, 128, 64), (1, 2, 256, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rotary(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    cos_np, sin_np, inv_freq = make_cos_sin(N, D)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    cos = mx.array(cos_np).astype(mx.bfloat16)
    sin = mx.array(sin_np).astype(mx.bfloat16)

    got = rotary(x, cos, sin)
    # mx.fast.rope uses inv_freq = 1/freqs[i], so pass wavelengths (1/inv_freq) to match
    # our angle = pos * inv_freq.
    exp = mx.fast.rope(x, dims=D, traditional=False, base=None, scale=1.0,
                       offset=0, freqs=mx.array(1.0 / inv_freq))
    mx.eval(got, exp)
    assert got.shape == x.shape
    assert mx.allclose(got, exp, atol=3e-2, rtol=3e-2), \
        f"vs mx.fast.rope: {mx.max(mx.abs(got.astype(mx.float32)-exp.astype(mx.float32))).item()}"

    # explicit split-half reference with the same tables
    xf = np.array(x.astype(mx.float32))
    x1, x2 = xf[..., :D // 2], xf[..., D // 2:]
    c, s = cos_np[None, None], sin_np[None, None]  # (1,1,N,D/2)
    ref = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    got_np = np.array(got.astype(mx.float32))
    assert np.max(np.abs(got_np - ref)) < 3e-2, np.max(np.abs(got_np - ref))


@pytest.mark.parametrize("shape", SHAPES)
def test_rotary_interleaved(shape):
    # GPT-J interleaved convention: rotates adjacent pairs (x[2p], x[2p+1]).
    B, H, N, D = shape
    mx.random.seed(1)
    cos_np, sin_np, inv_freq = make_cos_sin(N, D)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    cos = mx.array(cos_np).astype(mx.bfloat16)
    sin = mx.array(sin_np).astype(mx.bfloat16)

    got = rotary(x, cos, sin, interleaved=True)
    # vs mx.fast.rope(traditional=True) with the same inverse frequencies
    exp = mx.fast.rope(x, dims=D, traditional=True, base=None, scale=1.0,
                       offset=0, freqs=mx.array(1.0 / inv_freq))
    mx.eval(got, exp)
    assert got.shape == x.shape
    assert mx.allclose(got, exp, atol=3e-2, rtol=3e-2), \
        f"vs mx.fast.rope(traditional=True): {mx.max(mx.abs(got.astype(mx.float32)-exp.astype(mx.float32))).item()}"

    # explicit interleaved reference with the same tables
    xf = np.array(x.astype(mx.float32))
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    c, s = cos_np[None, None], sin_np[None, None]
    ref = np.empty_like(xf)
    ref[..., 0::2] = xe * c - xo * s
    ref[..., 1::2] = xe * s + xo * c
    got_np = np.array(got.astype(mx.float32))
    assert np.max(np.abs(got_np - ref)) < 3e-2, np.max(np.abs(got_np - ref))


def positioned_reference(x, cos, sin, positions, rotary_dim, interleaved=False):
    out = x.copy()
    B, _, N, _ = x.shape
    for b in range(B):
        pos = positions if positions.ndim == 1 else positions[b]
        c = cos[pos][None, :, :]
        s = sin[pos][None, :, :]
        if interleaved:
            a = x[b, ..., :rotary_dim:2]
            z = x[b, ..., 1:rotary_dim:2]
            out[b, ..., :rotary_dim:2] = a * c - z * s
            out[b, ..., 1:rotary_dim:2] = a * s + z * c
        else:
            rp = rotary_dim // 2
            a = x[b, ..., :rp]
            z = x[b, ..., rp:rotary_dim]
            out[b, ..., :rp] = a * c - z * s
            out[b, ..., rp:rotary_dim] = a * s + z * c
    return out


@pytest.mark.parametrize(
    "shape,rotary_dim,interleaved,batched",
    [
        ((1, 3, 17, 64), 32, False, False),
        ((2, 2, 13, 128), 64, True, True),
        ((1, 2, 9, 256), 192, False, True),
        ((1, 1, 7, 512), 128, True, False),
    ],
)
def test_rotary_positioned_partial(shape, rotary_dim, interleaved, batched):
    B, _, N, D = shape
    rng = np.random.default_rng(21 + D)
    max_pos = 4 * N + 3
    inv = 10000.0 ** (-(np.arange(rotary_dim // 2, dtype=np.float32) * 2 / rotary_dim))
    angle = np.arange(max_pos, dtype=np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    positions = ((np.arange(N, dtype=np.int32) * 3 + 1) % max_pos)
    if batched:
        positions = np.stack([(positions + b * 2) % max_pos for b in range(B)])
    x_np = rng.standard_normal(shape).astype(np.float32)
    x = mx.array(x_np).astype(mx.bfloat16)
    got = rotary_positioned(
        x, mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16),
        mx.array(positions), rotary_dim=rotary_dim, interleaved=interleaved)
    mx.eval(got)
    xb = np.array(x.astype(mx.float32))
    cb = np.array(mx.array(cos).astype(mx.bfloat16).astype(mx.float32))
    sb = np.array(mx.array(sin).astype(mx.bfloat16).astype(mx.float32))
    ref = positioned_reference(xb, cb, sb, positions, rotary_dim, interleaved)
    got_np = np.array(got.astype(mx.float32))
    assert np.max(np.abs(got_np - ref)) < 3e-2
    np.testing.assert_array_equal(got_np[..., rotary_dim:], xb[..., rotary_dim:])


def mrope_reference(x, cos, sin, positions, sections, rotary_dim, interleaved):
    out = x.copy()
    B, H, N, _ = x.shape
    rp = rotary_dim // 2
    boundaries = np.cumsum(sections)
    for b in range(B):
        p3 = positions if positions.ndim == 2 else positions[b]
        for p in range(rp):
            axis = p % 3 if interleaved else int(np.searchsorted(boundaries, p, side="right"))
            pos = p3[axis]
            c, s = cos[pos, p][None, :], sin[pos, p][None, :]
            a, z = x[b, :, :, p], x[b, :, :, rp + p]
            out[b, :, :, p] = a * c - z * s
            out[b, :, :, rp + p] = a * s + z * c
    return out


@pytest.mark.parametrize(
    "shape,rotary_dim,sections,interleaved,batched",
    [
        ((1, 2, 19, 64), 64, (8, 12, 12), False, False),
        ((2, 3, 11, 64), 64, (11, 11, 10), True, True),
        ((1, 2, 7, 128), 64, (8, 12, 12), False, True),
        ((1, 1, 5, 512), 128, (22, 21, 21), True, False),
    ],
)
def test_mrope(shape, rotary_dim, sections, interleaved, batched):
    B, _, N, D = shape
    rng = np.random.default_rng(41 + D)
    max_pos = 3 * N + 5
    inv = 10000.0 ** (-(np.arange(rotary_dim // 2, dtype=np.float32) * 2 / rotary_dim))
    angle = np.arange(max_pos, dtype=np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    p = np.stack([
        np.arange(N, dtype=np.int32) % max_pos,
        (2 * np.arange(N, dtype=np.int32) + 1) % max_pos,
        (3 * np.arange(N, dtype=np.int32) + 2) % max_pos,
    ])
    positions = p if not batched else np.stack([p, (p + 2) % max_pos])[:B]
    x_np = rng.standard_normal(shape).astype(np.float32)
    x = mx.array(x_np).astype(mx.bfloat16)
    c = mx.array(cos).astype(mx.bfloat16)
    s = mx.array(sin).astype(mx.bfloat16)
    got = mrope(
        x, c, s, mx.array(positions), sections, rotary_dim=rotary_dim,
        section_interleaved=interleaved)
    mx.eval(got)
    xb = np.array(x.astype(mx.float32))
    cb, sb = np.array(c.astype(mx.float32)), np.array(s.astype(mx.float32))
    ref = mrope_reference(xb, cb, sb, positions, sections, rotary_dim, interleaved)
    got_np = np.array(got.astype(mx.float32))
    assert np.max(np.abs(got_np - ref)) < 3e-2
    np.testing.assert_array_equal(got_np[..., rotary_dim:], xb[..., rotary_dim:])


@pytest.mark.parametrize("scaling", ["linear", "llama3"])
def test_rotary_positioned_consumes_scaled_tables(scaling):
    """Scaling policy stays in table generation; the kernel must preserve any
    validated linear or Llama-3 piecewise frequencies it is given."""
    B, H, N, D, rd = 1, 2, 23, 128, 64
    rng = np.random.default_rng(301)
    inv = 10000.0 ** (-(np.arange(rd // 2, dtype=np.float32) * 2 / rd))
    factor = 8.0
    if scaling == "linear":
        scaled = inv / factor
    else:
        old_context, low, high = 8192.0, 1.0, 4.0
        wavelength = 2 * np.pi / inv
        smooth = (old_context / wavelength - low) / (high - low)
        smooth = np.clip(smooth, 0.0, 1.0)
        scaled = np.where(
            wavelength > old_context / low, inv / factor,
            np.where(wavelength < old_context / high, inv,
                     (1.0 - smooth) * inv / factor + smooth * inv))
    max_pos = 4 * N
    angle = np.arange(max_pos, dtype=np.float32)[:, None] * scaled[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    positions = ((5 * np.arange(N) + 3) % max_pos).astype(np.int32)
    x = mx.array(rng.standard_normal((B, H, N, D)).astype(np.float32)).astype(mx.bfloat16)
    c, s = mx.array(cos).astype(mx.bfloat16), mx.array(sin).astype(mx.bfloat16)
    got = rotary_positioned(x, c, s, mx.array(positions), rotary_dim=rd)
    mx.eval(got)
    xb = np.array(x.astype(mx.float32))
    ref = positioned_reference(
        xb, np.array(c.astype(mx.float32)), np.array(s.astype(mx.float32)),
        positions, rd, False)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=3e-2, rtol=2e-2)


def test_vision_rope_2d_local_axis_pairing():
    rng = np.random.default_rng(731)
    B, H, N, D, P = 2, 3, 7, 128, 11
    x = mx.array((0.2 * rng.standard_normal((B, H, N, D))).astype(np.float32)).astype(mx.bfloat16)
    cos = mx.array(rng.uniform(-1, 1, (P, D // 4)).astype(np.float32)).astype(mx.bfloat16)
    sin = mx.array(rng.uniform(-1, 1, (P, D // 4)).astype(np.float32)).astype(mx.bfloat16)
    positions = rng.integers(0, P, (B, N, 2), dtype=np.int32)
    got = vision_rope_2d(x, cos, sin, mx.array(positions))
    mx.eval(got)
    xb = np.array(x.astype(mx.float32)); cb = np.array(cos.astype(mx.float32))
    sb = np.array(sin.astype(mx.float32)); pairs = D // 4
    ref = np.empty_like(xb)
    for b in range(B):
        for n in range(N):
            for axis in range(2):
                base = axis * (D // 2); pos = positions[b, n, axis]
                a = xb[b, :, n, base:base + pairs]
                z = xb[b, :, n, base + pairs:base + 2 * pairs]
                ref[b, :, n, base:base + pairs] = a * cb[pos] - z * sb[pos]
                ref[b, :, n, base + pairs:base + 2 * pairs] = a * sb[pos] + z * cb[pos]
    ref = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=0, rtol=0)


def test_qwen_vision_rope_2d_global_split_pairing():
    rng = np.random.default_rng(732)
    B, H, N, D, P = 2, 3, 7, 128, 11
    x = mx.array((0.2 * rng.standard_normal((B, H, N, D))).astype(np.float32)).astype(mx.bfloat16)
    cos = mx.array(rng.uniform(-1, 1, (P, D // 4)).astype(np.float32)).astype(mx.bfloat16)
    sin = mx.array(rng.uniform(-1, 1, (P, D // 4)).astype(np.float32)).astype(mx.bfloat16)
    positions = rng.integers(0, P, (B, N, 2), dtype=np.int32)
    got = qwen_vision_rope_2d(x, cos, sin, mx.array(positions))
    mx.eval(got)
    xb = np.array(x.astype(mx.float32)); cb = np.array(cos.astype(mx.float32))
    sb = np.array(sin.astype(mx.float32)); pairs = D // 4
    ref = np.empty_like(xb)
    for b in range(B):
        for n in range(N):
            c = np.concatenate((cb[positions[b, n, 0]], cb[positions[b, n, 1]]))
            s = np.concatenate((sb[positions[b, n, 0]], sb[positions[b, n, 1]]))
            a = xb[b, :, n, :2 * pairs]
            z = xb[b, :, n, 2 * pairs:]
            ref[b, :, n, :2 * pairs] = a * c - z * s
            ref[b, :, n, 2 * pairs:] = a * s + z * c
    ref = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=0, rtol=0)


if __name__ == "__main__":
    for shp in SHAPES:
        test_rotary(shp)
        test_rotary_interleaved(shp)
        print("ok", shp)
