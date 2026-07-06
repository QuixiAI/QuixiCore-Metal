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

from tk import rotary


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


if __name__ == "__main__":
    for shp in SHAPES:
        test_rotary(shp)
        test_rotary_interleaved(shp)
        print("ok", shp)
