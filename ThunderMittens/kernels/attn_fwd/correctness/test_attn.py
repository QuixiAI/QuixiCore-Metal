"""Correctness test for the ThunderMittens attn_fwd Metal kernel.

attn_fwd is a warp-level flash-attention forward (non-causal), bf16, D in {64,128}.
The kernel pre-scales Q by (1/sqrt(D)) * log2(e) and uses exp2, which is exactly a
softmax with scale 1/sqrt(D). Oracle: mx.fast.scaled_dot_product_attention with that
scale (NOT scale=1, which time_attn.py uses for perf only).
Run from the kernels/ directory:

    python -m pytest attn_fwd/correctness/test_attn.py -v
"""

import math

import mlx.core as mx
import pytest

from tk import attn_fwd

# (B, H, N, D); N must be a multiple of 8.
SHAPES = [(1, 2, 256, 64), (2, 4, 512, 64), (1, 2, 256, 128), (2, 2, 128, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_attn_fwd_matches_sdpa(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)

    got = attn_fwd(q, k, v)
    exp = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(D), mask=None
    )
    mx.eval(got, exp)

    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()
    assert mx.allclose(got, exp, atol=4e-2, rtol=4e-2), f"max diff: {diff}"


def _ref_attn(q, k, v, scale, softcap=0.0, sinks=None):
    """fp64 numpy oracle: optional Gemma-2 softcap + gpt-oss sink (denominator-only)."""
    import numpy as np
    qf = np.array(q.astype(mx.float32), dtype=np.float64)
    kf = np.array(k.astype(mx.float32), dtype=np.float64)
    vf = np.array(v.astype(mx.float32), dtype=np.float64)
    s = np.einsum("bhnd,bhmd->bhnm", qf, kf) * scale
    if softcap > 0:
        s = softcap * np.tanh(s / softcap)
    if sinks is not None:
        m = np.maximum(s.max(-1), sinks[None, :, None])   # sink participates in the max
    else:
        m = s.max(-1)
    e = np.exp(s - m[..., None])
    den = e.sum(-1)
    if sinks is not None:
        den = den + np.exp(sinks[None, :, None] - m)
    p = e / den[..., None]
    return np.einsum("bhnm,bhmd->bhnd", p, vf)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 2, 128, 128)])
@pytest.mark.parametrize("softcap,use_sink", [(30.0, False), (0.0, True), (50.0, True)])
def test_attn_fwd_softcap_sinks(shape, softcap, use_sink):
    import numpy as np
    B, H, N, D = shape
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    sinks_np = rng.standard_normal(H).astype("float32") * 2.0 if use_sink else None
    sinks = mx.array(sinks_np) if use_sink else None

    got = attn_fwd(q, k, v, softcap=softcap, sinks=sinks)
    mx.eval(got)
    ref = _ref_attn(q, k, v, 1.0 / math.sqrt(D), softcap, sinks_np)
    diff = np.abs(np.array(got.astype(mx.float32), dtype=np.float64) - ref).max()
    assert diff < 4e-2, f"max diff {diff} (softcap={softcap}, sink={use_sink})"


def test_attn_fwd_flagless_unchanged():
    """Regression guard: softcap=0 / no sinks must match the pre-flag kernel path exactly
    (same kernel, uniform branches off)."""
    B, H, N, D = 2, 4, 512, 64
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    exp = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D), mask=None)
    got = attn_fwd(q, k, v)
    mx.eval(got, exp)
    assert mx.allclose(got, exp, atol=4e-2, rtol=4e-2)


if __name__ == "__main__":
    for shp in SHAPES:
        test_attn_fwd_matches_sdpa(shp)
        print(f"ok {shp}")
