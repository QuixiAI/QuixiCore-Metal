"""Correctness tests for sliding-window causal attention (attn_window).

Oracle: numpy softmax over the banded mask j in [max(0, i-W+1), i]. Also asserts
W >= N is BIT-IDENTICAL to attn_causal (the band never cuts a tile), which
transitively validates the new make_windowed substrate helper's lane mapping.
Run from kernels/:  python -m pytest attn_causal/correctness/test_attn_window.py -v
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import attn_causal, attn_window

SHAPES = [(1, 2, 128, 64), (2, 2, 64, 128), (1, 1, 256, 64)]
# non-multiples of 8 exercise make_windowed's per-element band; W < 8 double-masks the
# diagonal tile; W >= N must equal attn_causal exactly.
WINDOWS = [1, 5, 8, 13, 100]


def _oracle(q, k, v, window):
    B, H, N, D = q.shape
    s = (q.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)) / math.sqrt(D)
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    mask = (j <= i) & (j >= i - window + 1) if window > 0 else (j <= i)
    s = np.where(mask, s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return p @ v.astype(np.float64)


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("shape", SHAPES)
def test_attn_window(shape, window):
    B, H, N, D = shape
    rng = np.random.default_rng(0)
    q = (0.5 * rng.standard_normal(shape)).astype(np.float32)
    k = (0.5 * rng.standard_normal(shape)).astype(np.float32)
    v = (0.5 * rng.standard_normal(shape)).astype(np.float32)
    qd = mx.array(q).astype(mx.bfloat16)
    kd = mx.array(k).astype(mx.bfloat16)
    vd = mx.array(v).astype(mx.bfloat16)
    got = np.array(attn_window(qd, kd, vd, window).astype(mx.float32))
    ref = _oracle(np.array(qd.astype(mx.float32)), np.array(kd.astype(mx.float32)),
                  np.array(vd.astype(mx.float32)), window)
    diff = np.abs(got - ref).max()
    scale = np.abs(ref).max() + 1e-9
    assert diff / scale < 0.04, f"W={window} relative diff {diff/scale}"


@pytest.mark.parametrize("shape", SHAPES)
def test_attn_window_full_equals_causal(shape):
    """W >= N: the band never bites — output must be bit-identical to attn_causal."""
    B, H, N, D = shape
    rng = np.random.default_rng(1)
    q = mx.array((0.5 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array((0.5 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array((0.5 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
    w = np.array(attn_window(q, k, v, N + 5).astype(mx.float32))
    c = np.array(attn_causal(q, k, v).astype(mx.float32))
    assert np.array_equal(w, c), "window >= N must match attn_causal exactly"
    d = np.array(attn_window(q, k, v, 0).astype(mx.float32))
    assert np.array_equal(d, c), "window=0 (disabled) must match attn_causal exactly"


if __name__ == "__main__":
    for shp in SHAPES:
        for w in WINDOWS:
            test_attn_window(shp, w)
        test_attn_window_full_equals_causal(shp)
        print("ok", shp)


def _ref_masked_attn(q, k, v, scale, window=0, softcap=0.0, sinks=None):
    """fp64 numpy oracle: causal (+optional sliding window) with optional Gemma-2 softcap
    and gpt-oss denominator-only sinks."""
    import numpy as np
    qf = np.array(q.astype(mx.float32), dtype=np.float64)
    kf = np.array(k.astype(mx.float32), dtype=np.float64)
    vf = np.array(v.astype(mx.float32), dtype=np.float64)
    B, H, N, D = qf.shape
    s = np.einsum("bhnd,bhmd->bhnm", qf, kf) * scale
    if softcap > 0:
        s = softcap * np.tanh(s / softcap)
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    mask = j > i
    if window > 0:
        mask |= j < (i - window + 1)
    s = np.where(mask[None, None], -np.inf, s)
    if sinks is not None:
        m = np.maximum(s.max(-1), sinks[None, :, None])
    else:
        m = s.max(-1)
    e = np.exp(s - m[..., None])
    den = e.sum(-1)
    if sinks is not None:
        den = den + np.exp(sinks[None, :, None] - m)
    p = e / den[..., None]
    return np.einsum("bhnm,bhmd->bhnd", p, vf)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 2, 128, 128)])
@pytest.mark.parametrize("window,softcap,use_sink", [
    (0, 30.0, False),    # Gemma-2 global layer: causal + softcap (via attn_causal)
    (64, 50.0, False),   # Gemma-2 local layer: window + softcap
    (64, 0.0, True),     # gpt-oss local layer: window + sink
    (0, 0.0, True),      # gpt-oss global layer: causal + sink
    (64, 30.0, True),    # everything at once
])
def test_attn_softcap_sinks(shape, window, softcap, use_sink):
    import numpy as np
    from tk import attn_causal
    B, H, N, D = shape
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((B, H, N, D)).astype("float32")).astype(mx.bfloat16)
    sinks_np = (rng.standard_normal(H) * 2.0).astype("float32") if use_sink else None
    sinks = mx.array(sinks_np) if use_sink else None

    if window > 0:
        got = attn_window(q, k, v, window, softcap=softcap, sinks=sinks)
    else:
        got = attn_causal(q, k, v, softcap=softcap, sinks=sinks)
    mx.eval(got)
    ref = _ref_masked_attn(q, k, v, 1.0 / math.sqrt(D), window, softcap, sinks_np)
    diff = np.abs(np.array(got.astype(mx.float32), dtype=np.float64) - ref).max()
    assert diff < 4e-2, f"max diff {diff} (w={window}, cap={softcap}, sink={use_sink})"
