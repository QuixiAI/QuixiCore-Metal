"""First-order autograd tests (Wave-7 #10, MLX side).

Each op's mx.vjp gradient must match the direct tk *_backward call. The norm/gelu forwards emit bf16,
so the cotangent flowing into the vjp is bf16 — tolerances reflect that.

Run from kernels/:  python -m pytest tk/tests/test_autograd.py -q
"""

import mlx.core as mx
import numpy as np
import pytest

import tk
from tk import autograd as tka


def _maxdiff(a, b):
    mx.eval(a, b)
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def test_autograd_gelu():
    x = mx.random.normal((8, 256))
    ct = mx.random.normal((8, 256))
    _, (gx,) = mx.vjp(tka.gelu, (x,), (ct,))
    assert _maxdiff(gx, tk.gelu_backward(x, ct)) < 3e-2


@pytest.mark.parametrize("mode", ["reglu", "geglu", "swiglu", "geglu_erf", "geglu_quick"])
def test_autograd_glu(mode):
    x = mx.random.normal((8, 256))
    g = mx.random.normal((8, 256))
    ct = mx.random.normal((8, 256))
    _, (gx, gg) = mx.vjp(lambda a, c: tka.glu(a, c, mode=mode), (x, g), (ct,))
    da, db = tk.glu_backward(x, g, ct, mode=mode)
    assert _maxdiff(gx, da) < 1e-3 and _maxdiff(gg, db) < 1e-3


def test_autograd_rms_norm():
    x = mx.random.normal((8, 256))
    w = mx.random.normal((256,))
    ct = mx.random.normal((8, 256))
    _, (gx, gw) = mx.vjp(lambda a, ww: tka.rms_norm(a, ww), (x, w), (ct,))
    dx, dw = tk.rms_norm_backward(x, w, ct)
    assert _maxdiff(gx, dx) < 3e-2 and _maxdiff(gw, dw) < 3e-2


def test_autograd_layernorm():
    x = mx.random.normal((8, 256))
    w = mx.random.normal((256,))
    b = mx.random.normal((256,))
    ct = mx.random.normal((8, 256))
    _, (gx, gw, gb) = mx.vjp(lambda a, ww, bb: tka.layernorm(a, ww, bb), (x, w, b), (ct,))
    dx, dw, db = tk.layernorm_backward(x, w, ct)
    assert _maxdiff(gx, dx) < 3e-2 and _maxdiff(gw, dw) < 3e-2 and _maxdiff(gb, db) < 3e-2


def test_autograd_dropout():
    x = mx.random.normal((8, 256))
    ct = mx.random.normal((8, 256))
    _, (gx,) = mx.vjp(lambda a: tka.dropout(a, 0.3, 7), (x,), (ct,))
    assert _maxdiff(gx, tk.dropout_backward(ct, 0.3, 7)) < 1e-4


def test_autograd_embedding_lookup():
    rng = np.random.default_rng(0)
    tok = mx.array(rng.integers(0, 50, size=(40,)).astype(np.int32))
    tab = mx.random.normal((50, 64))
    ct = mx.random.normal((40, 64))
    _, (gt,) = mx.vjp(lambda t: tka.embedding_lookup(tok, t, None, 1.5), (tab,), (ct,))
    assert _maxdiff(gt, tk.embedding_backward(tok, ct, 50, scale=1.5)) < 1e-4


def test_autograd_grad_end_to_end():
    # mx.grad of a scalar loss through a differentiable op reaches the input, and equals the manual
    # vjp of the *_backward (fp32-preserving glu keeps this exact; the chain-rule sum collapses to
    # summing the per-element backward with a ones cotangent).
    x = mx.random.normal((4, 256))
    g = mx.random.normal((4, 256))
    gx = mx.grad(lambda x: mx.sum(tka.glu(x, g, mode="swiglu")))(x)
    da, _ = tk.glu_backward(x, g, mx.ones((4, 256)), mode="swiglu")
    mx.eval(gx, da)
    assert gx.shape == x.shape and _maxdiff(gx, da) < 1e-3
