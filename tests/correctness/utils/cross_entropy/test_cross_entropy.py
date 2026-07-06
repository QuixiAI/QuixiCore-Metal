"""Correctness tests for the fused cross-entropy kernels.

tk.cross_entropy / cross_entropy_grad compute per-row loss + lse and grad_logits over the vocab
axis without materializing the (T, V) probabilities. Oracle: torch.nn.functional.cross_entropy
(reduction='none') + autograd; z-loss checked against a manual lse^2 term.

Run from kernels/:  python -m pytest cross_entropy/correctness/test_cross_entropy.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import tk  # noqa: E402

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _mk(T, V, seed, n_ignore=0, ignore_index=-100):
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((T, V)) * 2.0).astype(np.float32)
    tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
    if n_ignore:
        tgt[:n_ignore] = ignore_index
    return logits, tgt


@pytest.mark.parametrize("dtype,tol", [("float32", 3e-3), ("bfloat16", 6e-2)])
@pytest.mark.parametrize("T,V", [(8, 1000), (16, 32768), (8, 999), (4, 128256)])
def test_forward(dtype, tol, T, V):
    logits, tgt = _mk(T, V, seed=T + V)
    lm = mx.array(logits).astype(_MX[dtype])
    loss, lse = tk.cross_entropy(lm, mx.array(tgt), reduction="none", return_lse=True)
    mx.eval(loss, lse)
    ref = F.cross_entropy(torch.tensor(np.array(lm.astype(mx.float32))),
                          torch.tensor(tgt.astype(np.int64)), reduction="none").numpy()
    assert np.abs(np.array(loss) - ref).max() < tol
    # lse cross-check
    lse_ref = torch.logsumexp(torch.tensor(np.array(lm.astype(mx.float32))), dim=1).numpy()
    assert np.abs(np.array(lse) - lse_ref).max() < (1e-2 if dtype == "float32" else 0.3)


@pytest.mark.parametrize("dtype,tol", [("float32", 3e-3), ("bfloat16", 6e-2)])
def test_ignore_index(dtype, tol):
    T, V = 8, 2000
    logits, tgt = _mk(T, V, seed=1, n_ignore=3)
    lm = mx.array(logits).astype(_MX[dtype])
    loss = tk.cross_entropy(lm, mx.array(tgt), reduction="none")
    mx.eval(loss)
    ln = np.array(loss)
    assert np.all(ln[:3] == 0.0)   # ignored rows -> 0
    ref = F.cross_entropy(torch.tensor(np.array(lm.astype(mx.float32))),
                          torch.tensor(tgt.astype(np.int64)), ignore_index=-100,
                          reduction="none").numpy()
    assert np.abs(ln - ref).max() < tol


@pytest.mark.parametrize("eps", [0.1, 0.2])
def test_label_smoothing(eps):
    T, V = 8, 1000
    logits, tgt = _mk(T, V, seed=2)
    lm = mx.array(logits)
    loss = tk.cross_entropy(lm, mx.array(tgt), reduction="none", label_smoothing=eps)
    mx.eval(loss)
    ref = F.cross_entropy(torch.tensor(logits), torch.tensor(tgt.astype(np.int64)),
                          label_smoothing=eps, reduction="none").numpy()
    assert np.abs(np.array(loss) - ref).max() < 3e-3


def test_z_loss():
    T, V, z = 8, 1000, 1e-4
    logits, tgt = _mk(T, V, seed=3)
    lm = mx.array(logits)
    loss = tk.cross_entropy(lm, mx.array(tgt), reduction="none", z_loss=z)
    mx.eval(loss)
    lt = torch.tensor(logits)
    base = F.cross_entropy(lt, torch.tensor(tgt.astype(np.int64)), reduction="none").numpy()
    lse = torch.logsumexp(lt, dim=1).numpy()
    ref = base + z * lse ** 2
    assert np.abs(np.array(loss) - ref).max() < 3e-3


@pytest.mark.parametrize("dtype,tol", [("float32", 3e-3), ("bfloat16", 6e-2)])
@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_backward(dtype, tol, eps):
    T, V = 8, 4000
    logits, tgt = _mk(T, V, seed=4, n_ignore=2)
    lm = mx.array(logits).astype(_MX[dtype])
    _, lse = tk.cross_entropy(lm, mx.array(tgt), reduction="none", label_smoothing=eps,
                              return_lse=True)
    n = max(int((tgt != -100).sum()), 1)
    g = tk.cross_entropy_grad(lm, mx.array(tgt), lse, mx.full((T,), 1.0 / n, dtype=mx.float32),
                              label_smoothing=eps)
    mx.eval(g)
    lt = torch.tensor(np.array(lm.astype(mx.float32)), requires_grad=True)
    F.cross_entropy(lt, torch.tensor(tgt.astype(np.int64)), ignore_index=-100,
                    label_smoothing=eps, reduction="mean").backward()
    assert np.abs(np.array(g.astype(mx.float32)) - lt.grad.numpy()).max() < tol


@pytest.mark.parametrize("dtype,tol", [("float32", 3e-3), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("softcap", [30.0, 50.0])
def test_softcap(dtype, tol, softcap):
    # Gemma-2 logit softcap: z -> softcap*tanh(z/softcap) before the softmax; grad flows through tanh.
    T, V = 8, 4000
    logits, tgt = _mk(T, V, seed=6, n_ignore=2)
    logits = logits * 3.0   # widen so the cap actually bites
    lm = mx.array(logits).astype(_MX[dtype])
    loss, lse = tk.cross_entropy(lm, mx.array(tgt), reduction="none", softcap=softcap,
                                 return_lse=True)
    mx.eval(loss, lse)
    lt = torch.tensor(np.array(lm.astype(mx.float32)), requires_grad=True)
    z = softcap * torch.tanh(lt / softcap)
    ref = F.cross_entropy(z, torch.tensor(tgt.astype(np.int64)), ignore_index=-100,
                          reduction="none")
    assert np.abs(np.array(loss) - ref.detach().numpy()).max() < tol
    n = max(int((tgt != -100).sum()), 1)
    g = tk.cross_entropy_grad(lm, mx.array(tgt), lse, mx.full((T,), 1.0 / n, dtype=mx.float32),
                              softcap=softcap)
    mx.eval(g)
    F.cross_entropy(z, torch.tensor(tgt.astype(np.int64)), ignore_index=-100,
                    reduction="mean").backward()
    assert np.abs(np.array(g.astype(mx.float32)) - lt.grad.numpy()).max() < tol


def test_fused_linear_cross_entropy_softcap():
    T, K, V, softcap = 40, 256, 2000, 30.0
    rng = np.random.default_rng(7)
    h = (0.2 * rng.standard_normal((T, K))).astype(np.float32)
    W = (0.2 * rng.standard_normal((V, K))).astype(np.float32)
    tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
    loss, dh, dW = tk.fused_linear_cross_entropy(mx.array(h), mx.array(W), mx.array(tgt),
                                                 chunk_size=16, softcap=softcap)
    mx.eval(loss, dh, dW)
    ht = torch.tensor(h, requires_grad=True)
    Wt = torch.tensor(W, requires_grad=True)
    z = softcap * torch.tanh((ht @ Wt.T) / softcap)
    L = F.cross_entropy(z, torch.tensor(tgt.astype(np.int64)), reduction="mean")
    L.backward()
    assert abs(float(loss) - float(L)) < 1e-3
    assert np.abs(np.array(dh) - ht.grad.numpy()).max() < 1e-3
    assert np.abs(np.array(dW) - Wt.grad.numpy()).max() < 1e-3


@pytest.mark.parametrize("chunk", [8, 16, 64])
def test_fused_linear_cross_entropy(chunk):
    T, K, V = 40, 256, 2000
    rng = np.random.default_rng(5)
    h = (0.1 * rng.standard_normal((T, K))).astype(np.float32)
    W = (0.1 * rng.standard_normal((V, K))).astype(np.float32)
    tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
    loss, dh, dW = tk.fused_linear_cross_entropy(mx.array(h), mx.array(W), mx.array(tgt),
                                                 chunk_size=chunk)
    mx.eval(loss, dh, dW)
    ht = torch.tensor(h, requires_grad=True)
    Wt = torch.tensor(W, requires_grad=True)
    L = F.cross_entropy(ht @ Wt.T, torch.tensor(tgt.astype(np.int64)), reduction="mean")
    L.backward()
    assert abs(float(loss) - float(L)) < 1e-3
    assert np.abs(np.array(dh) - ht.grad.numpy()).max() < 1e-3
    assert np.abs(np.array(dW) - Wt.grad.numpy()).max() < 1e-3


if __name__ == "__main__":
    test_forward("float32", 3e-3, 8, 1000)
    test_backward("float32", 3e-3, 0.0)
    test_fused_linear_cross_entropy(16)
    print("ok")
