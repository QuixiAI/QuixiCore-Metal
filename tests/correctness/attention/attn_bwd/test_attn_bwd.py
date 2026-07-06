"""FlashAttention-2 backward (dQ/dK/dV), validated against PyTorch autograd.

Also checks attn_fwd_l's L = log2-domain logsumexp vs numpy.
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import attn_fwd_l, attn_bwd


def _ref_grads(q, k, v, do, causal):
    torch = pytest.importorskip("torch")
    qt = torch.tensor(q, dtype=torch.float32, requires_grad=True)
    kt = torch.tensor(k, dtype=torch.float32, requires_grad=True)
    vt = torch.tensor(v, dtype=torch.float32, requires_grad=True)
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = (qt @ kt.transpose(-1, -2)) * scale
    if causal:
        N = q.shape[2]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = p @ vt
    o.backward(torch.tensor(do, dtype=torch.float32))
    return o.detach().numpy(), qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_bwd(D, causal):
    B, H, N = 1, 2, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    do = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)

    o, L = attn_fwd_l(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                      mx.array(v).astype(mx.bfloat16), causal=causal)
    mx.eval(o, L)
    dq, dk, dv = attn_bwd(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                          mx.array(v).astype(mx.bfloat16), o, mx.array(do).astype(mx.bfloat16),
                          L, causal=causal)
    mx.eval(dq, dk, dv)

    o_ref, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do, causal)

    # forward O matches (sanity that L/o are consistent)
    assert np.abs(np.array(o.astype(mx.float32)) - o_ref).max() / (np.abs(o_ref).max() + 1e-9) < 0.05
    for name, got, ref in [("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)]:
        g = np.array(got.astype(mx.float32))
        rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
        assert rel < 0.06, f"{name} D{D} causal={causal} rel {rel}"


def test_attn_fwd_l_logsumexp():
    B, H, N, D = 1, 2, 64, 64
    rng = np.random.default_rng(1)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    _, L = attn_fwd_l(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                      mx.array(v).astype(mx.bfloat16))
    mx.eval(L)
    g = np.array(L.astype(mx.float32))
    scale = 1.0 / np.sqrt(D)
    s2 = (q @ np.swapaxes(k, -1, -2)) * scale * 1.44269504089            # log2-domain scores
    ref = s2.max(-1) + np.log2(np.exp2(s2 - s2.max(-1, keepdims=True)).sum(-1))
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-6) < 0.02


if __name__ == "__main__":
    test_attn_bwd(64, False); test_attn_bwd(64, True); test_attn_fwd_l_logsumexp()
    print("ok")
