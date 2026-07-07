import mlx.core as mx
import numpy as np
import pytest

from tk import kd_kl_topk_bwd, kd_kl_topk_fwd


def _logsumexp(z):
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def _ref(logits, idx, prob, grad_out, invtemp, tail_mode):
    z = logits * invtemp
    lse = _logsumexp(z)
    q = np.exp(z - lse[:, None])
    T, K = idx.shape
    loss = np.zeros(T, np.float32)
    grad = np.zeros_like(logits, np.float32)
    for t in range(T):
        valid = idx[t] >= 0
        ii = idx[t, valid]
        pp = prob[t, valid]
        P = pp.sum()
        S = q[t, ii].sum()
        if tail_mode == 0:
            pt = pp / max(P, 1e-30)
            loss[t] = np.sum(pt * (np.log(np.maximum(pt, 1e-30)) - np.log(q[t, ii])))
            grad[t] = q[t]
            grad[t, ii] -= pt
        else:
            loss[t] = np.sum(pp * (np.log(np.maximum(pp, 1e-30)) - np.log(q[t, ii])))
            tail = max(1.0 - P, 0.0)
            tail_c = tail / max(1.0 - S, 1e-30) if tail > 0 else 0.0
            if tail > 0:
                loss[t] += tail * (np.log(max(tail, 1e-30)) - np.log(max(1.0 - S, 1e-30)))
            grad[t] = (P - tail_c * S) * q[t]
            grad[t, ii] += -pp + tail_c * q[t, ii]
        grad[t] *= grad_out[t] * invtemp
    return loss, lse.astype(np.float32), grad


@pytest.mark.parametrize("tail_mode", [0, 1])
def test_kd_kl_topk_fwd_bwd_matches_dense_reference(tail_mode):
    rng = np.random.default_rng(120 + tail_mode)
    T, V, K = 4, 96, 9
    logits = (rng.standard_normal((T, V)) * 1.3).astype(np.float32)
    teacher = rng.random((T, V)).astype(np.float64)
    teacher /= teacher.sum(axis=-1, keepdims=True)
    idx = np.argsort(-teacher, axis=-1)[:, :K].astype(np.int32)
    prob = np.take_along_axis(teacher.astype(np.float32), idx, axis=-1)
    idx[1, -2:] = -1
    prob[1, -2:] = 0.0
    grad_out = rng.standard_normal(T).astype(np.float32)
    invtemp = 0.5

    loss, lse = kd_kl_topk_fwd(mx.array(logits), mx.array(idx), mx.array(prob),
                               invtemp=invtemp, tail_mode=tail_mode)
    grad = kd_kl_topk_bwd(mx.array(logits), mx.array(idx), mx.array(prob), lse,
                          mx.array(grad_out), invtemp=invtemp, tail_mode=tail_mode)
    mx.eval(loss, lse, grad)
    ref_loss, ref_lse, ref_grad = _ref(logits, idx, prob, grad_out, invtemp, tail_mode)
    np.testing.assert_allclose(np.array(loss), ref_loss, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(lse), ref_lse, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(grad), ref_grad, rtol=2e-5, atol=2e-5)
