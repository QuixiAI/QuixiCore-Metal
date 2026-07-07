import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import kd_kl_dense_bwd, kd_kl_dense_fwd


def _logsumexp(x):
    m = x.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def _kd_ref(t_logits, s_logits, grad_out, invtemp):
    zt = t_logits * invtemp
    zs = s_logits * invtemp
    lse_t = _logsumexp(zt).astype(np.float32)
    lse_s = _logsumexp(zs).astype(np.float32)
    pt = np.exp(zt - lse_t[:, None])
    q = np.exp(zs - lse_s[:, None])
    loss = (pt * ((zt - lse_t[:, None]) - (zs - lse_s[:, None]))).sum(axis=1)
    grad = (q - pt) * (grad_out[:, None] * invtemp)
    return loss.astype(np.float32), lse_t, lse_s, grad.astype(np.float32)


@pytest.mark.parametrize("shape", [(3, 257), (5, 512)])
def test_kd_kl_dense_fwd_bwd_matches_numpy(shape):
    rng = np.random.default_rng(91 + shape[1])
    t_np = (0.4 * rng.standard_normal(shape)).astype(np.float32)
    s_np = (0.4 * rng.standard_normal(shape)).astype(np.float32)
    go_np = rng.uniform(0.2, 1.3, size=(shape[0],)).astype(np.float32)
    invtemp = 0.7
    t = mx.array(t_np).astype(mx.float32)
    s = mx.array(s_np).astype(mx.float32)
    loss, lse_t, lse_s = kd_kl_dense_fwd(t, s, invtemp)
    grad = kd_kl_dense_bwd(t, s, lse_t, lse_s, mx.array(go_np), invtemp)
    mx.eval(loss, lse_t, lse_s, grad)
    ref_loss, ref_lse_t, ref_lse_s, ref_grad = _kd_ref(t_np, s_np, go_np, invtemp)
    np.testing.assert_allclose(np.array(loss), ref_loss, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(lse_t), ref_lse_t, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(lse_s), ref_lse_s, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.array(grad), ref_grad, rtol=2e-5, atol=2e-5)
