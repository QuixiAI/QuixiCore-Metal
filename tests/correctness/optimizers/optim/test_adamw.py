"""Correctness tests for the AdamW optimizer step.

Run from kernels/:  python -m pytest optim/correctness/test_adamw.py -q
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import adamw, adamw_masked

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _np_adamw(p, g, m, v, lr, b1, b2, eps, wd, t):
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    mhat = m / (1.0 - b1 ** t)
    vhat = v / (1.0 - b2 ** t)
    p = p - lr * (mhat / (np.sqrt(vhat) + eps) + wd * p)
    return p, m, v


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_adamw_multistep(dtype, wd):
    rng = np.random.default_rng(hash((dtype, wd)) & 0xFFFF)
    D = 4096
    lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
    p = (0.1 * rng.standard_normal(D)).astype(np.float32)
    m = np.zeros(D, np.float32)
    v = np.zeros(D, np.float32)
    tol = 2e-4 if dtype == "float32" else 3e-2
    for t in range(1, 6):
        g = rng.standard_normal(D).astype(np.float32)
        p_ref, m_ref, v_ref = _np_adamw(p.astype(np.float64), g.astype(np.float64),
                                        m.astype(np.float64), v.astype(np.float64),
                                        lr, b1, b2, eps, wd, t)
        pk, mk, vk = adamw(mx.array(p).astype(_MX[dtype]), mx.array(g), mx.array(m), mx.array(v),
                           lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd, step=t)
        pk = np.array(pk.astype(mx.float32))
        mk, vk = np.array(mk), np.array(vk)
        assert np.allclose(pk, p_ref, atol=tol), (dtype, wd, t, np.abs(pk - p_ref).max())
        assert np.allclose(mk, m_ref, atol=tol)
        assert np.allclose(vk, v_ref, atol=tol)
        # advance the running state from the canonical (fp64) reference
        p, m, v = p_ref.astype(np.float32), m_ref.astype(np.float32), v_ref.astype(np.float32)


def test_adamw_masked_modes():
    rng = np.random.default_rng(122)
    seg = 32
    nseg = 5
    n = seg * nseg
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.05, 3
    p = (0.1 * rng.standard_normal(n)).astype(np.float32)
    g = rng.standard_normal(n).astype(np.float32)
    m = rng.standard_normal(n).astype(np.float32) * 0.01
    v = np.abs(rng.standard_normal(n).astype(np.float32)) * 0.01
    mask = np.array([1, 0, 1, 0, 1], np.uint8)

    pm, mm, vm = adamw_masked(mx.array(p), mx.array(g), mx.array(m), mx.array(v),
                              lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd,
                              step=step, mask=mx.array(mask), seg_size=seg, mask_mode=0)
    pu, mu, vu = _np_adamw(p.astype(np.float64), g.astype(np.float64),
                           m.astype(np.float64), v.astype(np.float64),
                           lr, b1, b2, eps, wd, step)
    pm, mm, vm = np.array(pm), np.array(mm), np.array(vm)
    for s in range(nseg):
        sl = slice(s * seg, (s + 1) * seg)
        if mask[s]:
            np.testing.assert_allclose(pm[sl], pu[sl], atol=2e-4)
            np.testing.assert_allclose(mm[sl], mu[sl], atol=2e-4)
            np.testing.assert_allclose(vm[sl], vu[sl], atol=2e-4)
        else:
            np.testing.assert_array_equal(pm[sl], p[sl])
            np.testing.assert_array_equal(mm[sl], m[sl])
            np.testing.assert_array_equal(vm[sl], v[sl])

    pd, md, vd = adamw_masked(mx.array(p), mx.array(g), mx.array(m), mx.array(v),
                              lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd,
                              step=step, mask=mx.array(mask), seg_size=seg, mask_mode=1)
    pnodecay, mnodecay, vnodecay = _np_adamw(p.astype(np.float64), g.astype(np.float64),
                                             m.astype(np.float64), v.astype(np.float64),
                                             lr, b1, b2, eps, 0.0, step)
    pd, md, vd = np.array(pd), np.array(md), np.array(vd)
    for s in range(nseg):
        sl = slice(s * seg, (s + 1) * seg)
        refp = pu if mask[s] else pnodecay
        np.testing.assert_allclose(pd[sl], refp[sl], atol=2e-4)
        np.testing.assert_allclose(md[sl], mnodecay[sl], atol=2e-4)
        np.testing.assert_allclose(vd[sl], vnodecay[sl], atol=2e-4)


if __name__ == "__main__":
    test_adamw_multistep("float32", 0.01)
    print("ok")
