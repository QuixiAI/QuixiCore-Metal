"""Correctness tests for the AdamW optimizer step.

Run from kernels/:  python -m pytest optim/correctness/test_adamw.py -q
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import adamw

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


if __name__ == "__main__":
    test_adamw_multistep("float32", 0.01)
    print("ok")
