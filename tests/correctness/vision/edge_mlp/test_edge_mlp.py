import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import edge_mlp_256x7


def _gelu_erf(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


@pytest.mark.parametrize("use_kernel", [None, False, True])
def test_edge_mlp_matches_materialized_pairwise_reference(use_kernel):
    rng = np.random.default_rng(73)
    B, L = 1, 3
    hidden = (0.08 * rng.standard_normal((B, L, 256))).astype(np.float32)
    first_weight = (0.06 * rng.standard_normal((256, 512))).astype(np.float32)
    first_bias = (0.04 * rng.standard_normal(256)).astype(np.float32)
    second_weight = (0.05 * rng.standard_normal((7, 256))).astype(np.float32)
    second_bias = (0.03 * rng.standard_normal(7)).astype(np.float32)
    got = edge_mlp_256x7(*(mx.array(x) for x in (
        hidden, first_weight, first_bias, second_weight, second_bias)), use_kernel=use_kernel)
    mx.eval(got)

    left = hidden @ first_weight[:, :256].T
    right = hidden @ first_weight[:, 256:].T + first_bias
    joined = left[:, :, None, :] + right[:, None, :, :]
    activation = _gelu_erf(joined).astype(np.float32)
    ref = np.einsum("bijh,ch->bcij", activation, second_weight) + second_bias[None, :, None, None]
    np.testing.assert_allclose(np.array(got), ref, rtol=2e-4, atol=2e-4)


def test_edge_mlp_bfloat16_kernel_matches_framework_route():
    rng = np.random.default_rng(79)
    shapes = ((1, 2, 256), (256, 512), (256,), (7, 256), (7,))
    arrays = [
        mx.array((0.05 * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)
        for shape in shapes
    ]
    got = edge_mlp_256x7(*arrays, use_kernel=True)
    ref = edge_mlp_256x7(*arrays, use_kernel=False)
    mx.eval(got, ref)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=6e-2, atol=6e-2)
