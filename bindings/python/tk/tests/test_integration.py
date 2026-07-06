"""End-to-end integration test: compose the Wave-5/6 serving + training kernels in one step.

Runs the full serving decode chain (embedding -> LM-head logits -> grammar/bad-word/penalty masking
-> min-p / typical-p sample -> beam advance + zero-copy KV remap) and a training step (embedding
backward -> AdamW) on each available backend (MLX always; PyTorch-MPS if importable), asserting
shapes and basic validity. This is a smoke test that the kernels chain together, not a numeric oracle.

Run from kernels/:  python -m pytest tk/tests/test_integration.py -q
"""

import numpy as np
import pytest

import tk


def _backends():
    outs = []
    try:
        import mlx.core as mx  # noqa: F401
        outs.append("mlx")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            outs.append("torch")
    except Exception:  # noqa: BLE001
        pass
    return outs


class _B:
    """Tiny array adapter so one test body drives both backends."""

    def __init__(self, name):
        self.name = name
        if name == "mlx":
            import mlx.core as mx
            self.m = mx
        else:
            import torch
            self.t = torch

    def arr(self, x, dt="f32"):
        if self.name == "mlx":
            d = {"f32": self.m.float32, "bf16": self.m.bfloat16, "i32": self.m.int32}[dt]
            return self.m.array(x).astype(d)
        d = {"f32": self.t.float32, "bf16": self.t.bfloat16, "i32": self.t.int32}[dt]
        return self.t.from_numpy(np.ascontiguousarray(x)).to(d).to("mps")

    def np(self, x):
        if self.name == "mlx":
            return np.array(x.astype(self.m.float32)) if x.dtype != self.m.int32 else np.array(x)
        return x.detach().to("cpu").numpy()

    def matmul_T(self, a, b):
        return self.m.matmul(a, self.m.swapaxes(b, -1, -2)) if self.name == "mlx" else a @ b.transpose(-1, -2)


@pytest.mark.parametrize("backend", _backends())
def test_serving_decode_step(backend):
    be = _B(backend)
    rng = np.random.default_rng(0)
    T, D, V = 8, 512, 2000

    # 1. token embedding lookup: (T,) ids -> (T, D)
    ids = rng.integers(0, V, size=T).astype(np.int32)
    table = (0.05 * rng.standard_normal((V, D))).astype(np.float32)
    h = tk.embedding_lookup(be.arr(ids, "i32"), be.arr(table, "bf16"))
    assert tuple(h.shape) == (T, D)

    # 2. LM-head logits = h @ W.T  (matmul reads W once; the bandwidth-optimal path)
    W = (0.05 * rng.standard_normal((V, D))).astype(np.float32)
    logits = be.matmul_T(h, be.arr(W, "bf16"))          # (T, V), h reused from the embedding step
    logits = logits.astype(be.m.float32) if backend == "mlx" else logits.float()
    assert tuple(logits.shape) == (T, V)

    # 3. grammar bitmask (allow ~half) then a bad/stop-word deny-list
    allow = rng.integers(0, 2, size=(T, V)).astype(np.uint32)
    allow[:, 0] = 1
    nwords = (V + 31) // 32
    ap = np.zeros((T, nwords * 32), np.uint32); ap[:, :V] = allow
    ap = ap.reshape(T, nwords, 32)
    packed = np.zeros((T, nwords), np.uint32)
    for j in range(32):
        packed |= ap[:, :, j] << np.uint32(j)
    logits = tk.apply_token_bitmask(logits, be.arr(packed.view(np.int32).astype(np.int32), "i32"))
    bad_ids = rng.integers(0, V, size=(T, 4)).astype(np.int32)
    bad_lens = rng.integers(0, 5, size=T).astype(np.int32)
    logits = tk.apply_bad_words(logits, be.arr(bad_ids, "i32"), be.arr(bad_lens, "i32"))

    # 4. sample: min-p and typical-p both return valid token ids per row
    for tok in (tk.min_p_sample(logits, 0.02, seed=1), tk.typical_p_sample(logits, 0.9, seed=2)):
        t = be.np(tok).reshape(-1)
        assert t.shape == (T,)
        assert np.all((t >= 0) & (t < V))
        # sampled token must be allowed by the grammar mask (masking composes before the sampler)
        for r in range(T):
            assert allow[r, int(t[r])] == 1

    # 5. beam advance + zero-copy KV block-table remap (serving beam search)
    BM = 4
    beam_logits = be.arr((rng.standard_normal((2 * BM, V))).astype(np.float32), "bf16")
    cum = be.arr(rng.standard_normal((2, BM)).astype(np.float32), "f32")
    nt, parent, ncum = tk.beam_advance(beam_logits, cum, BM)
    assert tuple(nt.shape) == (2, BM) and tuple(parent.shape) == (2, BM)
    bt = np.arange(2 * BM * 16, dtype=np.int32).reshape(2 * BM, 16)
    new_bt = tk.beam_remap_block_table(be.arr(bt, "i32"), parent)
    assert tuple(new_bt.shape) == (2 * BM, 16)


@pytest.mark.parametrize("backend", _backends())
def test_training_step(backend):
    be = _B(backend)
    rng = np.random.default_rng(1)
    T, D, V = 16, 256, 500

    # embedding backward: scatter-add grad by token id -> (V, D) fp32 table
    ids = rng.integers(0, V, size=T).astype(np.int32)
    dY = (0.1 * rng.standard_normal((T, D))).astype(np.float32)
    dtable = tk.embedding_backward(be.arr(ids, "i32"), be.arr(dY, "bf16"), V, scale=1.0)
    assert tuple(dtable.shape) == (V, D)

    # AdamW step over that gradient table
    p = (0.02 * rng.standard_normal((V, D))).astype(np.float32)
    m = np.zeros((V, D), np.float32); v = np.zeros((V, D), np.float32)
    p2, m2, v2 = tk.adamw(be.arr(p, "f32"), dtable, be.arr(m, "f32"), be.arr(v, "f32"),
                          lr=1e-3, weight_decay=0.01, step=1)
    assert tuple(p2.shape) == (V, D)
    # a real update moved the params where the gradient was nonzero
    assert np.abs(be.np(p2) - p).max() > 0
