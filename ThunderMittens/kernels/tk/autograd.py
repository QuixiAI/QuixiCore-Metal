"""First-order autograd wrappers (Wave-7 #10) — opt-in differentiable variants of the tk forward ops.

Both backends, same unified dispatch as the plain routers:
  * MLX: an ``mx.custom_function`` registers a vjp that calls the tk backward kernel, so
    ``mx.grad`` / ``mx.vjp`` flow through the op.
  * torch: a ``torch.autograd.Function`` whose ``backward`` calls the tk backward, so
    ``loss.backward()`` populates ``.grad``.

The plain ``tk.<op>`` forward routers stay non-differentiable (no tracing overhead). Import these
when you want gradients::

    from tk import autograd as tka
    y = tka.gelu(x)          # mx.grad / .backward() now flow through

Scope is first order (no second-order / CPU eval) by design — the tk kernels have no CPU path.
"""

import tk as _tk


# --------------------------------------------------------------------------- MLX (custom_function)
# NOTE: mx.custom_function passes a mis-shaped `primals` to the vjp when the wrapped forward changes
# dtype (e.g. tk.gelu/rms_norm/layernorm cast fp32->bf16). So every vjp below CLOSES OVER the original
# input arrays (which have the correct shape/dtype) and ignores the `primals` argument.
def _mlx_diff(fwd, vjp, *inputs):
    """Build a one-shot mx.custom_function with the given vjp and apply it to `inputs`."""
    import mlx.core as mx
    f = mx.custom_function(fwd)
    f.vjp(vjp)
    return f(*inputs)


# --------------------------------------------------------------------------- torch (autograd.Function)
_TORCH = {}


def _torch_fns():
    if _TORCH:
        return _TORCH
    import torch

    class Gelu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return _tk.gelu(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return _tk.gelu_backward(x, g.contiguous())

    class Glu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gate, mode, alpha, limit):
            ctx.save_for_backward(x, gate)
            ctx.p = (mode, alpha, limit)
            return _tk.glu(x, gate, mode=mode, alpha=alpha, limit=limit)

        @staticmethod
        def backward(ctx, g):
            x, gate = ctx.saved_tensors
            mode, alpha, limit = ctx.p
            da, db = _tk.glu_backward(x, gate, g.contiguous(), mode=mode, alpha=alpha, limit=limit)
            return da, db, None, None, None

    class RmsNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return _tk.rms_norm(x, weight, eps)

        @staticmethod
        def backward(ctx, g):
            x, weight = ctx.saved_tensors
            dx, dw = _tk.rms_norm_backward(x, weight, g.contiguous(), eps=ctx.eps)
            return dx, dw, None

    class LayerNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return _tk.layernorm(x, weight, bias, eps)

        @staticmethod
        def backward(ctx, g):
            x, weight = ctx.saved_tensors
            dx, dw, db = _tk.layernorm_backward(x, weight, g.contiguous(), eps=ctx.eps)
            return dx, dw, db, None

    class EmbeddingLookup(torch.autograd.Function):
        @staticmethod
        def forward(ctx, table, token_ids, pos_table, scale):
            ctx.save_for_backward(token_ids)
            ctx.vocab = int(table.shape[0])
            ctx.scale = float(scale)
            return _tk.embedding_lookup(token_ids, table, pos_table, float(scale))

        @staticmethod
        def backward(ctx, g):
            (token_ids,) = ctx.saved_tensors
            dtable = _tk.embedding_backward(token_ids, g.contiguous(), ctx.vocab, scale=ctx.scale)
            return dtable, None, None, None

    class Dropout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, p, seed):
            ctx.p, ctx.seed = float(p), int(seed)
            return _tk.dropout(x, float(p), int(seed))

        @staticmethod
        def backward(ctx, g):
            return _tk.dropout_backward(g.contiguous(), ctx.p, ctx.seed), None, None

    _TORCH.update(gelu=Gelu, glu=Glu, rms_norm=RmsNorm, layernorm=LayerNorm,
                  embedding_lookup=EmbeddingLookup, dropout=Dropout)
    return _TORCH


# --------------------------------------------------------------------------- public differentiable ops
def gelu(x):
    """Differentiable GELU (tanh approx). mx.grad / .backward() flow through via gelu_backward."""
    if _tk._is_torch(x):
        return _torch_fns()["gelu"].apply(x)
    return _mlx_diff(lambda x: _tk.gelu(x),
                     lambda primals, cot, out: (_tk.gelu_backward(x, cot),), x)


def glu(x, gate, mode="swiglu", alpha=1.0, limit=1.0e20):
    """Differentiable GLU-family activation (grad wrt x and gate via glu_backward)."""
    if _tk._is_torch(x):
        return _torch_fns()["glu"].apply(x, gate, mode, alpha, limit)

    def vjp(primals, cot, out):
        return _tk.glu_backward(x, gate, cot, mode=mode, alpha=alpha, limit=limit)

    return _mlx_diff(lambda x, gate: _tk.glu(x, gate, mode=mode, alpha=alpha, limit=limit),
                     vjp, x, gate)


def rms_norm(x, weight, eps=1e-5):
    """Differentiable RMSNorm (grad wrt x and weight via rms_norm_backward)."""
    if _tk._is_torch(x):
        return _torch_fns()["rms_norm"].apply(x, weight, eps)

    def vjp(primals, cot, out):
        return _tk.rms_norm_backward(x, weight, cot, eps=eps)   # (dx, dw)

    return _mlx_diff(lambda x, weight: _tk.rms_norm(x, weight, eps), vjp, x, weight)


def layernorm(x, weight, bias, eps=1e-5):
    """Differentiable LayerNorm (grad wrt x, weight, bias via layernorm_backward)."""
    if _tk._is_torch(x):
        return _torch_fns()["layernorm"].apply(x, weight, bias, eps)

    def vjp(primals, cot, out):
        return _tk.layernorm_backward(x, weight, cot, eps=eps)  # (dx, dw, db)

    return _mlx_diff(lambda x, weight, bias: _tk.layernorm(x, weight, bias, eps), vjp,
                     x, weight, bias)


def embedding_lookup(token_ids, table, pos_table=None, scale=1.0):
    """Differentiable embedding lookup (grad wrt `table` via embedding_backward; ids are integer)."""
    vocab = int(table.shape[0])
    if _tk._is_torch(table):
        import torch
        pt = pos_table if pos_table is not None else torch.zeros(1, dtype=table.dtype,
                                                                 device=table.device)
        return _torch_fns()["embedding_lookup"].apply(table, token_ids, pt, float(scale))

    def vjp(primals, cot, out):
        dtable = _tk.embedding_backward(token_ids, cot, vocab, scale=float(scale))
        return (dtable.astype(primals[0].dtype),)

    return _mlx_diff(lambda table: _tk.embedding_lookup(token_ids, table, pos_table, float(scale)),
                     vjp, table)


def dropout(x, p, seed):
    """Differentiable dropout (grad wrt x recomputes the same seed/p keep-mask in dropout_backward)."""
    if _tk._is_torch(x):
        return _torch_fns()["dropout"].apply(x, p, seed)
    return _mlx_diff(lambda x: _tk.dropout(x, float(p), int(seed)),
                     lambda primals, cot, out: (_tk.dropout_backward(cot, float(p), int(seed)),), x)
