# Copyright © 2023 Apple Inc.
"""ThunderMittens kernels — unified Python API.

`tk.<kernel>(x, ...)` auto-routes by the input type:
  - mlx.core.array   -> the MLX backend (tk._ext, built via setup.py build_ext)
  - torch.Tensor     -> the PyTorch MPS backend (tk_torch)

Backends are imported lazily, so you only need the framework whose tensors you pass
(e.g. a PyTorch-only user never triggers the MLX import).
"""

# --- lazy backend loaders ---
_mlx_ext = None
_torch_backend = None


def _mlx():
    global _mlx_ext
    if _mlx_ext is None:
        from . import _ext as e  # compiled MLX extension
        _mlx_ext = e
    return _mlx_ext


def _torch():
    global _torch_backend
    if _torch_backend is None:
        import tk_torch  # standalone PyTorch MPS backend
        _torch_backend = tk_torch
    return _torch_backend


def _is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


# --- dispatching kernels ---
def layernorm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().layernorm(x, weight, bias, eps)
    return _mlx().layernorm(x, weight, bias, eps=eps)


def layernorm_backward(x, weight, dy, eps=1e-5):
    """LayerNorm backward. Returns (dx, dweight, dbias): dx has x's shape (fused dX kernel),
    dweight/dbias (D,) summed over rows. x/dy (..., D), weight (D,). Matches torch autograd. mean,
    rstd, dweight, dbias are framework reductions; the per-row dX is the kernel. mlx / torch (MPS)."""
    D = x.shape[-1]
    if _is_torch(x):
        import torch
        xf = x.reshape(-1, D).contiguous().float()
        dyf = dy.reshape(-1, D).contiguous()
        mean = xf.mean(-1, keepdim=True)
        rstd = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + eps)   # (rows,1)
        xhat = (xf - mean) * rstd
        dx = _torch().layernorm_bwd_dx(x.reshape(-1, D).contiguous(), weight, dyf,
                                       mean.squeeze(-1).contiguous(), rstd.squeeze(-1).contiguous())
        dw = (dyf.float() * xhat).sum(0).to(weight.dtype)
        db = dyf.float().sum(0).to(weight.dtype)
        return dx.reshape(x.shape), dw, db
    import mlx.core as mx
    xf = mx.reshape(x, (-1, D)).astype(mx.float32)
    dyf = mx.reshape(dy, (-1, D))
    mean = mx.mean(xf, axis=-1, keepdims=True)
    rstd = mx.rsqrt(mx.mean((xf - mean) ** 2, axis=-1, keepdims=True) + eps)  # (rows,1)
    xhat = (xf - mean) * rstd
    dx = _mlx().layernorm_bwd_dx(mx.reshape(x, (-1, D)), weight, dyf, mx.reshape(mean, (-1,)),
                                 mx.reshape(rstd, (-1,)))
    dw = mx.sum(dyf.astype(mx.float32) * xhat, axis=0).astype(weight.dtype)
    db = mx.sum(dyf.astype(mx.float32), axis=0).astype(weight.dtype)
    return mx.reshape(dx, x.shape), dw, db


def add_rt(x, y):
    """Elementwise x + y. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().add_rt(x, y)
    return _mlx().add_rt(x, y)


def _ceil(a, m):
    return ((a + m - 1) // m) * m


def _scale_vec(scale, num, ref):
    """Broadcast a python scalar to a (num,) float32 scale array on ref's backend.

    Per-head callers pass a length-`num` array (returned as-is); per-tensor callers pass a
    plain float, which is broadcast into every head slot.
    """
    if isinstance(scale, (int, float)):
        if _is_torch(ref):
            import torch
            return torch.full((num,), float(scale), dtype=torch.float32, device=ref.device)
        import mlx.core as mx
        return mx.full((num,), float(scale), dtype=mx.float32)
    return scale


def matmul_custom(x, y):
    """(N,K) @ (K,M) GEMM, arbitrary shapes. Accepts mlx.array or torch.Tensor (MPS).

    The kernel is tile-blocked (needs N%32, M%32, K%16); arbitrary shapes are handled by
    zero-padding to the next tile multiple and slicing the result (shared-tile staging /
    a truly general kernel is a perf follow-up)."""
    if _is_torch(x):
        return _torch().matmul_custom(x, y)  # tk_torch pads/slices
    import mlx.core as mx

    N, K = x.shape[-2], x.shape[-1]
    M = y.shape[-1]
    Np, Kp, Mp = _ceil(N, 32), _ceil(K, 16), _ceil(M, 32)
    xp = mx.pad(x, [(0, Np - N), (0, Kp - K)]) if (Np != N or Kp != K) else x
    yp = mx.pad(y, [(0, Kp - K), (0, Mp - M)]) if (Kp != K or Mp != M) else y
    out = _mlx().matmul_custom(xp, yp)
    return out[:N, :M]


def attn_fwd(q, k, v):
    """Non-causal attention forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_fwd(q, k, v)
    return _mlx().attn_fwd(q, k, v)


def _varlen_worklist(cu_seqlens_q):
    """Host-side (numpy) prefill tile plan. Returns qlens, padded qlens, pad offsets, and the
    per-tile (batch, local-row-0) worklist — all derived from the host cu_seqlens ints."""
    import numpy as np
    cu = np.asarray(cu_seqlens_q, dtype=np.int64)
    B = len(cu) - 1
    qlens = np.diff(cu).astype(np.int32)
    padded = (((qlens + 7) // 8) * 8).astype(np.int32)
    pad_off = np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)
    tile_seq, tile_local0 = [], []
    for b in range(B):
        for t in range(int(padded[b]) // 8):
            tile_seq.append(b)
            tile_local0.append(t * 8)
    return (qlens, padded, pad_off,
            np.asarray(tile_seq, np.int32), np.asarray(tile_local0, np.int32))


def varlen_build_worklist(cu_seqlens, max_tiles):
    """On-device version of the varlen prefill scheduler: builds the tile worklist from a DEVICE
    cu_seqlens (B+1,) int with no host loop. Returns (qlens (B,), pad_off (B+1,), tile_seq
    (max_tiles,), tile_local0 (max_tiles,), n_tiles (1,)); tile_seq is -1 past n_tiles. max_tiles is
    a host upper bound on sum(ceil(qlen/8)) — Metal cannot size a grid from device data, so the
    caller provides the bound and reads n_tiles / pad_off[-1] to drive the downstream dispatch.
    B <= 256 (single-threadgroup scan). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(cu_seqlens):
        return tuple(_torch().varlen_build_worklist(cu_seqlens, int(max_tiles)))
    out = _mlx().varlen_build_worklist(cu_seqlens, int(max_tiles))
    return out[0], out[1], out[2], out[3], out[4]


def attn_varlen_prefill(q_packed, key_cache, value_cache, block_table, context_lens,
                        cu_seqlens_q, scale=0.0):
    """Varlen / paged-prefill causal attention: ragged packed queries reading K/V from the paged
    cache, no dense (B,H,N,D) materialization. Supports a cached prefix (context_len >= q_len),
    GQA, and D in {64,128}.

    q_packed (total_q, H, D) bf16 with sequences concatenated per cu_seqlens_q (a host int
    sequence of length B+1). key_cache/value_cache (num_blocks, block_size, H_KV, D) bf16;
    block_table (B, max_blocks) int32; context_lens (B,) int32. scale defaults to 1/sqrt(D).
    Returns (total_q, H, D) bf16. KV insertion is separate (call rope_kv_insert / kv_cache_scatter
    before this). Accepts mlx.array or torch.Tensor (MPS)."""
    import numpy as np
    total_q, H, D = q_packed.shape
    if scale == 0.0:
        scale = 1.0 / (float(D) ** 0.5)
    cu = np.asarray(cu_seqlens_q, dtype=np.int64)
    B = len(cu) - 1
    qlens, padded, pad_off, tile_seq, tile_local0 = _varlen_worklist(cu_seqlens_q)
    total_padded = int(pad_off[-1])
    is_t = _is_torch(q_packed)

    if is_t:
        import torch
        dev = q_packed.device
        parts = []
        for b in range(B):
            seg = q_packed[int(cu[b]):int(cu[b + 1])]
            pad = int(padded[b]) - int(qlens[b])
            if pad:
                seg = torch.cat(
                    [seg, torch.zeros(pad, H, D, dtype=seg.dtype, device=dev)], 0)
            parts.append(seg)
        q_hm = torch.cat(parts, 0).permute(1, 0, 2).contiguous()   # (H, total_padded, D)
        ts = torch.from_numpy(tile_seq).to(dev)
        tl = torch.from_numpy(tile_local0).to(dev)
        sq = torch.from_numpy(qlens).to(dev)
        o_hm = _torch().attn_varlen_prefill(
            q_hm, key_cache, value_cache, block_table, context_lens, ts, tl, sq, float(scale))
        o_pad = o_hm.permute(1, 0, 2).contiguous()                 # (total_padded, H, D)
        outs = [o_pad[int(pad_off[b]):int(pad_off[b]) + int(qlens[b])] for b in range(B)]
        return torch.cat(outs, 0)

    import mlx.core as mx
    parts = []
    for b in range(B):
        seg = q_packed[int(cu[b]):int(cu[b + 1])]
        pad = int(padded[b]) - int(qlens[b])
        if pad:
            seg = mx.concatenate([seg, mx.zeros((pad, H, D), dtype=seg.dtype)], axis=0)
        parts.append(seg)
    q_hm = mx.transpose(mx.concatenate(parts, axis=0), (1, 0, 2))   # (H, total_padded, D)
    ts, tl, sq = mx.array(tile_seq), mx.array(tile_local0), mx.array(qlens)
    o_hm = _mlx().attn_varlen_prefill(
        q_hm, key_cache, value_cache, block_table, context_lens, ts, tl, sq, float(scale))
    o_pad = mx.transpose(o_hm, (1, 0, 2))                           # (total_padded, H, D)
    outs = [o_pad[int(pad_off[b]):int(pad_off[b]) + int(qlens[b])] for b in range(B)]
    return mx.concatenate(outs, axis=0)


def rope_kv_insert(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache):
    """Fused RoPE (split-half) on K + paged-KV insert. Returns updated (key_cache, value_cache).

    Accepts mlx.array or torch.Tensor (MPS). k/v (num_tokens, num_kv_heads, D);
    caches (num_blocks, block_size, num_kv_heads, D); D in {64,128}.
    """
    if _is_torch(k):
        return _torch().rope_kv_insert(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache)
    return _mlx().rope_kv_insert(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache)


def rope_kv_insert_norm(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache,
                        norm_weight, eps=1e-5, gemma=False):
    """Fused K RMSNorm + RoPE (split-half) + paged-KV insert. gemma=True uses (1+weight).

    Returns updated (key_cache, value_cache). Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(k):
        return _torch().rope_kv_insert_norm(k, v, cos, sin, positions, slot_mapping,
                                            key_cache, value_cache, norm_weight, eps, gemma)
    return _mlx().rope_kv_insert_norm(k, v, cos, sin, positions, slot_mapping,
                                      key_cache, value_cache, norm_weight, eps, gemma)


def rope_q(q, cos, sin, positions, norm_weight=None, gemma=False, eps=1e-6):
    """Rotate (split-half) and optionally weighted-RMSNorm Q into a contiguous q_out (Q is not
    paged). q (num_tokens, num_q_heads, D); cos/sin (P, D/2). If norm_weight is given, RMSNorm over
    D is applied (gemma=True uses 1+weight). Accepts mlx.array or torch.Tensor (MPS). {f16,bf16,f32}.
    """
    D = q.shape[-1]
    do_norm = norm_weight is not None
    if norm_weight is None:   # dummy (unused)
        if _is_torch(q):
            import torch
            norm_weight = torch.ones(D, dtype=q.dtype, device=q.device)
        else:
            import mlx.core as mx
            norm_weight = mx.ones((D,), dtype=q.dtype)
    if _is_torch(q):
        return _torch().rope_q(q, cos, sin, positions, norm_weight, do_norm, gemma, eps)
    return _mlx().rope_q(q, cos, sin, positions, norm_weight, do_norm, gemma, eps)


def mla_q_norm_rope(q, cos, sin, positions, num_heads, nope_dim, rope_dim,
                    norm_mode=0, eps=1e-6, norm_weight=None):
    """DeepSeek MLA Q-path: optional RMSNorm over the full head dim (norm_mode 0=none, 1=rms
    no-weight, 2=rms + norm_weight) then GPT-J interleaved RoPE on the last rope_dim dims.

    q (…, head_dim) bf16, head_dim=nope_dim+rope_dim (%64==0); cos/sin (max_pos, rope_dim/2);
    positions (num_tokens,). Accepts mlx.array or torch.Tensor (MPS).
    """
    head_dim = q.shape[-1]
    if norm_weight is None:   # dummy (unused unless norm_mode==2)
        if _is_torch(q):
            import torch
            norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=q.device)
        else:
            import mlx.core as mx
            norm_weight = mx.ones((head_dim,), dtype=mx.bfloat16)
    if _is_torch(q):
        return _torch().mla_q_norm_rope(q, cos, sin, positions, norm_weight, num_heads,
                                        nope_dim, rope_dim, norm_mode, eps)
    return _mlx().mla_q_norm_rope(q, cos, sin, positions, norm_weight, num_heads,
                                  nope_dim, rope_dim, norm_mode, eps)


def mla_kv_insert(kv_c, k_pe, cos, sin, positions, slot_mapping, kv_cache,
                  rope_dim=None, norm_mode=0, eps=1e-6, norm_weight=None):
    """DeepSeek MLA classic KV-insert: writes the (optionally kv_a-RMSNormed, norm_mode 0/2) latent
    kv_c + interleaved-RoPE'd k_pe into a paged bf16 cache (num_blocks, block_size, LATENT+rope_dim).

    Returns the updated kv_cache. Accepts mlx.array or torch.Tensor (MPS).
    """
    latent = kv_c.shape[-1]
    if rope_dim is None:
        rope_dim = k_pe.shape[-1]
    if norm_weight is None:   # dummy (unused unless norm_mode==2)
        if _is_torch(kv_c):
            import torch
            norm_weight = torch.ones(latent, dtype=torch.bfloat16, device=kv_c.device)
        else:
            import mlx.core as mx
            norm_weight = mx.ones((latent,), dtype=mx.bfloat16)
    if _is_torch(kv_c):
        return _torch().mla_kv_insert(kv_c, k_pe, cos, sin, positions, slot_mapping, kv_cache,
                                      norm_weight, rope_dim, norm_mode, eps)
    return _mlx().mla_kv_insert(kv_c, k_pe, cos, sin, positions, slot_mapping, kv_cache,
                                norm_weight, rope_dim, norm_mode, eps)


def mla_decode(q, kv_cache, block_table, context_lens, scale=0.0):
    """DeepSeek MLA absorb-path latent flash-decode (MQA). q (B,N,576)=[ql_nope(512)|q_pe(64)]
    (ql_nope = q_nope @ W_UK_T, applied by the caller) attends the shared latent cache
    (num_blocks, block_size, 576); the value accumulate is over the 512 latent only. Returns
    o (B,N,512); the caller then up-projects with W_UV. Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(q):
        return _torch().mla_decode(q, kv_cache, block_table, context_lens, scale)
    return _mlx().mla_decode(q, kv_cache, block_table, context_lens, scale)


def mla_decode_fp8(q, data_cache, scale_cache, block_table, context_lens, scale=0.0):
    """DeepSeek-V4 dense latent decode over the UE8M0-packed cache (from mla_kv_insert_fp8). q
    (B,N,512) [the absorbed query] attends the packed cache with dequant-on-read; score and value
    both over the full 512, scale 512^-0.5. Returns o (B,N,512); the inverse-RoPE of o[448:512] +
    the wo_a/wo_b output projection are caller-applied. Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(q):
        return _torch().mla_decode_fp8(q, data_cache, scale_cache, block_table, context_lens, scale)
    return _mlx().mla_decode_fp8(q, data_cache, scale_cache, block_table, context_lens, scale)


def mla_decode_fp8_sparse(q, data_cache, scale_cache, block_table, indices, topk_length, scale=0.0):
    """DeepSeek-V4 sparse latent decode over the UE8M0-packed cache: like mla_decode_fp8 but each
    query attends only the token positions indices[b, 0:topk_length[b]] (the Lightning Indexer's
    top-k set). indices (B, max_topk) int; topk_length (B,) int. Returns o (B,N,512). MPS/MLX.
    """
    if _is_torch(q):
        return _torch().mla_decode_fp8_sparse(q, data_cache, scale_cache, block_table, indices,
                                              topk_length, scale)
    return _mlx().mla_decode_fp8_sparse(q, data_cache, scale_cache, block_table, indices,
                                        topk_length, scale)


def mla_kv_insert_fp8(kv, cos, sin, positions, slot_mapping, data_cache, scale_cache):
    """DeepSeek-V4 packed MLA KV-insert: kv (…, 512) = [448 NoPE | 64 RoPE]; NoPE is quantized to
    e4m3 fp8 with per-64-block UE8M0 (power-of-2) scales, RoPE gets interleaved RoPE bf16. Writes
    into a paged data_cache (nb, bs, 576) uint8 + scale_cache (nb, bs, 8) uint8; returns the
    updated (data_cache, scale_cache). Dequant: e4m3_decode(code) * 2**(scale_byte-127). MPS/MLX.
    """
    if _is_torch(kv):
        return _torch().mla_kv_insert_fp8(kv, cos, sin, positions, slot_mapping, data_cache, scale_cache)
    data, scale = _mlx().mla_kv_insert_fp8(kv, cos, sin, positions, slot_mapping, data_cache, scale_cache)
    return data, scale


def rms_norm_add(x, residual, weight, eps=1e-5):
    """Fused residual-add + RMSNorm. Returns (out, x+residual).

    out = rms_norm(x + residual) * weight. Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(x):
        return _torch().rms_norm_add(x, residual, weight, eps)
    return _mlx().rms_norm_add(x, residual, weight, eps=eps)


def layernorm_add(x, residual, weight, bias, eps=1e-5):
    """Fused residual-add + LayerNorm. Returns (out, x+residual).

    out = layernorm(x + residual) * weight + bias. Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(x):
        return _torch().layernorm_add(x, residual, weight, bias, eps)
    return _mlx().layernorm_add(x, residual, weight, bias, eps=eps)


def rms_norm_add_fp8(x, residual, weight, eps=1e-5, scale=None):
    """Fused add + rms_norm + fp8. scale=None -> dynamic per-row (returns codes, x+residual, scale);
    else static per-tensor (returns codes, x+residual). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rms_norm_add_fp8(x, residual, weight, eps, scale)
    if scale is None:
        return _mlx().rms_norm_add_fp8_dyn(x, residual, weight, eps=eps)
    return _mlx().rms_norm_add_fp8(x, residual, weight, eps=eps, scale=scale)


def layernorm_add_fp8(x, residual, weight, bias, eps=1e-5, scale=None):
    """Fused add + layernorm + fp8. scale=None -> dynamic (codes, x+residual, scale); else static.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().layernorm_add_fp8(x, residual, weight, bias, eps, scale)
    if scale is None:
        return _mlx().layernorm_add_fp8_dyn(x, residual, weight, bias, eps=eps)
    return _mlx().layernorm_add_fp8(x, residual, weight, bias, eps=eps, scale=scale)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rms_norm(x, weight, eps)
    return _mlx().rms_norm(x, weight, eps=eps)


def rms_norm_backward(x, weight, dy, eps=1e-5):
    """RMSNorm backward. Returns (dx, dweight): dx has x's shape (fused dX kernel), dweight (D,) is
    summed over all rows. x/dy (..., D), weight (D,). Matches torch autograd. The row std (rstd) and
    dweight are cheap framework reductions; the per-row dX (reduction + Liger-factored combine) is
    the kernel. Accepts mlx.array or torch.Tensor (MPS)."""
    D = x.shape[-1]
    if _is_torch(x):
        import torch
        xf = x.reshape(-1, D).contiguous()
        dyf = dy.reshape(-1, D).contiguous()
        rstd = torch.rsqrt((xf.float() ** 2).mean(-1, keepdim=True) + eps)   # (rows, 1)
        dx = _torch().rms_norm_bwd_dx(xf, weight, dyf, rstd.squeeze(-1).contiguous())
        dw = (dyf.float() * xf.float() * rstd).sum(0).to(weight.dtype)
        return dx.reshape(x.shape), dw
    import mlx.core as mx
    xf = mx.reshape(x, (-1, D))
    dyf = mx.reshape(dy, (-1, D))
    rstd = mx.rsqrt(mx.mean(xf.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps)  # (rows,1)
    dx = _mlx().rms_norm_bwd_dx(xf, weight, dyf, mx.reshape(rstd, (-1,)))
    dw = mx.sum(dyf.astype(mx.float32) * xf.astype(mx.float32) * rstd, axis=0).astype(weight.dtype)
    return mx.reshape(dx, x.shape), dw


def rms_norm_add_backward(hidden, weight, dout, dresidual=None, eps=1e-5):
    """Backward of the fused residual-add + RMSNorm (rms_norm_add forward: out = rms_norm(x+residual)*w,
    residual_out = x+residual). Pass `hidden` = the saved x+residual. Since residual_out == hidden and
    hidden = x + residual, the grad wrt x and wrt residual are identical: dhidden = rms_norm_bwd_dx(
    hidden, w, dout) (+ dresidual if a grad flows into the residual output). Returns (dx, dresidual,
    dweight) with dx IS dresidual. Ref: Liger fused_add_rms_norm. Accepts mlx / torch (MPS)."""
    dx, dweight = rms_norm_backward(hidden, weight, dout, eps=eps)
    if dresidual is not None:
        dx = dx + dresidual   # the residual-output branch passes straight through to hidden
    return dx, dx, dweight


def softmax(x):
    """Softmax over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().softmax(x)
    return _mlx().softmax(x)


def rotary(x, cos, sin, interleaved=False):
    """RoPE. x is (B,H,N,D), cos/sin (N,D/2). mlx.array or torch.Tensor (MPS).
    interleaved=False: split-half (GPT-NeoX); True: GPT-J adjacent pairs."""
    if _is_torch(x):
        return _torch().rotary(x, cos, sin, interleaved)
    return _mlx().rotary(x, cos, sin, interleaved)


def gelu(x):
    """GELU (tanh approx) over the last axis. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().gelu(x)
    return _mlx().gelu(x)


def dropout(x, p, seed):
    """Inverted dropout (training): out = keep ? x/(1-p) : 0, keep_i = rng_uniform(seed, i) >= p.
    The keep-mask is a pure function of (seed, index), so dropout_backward with the SAME seed/p
    recomputes it exactly (no mask storage). p in [0, 1). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().dropout(x, float(p), int(seed))
    return _mlx().dropout(x, float(p), int(seed))


def dropout_backward(dy, p, seed):
    """Dropout backward: dx = keep ? dy/(1-p) : 0, same mask recomputed from (seed, p) as the
    matching dropout forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(dy):
        return _torch().dropout_backward(dy, float(p), int(seed))
    return _mlx().dropout_backward(dy, float(p), int(seed))


def gelu_backward(x, dy):
    """GELU (tanh approx) backward: dx = dy * gelu'(x). Elementwise; returns x's shape. Matches
    torch autograd for F.gelu(approximate='tanh'). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().gelu_bwd(x, dy)
    return _mlx().gelu_bwd(x, dy)


def glu(x, gate, mode="swiglu", alpha=1.0, limit=1.0e20):
    """GLU-family activation. mode in reglu/geglu/swiglu/swiglu_oai/geglu_erf/geglu_quick."""
    if _is_torch(x):
        return _torch().glu(x, gate, mode, alpha, limit)
    return _mlx().glu(x, gate, mode=mode, alpha=alpha, limit=limit)


def glu_backward(x, gate, dc, mode="swiglu", alpha=1.0, limit=1.0e20):
    """GLU-family backward: given the upstream grad dc (wrt out=act(x)*gate), returns (da, db) =
    grads wrt x, gate. da = dc*gate*act'(x), db = dc*act(x). mode in reglu/geglu/swiglu/swiglu_oai/
    geglu_erf/geglu_quick. Matches the gradient of the tk glu forward. Accepts mlx / torch (MPS)."""
    if _is_torch(x):
        return _torch().glu_backward(x, gate, dc, mode, alpha, limit)
    out = _mlx().glu_backward(x, gate, dc, mode=mode, alpha=alpha, limit=limit)
    return out[0], out[1]


def reglu(x, gate):
    return glu(x, gate, mode="reglu")


def geglu(x, gate):
    return glu(x, gate, mode="geglu")


def swiglu(x, gate):
    return glu(x, gate, mode="swiglu")


def swiglu_oai(x, gate, alpha=1.0, limit=1.0e20):
    return glu(x, gate, mode="swiglu_oai", alpha=alpha, limit=limit)


def geglu_erf(x, gate):
    return glu(x, gate, mode="geglu_erf")


def geglu_quick(x, gate):
    return glu(x, gate, mode="geglu_quick")


def hadamard(x, scale=0.0):
    """Walsh-Hadamard transform over the final axis. Default scale is 1/sqrt(D)."""
    if _is_torch(x):
        return _torch().hadamard(x, scale)
    return _mlx().hadamard(x, scale=scale)


def kv_cache_scatter(key, value, slot_mapping, num_blocks, block_size):
    """Scatter key/value rows (T,H,D) into paged KV caches (num_blocks, block_size, H, D)."""
    if _is_torch(key):
        return _torch().kv_cache_scatter(key, value, slot_mapping, num_blocks, block_size)
    return _mlx().kv_cache_scatter(key, value, slot_mapping, num_blocks, block_size)


def kv_cache_gather(key_cache, value_cache, block_table, cu_seq_lens, num_tokens):
    """Gather paged KV caches back to contiguous key/value tensors."""
    if _is_torch(key_cache):
        return _torch().kv_cache_gather(key_cache, value_cache, block_table, cu_seq_lens, num_tokens)
    return _mlx().kv_cache_gather(key_cache, value_cache, block_table, cu_seq_lens, num_tokens)


def kv_cache_copy_blocks(key_cache, value_cache, block_mapping):
    """Copy paged KV cache blocks according to (src, dst) pairs."""
    if _is_torch(key_cache):
        return _torch().kv_cache_copy_blocks(key_cache, value_cache, block_mapping)
    return _mlx().kv_cache_copy_blocks(key_cache, value_cache, block_mapping)


def kv_cache_scales(key, value):
    """Return fp8 KV-cache scales `(key_scale, value_scale)` as absmax / 240."""
    if _is_torch(key):
        return _torch().kv_cache_scales(key, value)
    return _mlx().kv_cache_scales(key, value)


def paged_attention(q, key_cache, value_cache, block_table, context_lens, scale=0.0, window=0):
    """Decode paged attention. q/out (B,H,D), caches (num_blocks, block_size, H, D).
    window > 0 restricts to the `window` most recent keys (Mistral sliding window)."""
    if _is_torch(q):
        return _torch().paged_attention(q, key_cache, value_cache, block_table, context_lens,
                                        scale, window)
    return _mlx().paged_attention(q, key_cache, value_cache, block_table, context_lens, scale,
                                  window=window)


def paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens, alibi_slopes,
                          scale=0.0, window=0):
    """Paged decode with a per-head ALiBi linear position bias. alibi_slopes is (num_heads,);
    each score gets slope[h]*(t - context_len + 1). window > 0 restricts to the `window` most
    recent keys. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().paged_attention_alibi(q, key_cache, value_cache, block_table,
                                              context_lens, alibi_slopes, scale, window)
    return _mlx().paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens,
                                        alibi_slopes, scale, window=window)


def paged_attention_block_sparse(q, key_cache, value_cache, block_table, context_lens, block_mask,
                                 scale=0.0, window=0):
    """Block-sparse paged decode: a query skips entire KV blocks it doesn't attend to.
    block_mask is (batch, max_blocks) int (1=attend, 0=skip), sharing block_table's layout.
    window > 0 restricts to the `window` most recent keys. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().paged_attention_block_sparse(q, key_cache, value_cache, block_table,
                                                     context_lens, block_mask, scale, window)
    return _mlx().paged_attention_block_sparse(q, key_cache, value_cache, block_table,
                                               context_lens, block_mask, scale, window=window)


def paged_attention_xcache(q, key_cache, value_cache, block_table, context_lens, scale=0.0):
    """Paged decode over a vLLM x-packed KV cache (so a vLLM cache can be consumed directly):
    key_cache (num_blocks, num_kv_heads, head_size/x, block_size, x), value_cache
    (num_blocks, num_kv_heads, head_size, block_size), x = 16/sizeof(dtype). Bit-equivalent to
    paged_attention on the same values. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().paged_attention_xcache(q, key_cache, value_cache, block_table,
                                               context_lens, scale)
    return _mlx().paged_attention_xcache(q, key_cache, value_cache, block_table, context_lens,
                                         scale)


def paged_attention_staged(q, key_cache, value_cache, block_table, context_lens, scale=0.0):
    """GQA KV-reuse staged decode: bit-equivalent to paged_attention, but stages each KV vector
    once into threadgroup memory and reuses it across the query heads sharing that kv_head
    (amortizes cache bandwidth by group_size). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().paged_attention_staged(q, key_cache, value_cache, block_table, context_lens, scale)
    return _mlx().paged_attention_staged(q, key_cache, value_cache, block_table, context_lens, scale)


def _fmt_code(fmt):
    """Map an fp8 format ('e4m3'/'e5m2' or 0/1) to the kernel's integer format code."""
    return {"e4m3": 0, "e5m2": 1}.get(fmt, fmt) if isinstance(fmt, str) else int(fmt)


def kv_cache_scatter_fp8(key, value, slot_mapping, num_blocks, block_size, k_scale, v_scale,
                         fmt="e4m3"):
    """Scatter K/V into a uint8 paged cache. Returns (kc, vc).

    k_scale/v_scale may be a plain float (per-tensor, broadcast to every head) or a
    (num_heads,) array (per-head). fmt: 'e4m3' (default) or 'e5m2'.
    Accepts mlx.array or torch.Tensor (MPS).
    """
    H = key.shape[1]
    k_scale, v_scale = _scale_vec(k_scale, H, key), _scale_vec(v_scale, H, key)
    if _is_torch(key):
        return _torch().kv_cache_scatter_fp8(key, value, slot_mapping, num_blocks, block_size,
                                             k_scale, v_scale, fmt)
    return _mlx().kv_cache_scatter_fp8(key, value, slot_mapping, num_blocks, block_size,
                                       k_scale, v_scale, _fmt_code(fmt))


def paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                        k_scale, v_scale, scale=0.0, fmt="e4m3", window=0):
    """Decode paged attention over fp8 (uint8) caches, dequantized on read. GQA aware.

    k_scale/v_scale may be a plain float (per-tensor) or a (num_kv_heads,) array (per-head).
    fmt: 'e4m3' (default) or 'e5m2' — must match the format the cache was written with.
    window > 0 restricts to the `window` most recent keys. Accepts mlx.array or torch.Tensor (MPS).
    """
    H_KV = key_cache.shape[2]
    k_scale, v_scale = _scale_vec(k_scale, H_KV, q), _scale_vec(v_scale, H_KV, q)
    if _is_torch(q):
        return _torch().paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                                            k_scale, v_scale, scale, fmt, window)
    return _mlx().paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                                      k_scale, v_scale, scale, _fmt_code(fmt), window=window)


def paged_attention_v2(q, key_cache, value_cache, block_table, context_lens,
                       scale=0.0, partition_size=256, window=0):
    """Long-context paged decode attention (partition/reduce). GQA/MQA aware.

    q/out (B,H,D); caches (num_blocks, block_size, num_kv_heads, D). Accepts
    mlx.array or torch.Tensor (MPS). partition_size must be a multiple of block_size.
    window > 0 restricts to the `window` most recent keys.
    """
    if _is_torch(q):
        return _torch().paged_attention_v2(
            q, key_cache, value_cache, block_table, context_lens, scale, partition_size, window)
    return _mlx().paged_attention_v2(
        q, key_cache, value_cache, block_table, context_lens,
        scale=scale, partition_size=partition_size, window=window)


def cascade_attention(q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
                      scale=0.0, partition_size=256):
    """Cascade / shared-prefix attention: all B requests attend a single SHARED contiguous prefix KV
    (prefix_k/prefix_v (prefix_len, H_KV, D)) plus their own paged suffix (key_cache/value_cache +
    block_table + context_lens). The two levels' softmax states are merged via the shared
    log-sum-exp reduce == full attention over [prefix ++ suffix] per request. Amortizes the shared
    system prompt across a batch. q/out (B,H,D); D in {64,128}; GQA/MQA aware. partition_size must be
    a multiple of block_size. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().cascade_attention(
            q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
            scale, partition_size)
    return _mlx().cascade_attention(
        q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
        scale=scale, partition_size=partition_size)


def paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                           k_scale, v_scale, scale=0.0, partition_size=256, fmt="e4m3", window=0):
    """Long-context paged decode over an fp8 (uint8) cache, dequantized on read. GQA/MQA aware.

    k_scale/v_scale: plain float (per-tensor) or a (num_kv_heads,) array (per-head).
    fmt: 'e4m3' (default) or 'e5m2'. window > 0 restricts to the `window` most recent keys.
    Accepts mlx.array or torch.Tensor (MPS).
    """
    H_KV = key_cache.shape[2]
    k_scale, v_scale = _scale_vec(k_scale, H_KV, q), _scale_vec(v_scale, H_KV, q)
    if _is_torch(q):
        return _torch().paged_attention_v2_fp8(
            q, key_cache, value_cache, block_table, context_lens, k_scale, v_scale,
            scale, partition_size, fmt, window)
    return _mlx().paged_attention_v2_fp8(
        q, key_cache, value_cache, block_table, context_lens, k_scale, v_scale,
        scale=scale, partition_size=partition_size, fmt=_fmt_code(fmt), window=window)


def moe_route_topk(logits, k):
    """MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).

    logits (num_tokens, num_experts); k <= min(16, num_experts). Accepts mlx.array or torch.Tensor.
    """
    if _is_torch(logits):
        return _torch().moe_route_topk(logits, k)
    return _mlx().moe_route_topk(logits, k)


def moe_permute(topk_ids, num_experts):
    """Group T*k routing rows by expert. Returns (sorted_row_idx, offsets, inv_idx) int32.

    Accepts mlx.array or torch.Tensor (MPS). A flat row r maps to token r//k, slot r%k.
    """
    if _is_torch(topk_ids):
        return _torch().moe_permute(topk_ids, num_experts)
    sorted_idx, offsets, inv_idx = _mlx().moe_permute(topk_ids, num_experts)[:3]
    return sorted_idx, offsets, inv_idx


def moe_pad_schedule(sorted_row_idx, offsets, k):
    """32-row-padded per-expert schedule for the grouped GEMMs (all-GPU, no host sync).

    Turns moe_permute's compact layout into padded segments. Returns int32 arrays
    (expert_of_tile (max_tiles,), gather_idx (total_pad_max,), inv_pad (T*k,),
    off_pad (E+1,)); -1 sentinels mark pad tiles/rows beyond the real total.
    inv_pad[r] is the padded row moe_finalize must read for routing row r.
    Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(sorted_row_idx):
        return _torch().moe_pad_schedule(sorted_row_idx, offsets, k)
    out = _mlx().moe_pad_schedule(sorted_row_idx, offsets, k)
    return out[0], out[1], out[2], out[3]


def moe_gather(x, gather_idx):
    """Gather padded rows: out[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0).

    x (T, H) float32/bfloat16; returns (len(gather_idx), H). Accepts mlx.array or
    torch.Tensor (MPS).
    """
    if _is_torch(x):
        return _torch().moe_gather(x, gather_idx)
    return _mlx().moe_gather(x, gather_idx)


def moe_mlp(x, router_logits, W1, W2, k):
    """End-to-end MoE MLP: route → permute → pad-schedule → gather → SwiGLU GEMM →
    down-proj GEMM → weighted combine. All-GPU, no host sync.

    x (T, H); router_logits (T, E); W1 (E, H, 2*I) laid out [gate | up]; W2 (E, I, H);
    k experts per token. Returns (T, H) in x's dtype. H%16, I%32 required.
    Accepts mlx.array or torch.Tensor (MPS).
    """
    topk_ids, topk_weights = moe_route_topk(router_logits, k)
    sorted_row_idx, offsets, _inv_idx = moe_permute(topk_ids, W1.shape[0])
    expert_of_tile, gather_idx, inv_pad, _off_pad = moe_pad_schedule(sorted_row_idx, offsets, k)
    permuted = moe_gather(x, gather_idx)
    inter = moe_grouped_gemm_swiglu(permuted, W1, expert_of_tile)
    expert_out = moe_grouped_gemm_rect(inter, W2, expert_of_tile)
    return moe_finalize(expert_out, inv_pad, topk_weights, k)


def moe_grouped_gemm(permuted_input, W, expert_of_tile):
    """Fused grouped expert GEMM: out = permuted_input @ W[expert]. Returns (total_rows, H).

    permuted_input (total_rows, H) grouped by expert (segments padded to 32); W (E, H, H);
    expert_of_tile (total_rows/32,). Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(permuted_input):
        return _torch().moe_grouped_gemm(permuted_input, W, expert_of_tile)
    return _mlx().moe_grouped_gemm(permuted_input, W, expert_of_tile)


def moe_grouped_gemm_rect(A, W, expert_of_tile):
    """Rectangular grouped expert GEMM: out(rows,N_out) = A(rows,K_dim) @ W[e](K_dim,N_out).
    W (E,K_dim,N_out); K_dim%16, N_out%32, rows%32. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(A):
        return _torch().moe_grouped_gemm_rect(A, W, expert_of_tile)
    return _mlx().moe_grouped_gemm_rect(A, W, expert_of_tile)


def moe_grouped_gemm_swiglu(A, W1, expert_of_tile):
    """Fused SiLU-GLU GEMM1: out(rows,inter) = silu(A@W1_gate) * (A@W1_up). W1[e] (H, 2*inter),
    laid out [gate | up]. H%16, inter%32, rows%32. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(A):
        return _torch().moe_grouped_gemm_swiglu(A, W1, expert_of_tile)
    return _mlx().moe_grouped_gemm_swiglu(A, W1, expert_of_tile)


def moe_finalize(expert_out, inv_idx, topk_weights, k):
    """out[t] = sum_k weight[t,k] * expert_out[inv_idx[t*k+k]]. Returns (T, Hdim).

    Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(expert_out):
        return _torch().moe_finalize(expert_out, inv_idx, topk_weights, k)
    return _mlx().moe_finalize(expert_out, inv_idx, topk_weights, k)


def argmax_sample(logits):
    """Greedy sampling: argmax token index over the last (vocab) axis. Returns int32.

    Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(logits):
        return _torch().argmax_sample(logits)
    return _mlx().argmax_sample(logits)


def sample_categorical(logits, temperature=1.0, seed=0):
    """Gumbel-max categorical sampling from softmax(logits/temperature). Returns int32.

    Accepts mlx.array or torch.Tensor (MPS). The draw is reproducible given (seed, row).
    """
    if _is_torch(logits):
        return _torch().sample_categorical(logits, temperature, seed)
    return _mlx().sample_categorical(logits, temperature=temperature, seed=seed)


def top_k_sample(logits, k, temperature=1.0, seed=0):
    """Top-k sampling: Gumbel-max from softmax over the k highest logits. Returns int32.

    Accepts mlx.array or torch.Tensor (MPS). Reproducible given (seed, row). k <= 64.
    """
    if _is_torch(logits):
        return _torch().top_k_sample(logits, k, temperature, seed)
    return _mlx().top_k_sample(logits, k, temperature=temperature, seed=seed)


def top_p_sample(logits, p, temperature=1.0, seed=0):
    """Top-p (nucleus) sampling: Gumbel-max from the smallest top-prob set with mass >= p. int32.

    Accepts mlx.array or torch.Tensor (MPS). Reproducible given (seed, row).
    """
    if _is_torch(logits):
        return _torch().top_p_sample(logits, p, temperature, seed)
    return _mlx().top_p_sample(logits, p, temperature=temperature, seed=seed)


def min_p_sample(logits, min_p, temperature=1.0, seed=0):
    """min-p sampling: Gumbel-max over tokens with (tempered) prob >= min_p * max_prob. Returns the
    token index per row (int32). min_p in (0, 1]. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().min_p_sample(logits, min_p, temperature, seed)
    return _mlx().min_p_sample(logits, min_p, temperature=temperature, seed=seed)


def apply_token_bitmask(logits, bitmask):
    """Grammar / structured-output masking: set logits[v] = -inf where the packed allow-bitmask bit
    for token v is 0. logits (T, V); bitmask (T, ceil(V/32)) int32 packed words (bit v of row t
    allows token t*32-block... i.e. word[v>>5] bit (v&31)). Returns masked logits, same dtype.
    Composes before any sampler. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().apply_token_bitmask(logits, bitmask)
    return _mlx().apply_token_bitmask(logits, bitmask)


def embedding_lookup(token_ids, table, pos_table=None, scale=1.0):
    """Token embedding lookup: out[t] = scale*table[token_ids[t]] (+ pos_table[t] if given). A
    negative / out-of-range token id emits zeros (padding). token_ids (num_tok,) int; table
    (vocab, D); optional pos_table (num_tok, D). Returns (num_tok, D). Accepts mlx / torch (MPS)."""
    if _is_torch(table):
        import torch
        pt = pos_table if pos_table is not None else torch.zeros(1, dtype=table.dtype,
                                                                 device=table.device)
        return _torch().embedding_lookup(token_ids, table, pt, float(scale))
    import mlx.core as mx
    pt = pos_table if pos_table is not None else mx.zeros((1,), dtype=table.dtype)
    return _mlx().embedding_lookup(token_ids, table, pt, float(scale))


def embedding_backward(token_ids, dY, vocab, scale=1.0):
    """Embedding backward: scatter-add the upstream grad dY (num_tok, D) into a (vocab, D) fp32
    gradient table by token id (dtable[token_ids[t]] += scale*dY[t]). A negative / out-of-range id
    contributes nothing (matches the padding-zeros forward). token_ids (num_tok,) int; dY float.
    Returns (vocab, D) float32. Matches nn.Embedding autograd. Accepts mlx / torch (MPS)."""
    if _is_torch(dY):
        return _torch().embedding_backward(token_ids, dY, int(vocab), float(scale))
    return _mlx().embedding_backward(token_ids, dY, vocab=int(vocab), scale=float(scale))


def merge_multimodal_spans(text, modal, src):
    """Multimodal span merge: out[t] = modal[src[t]] if src[t] >= 0 else text[t]. text (num_tok, D),
    modal (num_modal, D) same dtype, src (num_tok,) int (-1 keeps the text embedding, >=0 gathers a
    modal row). The src map is the flattened placeholder->modal index list (build it host-side from
    the image/audio span offsets/lengths). Returns (num_tok, D). Accepts mlx / torch (MPS)."""
    if _is_torch(text):
        return _torch().merge_multimodal_spans(text, modal, src)
    return _mlx().merge_multimodal_spans(text, modal, src)


def cross_entropy(logits, targets, ignore_index=-100, reduction="mean", label_smoothing=0.0,
                  z_loss=0.0, softcap=0.0, return_lse=False):
    """Fused cross-entropy over the vocab axis WITHOUT storing the (T, V) probabilities.

    logits (T, V) f32/f16/bf16, targets (T,) int. reduction in {"none", "mean", "sum"}. Supports
    label_smoothing, a z-loss (z_loss * lse^2), and a Gemma-2 logit softcap (softcap>0 caps each
    logit to softcap*tanh(z/softcap) before the softmax). Returns the reduced loss (scalar for
    mean/sum, (T,) for none); if return_lse, also returns per-row lse (needed by
    cross_entropy_grad). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        loss, lse = _torch().cross_entropy_fwd(logits, targets, int(ignore_index),
                                               float(label_smoothing), float(z_loss), float(softcap))
        if reduction == "mean":
            n = (targets != ignore_index).sum().clamp(min=1).to(loss.dtype)
            out = loss.sum() / n
        elif reduction == "sum":
            out = loss.sum()
        else:
            out = loss
        return (out, lse) if return_lse else out
    import mlx.core as mx
    loss, lse = _mlx().cross_entropy_fwd(logits, targets, int(ignore_index),
                                         float(label_smoothing), float(z_loss), float(softcap))
    if reduction == "mean":
        n = mx.maximum((targets != ignore_index).sum(), 1).astype(loss.dtype)
        out = loss.sum() / n
    elif reduction == "sum":
        out = loss.sum()
    else:
        out = loss
    return (out, lse) if return_lse else out


def cross_entropy_grad(logits, targets, lse, grad_out, ignore_index=-100, label_smoothing=0.0,
                       z_loss=0.0, softcap=0.0):
    """Fused cross-entropy backward: grad_logits (T, V), out-of-place. grad_out is the per-row
    upstream gradient (T,) (e.g. 1/n_non_ignore for a mean reduction, or a scalar broadcast).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        import torch
        if not torch.is_tensor(grad_out) or grad_out.dim() == 0:
            grad_out = torch.full((logits.shape[0],), float(grad_out) if not torch.is_tensor(grad_out)
                                  else float(grad_out.item()), dtype=torch.float32,
                                  device=logits.device)
        return _torch().cross_entropy_bwd(logits, targets, lse, grad_out, int(ignore_index),
                                          float(label_smoothing), float(z_loss), float(softcap))
    import mlx.core as mx
    if not isinstance(grad_out, mx.array) or grad_out.ndim == 0:
        val = float(grad_out) if not isinstance(grad_out, mx.array) else float(grad_out.item())
        grad_out = mx.full((logits.shape[0],), val, dtype=mx.float32)
    return _mlx().cross_entropy_bwd(logits, targets, lse, grad_out, int(ignore_index),
                                    float(label_smoothing), float(z_loss), float(softcap))


def fused_linear_cross_entropy(h, W, targets, chunk_size=4096, ignore_index=-100,
                               label_smoothing=0.0, z_loss=0.0, softcap=0.0):
    """Liger-style chunked fused-linear-CE: loss + grads for (h @ W.T) vs targets without ever
    materializing the full (T, V) logits. Computes logits chunk by chunk over the token axis.
    h (T, K), W (V, K); targets (T,). Returns (loss (mean over non-ignored), dh (T,K), dW (V,K)).
    Accepts mlx.array or torch.Tensor (MPS)."""
    is_t = _is_torch(h)
    T = h.shape[0]
    if is_t:
        import torch
        n = int((targets != ignore_index).sum().clamp(min=1).item())
        dh = torch.zeros_like(h)
        dW = torch.zeros_like(W)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        for c0 in range(0, T, chunk_size):
            c1 = min(c0 + chunk_size, T)
            hc = h[c0:c1]
            tc = targets[c0:c1]
            logits = hc @ W.T
            loss_c, lse_c = _torch().cross_entropy_fwd(logits, tc, int(ignore_index),
                                                       float(label_smoothing), float(z_loss),
                                                       float(softcap))
            total = total + loss_c.sum()
            go = torch.full((c1 - c0,), 1.0 / n, dtype=torch.float32, device=h.device)
            g = _torch().cross_entropy_bwd(logits, tc, lse_c, go, int(ignore_index),
                                           float(label_smoothing), float(z_loss), float(softcap))
            dh[c0:c1] = g @ W
            dW = dW + g.T @ hc
        return total / n, dh, dW
    import mlx.core as mx
    n = int(mx.maximum((targets != ignore_index).sum(), 1).item())
    dh_parts = []
    dW = mx.zeros(W.shape, dtype=W.dtype)
    total = mx.zeros((), dtype=mx.float32)
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        hc = h[c0:c1]
        tc = targets[c0:c1]
        logits = hc @ mx.swapaxes(W, 0, 1)
        loss_c, lse_c = _mlx().cross_entropy_fwd(logits, tc, int(ignore_index),
                                                 float(label_smoothing), float(z_loss),
                                                 float(softcap))
        total = total + loss_c.sum()
        go = mx.full((c1 - c0,), 1.0 / n, dtype=mx.float32)
        g = _mlx().cross_entropy_bwd(logits, tc, lse_c, go, int(ignore_index),
                                     float(label_smoothing), float(z_loss), float(softcap))
        dh_parts.append(mx.matmul(g, W))
        dW = dW + mx.matmul(mx.swapaxes(g, 0, 1), hc)
    return total / n, mx.concatenate(dh_parts, axis=0), dW


_LM_HEAD_MODES = {"argmax": 0, "categorical": 1, "topk": 2}


_LM_QUANT_BLOCK_K = {"q8_0": 32, "q4_0": 32}


def lm_head_sample(h, W, mode="argmax", k=0, temperature=1.0, seed=0, bias=None, fused=False,
                   format=None):
    """LM-head + sampling: a decode token per row of h. h (T, K), W (V, K) row-major, both
    fp16/bf16/f32. mode in {"argmax", "categorical", "topk"} (top-p -> matmul + top_p_sample).

    format ("q8_0"/"q4_0") selects the fused quantized-weight path: W is the packed weight tensor
    (dequantized on read, no logits materialization); supports argmax/categorical/topk.
    bias is an optional (V,) additive logit bias. Returns (T,) int32 token ids. The Gumbel noise is
    indexed by the global vocab id, so the draw equals the unfused sampler on the same logits + seed.
    Accepts mlx.array or torch.Tensor (MPS).

    Default path: logits = h @ W.T (the framework matmul reads W once and is at the bandwidth floor)
    then the fast fused sampler. This is ~2.5x faster than a hand-written fused GEMV on Apple, where
    the head GEMV is bandwidth-bound and matmul is already optimal; the saved (T,V) logits traffic is
    negligible. fused=True runs the single-kernel no-materialization variant instead (a capability for
    memory-pressured huge-V decode; slower here — matmul wins)."""
    if mode not in _LM_HEAD_MODES:
        raise ValueError(f"lm_head_sample: mode must be one of {list(_LM_HEAD_MODES)}")
    if format is not None:
        if format not in _LM_QUANT_BLOCK_K:
            raise ValueError(f"lm_head_sample: quant format must be one of {list(_LM_QUANT_BLOCK_K)}")
        m = _LM_HEAD_MODES[mode]
        V = W.shape[0]
        K = W.shape[1] * _LM_QUANT_BLOCK_K[format]   # packed Wq is (V, K/block_k, block_bytes)
        if _is_torch(h):
            import torch
            b = bias if bias is not None else torch.zeros(1, dtype=torch.float32, device=h.device)
            return _torch().lm_head_sample_q(h, W, b, V, K, format, m, int(k), float(temperature),
                                             int(seed))
        import mlx.core as mx
        b = bias if bias is not None else mx.zeros((1,), dtype=mx.float32)
        return _mlx().lm_head_sample_q(h, W, b, V, K, format, m, int(k), float(temperature),
                                       int(seed))
    if fused:
        m = _LM_HEAD_MODES[mode]
        if _is_torch(h):
            import torch
            b = bias if bias is not None else torch.zeros(1, dtype=torch.float32, device=h.device)
            return _torch().lm_head_sample(h, W, b, m, int(k), float(temperature), int(seed))
        import mlx.core as mx
        b = bias if bias is not None else mx.zeros((1,), dtype=mx.float32)
        return _mlx().lm_head_sample(h, W, b, m, int(k), float(temperature), int(seed))

    # matmul + fast sampler (the default): reads W once, reused across T; bandwidth-optimal.
    if _is_torch(h):
        import torch
        logits = torch.matmul(h, W.transpose(-1, -2))
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
    else:
        import mlx.core as mx
        logits = mx.matmul(h, mx.swapaxes(W, -1, -2))
        if bias is not None:
            logits = logits + bias.astype(logits.dtype)
    if mode == "argmax":
        return argmax_sample(logits)
    if mode == "categorical":
        return sample_categorical(logits, temperature=temperature, seed=seed)
    return top_k_sample(logits, k, temperature=temperature, seed=seed)


def beam_advance(logits, cum_log_probs, beam_width):
    """Beam-search advance: one fused log-softmax + cumulative-score + top-beam_width step with
    parent tracking. logits (B*BM, V), cum_log_probs (B, BM). Returns (next_token (B,BM) int32,
    parent_beam (B,BM) int32, cum_log_probs' (B,BM) f32). beam_width <= 16. Step-0 convention: set
    cum_log_probs[:, 1:] = -inf. Accepts mlx.array or torch.Tensor (MPS).

    The candidate search is single-pass: one scan of the vocab builds each beam's log-sum-exp and its
    top-2*beam_width via a per-lane register heap, then a small cross-lane merge — no (B*BM, V)
    log_softmax or giant argpartition is materialized. Competitive with the framework path and faster
    for larger batch (it produces the fused exact (token, parent, score) in one call)."""
    if _is_torch(logits):
        return _torch().beam_advance(logits, cum_log_probs, int(beam_width))
    out = _mlx().beam_advance(logits, cum_log_probs, int(beam_width))
    return out[0], out[1], out[2]


def beam_build_copy_pairs(parent_beam, block_table, seq_lens, block_size):
    """Build the (src,dst) block-copy pairs for a beam KV reorder ON-DEVICE — no host readback.
    Returns a fixed (B*BM*max_blocks, 2) int64 buffer of pairs (sentinel (-1,-1) for empty slots),
    ready to feed kv_cache_copy_blocks. parent_beam (B,BM) int, block_table (B*BM,max_blocks) int,
    seq_lens (B*BM,) int. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(parent_beam):
        return _torch().beam_build_copy_pairs(parent_beam, block_table, seq_lens, int(block_size))
    return _mlx().beam_build_copy_pairs(parent_beam, block_table, seq_lens, int(block_size))


def spec_verify_linear(draft_tokens, draft_probs, target_probs, bonus_tokens, accept_u, seed):
    """Speculative decoding: linear (non-tree) rejection-sampling verification (vLLM contract).
    draft_tokens (B,S) int; draft_probs (B,S,V) f32; target_probs (B,S+1,V) f32; bonus_tokens (B,)
    int; accept_u (B,S) f32 uniforms. Returns (out_tokens (B,S+1) int32, accepted_cnt (B,) int32):
    draft dt is accepted iff accept_u <= p_target/p_draft; the first rejection emits a token sampled
    from the residual (p_target-p_draft)+ (seed drives the Gumbel-max resample); all-accept appends
    the bonus token; positions after the recovered token are -1. Accepts mlx.array / torch (MPS)."""
    if _is_torch(draft_tokens):
        return tuple(_torch().spec_verify_linear(
            draft_tokens, draft_probs, target_probs, bonus_tokens, accept_u, int(seed)))
    out = _mlx().spec_verify_linear(draft_tokens, draft_probs, target_probs, bonus_tokens,
                                    accept_u, int(seed))
    return out[0], out[1]


def beam_reorder_kv(key_cache, value_cache, block_table, parent_beam, seq_lens):
    """Reorder a paged KV cache after a beam step so each new beam's physical blocks hold its
    parent beam's KV history. Returns (key_cache', value_cache') (new caches — the copy op clones
    then applies src->dst, so it is a safe parallel scatter even for fan-out).

    key_cache/value_cache (num_blocks, block_size, H_KV, D); block_table (B*BM, max_blocks) int
    (row = global beam b*BM+i); parent_beam (B, BM) int32 from tk.beam_advance (per-batch-local
    parent index in [0, BM)); seq_lens gives the current length per beam (scalar, (B*BM,), or
    (B,BM)) to bound how many blocks are copied. Requires distinct physical blocks per beam (the
    zero-copy block-table-remap alternative is a cache-manager concern, out of scope). Fully
    GPU-resident: the copy pairs are built on-device (no parent_beam/block_table readback, so no
    per-step decode sync). Accepts mlx.array or torch.Tensor (MPS)."""
    block_size = key_cache.shape[1]
    B, BM = int(parent_beam.shape[0]), int(parent_beam.shape[1])
    is_t = _is_torch(key_cache)

    # Normalize seq_lens to a device (B*BM,) int array with NO host sync.
    if isinstance(seq_lens, (int, float)):
        if is_t:
            import torch
            sl = torch.full((B * BM,), int(seq_lens), dtype=torch.int32, device=key_cache.device)
        else:
            import mlx.core as mx
            sl = mx.full((B * BM,), int(seq_lens), dtype=mx.int32)
    elif getattr(seq_lens, "ndim", 1) == 0:      # 0-dim tensor: broadcast on-device (no readback)
        if is_t:
            import torch
            sl = seq_lens.reshape(1).expand(B * BM).to(torch.int32)
        else:
            import mlx.core as mx
            sl = mx.broadcast_to(mx.reshape(seq_lens, (1,)), (B * BM,)).astype(mx.int32)
    else:
        if is_t:
            import torch
            sl = seq_lens.reshape(B * BM).to(torch.int32)
        else:
            import mlx.core as mx
            sl = mx.reshape(seq_lens, (B * BM,)).astype(mx.int32)

    pairs = beam_build_copy_pairs(parent_beam, block_table, sl, block_size)
    return kv_cache_copy_blocks(key_cache, value_cache, pairs)


def beam_length_penalty(cum_log_probs, lengths, alpha=1.0):
    """Length-penalized beam score (FasterTransformer rule): cum_log_probs / ((5+len)/6)^alpha.
    lengths broadcasts against cum_log_probs (scalar or (B, BM)). Pure framework ops; the finished-
    beam (CBA) / EOS bookkeeping stays host-side policy. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(cum_log_probs):
        import torch
        ln = lengths if torch.is_tensor(lengths) else torch.tensor(
            float(lengths), dtype=cum_log_probs.dtype, device=cum_log_probs.device)
        pen = ((5.0 + ln.to(cum_log_probs.dtype)) / 6.0) ** float(alpha)
        return cum_log_probs / pen
    import mlx.core as mx
    ln = lengths if isinstance(lengths, mx.array) else mx.array(float(lengths))
    pen = ((5.0 + ln.astype(cum_log_probs.dtype)) / 6.0) ** float(alpha)
    return cum_log_probs / pen


def apply_penalty(logits, prev_tokens, temperature=1.0, repetition_penalty=1.0,
                  presence_penalty=0.0, frequency_penalty=0.0, bias=None, eos_id=-1,
                  min_length=0, gen_len=0, parent_ids=None):
    """Temperature + rep/presence/freq penalties + logit bias + min-length EOS mask.

    logits (T,V); prev_tokens (T,L) int (out-of-range = ignored padding); bias (V,) or None;
    forbids eos_id while gen_len < min_length. parent_ids (T,) int redirects each row's occurrence
    history (beam search: beam inherits its parent beam's history; None = identity). Returns
    penalized logits (T,V). Accepts mlx.array or torch.Tensor (MPS).
    """
    T = logits.shape[0]
    if _is_torch(logits):
        return _torch().apply_penalty(logits, prev_tokens, temperature=temperature,
                                      repetition_penalty=repetition_penalty,
                                      presence_penalty=presence_penalty,
                                      frequency_penalty=frequency_penalty, bias=bias, eos_id=eos_id,
                                      min_length=min_length, gen_len=gen_len, parent_ids=parent_ids)
    import mlx.core as mx
    if bias is None:
        bias = mx.zeros((logits.shape[-1],), dtype=mx.float32)
    if parent_ids is None:
        parent_ids = mx.arange(T, dtype=mx.int32)
    return _mlx().apply_penalty(
        logits, prev_tokens, bias, parent_ids, temperature=temperature,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty, eos_id=eos_id, min_length=min_length,
        gen_len=gen_len)[0]


def quantize_per_tensor_fp8(x):
    """Per-tensor fp8 e4m3 quant (global absmax/448 via atomic-max). Returns (codes uint8, scale).

    Accepts mlx.array or torch.Tensor (MPS). Reconstruct as scale * e4m3_decode(codes).
    """
    if _is_torch(x):
        return _torch().quantize_per_tensor_fp8(x)
    codes, scale, _ = _mlx().quantize_per_tensor_fp8(x)[:3]
    return codes, scale


def quantize_per_tensor_int8(x):
    """Per-tensor symmetric int8 quant (global absmax/127). Returns (codes int8, scale).

    Accepts mlx.array or torch.Tensor (MPS).
    """
    if _is_torch(x):
        return _torch().quantize_per_tensor_int8(x)
    codes, scale, _ = _mlx().quantize_per_tensor_int8(x)[:3]
    return codes, scale


def quantize_per_token_fp8(x):
    """Per-row fp8 e4m3 quant. Returns (codes uint8, scale f32), scale=absmax/448.

    Accepts mlx.array or torch.Tensor (MPS). Reconstruct as scale[...,None] * e4m3_decode(codes).
    """
    if _is_torch(x):
        return _torch().quantize_per_token_fp8(x)
    return _mlx().quantize_per_token_fp8(x)


def quantize_per_token_int8(x):
    """Per-row symmetric int8 quant. Returns (codes int8, scale f32), scale=absmax/127.

    Accepts mlx.array or torch.Tensor (MPS). Reconstruct as scale[...,None] * codes.
    """
    if _is_torch(x):
        return _torch().quantize_per_token_int8(x)
    return _mlx().quantize_per_token_int8(x)


def attn_causal(q, k, v):
    """Causal attention forward. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_causal(q, k, v)
    return _mlx().attn_causal(q, k, v)


def attn_window(q, k, v, window):
    """Sliding-window causal attention (Mistral/Gemma-style local attention): a query at
    position i attends keys [max(0, i-window+1), i] — the `window` most recent tokens
    including self. window <= 0 disables the window (== attn_causal). q,k,v (B,H,N,D) bf16,
    D in {64,128}, N%8==0. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_window(q, k, v, window)
    return _mlx().attn_window(q, k, v, window)


def flux_gelu(x, w, bias):
    """Fused gelu(x @ w + bias). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().flux_gelu(x, w, bias)
    return _mlx().flux_gelu(x, w, bias)


def flux_gate(x, w, bias, gate, residual):
    """Fused (x @ w + bias) * gate + residual. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().flux_gate(x, w, bias, gate, residual)
    return _mlx().flux_gate(x, w, bias, gate, residual)


def gemm_staged(x, y):
    """Multi-simdgroup threadgroup-staged GEMM (x @ y), tile-multiple shapes.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().gemm_staged(x, y)
    return _mlx().gemm_staged(x, y)


def attn_multiwarp(q, k, v):
    """Multi-warp flash attention forward (shared K/V). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_multiwarp(q, k, v)
    return _mlx().attn_multiwarp(q, k, v)


def linear_attn(q, k, v, use_kernel=False):
    """Non-causal linear attention Q@(K^T@V). Accepts mlx.array or torch.Tensor (MPS).

    Default routes to two framework matmuls: the non-causal form has no scan, and the
    composition uses the whole GPU per GEMM — measured ~15x faster than the
    one-simdgroup-per-(batch,head) TK kernel at (2,8,4096,64). use_kernel=True runs the
    ported TK kernel (parity/porting path)."""
    if use_kernel:
        if _is_torch(q):
            return _torch().linear_attn(q, k, v)
        return _mlx().linear_attn(q, k, v)
    if _is_torch(q):
        return q @ (k.transpose(-1, -2) @ v)
    import mlx.core as mx
    return mx.matmul(q, mx.matmul(k.swapaxes(-1, -2), v))


def hedgehog(q, k, v, use_kernel=False):
    """Hedgehog feature-map linear attention: out = phi(Q) @ (phi(K)^T @ V) with
    phi(x) = exp(x - rowmax(x)). Accepts mlx.array or torch.Tensor (MPS).

    Default routes to framework ops (feature map + two matmuls), measured ~3x faster than
    the one-simdgroup-per-(batch,head) TK kernel; use_kernel=True runs the ported kernel."""
    if use_kernel:
        if _is_torch(q):
            return _torch().hedgehog(q, k, v)
        return _mlx().hedgehog(q, k, v)
    if _is_torch(q):
        fq = (q.float() - q.float().amax(-1, keepdim=True)).exp()
        fk = (k.float() - k.float().amax(-1, keepdim=True)).exp()
        return (fq @ (fk.transpose(-1, -2) @ v.float())).to(q.dtype)
    import mlx.core as mx
    fq = mx.exp(q.astype(mx.float32) - mx.max(q.astype(mx.float32), axis=-1, keepdims=True))
    fk = mx.exp(k.astype(mx.float32) - mx.max(k.astype(mx.float32), axis=-1, keepdims=True))
    out = mx.matmul(fq, mx.matmul(fk.swapaxes(-1, -2), v.astype(mx.float32)))
    return out.astype(q.dtype)


def lin_attn_causal(q, k, v):
    """Causal linear attention (chunked scan). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().lin_attn_causal(q, k, v)
    return _mlx().lin_attn_causal(q, k, v)


def mamba2(C, B, X, cumlog):
    """Mamba-2 / SSD forward. cumlog = cumsum(log a). Accepts mlx.array or torch.Tensor (MPS).
    D in {64,128}; auto-routed between the quadratic kernel and the chunked linear-time pipeline
    at the MEASURED crossovers (N>=2048 for D=64, N>=4096 for D=128; chunked needs N%64==0)."""
    if _is_torch(C):
        return _torch().mamba2(C, B, X, cumlog)
    return _mlx().mamba2(C, B, X, cumlog)


def mamba2_chunked(C, B, X, cumlog):
    """Mamba-2 / SSD forward, forced chunked linear-time route (tests/benchmarks). N%64==0,
    N>=128, D in {64,128}. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(C):
        return _torch().mamba2_chunked(C, B, X, cumlog)
    return _mlx().mamba2_chunked(C, B, X, cumlog)


def mamba2_bwd(C, B, X, cumlog, dY, force_quadratic=False):
    """Mamba-2 / SSD backward. Given dY, returns (dC, dB, dX, dcumlog) matching the forward shapes
    (dcumlog = rowsum(M) - colsum(M), the gradient w.r.t. cumlog). D in {64,128}. Auto-routed like
    the forward (chunked linear-time above the measured crossovers); force_quadratic pins the
    O(N^2) route (both backends, for testing route agreement). Use mamba2_dcl_to_da to turn dcumlog
    into d(log a) / da. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(C):
        return _torch().mamba2_bwd(C, B, X, cumlog, dY, force_quadratic=force_quadratic)
    out = _mlx().mamba2_bwd(C, B, X, cumlog, dY, force_quadratic=force_quadratic)
    return out[0], out[1], out[2], out[3]


def mamba2_bwd_chunked(C, B, X, cumlog, dY):
    """Mamba-2 / SSD backward, forced chunked linear-time route (tests/benchmarks). N%64==0,
    N>=128, D in {64,128}. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(C):
        return _torch().mamba2_bwd_chunked(C, B, X, cumlog, dY)
    out = _mlx().mamba2_bwd_chunked(C, B, X, cumlog, dY)
    return out[0], out[1], out[2], out[3]


def ssd_decode(S, alpha, x, k, q):
    """Single-token SSD decode step: S' = alpha*S + x⊗k ; y = S'·q (readout after the write) —
    the O(D^2) generation step for mamba2 / lin_attn_decay (q=C_t, k=B_t, x=X_t; alpha=1 for
    undecayed linear attention). S (B,H,D,D) fp32, alpha (B,H), x/k/q (B,H,D); D in {64,128}.
    Returns (y, S'). On torch (MPS) the state is updated IN PLACE (S' is S); on MLX the update
    is functional (S' is a fresh array). Accepts mlx.array or torch.Tensor."""
    if _is_torch(S):
        return _torch().ssd_decode(S, alpha, x, k, q)
    out = _mlx().ssd_decode(S, alpha, x, k, q)
    return out[0], out[1]


def mamba2_dcl_to_da(dcumlog, a):
    """Convert the gradient w.r.t. cumlog into the gradient w.r.t. the decay a. Since
    cumlog = cumsum(log a) along the sequence, d(log a)_t = reverse_cumsum(dcumlog)_t and
    da = d(log a) / a. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(dcumlog):
        import torch
        dloga = torch.flip(torch.cumsum(torch.flip(dcumlog, [-1]), dim=-1), [-1])
        return dloga / a
    import mlx.core as mx
    rev = dcumlog[..., ::-1]
    dloga = mx.cumsum(rev, axis=-1)[..., ::-1]
    return dloga / a


def lin_attn_decay(q, k, v, slopes):
    """Decay / retention linear attention (RetNet / Lightning-Attention-2):
    out_i = sum_{j<=i} exp(-slope_h*(i-j)) * (q_i.k_j) * v_j. q,k,v (B,H,N,D) bf16, D=64; `slopes`
    is the per-head decay rate (H,). Builds the decay-log ramp cl=-slope*position on-device (the
    former numpy build ran on the host per call) and runs the retention kernel.
    Accepts mlx.array or torch.Tensor (MPS)."""
    import numpy as np
    B, H, N, _ = q.shape
    if _is_torch(q):
        import torch
        sl = torch.as_tensor(np.asarray(slopes), dtype=torch.float32, device=q.device).reshape(int(H))
        pos = torch.arange(int(N), dtype=torch.float32, device=q.device)
        cl = (-(sl[:, None] * pos[None, :])).unsqueeze(0).expand(int(B), int(H), int(N)).contiguous()
        return _torch().lin_attn_decay(q, k, v, cl)
    import mlx.core as mx
    sl = mx.array(np.asarray(slopes, np.float32)).reshape((int(H),))
    pos = mx.arange(int(N)).astype(mx.float32)
    # + zeros materializes the broadcast (the kernel needs a contiguous (B,H,N) buffer)
    cl = (-(sl[:, None] * pos[None, :]))[None] + mx.zeros((int(B), int(H), int(N)), dtype=mx.float32)
    return _mlx().lin_attn_decay(q, k, v, cl)


def based(q, k, v):
    """Based 2nd-order Taylor feature-map linear attention (causal):
    out_i = sum_{j<=i} (1 + x + x^2/2) * v_j, x = (q_i.k_j)/sqrt(D_QK). q,k (B,H,N,16); v (B,H,N,64)
    bf16 -> (B,H,N,64). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().based(q, k, v)
    return _mlx().based(q, k, v)


def attn_fwd_l(q, k, v, causal=False):
    """Flash-attention forward returning (o, L). o is (B,H,N,D) bf16; L is (B,H,N) fp32 — the
    log2-domain logsumexp per query row, needed by the backward. `causal` masks future positions.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_fwd_l(q, k, v, causal)
    return _mlx().attn_fwd_l(q, k, v, causal=causal)


def attn_bwd(q, k, v, o, do, L, causal=False):
    """FlashAttention-2 backward -> (dq, dk, dv). q,k,v,o,do are (B,H,N,D) bf16; L (B,H,N) fp32 from
    the forward (tk.attn_fwd_l). D in {64,128}, N%8==0. Accepts mlx.array or torch.Tensor (MPS)."""
    be = _torch() if _is_torch(q) else _mlx()
    delta = be.attn_bwd_prep(o, do)
    dq = be.attn_bwd_dq(q, k, v, do, L, delta, causal)
    dk, dv = be.attn_bwd_dkv(q, k, v, do, L, delta, causal)
    return dq, dk, dv


def cmplx_matmul(a, b):
    """Complex GEMM D=A@B; operands carry a leading size-2 (real,imag) axis: a (2,N,K),
    b (2,K,M) -> (2,N,M). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(a):
        return _torch().cmplx_matmul(a, b)
    return _mlx().cmplx_matmul(a, b)


def fftconv(x, fmat, twf, finv, twi, kf):
    """Monarch FFT convolution (N=S*S). Complex inputs with a leading size-2 (real,imag) axis:
    x (2,B,H,S,S), fmat/twf/finv/twi (2,S,S), kf (2,H,S,S) -> real (B,H,S,S).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().fftconv(x, fmat, twf, finv, twi, kf)
    return _mlx().fftconv(x, fmat, twf, finv, twi, kf)


def qgemm(wq, x, format="q8_0"):
    """Quantized GEMM (Marlin's method): out = dequantize(wq) @ x. wq is packed weight blocks
    (N, K//block_k, block_bytes) uint8; x is (K, M) float16 -> (N, M) float16.
    Routes batch-1 (M==1) to the qgemv decode path. Accepts mlx.array or torch.Tensor (MPS)."""
    if x.shape[-1] == 1:                       # batch-1 decode -> GEMV
        return qgemv(wq, x, format)
    if _is_torch(wq):
        return _torch().qgemm(wq, x, format)
    return _mlx().qgemm(wq, x, format=format)


def qgemm_direct(wq, x, format="q8_0"):
    """qgemm with dequant-direct-to-fragment (Marlin zero-shuffle, no threadgroup staging). MLX
    only (experimental perf variant of qgemm; same result). Falls back to qgemm on torch."""
    if _is_torch(wq):
        return _torch().qgemm(wq, x, format)
    return _mlx().qgemm_direct(wq, x, format=format)


def attn_q(q, kq, vq, format="q8_0", causal=False, multiwarp="auto"):
    """Quantized-KV flash attention: softmax(QK^T)·V with K,V given as quantized blocks (format).
    q bf16 (B,H,N,D); kq/vq uint8 (B,H,N,D/block_k,block_bytes) -> bf16 (B,H,N,D). D in {64,128}.
    multiwarp="auto" (default) uses the 4-warp variant whenever legal (non-causal, N%32==0) —
    it stages 4 KV tiles per barrier pair and measures ~2x faster than single-warp.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if multiwarp == "auto":
        multiwarp = (not causal) and q.shape[2] % 32 == 0
    if _is_torch(q):
        return _torch().attn_q(q, kq, vq, format, causal, multiwarp)
    return _mlx().attn_q(q, kq, vq, format=format, causal=causal, multiwarp=multiwarp)


def qgemm_actorder(wq, x, perm, w_format="kU4B8", fused=False):
    """GPTQ act-order (desc_act): the weight is quantized in g_idx-permuted column (K) order so its
    groups are contiguous; recover W@X by gathering the activation rows by the same permutation, then
    running the standard qgemm. `perm` is a length-K index array (= argsort(g_idx)). A load-time
    reordering layer, not a new format. `fused=True` (MLX/torch) instead gathers the X K-rows inside
    the kernel (qgemm_actorder_k) — no materialized permuted-X copy; needs M%32==0, N%32==0, x fp16.
    Accepts mlx.array or torch.Tensor (MPS)."""
    import numpy as np
    if fused:
        if _is_torch(x):
            import torch
            p = torch.as_tensor(np.asarray(perm), dtype=torch.int32, device=x.device)
            return _torch().qgemm_actorder_k(wq, x.to(torch.float16), p, w_format)
        import mlx.core as mx
        return _mlx().qgemm_actorder_k(wq, x.astype(mx.float16),
                                       mx.array(np.asarray(perm, np.int32)), format=w_format)
    if _is_torch(x):
        import torch
        idx = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=x.device)
        return qgemm(wq, x.index_select(0, idx), w_format)
    import mlx.core as mx
    return qgemm(wq, mx.take(x, mx.array(np.asarray(perm, np.int32)), axis=0), w_format)


def qgemm_w8a8(wq, xq, w_scale, a_scale):
    """W8A8 prefill GEMM (M>1, bit-exact int32): out[n,m]=w_scale[n]*a_scale[m]*sum_k Wq[n,k]*Xq[m,k].
    wq int8 (N,K); xq int8 (M,K) token-major; w_scale (N,) half; a_scale (M,) half -> (N,M) half.
    NOTE: int prefill is perf-negative on Apple (no int matmul); use for exact int32 numerics."""
    if _is_torch(wq):
        return _torch().qgemm_w8a8(wq, xq, w_scale, a_scale)
    return _mlx().qgemm_w8a8(wq, xq, w_scale, a_scale)


def qgemm_w2a8(wq, xq, a_scale):
    """BitNet W2A8 prefill GEMM (M>1): ternary 2-bit weight x int8 act (M,K), per-group absmean scale
    * a_scale[m] -> (N,M) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_w2a8(wq, xq, a_scale)
    return _mlx().qgemm_w2a8(wq, xq, a_scale)


def qgemm_fp8_block2d(wq, x, scale2d):
    """fp8_block2d GEMM: codes-only fp8 weights (N,K/128,128) + a separate (N/128,K/128) tile scale
    (storage-optimal fp8_block). x (K,M) f16 -> (N,M) f16. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_blockscale(wq, x, scale2d)
    return _mlx().qgemm_blockscale(wq, x, scale2d)


def qgemm_fp8_scaled(wq, xq, w_scale, a_scale):
    """fp8 rank-1 scaled GEMM: BOTH operands fp8 e4m3 codes (wq (N,K), xq (K,M)), per-channel w_scale (N,)
    and per-token a_scale (M,) f16 -> (N,M) f16. out[n,m]=w_scale[n]*a_scale[m]*sum_k dequant·dequant.
    The fp8 analog of W8A8/SmoothQuant. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_fp8_scaled(wq, xq, w_scale, a_scale)
    return _mlx().qgemm_fp8_scaled(wq, xq, w_scale, a_scale)


def qgemv_w8a8(wq, xq, w_scale, a_scale):
    """W8A8/SmoothQuant decode GEMV: int8 weight (N,K) x int8 act (K,1) -> int32, *w_scale[n]*a_scale.
    w_scale (N,) half, a_scale (1,) half -> (N,1) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv_w8a8(wq, xq, w_scale, a_scale)
    return _mlx().qgemv_w8a8(wq, xq, w_scale, a_scale)


def qgemv_w2a8(wq, xq, a_scale):
    """BitNet W2A8 decode GEMV: ternary 2-bit weight (bitnet blocks) x int8 act (K,1) -> int32,
    per-group absmean scale * a_scale -> (N,1) half. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv_w2a8(wq, xq, a_scale)
    return _mlx().qgemv_w2a8(wq, xq, a_scale)


def qgemv(wq, x, format="q8_0"):
    """Quantized GEMV (batch-1 decode): out = dequantize(wq) @ x. wq packed weight blocks
    (N, K//block_k, block_bytes) uint8; x is (K, 1) float16 -> (N, 1) float16.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qgemv(wq, x, format)
    return _mlx().qgemv(wq, x, format=format)


def qflux_gelu(wq, x, bias, format="q8_0"):
    """Quantized fused GEMM+GELU: gelu(dequantize(wq) @ x + bias). wq packed weight blocks;
    x (K,M) float16; bias (M,) float16 -> (N,M) float16. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(wq):
        return _torch().qflux_gelu(wq, x, bias, format)
    return _mlx().qflux_gelu(wq, x, bias, format=format)


def _round_activation(x, act):
    """Snap activations x (K,M) to the 8-bit grid (int8/fp8), returning a fp16 array of the same
    framework. On Apple there's no int8/fp8 matmul, so W·A8 = round activations then the half GEMM
    (parity numerics). Rounding is done in numpy (a parity tool, not a perf path)."""
    import numpy as np
    from .quant import ACT_FORMATS
    if act not in ACT_FORMATS:
        raise ValueError(f"act must be one of {list(ACT_FORMATS)} or None, got {act!r}")
    if _is_torch(x):
        import torch
        xr = ACT_FORMATS[act](x.detach().float().cpu().numpy())[0]
        return torch.from_numpy(xr).to(x.device, torch.float16)
    import mlx.core as mx
    xr = ACT_FORMATS[act](np.array(x.astype(mx.float32)))[0]
    return mx.array(xr).astype(mx.float16)


def qmm(wq, x, w_format="q8_0", act=None):
    """Quantized matmul = dequantize(wq) @ x. Weight quantized via `w_format`; if `act` is
    "int8"/"fp8" the activations are also quantized (W·A8 parity: fp8 W8A8, int8 W8A8, int8 W4A8),
    else they stay fp16 (W·A16). Routes batch-1 (M==1) to the GEMV decode path. wq (N,K/bk,bytes)
    uint8; x (K,M) -> (N,M) float16. Accepts mlx.array or torch.Tensor (MPS)."""
    if act is not None:
        xq = _round_activation(x, act)
    elif _is_torch(x):
        import torch
        xq = x.to(torch.float16)
    else:
        import mlx.core as mx
        xq = x.astype(mx.float16)
    return qgemm(wq, xq, w_format)
