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
    """LayerNorm backward. Returns (dx, dweight, dbias): dx has x's shape, dweight/dbias (D,) fp32
    summed over rows. x/dy (..., D), weight (D,). Matches torch autograd. A single fused kernel
    computes mean+rstd in-kernel, writes dX, and accumulates dweight/dbias via device atomics in one
    pass (measured ~2.5x faster than the old mean/rstd + dX + dweight/dbias hybrid, on par with
    mx.fast). Accepts mlx.array or torch.Tensor (MPS)."""
    D = x.shape[-1]
    if _is_torch(x):
        xf = x.reshape(-1, D).contiguous()
        dyf = dy.reshape(-1, D).contiguous()
        dx, dw, db = _torch().layernorm_bwd_fused(xf, weight, dyf, float(eps))
        return dx.reshape(x.shape), dw.to(weight.dtype), db.to(weight.dtype)
    import mlx.core as mx
    xf = mx.reshape(x, (-1, D))
    dyf = mx.reshape(dy, (-1, D))
    dx, dw, db = _mlx().layernorm_bwd_fused(xf, weight, dyf, float(eps))
    return mx.reshape(dx, x.shape), dw.astype(weight.dtype), db.astype(weight.dtype)


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


def attn_fwd(q, k, v, softcap=0.0, sinks=None):
    """Non-causal attention forward. softcap > 0 applies Gemma-style logit soft-capping
    (softcap*tanh(s/softcap)); sinks (H,) adds gpt-oss-style per-head attention-sink logits
    to the softmax denominator. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_fwd(q, k, v, softcap=softcap, sinks=sinks)
    return _mlx().attn_fwd(q, k, v, softcap=float(softcap), sinks=sinks)


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
    Any B (single-threadgroup chunked scan: each thread owns a contiguous batch chunk). Accepts
    mlx.array or torch.Tensor (MPS)."""
    if _is_torch(cu_seqlens):
        return tuple(_torch().varlen_build_worklist(cu_seqlens, int(max_tiles)))
    out = _mlx().varlen_build_worklist(cu_seqlens, int(max_tiles))
    return out[0], out[1], out[2], out[3], out[4]


def attn_varlen_prefill(q_packed, key_cache, value_cache, block_table, context_lens,
                        cu_seqlens_q, scale=0.0, softcap=0.0, sinks=None):
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
            q_hm, key_cache, value_cache, block_table, context_lens, ts, tl, sq, float(scale),
            softcap=softcap, sinks=sinks)
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
        q_hm, key_cache, value_cache, block_table, context_lens, ts, tl, sq, float(scale),
        softcap=float(softcap), sinks=sinks)
    o_pad = mx.transpose(o_hm, (1, 0, 2))                           # (total_padded, H, D)
    outs = [o_pad[int(pad_off[b]):int(pad_off[b]) + int(qlens[b])] for b in range(B)]
    return mx.concatenate(outs, axis=0)


def varlen_pad_q(q_packed, cu_seqlens, pad_off, total_padded):
    """Device varlen Q pad/gather: packed q (total_q, H, D) bf16 -> padded head-major q_hm
    (H, total_padded, D) via cu_seqlens + pad_off (pad rows zeroed). Accepts mlx / torch (MPS)."""
    if _is_torch(q_packed):
        return _torch().varlen_pad_q(q_packed, cu_seqlens, pad_off, int(total_padded))
    return _mlx().varlen_pad_q(q_packed, cu_seqlens, pad_off, int(total_padded))


def varlen_regather_o(o_hm, cu_seqlens, pad_off, total_q):
    """Device varlen output re-gather: head-major o_hm (H, total_padded, D) -> packed (total_q, H, D).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(o_hm):
        return _torch().varlen_regather_o(o_hm, cu_seqlens, pad_off, int(total_q))
    return _mlx().varlen_regather_o(o_hm, cu_seqlens, pad_off, int(total_q))


def attn_varlen_prefill_device(q_packed, key_cache, value_cache, block_table, context_lens,
                               cu_seqlens, max_tiles, scale=0.0, softcap=0.0, sinks=None):
    """FULLY DEVICE-RESIDENT varlen prefill: builds the tile worklist, pads/gathers Q into the
    head-major layout, runs the paged-prefill attention, and re-gathers the output back to packed —
    all on-device from a DEVICE cu_seqlens (B+1,) int, with only ONE scalar readback (total_padded,
    to size the padded buffer; Metal can't size a grid from device data). Same result as the host
    attn_varlen_prefill path, but no O(B) host loop / host pad+transpose. Any B (chunked worklist
    scan). q_packed (total_q, H, D) bf16. Accepts mlx.array or torch.Tensor (MPS)."""
    total_q, H, D = q_packed.shape
    if scale == 0.0:
        scale = 1.0 / (float(D) ** 0.5)
    qlens, pad_off, tile_seq, tile_local0, _n_tiles = varlen_build_worklist(cu_seqlens, max_tiles)
    total_padded = int(pad_off[-1])          # the one unavoidable scalar readback (padded length)
    q_hm = varlen_pad_q(q_packed, cu_seqlens, pad_off, total_padded)
    if _is_torch(q_packed):
        o_hm = _torch().attn_varlen_prefill(q_hm, key_cache, value_cache, block_table, context_lens,
                                            tile_seq, tile_local0, qlens, float(scale),
                                            softcap=softcap, sinks=sinks)
    else:
        o_hm = _mlx().attn_varlen_prefill(q_hm, key_cache, value_cache, block_table, context_lens,
                                          tile_seq, tile_local0, qlens, float(scale),
                                          softcap=float(softcap), sinks=sinks)
    return varlen_regather_o(o_hm, cu_seqlens, pad_off, total_q)


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
    """RMSNorm backward. Returns (dx, dweight): dx has x's shape, dweight (D,) fp32 is summed over all
    rows. x/dy (..., D), weight (D,). Matches torch autograd. A single fused kernel computes rstd
    in-kernel, writes dX, and accumulates dweight via a device atomic in ONE pass over x/dy/W —
    measured ~2.3x faster than the old rstd+dX+dweight hybrid and on par with mx.fast's fused VJP.
    Accepts mlx.array or torch.Tensor (MPS)."""
    D = x.shape[-1]
    if _is_torch(x):
        xf = x.reshape(-1, D).contiguous()
        dyf = dy.reshape(-1, D).contiguous()
        dx, dw = _torch().rms_norm_bwd_fused(xf, weight, dyf, float(eps))
        return dx.reshape(x.shape), dw.to(weight.dtype)
    import mlx.core as mx
    xf = mx.reshape(x, (-1, D))
    dyf = mx.reshape(dy, (-1, D))
    dx, dw = _mlx().rms_norm_bwd_fused(xf, weight, dyf, float(eps))
    return mx.reshape(dx, x.shape), dw.astype(weight.dtype)


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


def adamw(param, grad, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1):
    """AdamW optimizer step (decoupled weight decay). Given param/grad and fp32 moment state m, v and
    the 1-based step t, returns (param', m', v'):
        m' = b1 m + (1-b1) g;  v' = b2 v + (1-b2) g^2;  mhat = m'/(1-b1^t);  vhat = v'/(1-b2^t)
        param' = param - lr*( mhat/(sqrt(vhat)+eps) + wd*param )
    Matches torch.optim.AdamW. m, v must be float32. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(param):
        return _torch().adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step)
    out = _mlx().adamw(param, grad, m, v, float(lr), float(beta1), float(beta2), float(eps),
                       float(weight_decay), int(step))
    return out[0], out[1], out[2]


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


def kv_cache_gather_fp8(key_cache, value_cache, block_table, cu_seq_lens, k_scale, v_scale,
                        num_tokens, fmt=0):
    """fp8 KV gather + upconvert — the read path for a paged fp8 prefix cache. Reads e4m3
    (fmt=0) or e5m2 (fmt=1) codes and dequantizes to bf16 via code * scale[kv_head]; the
    per-kv_head scales round-trip with tk.kv_cache_scatter_fp8. Caches are uint8
    (num_blocks, block_size, num_kv_heads, head_size); k_scale/v_scale are (num_kv_heads,).
    Returns (key_out, value_out) bf16 (num_tokens, num_kv_heads, head_size). Accepts
    mlx.array or torch.Tensor (MPS)."""
    if _is_torch(key_cache):
        return _torch().kv_cache_gather_fp8(key_cache, value_cache, block_table, cu_seq_lens,
                                            k_scale, v_scale, num_tokens, fmt)
    return _mlx().kv_cache_gather_fp8(key_cache, value_cache, block_table, cu_seq_lens,
                                      k_scale, v_scale, int(num_tokens), int(fmt))


def kv_cache_scale_update(key, value, old_key_scale, old_value_scale):
    """Incremental per-tensor KV scale update (running max): new = max(old, absmax/240) — the
    streaming-decode analogue of kv_cache_scales. Returns (new_key_scale, new_value_scale)
    (1,) f32. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(key):
        return _torch().kv_cache_scale_update(key, value, old_key_scale, old_value_scale)
    return _mlx().kv_cache_scale_update(key, value, old_key_scale, old_value_scale)


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
                       scale=0.0, partition_size=256, window=0, softcap=0.0, sinks=None):
    """Long-context paged decode attention (partition/reduce). GQA/MQA aware.

    q/out (B,H,D); caches (num_blocks, block_size, num_kv_heads, D). Accepts
    mlx.array or torch.Tensor (MPS). partition_size must be a multiple of block_size.
    window > 0 restricts to the `window` most recent keys. softcap > 0 applies Gemma-style
    logit capping (in the partitions); sinks (H,) adds gpt-oss attention-sink logits to the
    softmax denominator (merged exactly once, in the reduce).
    """
    if _is_torch(q):
        return _torch().paged_attention_v2(
            q, key_cache, value_cache, block_table, context_lens, scale, partition_size, window,
            softcap=softcap, sinks=sinks)
    return _mlx().paged_attention_v2(
        q, key_cache, value_cache, block_table, context_lens,
        scale=scale, partition_size=partition_size, window=window,
        softcap=float(softcap), sinks=sinks)


def cascade_attention(q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
                      scale=0.0, partition_size=256):
    """Cascade / shared-prefix attention: all B requests attend a single SHARED contiguous prefix KV
    (prefix_k/prefix_v (prefix_len, H_KV, D)) plus their own paged suffix (key_cache/value_cache +
    block_table + context_lens). The two levels' softmax states are merged via the shared
    log-sum-exp reduce == full attention over [prefix ++ suffix] per request. Amortizes the shared
    system prompt across a batch. q/out (B,H,D); D in {64,128}; GQA/MQA aware. partition_size must be
    a multiple of block_size.

    N-level cascade: pass prefix_k/prefix_v as a LIST of levels ([(prefix_len_i, H_KV, D), ...]) — all
    levels' + the suffix's partials are concatenated and merged in one reduce == full attention over
    [level0 ++ ... ++ suffix]. Accepts mlx.array or torch.Tensor (MPS)."""
    if isinstance(prefix_k, (list, tuple)):
        if _is_torch(q):
            return _torch().cascade_attention_multi(
                q, list(prefix_k), list(prefix_v), key_cache, value_cache, block_table,
                context_lens, scale, partition_size)
        return _mlx().cascade_attention_multi(
            q, list(prefix_k), list(prefix_v), key_cache, value_cache, block_table, context_lens,
            scale=scale, partition_size=partition_size)
    if _is_torch(q):
        return _torch().cascade_attention(
            q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
            scale, partition_size)
    return _mlx().cascade_attention(
        q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
        scale=scale, partition_size=partition_size)


def cascade_attention_fp8(q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
                          k_scale, v_scale, scale=0.0, partition_size=256, fmt="e4m3"):
    """Cascade attention over a uint8 fp8 (e4m3/e5m2) SHARED prefix (per-kv-head dequant on read) +
    the regular (bf16/fp16) paged suffix, merged in one reduce. Fp8-compresses the large shared
    system prompt while keeping the suffix in the native cache dtype. prefix_k/prefix_v uint8;
    k_scale/v_scale a plain float (per-tensor, broadcast) or a (num_kv_heads,) array (per-head).
    fmt 'e4m3' (default) or 'e5m2'. Accepts mlx.array or torch.Tensor (MPS)."""
    H_KV = key_cache.shape[2]
    k_scale, v_scale = _scale_vec(k_scale, H_KV, q), _scale_vec(v_scale, H_KV, q)
    if _is_torch(q):
        return _torch().cascade_attention_fp8(q, prefix_k, prefix_v, key_cache, value_cache,
                                              block_table, context_lens, k_scale, v_scale, scale,
                                              partition_size, _fmt_code(fmt))
    return _mlx().cascade_attention_fp8(q, prefix_k, prefix_v, key_cache, value_cache, block_table,
                                        context_lens, k_scale, v_scale, scale=scale,
                                        partition_size=partition_size, fmt=_fmt_code(fmt))


def paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                           k_scale, v_scale, scale=0.0, partition_size=256, fmt="e4m3", window=0,
                           softcap=0.0, sinks=None):
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
            scale, partition_size, fmt, window, softcap=softcap, sinks=sinks)
    return _mlx().paged_attention_v2_fp8(
        q, key_cache, value_cache, block_table, context_lens, k_scale, v_scale,
        scale=scale, partition_size=partition_size, fmt=_fmt_code(fmt), window=window,
        softcap=float(softcap), sinks=sinks)


def moe_route_topk(logits, k):
    """MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).

    logits (num_tokens, num_experts); k <= min(16, num_experts). Accepts mlx.array or torch.Tensor.
    """
    if _is_torch(logits):
        return _torch().moe_route_topk(logits, k)
    return _mlx().moe_route_topk(logits, k)



def tau_tail(qkv, tok_qv_lin, tau_pos_table, positions, n_heads, head_dim):
    """tau_tail tail scaling: multiply the Q and V slices of a packed (T, 3*q_dim) QKV by
    tanh(tok_qv_lin[:, head]) + tau_pos_table[positions, head] (Q uses the first n_heads gate
    columns, V the next n_heads); the K slice passes through. tok_qv_lin (T, 2*n_heads),
    tau_pos_table (max_pos, n_heads), positions (T,) int. Returns a new QKV (functional).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(qkv):
        return _torch().tau_tail(qkv, tok_qv_lin, tau_pos_table, positions, n_heads, head_dim)
    return _mlx().tau_tail(qkv, tok_qv_lin, tau_pos_table, positions, int(n_heads),
                           int(head_dim))


def packbits(x, bit_order_big=True):
    """Pack a bool/uint8 array (row-major flattened) into bits, np.packbits semantics
    (bit_order_big=True matches numpy's default). Returns uint8 (ceil(N/8),).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().packbits(x, bit_order_big)
    return _mlx().packbits(x, bool(bit_order_big))


def segment_packbits(x, input_indptr, output_indptr, total_output_bytes, bit_order_big=True):
    """Ragged per-row packbits: pack each segment [input_indptr[i], input_indptr[i+1]) of the
    flat uint8 input into output_indptr[i] onward. output_indptr = host cumsum of
    ceil(seg_len/8); total_output_bytes = output_indptr[-1]. Accepts mlx/torch (MPS)."""
    if _is_torch(x):
        return _torch().segment_packbits(x, input_indptr, output_indptr, total_output_bytes,
                                         bit_order_big)
    return _mlx().segment_packbits(x, input_indptr, output_indptr, int(total_output_bytes),
                                   bool(bit_order_big))


def permute_cols(x, perm):
    """Column gather output[:, c] = x[:, perm[c]] on a 16-bit dtype (f16/bf16/int16/uint16 —
    Marlin's act-order weight repermutation). x (rows, cols), perm (cols,) int.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().permute_cols(x, perm)
    return _mlx().permute_cols(x, perm)


def tq_encode(key, value, key_cache, value_cache, key_scale, value_scale, key_zero,
              slot_mapping, v_centroids, signs, block_size, k_bits, k_signed, v_bits):
    """TurboQuant KV-cache encode: K asymmetric-uniform (per-32 fp16 scale+zp, 2-8 bits);
    V random-sign FWHT rotation -> per-32 fp16 RMS scale -> Lloyd-Max nearest-centroid
    (2/3/4/8 bits). key/value (tokens, num_kv_heads, head_size in {64,128,256}); caches
    paged (num_blocks, block_size, num_kv_heads, packed/scale-groups); v_centroids
    (2^v_bits,) and signs (head_size,) from tk.quant.lloyd_max_centroids / tq_signs.
    Returns the 5 updated cache arrays (functional). Accepts mlx.array or torch (MPS)."""
    if _is_torch(key):
        return _torch().tq_encode(key, value, key_cache, value_cache, key_scale, value_scale,
                                  key_zero, slot_mapping, v_centroids, signs, block_size,
                                  k_bits, k_signed, v_bits)
    return list(_mlx().tq_encode(key, value, key_cache, value_cache, key_scale, value_scale,
                                 key_zero, slot_mapping, v_centroids, signs, int(block_size),
                                 int(k_bits), bool(k_signed), int(v_bits)))


def tq_decode(key_cache, value_cache, key_scale, value_scale, key_zero, slots, v_centroids,
              signs, num_kv_heads, head_size, block_size, k_bits, k_signed, v_bits):
    """Inverse of tq_encode: gather a slot list and dequantize (V inverse-FWHT'd) to
    [k_out, v_out] float32 (n, num_kv_heads, head_size). Accepts mlx.array or torch (MPS)."""
    if _is_torch(key_cache):
        return _torch().tq_decode(key_cache, value_cache, key_scale, value_scale, key_zero,
                                  slots, v_centroids, signs, num_kv_heads, head_size,
                                  block_size, k_bits, k_signed, v_bits)
    return list(_mlx().tq_decode(key_cache, value_cache, key_scale, value_scale, key_zero,
                                 slots, v_centroids, signs, int(num_kv_heads), int(head_size),
                                 int(block_size), int(k_bits), bool(k_signed), int(v_bits)))


def indexer_k_quant_and_cache(k, slot_mapping, code_cache, scale_cache, quant_block_size=128,
                             ue8m0=False):
    """DeepSeek-V3.2 (DSA/NSA) indexer K quant-and-cache: quantize the indexer K per
    quant_block_size (canonical 128) into an e4m3 code cache (num_slots, head_dim) + fp32 scale
    cache (num_slots, head_dim/qbs) that the sparse-attention top-k selector reads cheaply.
    k (tokens, head_dim); slot_mapping (tokens,) int (<0 skips); ue8m0 rounds scales to powers
    of two. Functional — returns (code_cache, scale_cache), untouched slots preserved. Accepts
    mlx.array or torch.Tensor (MPS)."""
    if _is_torch(k):
        return _torch().indexer_k_quant_and_cache(k, slot_mapping, code_cache, scale_cache,
                                                  quant_block_size, ue8m0)
    out = _mlx().indexer_k_quant_and_cache(k, slot_mapping, code_cache, scale_cache,
                                           int(quant_block_size), bool(ue8m0))
    return out[0], out[1]


def indexer_k_gather(code_cache, scale_cache, slots, head_dim, quant_block_size=128):
    """Gather + dequantize the indexer cache back to bf16 K for a slot list: k_out[row] =
    decode(code_cache[slot]) * scale_cache[slot, qblock]. Returns bf16 (n, head_dim). Accepts
    mlx.array or torch.Tensor (MPS)."""
    if _is_torch(code_cache):
        return _torch().indexer_k_gather(code_cache, scale_cache, slots, head_dim, quant_block_size)
    return _mlx().indexer_k_gather(code_cache, scale_cache, slots, int(head_dim),
                                   int(quant_block_size))


def minference_block_mask(vertical_indexes, slash_indexes, context_lens, max_blocks,
                          block_size, vertical_topk=1 << 30, slash_topk=1 << 30,
                          last_n_blocks=1):
    """MInference decode block-mask builder: per-head vertical column indexes + slash
    diagonal offsets ((B, H, nnz) int32, -1 pad) -> per-head KV block mask
    (B, H, max_blocks) int32 0/1 consumed directly by tk.paged_attention_block_sparse
    (which accepts 2-D per-batch or 3-D per-head masks). vertical/slash_topk cap how many
    index entries are used; last_n_blocks recent blocks are always attended.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(vertical_indexes):
        return _torch().minference_block_mask(vertical_indexes, slash_indexes, context_lens,
                                              max_blocks, block_size, vertical_topk,
                                              slash_topk, last_n_blocks)
    return _mlx().minference_block_mask(vertical_indexes, slash_indexes, context_lens,
                                        int(max_blocks), int(block_size), int(vertical_topk),
                                        int(slash_topk), int(last_n_blocks))


def quadratic_transform(logits, factor, curve=1.0, temperature=1.0):
    """Quadratic / smoothing sampling transform: diff = l - max; diff -= diff^2(s*diff - k)
    with k = factor(3-curve)/2, s = factor(curve-1)/2; factor 0 = identity. Writes tempered
    logits. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().quadratic_transform(logits, factor, curve, temperature)
    return _mlx().quadratic_transform(logits, float(factor), float(curve), float(temperature))


def top_nsigma_mask(logits, nsigma, temperature=1.0):
    """Top-nsigma sampling: mask logits below max - nsigma * stddev (finite logits assumed —
    compose BEFORE other -inf masks). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().top_nsigma_mask(logits, nsigma, temperature)
    return _mlx().top_nsigma_mask(logits, float(nsigma), float(temperature))


def top_a_mask(logits, top_a, temperature=1.0):
    """Top-A sampling: mask tokens with prob < top_a * pmax^2 (computed in log space, no
    softmax materialization; the argmax always survives). Accepts mlx/torch (MPS)."""
    if _is_torch(logits):
        return _torch().top_a_mask(logits, top_a, temperature)
    return _mlx().top_a_mask(logits, float(top_a), float(temperature))


def epsilon_cutoff_mask(logits, epsilon, temperature=1.0):
    """Epsilon cutoff: mask tokens with prob < epsilon; the argmax (and exact ties) always
    survive. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().epsilon_cutoff_mask(logits, epsilon, temperature)
    return _mlx().epsilon_cutoff_mask(logits, float(epsilon), float(temperature))


def eta_cutoff_mask(logits, eta, temperature=1.0):
    """Eta sampling: mask tokens with prob < min(eta, sqrt(eta) * exp(-entropy)).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().eta_cutoff_mask(logits, eta, temperature)
    return _mlx().eta_cutoff_mask(logits, float(eta), float(temperature))


def xtc_mask(logits, threshold, probability, seed=0, temperature=1.0):
    """XTC (exclude top choices): with an on-device per-row coin < probability, remove every
    token with prob >= threshold EXCEPT the least likely such token (keeps output diverse
    without touching the tail). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().xtc_mask(logits, threshold, probability, seed, temperature)
    return _mlx().xtc_mask(logits, float(threshold), float(probability), int(seed),
                           float(temperature))


def skew_transform(probs, skew):
    """Skew sampling over PROBS: out_i = pow(cdf_i, exp(skew)) - pow(cdf_{i-1}, exp(skew)) on
    the index-order CDF (metal-forge contract; diverges from exllamav2's sorted-CDF skew,
    which needs a sort — deferred). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(probs):
        return _torch().skew_transform(probs, skew)
    return _mlx().skew_transform(probs, float(skew))


def top_k_renorm(probs, k):
    """Keep the top-k probabilities (ties -> smaller id), renormalize to sum 1, zero the rest
    (spec-decode distribution utility; k <= 64). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(probs):
        return _torch().top_k_renorm(probs, k)
    return _mlx().top_k_renorm(probs, int(k))


def top_p_renorm(probs, p):
    """Keep the smallest set of probabilities with mass >= p (32-iter threshold bisection —
    no sort), renormalize to 1, zero the rest. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(probs):
        return _torch().top_p_renorm(probs, p)
    return _mlx().top_p_renorm(probs, float(p))


def no_repeat_ngram_mask(logits, prev_tokens, lens, ngram_size, temperature=1.0):
    """Ban every token that would complete an already-seen ngram_size-gram of the history
    (prev_tokens (rows, L) int32 + lens (rows,)); n >= 2. Accepts mlx/torch (MPS)."""
    if _is_torch(logits):
        return _torch().no_repeat_ngram_mask(logits, prev_tokens, lens, ngram_size, temperature)
    return _mlx().no_repeat_ngram_mask(logits, prev_tokens, lens, int(ngram_size),
                                       float(temperature))


def dry_penalty(logits, prev_tokens, lens, breakers, multiplier, base=1.75,
                allowed_length=2, range=0, max_ngram=64, max_occurrences=64,
                early_exit_match_len=64, temperature=1.0):
    """DRY ("don't repeat yourself") penalty: for each earlier occurrence of the last token
    whose preceding context matches the current suffix (length match_len, reset by
    sequence breakers), penalize the token that followed it by
    multiplier * base^(match_len+1 - allowed_length), min'd into the logit. breakers is a
    single (NB,) int32 list (pad -1), shared across rows (TM scalar-param convention).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().dry_penalty(logits, prev_tokens, lens, breakers, multiplier, base,
                                    allowed_length, range, max_ngram, max_occurrences,
                                    early_exit_match_len, temperature)
    return _mlx().dry_penalty(logits, prev_tokens, lens, breakers, float(multiplier),
                              float(base), int(allowed_length), int(range), int(max_ngram),
                              int(max_occurrences), int(early_exit_match_len),
                              float(temperature))


def rms_norm_add_int8(x, residual, weight, eps=1e-5):
    """Fused residual-add + RMSNorm + dynamic per-row int8 (the W8A8 residual-stream epilogue;
    int8 sibling of rms_norm_add_fp8 dynamic). Returns (codes i8, x+residual, scale (rows,)).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rms_norm_add_int8(x, residual, weight, eps)
    out = _mlx().rms_norm_add_int8_dyn(x, residual, weight, float(eps))
    return out[0], out[1], out[2]


def layernorm_add_int8(x, residual, weight, bias, eps=1e-5):
    """Fused residual-add + LayerNorm + dynamic per-row int8 (int8 sibling of layernorm_add_fp8
    dynamic). Returns (codes i8, x+residual, scale (rows,)). Accepts mlx.array or torch (MPS)."""
    if _is_torch(x):
        return _torch().layernorm_add_int8(x, residual, weight, bias, eps)
    out = _mlx().layernorm_add_int8_dyn(x, residual, weight, bias, float(eps))
    return out[0], out[1], out[2]


def rms_norm_add_per_block(x, residual, weight, eps=1e-5, int8=False, ue8m0=False):
    """Fused residual-add + RMSNorm + per-128-block dynamic quant — emits (rows, D/128) group
    scales directly, so the codes feed the block-quant expert GEMMs (moe_grouped_gemm_*_q) with
    no separate quantize pass. int8=False -> fp8 e4m3 (ue8m0 rounds group scales to powers of
    two); int8=True -> symmetric int8. Returns (codes, x+residual, scale (rows, D/128)). D%128==0.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().rms_norm_add_per_block(x, residual, weight, eps, int8, ue8m0)
    out = (_mlx().rms_norm_add_per_block_int8(x, residual, weight, float(eps)) if int8
           else _mlx().rms_norm_add_per_block_fp8(x, residual, weight, float(eps), ue8m0=ue8m0))
    return out[0], out[1], out[2]


def layernorm_add_per_block(x, residual, weight, bias, eps=1e-5, int8=False, ue8m0=False):
    """Fused residual-add + LayerNorm + per-128-block dynamic quant (the LayerNorm sibling of
    rms_norm_add_per_block). Returns (codes, x+residual, scale (rows, D/128)). D%128==0.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().layernorm_add_per_block(x, residual, weight, bias, eps, int8, ue8m0)
    out = (_mlx().layernorm_add_per_block_int8(x, residual, weight, bias, float(eps)) if int8
           else _mlx().layernorm_add_per_block_fp8(x, residual, weight, bias, float(eps),
                                                   ue8m0=ue8m0))
    return out[0], out[1], out[2]


def silu_mul_quant_fp8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated activation -> dynamic per-token fp8 e4m3: act = silu(x)*gate ("swiglu") or
    the gpt-oss clamped variant ("swiglu_oai", alpha/limit); codes = e4m3(act/scale) with
    scale = rowmax|act|/448 — the (rows, D) bf16 intermediate never hits memory. Feeds
    qgemm_fp8_scaled. Returns (codes u8, scale (rows,)). Accepts mlx.array or torch (MPS)."""
    mode = {"swiglu": 0, "swiglu_oai": 1}[act]
    if _is_torch(x):
        return _torch().silu_mul_quant_fp8(x, gate, act=act, alpha=alpha, limit=limit)
    out = _mlx().silu_mul_quant_fp8(x, gate, mode, float(alpha), float(limit))
    return out[0], out[1]


def silu_mul_quant_int8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated activation -> dynamic per-token symmetric int8 (feeds qgemm_w8a8).
    Returns (codes i8, scale (rows,)). Accepts mlx.array or torch.Tensor (MPS)."""
    mode = {"swiglu": 0, "swiglu_oai": 1}[act]
    if _is_torch(x):
        return _torch().silu_mul_quant_int8(x, gate, act=act, alpha=alpha, limit=limit)
    out = _mlx().silu_mul_quant_int8(x, gate, mode, float(alpha), float(limit))
    return out[0], out[1]


def silu_mul_quant_fp8_group(x, gate, group_size=128, ue8m0=False, act="swiglu",
                             alpha=1.702, limit=7.0):
    """Fused gated activation -> per-group fp8 (canonical group 128, the block-quant GEMM
    activation layout; ue8m0 rounds scales up to powers of two). Returns
    (codes u8, scale (rows, D/G)). Accepts mlx.array or torch.Tensor (MPS)."""
    mode = {"swiglu": 0, "swiglu_oai": 1}[act]
    if _is_torch(x):
        return _torch().silu_mul_quant_fp8_group(x, gate, group_size=group_size, ue8m0=ue8m0,
                                                 act=act, alpha=alpha, limit=limit)
    out = _mlx().silu_mul_quant_fp8_group(x, gate, group_size, bool(ue8m0), mode,
                                          float(alpha), float(limit))
    return out[0], out[1]


def quantize_per_group_fp8(x, group_size=128, ue8m0=False):
    """Per-group dynamic fp8 e4m3 along the last axis (canonical group 128 — the activation
    side of block-quantized GEMMs). Returns (codes u8, scale (rows, D/G) f32); ue8m0 rounds
    scales up to powers of two (MX convention). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().quantize_per_group_fp8(x, group_size, ue8m0)
    out = _mlx().quantize_per_group_fp8(x, group_size=group_size, ue8m0=ue8m0)
    return out[0], out[1]


def quantize_per_group_int8(x, group_size=128):
    """Per-group dynamic symmetric int8. Returns (codes i8, scale (rows, D/G) f32).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().quantize_per_group_int8(x, group_size)
    out = _mlx().quantize_per_group_int8(x, group_size=group_size)
    return out[0], out[1]


def quantize_per_token_int8_azp(x):
    """ASYMMETRIC per-token int8 (vLLM azp): scale=(max-min)/255, azp=rint(-128-min/scale),
    q=clamp(rint(x/scale)+azp). Reconstruct scale*(q-azp). Returns (codes, scale, azp).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(x):
        return _torch().quantize_per_token_int8_azp(x)
    out = _mlx().quantize_per_token_int8_azp(x)
    return out[0], out[1], out[2]


def qgemm_w8a8_azp(wq, xq, w_scale, a_scale, w_rowsum, azp):
    """azp-corrected W8A8 GEMM: y[n,m] = s_w[n]*s_a[m]*(sum_k W*Xq - azp[m]*w_rowsum[n]).
    w_rowsum = W.sum(axis=1) int32, host-precomputed. Accepts mlx.array or torch (MPS)."""
    if _is_torch(wq):
        return _torch().qgemm_w8a8_azp(wq, xq, w_scale, a_scale, w_rowsum, azp)
    return _mlx().qgemm_w8a8_azp(wq, xq, w_scale, a_scale, w_rowsum, azp)


def gdn_recur(q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, load_initial=True):
    """GDN / GatedDeltaNet linear attention (the Qwen3-Next / Kimi-Linear hybrid mixer):
    per-timestep delta rule S = g*S + k*beta*(v - k.S); y = q.S, over varlen packed
    sequences. q/k (total_tokens, Hk, Dk in {64,128}); v (total_tokens, Hv, Dv);
    g/beta (total_tokens, Hv) with g the decay MULTIPLIER (not log); cu_seqlens (R+1,);
    state_pool (num_slots, Hv, Dv, Dk) fp32 indexed by slot_mapping (R,). load_initial:
    continue from the pool (decode) vs fresh S=0 (prefill). Returns (y, new_state_pool) —
    functional, untouched slots preserved. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().gdn_recur(q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping,
                                  load_initial=load_initial)
    out = _mlx().gdn_recur(q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping,
                           load_initial)
    return out[0], out[1]


def selective_scan(u, delta, A, B, C, state, D=None, delta_bias=None, z=None,
                   delta_softplus=True):
    """Mamba-1 (S6) selective scan forward, dense batch. Channel-major layouts:
    u/delta/z/out (batch, dim, seqlen); B/C (batch, n_groups, dstate, seqlen); A (dim, dstate)
    fp32 (A < 0); D/delta_bias (dim,) fp32 optional; state (batch, dim, dstate) fp32.
    Returns (out, new_state) — functional (the input state is not mutated). dstate <= 256.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(u):
        return _torch().selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, z=z,
                                       state=state, delta_softplus=delta_softplus)
    out = _mlx().selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, z=z,
                                state=state, delta_softplus=delta_softplus)
    return out[0], out[1]


def selective_scan_varlen(u, delta, A, B, C, query_start_loc, state, D=None, delta_bias=None,
                          z=None, cache_indices=None, has_initial_state=None,
                          delta_softplus=True, null_block_id=-1):
    """Varlen Mamba-1 scan over a flattened token axis with a per-request paged state pool:
    u/delta/z/out (dim, total_tokens); B/C (n_groups, dstate, total_tokens);
    query_start_loc (B+1,) int32; state (num_slots, dim, dstate) fp32 indexed per request by
    cache_indices (optional; identity when absent; == null_block_id skips the request);
    has_initial_state (B,) uint8 optional (fresh prefill starts at h = 0). Returns
    (out, new_state_pool) with untouched slots preserved. Accepts mlx.array or torch (MPS)."""
    if _is_torch(u):
        return _torch().selective_scan_varlen(u, delta, A, B, C, query_start_loc, state, D=D,
                                              delta_bias=delta_bias, z=z,
                                              cache_indices=cache_indices,
                                              has_initial_state=has_initial_state,
                                              delta_softplus=delta_softplus,
                                              null_block_id=null_block_id)
    out = _mlx().selective_scan_varlen(u, delta, A, B, C, D=D, delta_bias=delta_bias, z=z,
                                       query_start_loc=query_start_loc,
                                       cache_indices=cache_indices,
                                       has_initial_state=has_initial_state, state=state,
                                       delta_softplus=delta_softplus,
                                       null_block_id=null_block_id)
    return out[0], out[1]


def selective_scan_varlen_apc(u, delta, A, B, C, query_start_loc, cache_indices,
                              has_initial_state, state, block_idx_first_scheduled_token,
                              block_idx_last_scheduled_token, initial_state_idx,
                              cu_chunk_seqlen, last_chunk_indices, block_size,
                              cache_indices_stride, use_chunk_metadata, D=None,
                              delta_bias=None, z=None, delta_softplus=True, null_block_id=-1):
    """Varlen Mamba-1 scan with automatic prefix caching (the vLLM mamba paged-scan path):
    same S6 recurrence as selective_scan_varlen, but the running state is checkpointed into
    the paged state pool at chunk boundaries and the initial state is read from a
    (possibly cached) prefix block indexed by initial_state_idx. cache_indices is
    (B, cache_indices_stride) int32 mapping (request, block) -> pool slot; the chunk-metadata
    arrays (block_idx_first/last_scheduled_token (B,), cu_chunk_seqlen, last_chunk_indices)
    describe the logical chunk boundaries — pass use_chunk_metadata=False to chunk uniformly
    by block_size. Returns (out, new_state_pool). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(u):
        return _torch().selective_scan_varlen_apc(
            u, delta, A, B, C, query_start_loc, cache_indices, has_initial_state, state,
            block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx,
            cu_chunk_seqlen, last_chunk_indices, block_size, cache_indices_stride,
            use_chunk_metadata, D=D, delta_bias=delta_bias, z=z, delta_softplus=delta_softplus,
            null_block_id=null_block_id)
    out = _mlx().selective_scan_varlen_apc(
        u, delta, A, B, C, query_start_loc, cache_indices, has_initial_state, state,
        block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx,
        cu_chunk_seqlen, last_chunk_indices, int(block_size), int(cache_indices_stride),
        bool(use_chunk_metadata), D=D, delta_bias=delta_bias, z=z,
        delta_softplus=delta_softplus, null_block_id=null_block_id)
    return out[0], out[1]


def qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions, num_heads_q, num_heads_k,
                 num_heads_v, eps=1e-6, interleaved=False, gemma=False):
    """Fused per-head QK-RMSNorm + RoPE over a packed QKV buffer (the Qwen3 attention-prep
    pattern): every Q/K head is RMSNormed over its head_dim (q_weight/k_weight (D,)) then
    rotated at positions[token]; V heads are copied through. qkv (T, (Hq+Hk+Hv)*D) bf16;
    cos/sin (max_pos, D/2). interleaved: False = NeoX split-half, True = GPT-J pairs;
    gemma weights by (1+w). Full rotary only; D in {64,128,256}. Returns the new qkv.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(qkv):
        return _torch().qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                                     num_heads_q, num_heads_k, num_heads_v,
                                     eps=eps, interleaved=interleaved, gemma=gemma)
    return _mlx().qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                               num_heads_q, num_heads_k, num_heads_v,
                               eps=float(eps), interleaved=bool(interleaved),
                               gemma=bool(gemma))


def moe_route_grouped(logits, k, n_group, topk_group, bias=None, renormalize=True,
                      routed_scaling_factor=1.0, scoring="sigmoid"):
    """DeepSeek-style grouped (node-limited) MoE routing (HF noaux_tc semantics).

    score = scoring(logit) ("softmax" | "sigmoid" | "softplus_sqrt"); selection uses
    score + bias[e] (the e_score_correction_bias; optional); each group is ranked by the sum
    of its top-2 biased scores and only the best `topk_group` groups keep their experts; the
    emitted weight is the UNBIASED score, renormalized over the selected set when
    `renormalize`, times routed_scaling_factor. Returns (ids int32, weights f32), (T, k).
    E <= 512, E % n_group == 0, n_group <= 32, k <= 16. DeepSeek-V3: E=256, n_group=8,
    topk_group=4, k=8, scoring="sigmoid". Output contract == moe_route_topk (feeds
    moe_permute / moe_mlp unchanged). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().moe_route_grouped(logits, k, n_group, topk_group, bias=bias,
                                          renormalize=renormalize,
                                          routed_scaling_factor=routed_scaling_factor,
                                          scoring=scoring)
    import mlx.core as mx
    sf = {"softmax": 0, "sigmoid": 1, "softplus_sqrt": 2}[scoring]
    has_bias = bias is not None
    if bias is None:
        bias = mx.zeros((1,), dtype=mx.float32)
    return _mlx().moe_route_grouped(logits, bias, has_bias, int(k), int(n_group),
                                    int(topk_group), bool(renormalize),
                                    float(routed_scaling_factor), sf)


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


def moe_mlp(x, router_logits, W1, W2, k, quant_format=None, w1_bias=None, w2_bias=None,
            act="swiglu", alpha=1.702, limit=7.0):
    """End-to-end MoE MLP: route → permute → pad-schedule → gather → SwiGLU GEMM →
    down-proj GEMM → weighted combine. All-GPU, no host sync.

    Dense path (quant_format=None): W1 (E, H, 2*I) laid out [gate | up]; W2 (E, I, H).
    Quantized path: W1/W2 are packed uint8 expert stacks from tk.quant.quantize_expert_stack
    — W1 (E, 2*I, row_bytes over K=H), W2 (E, H, row_bytes over K=I) — with optional
    pre-activation w1_bias (E, 2*I) and down-proj w2_bias (E, H); act "swiglu" or
    "swiglu_oai" (gpt-oss). x (T, H); router_logits (T, E); k experts per token.
    Returns (T, H) in x's dtype. Accepts mlx.array or torch.Tensor (MPS).
    """
    topk_ids, topk_weights = moe_route_topk(router_logits, k)
    sorted_row_idx, offsets, _inv_idx = moe_permute(topk_ids, W1.shape[0])
    expert_of_tile, gather_idx, inv_pad, _off_pad = moe_pad_schedule(sorted_row_idx, offsets, k)
    permuted = moe_gather(x, gather_idx)
    if quant_format is None:
        inter = moe_grouped_gemm_swiglu(permuted, W1, expert_of_tile)
        expert_out = moe_grouped_gemm_rect(inter, W2, expert_of_tile)
    else:
        inter = moe_grouped_gemm_swiglu_q(permuted, W1, expert_of_tile, format=quant_format,
                                          bias=w1_bias, act=act, alpha=alpha, limit=limit)
        expert_out = moe_grouped_gemm_rect_q(inter, W2, expert_of_tile, format=quant_format,
                                             bias=w2_bias)
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


def moe_grouped_gemm_rect_q(A, Wq, expert_of_tile, format="mxfp4", bias=None):
    """Quantized grouped expert GEMM: out(rows, N_out) = A @ dequant(Wq[e])^T [+ bias[e]].

    A (rows, K_dim) bfloat16; Wq (E, N_out, row_bytes) uint8 packed by
    tk.quant.quantize_expert_stack (quant groups along K_dim); bias optional (E, N_out).
    format in {mxfp4, kU4, fp8_e4m3, q8_0, nvfp4, q4_K}. rows%32, K_dim%32 (and %block_k),
    N_out%32. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(A):
        return _torch().moe_grouped_gemm_rect_q(A, Wq, expert_of_tile, format=format, bias=bias)
    import mlx.core as mx
    has_bias = bias is not None
    if bias is None:
        bias = mx.zeros((1,), dtype=mx.bfloat16)
    return _mlx().moe_grouped_gemm_rect_q(A, Wq, expert_of_tile, bias, has_bias,
                                          A.shape[1], Wq.shape[1], format)


def moe_grouped_gemm_swiglu_q(A, W1q, expert_of_tile, format="mxfp4", bias=None,
                              act="swiglu", alpha=1.702, limit=7.0):
    """Quantized fused SwiGLU GEMM1: out(rows, inter) from a packed [gate | up] expert stack.

    A (rows, H) bfloat16; W1q (E, 2*inter, row_bytes) uint8 (quantize_expert_stack of the
    dense (E, H, 2*inter) W1); bias optional (E, 2*inter), added pre-activation.
    act "swiglu" (silu(gate)*up) or "swiglu_oai" (gpt-oss: clamp by `limit`,
    gate*sigmoid(alpha*gate)*(1+up)). Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(A):
        return _torch().moe_grouped_gemm_swiglu_q(A, W1q, expert_of_tile, format=format,
                                                  bias=bias, act=act, alpha=alpha, limit=limit)
    import mlx.core as mx
    act_mode = {"swiglu": 0, "swiglu_oai": 1}[act]
    has_bias = bias is not None
    if bias is None:
        bias = mx.zeros((1,), dtype=mx.bfloat16)
    return _mlx().moe_grouped_gemm_swiglu_q(A, W1q, expert_of_tile, bias, has_bias,
                                            W1q.shape[1] // 2, act_mode, float(alpha),
                                            float(limit), format)


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


def typical_p_sample(logits, typical_p, temperature=1.0, seed=0):
    """Typical-p (locally-typical) sampling: keep the smallest-surprise tokens (surprise =
    |(-log p_v) - H|, H = row entropy) until their cumulative prob reaches typical_p, then Gumbel-max
    sample among them. Returns the token index per row (int32). typical_p in (0, 1]. Accepts
    mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().typical_p_sample(logits, typical_p, temperature, seed)
    return _mlx().typical_p_sample(logits, typical_p, temperature=temperature, seed=seed)


def apply_token_bitmask(logits, bitmask):
    """Grammar / structured-output masking: set logits[v] = -inf where the packed allow-bitmask bit
    for token v is 0. logits (T, V); bitmask (T, ceil(V/32)) int32 packed words (bit v of row t
    allows token t*32-block... i.e. word[v>>5] bit (v&31)). Returns masked logits, same dtype.
    Composes before any sampler. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().apply_token_bitmask(logits, bitmask)
    return _mlx().apply_token_bitmask(logits, bitmask)


def apply_bad_words(logits, bad_ids, bad_lens):
    """Bad / stop-word masking: set logits[t, bad_ids[t,j]] = -inf for each j < bad_lens[t]. logits
    (T, V); bad_ids (T, maxbad) int (pad unused slots with anything, bad_lens gates them); bad_lens
    (T,) int. Returns masked logits, same dtype. Composes with the penalty/bitmask chain before a
    sampler. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(logits):
        return _torch().apply_bad_words(logits, bad_ids, bad_lens)
    return _mlx().apply_bad_words(logits, bad_ids, bad_lens)


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


def embedding_backward(token_ids, dY, vocab, scale=1.0, method="atomic"):
    """Embedding backward: scatter-add the upstream grad dY (num_tok, D) into a (vocab, D) fp32
    gradient table by token id (dtable[token_ids[t]] += scale*dY[t]). A negative / out-of-range id
    contributes nothing (matches the padding-zeros forward). token_ids (num_tok,) int; dY float.
    Returns (vocab, D) float32. Matches nn.Embedding autograd.

    method="atomic" (default): one relaxed float atomic-add per (token, d); simple, best when ids are
    mostly distinct. method="sorted": presort tokens by id (argsort) so each id's gradient is summed
    by a single threadgroup (atomic-free) — wins under heavy id duplication (high atomic contention).
    Both give the same result. Accepts mlx / torch (MPS)."""
    if method not in ("atomic", "sorted"):
        raise ValueError(f"embedding_backward: method must be 'atomic' or 'sorted', got {method!r}")
    if _is_torch(dY):
        return _torch().embedding_backward(token_ids, dY, int(vocab), float(scale), method=method)
    if method == "sorted":
        import mlx.core as _mx
        tok = token_ids.astype(_mx.int32)
        perm = _mx.argsort(tok)
        sorted_ids = tok[perm]
        return _mlx().embedding_backward_sorted(sorted_ids, perm, dY, vocab=int(vocab),
                                                scale=float(scale))
    return _mlx().embedding_backward(token_ids, dY, vocab=int(vocab), scale=float(scale))


def build_multimodal_src(span_offsets, span_lengths, modal_starts, num_tok):
    """Build the multimodal `src` map on-device (the input to merge_multimodal_spans), removing the
    host span loop. Span k covers text positions [span_offsets[k], +span_lengths[k]) and maps them to
    modal rows [modal_starts[k], +span_lengths[k]); returns src (num_tok,) int32 with
    src[t] = modal_starts[k]+offset for a token in span k, else -1. Accepts mlx / torch (MPS)."""
    if _is_torch(span_offsets):
        return _torch().build_multimodal_src(span_offsets, span_lengths, modal_starts, int(num_tok))
    return _mlx().build_multimodal_src(span_offsets, span_lengths, modal_starts, int(num_tok))


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


_LM_HEAD_MODES = {"argmax": 0, "categorical": 1, "topk": 2, "topp": 3}


_LM_QUANT_BLOCK_K = {"q8_0": 32, "q4_0": 32}


def lm_head_sample(h, W, mode="argmax", k=0, temperature=1.0, seed=0, bias=None, fused=False,
                   format=None, top_p=0.9):
    """LM-head + sampling: a decode token per row of h. h (T, K), W (V, K) row-major, both
    fp16/bf16/f32. mode in {"argmax", "categorical", "topk", "topp"}.

    format ("q8_0"/"q4_0") selects the fused quantized-weight path: W is the packed weight tensor
    (dequantized on read, no logits materialization); supports all four modes. For "topp" the fused
    quant path over-selects the top-k' candidate pool (k = the cap, default 32) then does a nucleus
    (top-p, threshold top_p) reduce over that pool — the standard "top-k then top-p" (the pool's
    softmax approximates the full normalizer for the peaked LM-head distribution). bias is an
    optional (V,) additive logit bias. Returns (T,) int32 token ids. The Gumbel noise is indexed by
    the global vocab id, so the draw equals the unfused sampler on the same logits + seed.
    Accepts mlx.array or torch.Tensor (MPS).

    Default path: logits = h @ W.T then the fast fused sampler (dense top-p = matmul + top_p_sample).
    fused=True runs the single-kernel no-materialization variant instead."""
    if mode not in _LM_HEAD_MODES:
        raise ValueError(f"lm_head_sample: mode must be one of {list(_LM_HEAD_MODES)}")
    if format is not None:
        if format not in _LM_QUANT_BLOCK_K:
            raise ValueError(f"lm_head_sample: quant format must be one of {list(_LM_QUANT_BLOCK_K)}")
        m = _LM_HEAD_MODES[mode]
        kk = int(k) if int(k) > 0 else (32 if mode in ("topk", "topp") else 0)
        V = W.shape[0]
        K = W.shape[1] * _LM_QUANT_BLOCK_K[format]   # packed Wq is (V, K/block_k, block_bytes)
        if _is_torch(h):
            import torch
            b = bias if bias is not None else torch.zeros(1, dtype=torch.float32, device=h.device)
            return _torch().lm_head_sample_q(h, W, b, V, K, format, m, kk, float(temperature),
                                             int(seed), float(top_p))
        import mlx.core as mx
        b = bias if bias is not None else mx.zeros((1,), dtype=mx.float32)
        return _mlx().lm_head_sample_q(h, W, b, V, K, format, m, kk, float(temperature),
                                       int(seed), float(top_p))
    if mode == "topp":
        # dense top-p: materialize logits (matmul reads W once, bandwidth-optimal) then top_p_sample.
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
        return top_p_sample(logits, top_p, temperature=temperature, seed=seed)
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


def beam_remap_block_table(block_table, parent_beam):
    """Zero-copy beam KV reorder: return a new block table (B*BM, max_blocks) where each child beam's
    rows point at its parent beam's PHYSICAL blocks (new[b*BM+k] = block_table[b*BM+parent_beam[b,k]])
    — no KV copy (the alternative to beam_reorder_kv). Children SHARE physical blocks, so the cache
    manager must refcount / copy-on-write a block before a beam mutates it (out of scope). block_table
    (B*BM, max_blocks) int, parent_beam (B, BM) int. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(block_table):
        return _torch().beam_remap_block_table(block_table, parent_beam)
    return _mlx().beam_remap_block_table(block_table, parent_beam)


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


def spec_build_tree_pointers(parents, num_nodes):
    """Host helper: build (retrieve_next_token, retrieve_next_sibling) int32 arrays from a per-node
    parent list `parents` (len num_nodes, parents[0] = -1 for the root; children keep input order).
    first-child / next-sibling pointers, -1 = none. Returns two numpy int32 arrays of length
    num_nodes. Feed these to spec_verify_tree (build once per tree topology on the host)."""
    import numpy as np
    nxt_tok = np.full(num_nodes, -1, np.int32)   # first child
    nxt_sib = np.full(num_nodes, -1, np.int32)   # next sibling
    last_child = np.full(num_nodes, -1, np.int32)
    for c in range(1, num_nodes):
        p = int(parents[c])
        if last_child[p] < 0:
            nxt_tok[p] = c
        else:
            nxt_sib[last_child[p]] = c
        last_child[p] = c
    return nxt_tok, nxt_sib


def build_dynamic_tree(parents):
    """Device-resident draft-tree builder: from a per-node parent list `parents` (B, N) int
    (parents[b,0] = -1 root, parents[c] < c topological) build the pointers spec_verify_tree consumes.
    Returns (retrieve_next_token (B,N), retrieve_next_sibling (B,N), positions (B,N)) int32:
    first-child / next-sibling pointers (-1 = none) and positions[c] = depth from the root. The
    on-device analogue of the host spec_build_tree_pointers. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(parents):
        return tuple(_torch().build_dynamic_tree(parents))
    out = _mlx().build_dynamic_tree(parents)
    return out[0], out[1], out[2]


def spec_verify_tree(draft_tokens, target_probs, retrieve_next_token, retrieve_next_sibling, seed,
                     tree_valid=None):
    """Speculative TREE verification (target-only rejection, TRT-LLM dynamicTree). draft_tokens
    (B, N-1) int (node c>=1 carries draft_tokens[c-1]); target_probs (B, N, V) f32 (target dist AT
    each node's position); retrieve_next_token / retrieve_next_sibling (B, N) int (first-child /
    next-sibling pointers, -1=none; use spec_build_tree_pointers). Walks each request's tree from the
    root accepting the first sibling whose cumulative target prob exceeds a coin, else a residual /
    bonus correction token (exact for any sibling count). tree_valid (B,) int (optional, default all
    ones): where 0 the request has no tree this step, so it samples the target root token with
    accept_num=0 (TRT-LLM first-generation fallback). Returns (accept_index (B,N), accept_token
    (B,N), accept_num (B,)) int32, -1-padded. Accepts mlx.array or torch.Tensor (MPS)."""
    B = target_probs.shape[0]
    if _is_torch(target_probs):
        import torch
        if tree_valid is None:
            tree_valid = torch.ones(B, dtype=torch.int32, device=target_probs.device)
        return tuple(_torch().spec_verify_tree(draft_tokens, target_probs, retrieve_next_token,
                                               retrieve_next_sibling, int(seed), tree_valid))
    import mlx.core as _mx
    if tree_valid is None:
        tree_valid = _mx.ones((B,), dtype=_mx.int32)
    out = _mlx().spec_verify_tree(draft_tokens, target_probs, retrieve_next_token,
                                  retrieve_next_sibling, tree_valid, int(seed))
    return out[0], out[1], out[2]


def spec_compact(out_tokens, accepted_cnt, seq_lens):
    """Compact accepted spec tokens: gather each request's valid tokens (accepted drafts + the
    recovered/bonus token, vlen=accepted_cnt+1) from out_tokens (B, S+1) into packed buffers.
    Returns (packed_tokens (B*(S+1),), packed_pos (B*(S+1),), cu_accepted (B+1,)), all int32:
    packed_pos[k] = seq_lens[b]+j (absolute KV position), cu_accepted[B] = total, tail = -1. Any B
    (chunked scan). Ref: vLLM rejection_sampler parse_output. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(out_tokens):
        return tuple(_torch().spec_compact(out_tokens, accepted_cnt, seq_lens))
    out = _mlx().spec_compact(out_tokens, accepted_cnt, seq_lens)
    return out[0], out[1], out[2]


def rejection_greedy_sample(cu_num_draft_tokens, draft_token_ids, target_argmax,
                            bonus_token_ids, max_draft, is_greedy=None):
    """vLLM v1 ragged greedy rejection verify: accept while draft_id == target_argmax, else stop;
    all-accept appends the bonus token. cu_num_draft_tokens (B+1,) int32 with a leading 0; all
    ids int32. is_greedy (B,) uint8 optional gate. Returns out (B, max_draft+1) int32, each row
    cleared to -1. Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(cu_num_draft_tokens):
        return _torch().rejection_greedy_sample(cu_num_draft_tokens, draft_token_ids,
                                                target_argmax, bonus_token_ids, max_draft, is_greedy)
    return _mlx().rejection_greedy_sample(cu_num_draft_tokens, draft_token_ids, target_argmax,
                                          bonus_token_ids, int(max_draft), is_greedy=is_greedy)


def rejection_random_sample(cu_num_draft_tokens, draft_token_ids, target_probs, bonus_token_ids,
                            recovered_token_ids, uniform_probs, max_draft, draft_probs=None,
                            is_greedy=None):
    """vLLM v1 ragged stochastic rejection verify: accept iff uniform <= p_target/q_draft (per
    draft token), else emit the precomputed recovered_token_ids and stop; all-accept appends the
    bonus. draft_probs optional (absent -> q=1). target_probs/draft_probs (total, V); uniform_probs
    (total,) host-generated. Returns out (B, max_draft+1) int32. Accepts mlx/torch (MPS)."""
    if _is_torch(cu_num_draft_tokens):
        return _torch().rejection_random_sample(cu_num_draft_tokens, draft_token_ids, target_probs,
                                                bonus_token_ids, recovered_token_ids, uniform_probs,
                                                max_draft, draft_probs, is_greedy)
    return _mlx().rejection_random_sample(cu_num_draft_tokens, draft_token_ids, target_probs,
                                          bonus_token_ids, recovered_token_ids, uniform_probs,
                                          int(max_draft), draft_probs=draft_probs, is_greedy=is_greedy)


def sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids, target_probs, inv_q,
                            draft_probs=None):
    """Sample the recovered token for each draft position from the adjusted residual:
    argmax_v (max(0, p_target - q_draft) * inv_q[req, v]) — inv_q (B, V) is the per-request
    exponential-race noise (equivalent to argmax(log residual + gumbel)). Produces the
    recovered_token_ids that rejection_random_sample consumes. Returns (total_draft,) int32.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(cu_num_draft_tokens):
        return _torch().sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids,
                                                target_probs, inv_q, draft_probs)
    return _mlx().sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids, target_probs,
                                          inv_q, draft_probs=draft_probs)


def spec_update_kv_meta(seq_lens, accepted_cnt):
    """Post-verify KV length: new_seq_lens[b] = seq_lens[b] + accepted_cnt[b] + 1. Returns (B,) int32.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(seq_lens):
        return _torch().spec_update_kv_meta(seq_lens, accepted_cnt)
    return _mlx().spec_update_kv_meta(seq_lens, accepted_cnt)


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


def attn_causal(q, k, v, softcap=0.0, sinks=None):
    """Causal attention forward. softcap > 0 applies Gemma-style logit soft-capping;
    sinks (H,) adds gpt-oss-style attention-sink logits to the softmax denominator.
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_causal(q, k, v, softcap=softcap, sinks=sinks)
    return _mlx().attn_causal(q, k, v, softcap=float(softcap), sinks=sinks)


def attn_window(q, k, v, window, softcap=0.0, sinks=None):
    """Sliding-window causal attention (Mistral/Gemma-style local attention): a query at
    position i attends keys [max(0, i-window+1), i] — the `window` most recent tokens
    including self. window <= 0 disables the window (== attn_causal). q,k,v (B,H,N,D) bf16,
    D in {64,128}, N%8==0. softcap > 0 = Gemma-2 logit capping; sinks (H,) = gpt-oss
    attention sinks (the window+sink combination is the gpt-oss layer config).
    Accepts mlx.array or torch.Tensor (MPS)."""
    if _is_torch(q):
        return _torch().attn_window(q, k, v, window, softcap=softcap, sinks=sinks)
    return _mlx().attn_window(q, k, v, window, softcap=float(softcap), sinks=sinks)


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


# First-order autograd wrappers (opt-in differentiable ops, both backends). See tk/autograd.py.
from . import autograd  # noqa: E402,F401
