// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "kv_cache/kv_cache.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool is_supported_float(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}

static Dtype promoted_float_dtype(const array& a, const array& b, const char* name) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument(std::string(name) + ": dtype must be float32, float16, or bfloat16");
  }
  return dtype;
}

static array contiguous_cast(const array& x, Dtype dtype, StreamOrDevice s) {
  return contiguous(astype(x, dtype, s), false, s);
}

std::vector<array> kv_cache_scatter(
    const array& key,
    const array& value,
    const array& slot_mapping,
    int num_blocks,
    int block_size,
    StreamOrDevice s) {
  if (key.ndim() != 3 || value.ndim() != 3 || key.shape() != value.shape()) {
    throw std::invalid_argument("kv_cache_scatter: key/value must have shape (num_tokens, num_heads, head_size)");
  }
  if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != key.shape(0)) {
    throw std::invalid_argument("kv_cache_scatter: slot_mapping must have shape (num_tokens,)");
  }
  if (num_blocks <= 0 || block_size <= 0) {
    throw std::invalid_argument("kv_cache_scatter: num_blocks and block_size must be positive");
  }

  const auto dtype = promoted_float_dtype(key, value, "kv_cache_scatter");
  auto key_c = contiguous_cast(key, dtype, s);
  auto value_c = contiguous_cast(value, dtype, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);

  const int H = key.shape(1);
  const int D = key.shape(2);
  std::vector<int> cache_shape = {num_blocks, block_size, H, D};
  return array::make_arrays(
      {cache_shape, cache_shape},
      {dtype, dtype},
      std::make_shared<KvCacheScatter>(to_stream(s), block_size),
      {key_c, value_c, slot_c});
}

std::vector<array> kv_cache_gather(
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& cu_seq_lens,
    int num_tokens,
    StreamOrDevice s) {
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 ||
      key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("kv_cache_gather: caches must have shape (num_blocks, block_size, num_heads, head_size)");
  }
  if (block_table.ndim() != 2) {
    throw std::invalid_argument("kv_cache_gather: block_table must be 2D");
  }
  if (cu_seq_lens.ndim() != 1 || cu_seq_lens.shape(0) != block_table.shape(0) + 1) {
    throw std::invalid_argument("kv_cache_gather: cu_seq_lens must have shape (num_seqs + 1,)");
  }
  if (num_tokens < 0) {
    throw std::invalid_argument("kv_cache_gather: num_tokens must be non-negative");
  }
  if (key_cache.dtype() != value_cache.dtype() || !is_supported_float(key_cache.dtype())) {
    throw std::invalid_argument("kv_cache_gather: caches must share a supported floating dtype");
  }

  auto key_c = contiguous(key_cache, false, s);
  auto value_c = contiguous(value_cache, false, s);
  auto block_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(cu_seq_lens, int32, s), false, s);

  const int H = key_cache.shape(2);
  const int D = key_cache.shape(3);
  std::vector<int> out_shape = {num_tokens, H, D};
  return array::make_arrays(
      {out_shape, out_shape},
      {key_cache.dtype(), key_cache.dtype()},
      std::make_shared<KvCacheGather>(to_stream(s), num_tokens),
      {key_c, value_c, block_c, lens_c});
}

std::vector<array> kv_cache_copy_blocks(
    const array& key_cache,
    const array& value_cache,
    const array& block_mapping,
    StreamOrDevice s) {
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 ||
      key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("kv_cache_copy_blocks: caches must have shape (num_blocks, block_size, num_heads, head_size)");
  }
  if (block_mapping.ndim() != 2 || block_mapping.shape(1) != 2) {
    throw std::invalid_argument("kv_cache_copy_blocks: block_mapping must have shape (num_pairs, 2)");
  }
  if (key_cache.dtype() != value_cache.dtype() || !is_supported_float(key_cache.dtype())) {
    throw std::invalid_argument("kv_cache_copy_blocks: caches must share a supported floating dtype");
  }

  auto key_c = contiguous(key_cache, false, s);
  auto value_c = contiguous(value_cache, false, s);
  auto map_c = contiguous(astype(block_mapping, int64, s), false, s);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()},
      {key_cache.dtype(), value_cache.dtype()},
      std::make_shared<KvCacheCopyBlocks>(to_stream(s)),
      {key_c, value_c, map_c});
}

array beam_build_copy_pairs(
    const array& parent_beam,
    const array& block_table,
    const array& seq_lens,
    int block_size,
    StreamOrDevice s) {
  if (parent_beam.ndim() != 2) {
    throw std::invalid_argument("beam_build_copy_pairs: parent_beam must have shape (B, BM)");
  }
  if (block_table.ndim() != 2) {
    throw std::invalid_argument("beam_build_copy_pairs: block_table must have shape (B*BM, max_blocks)");
  }
  const int B = parent_beam.shape(0), BM = parent_beam.shape(1);
  const int max_blocks = block_table.shape(1);
  auto pb_c = contiguous(astype(parent_beam, int32, s), false, s);
  auto bt_c = contiguous(astype(block_table, int32, s), false, s);
  auto sl_c = contiguous(astype(reshape(seq_lens, {B * BM}, s), int32, s), false, s);
  return array(
      {B * BM * max_blocks, 2}, int64,
      std::make_shared<BeamBuildCopyPairs>(to_stream(s), block_size),
      {pb_c, bt_c, sl_c});
}

std::vector<array> kv_cache_scales(
    const array& key,
    const array& value,
    StreamOrDevice s) {
  if (key.shape() != value.shape()) {
    throw std::invalid_argument("kv_cache_scales: key and value must have the same shape");
  }
  const auto dtype = promoted_float_dtype(key, value, "kv_cache_scales");
  auto key_c = contiguous_cast(key, dtype, s);
  auto value_c = contiguous_cast(value, dtype, s);

  return array::make_arrays(
      {{1}, {1}},
      {float32, float32},
      std::make_shared<KvCacheScales>(to_stream(s)),
      {key_c, value_c});
}

array paged_attention(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale,
    int window,
    StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 ||
      key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("paged_attention: caches must have shape (num_blocks, block_size, num_heads, head_size)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument(
        "paged_attention: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention: block_table must have shape (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention: context_lens must have shape (batch,)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention: head_size must be 64 or 128");
  }

  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument("paged_attention: dtype must be float32, float16, or bfloat16");
  }

  auto q_c = contiguous_cast(q, dtype, s);
  auto key_c = contiguous_cast(key_cache, dtype, s);
  auto value_c = contiguous_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  auto no_alibi = zeros({1}, float32, s);   // buffer 11 placeholder; use_alibi=0 ignores it
  auto no_mask = zeros({1}, int32, s);      // buffer 13 placeholder; use_mask=0 ignores it

  return array(
      q.shape(),
      dtype,
      std::make_shared<PagedAttention>(to_stream(s), scale, /*use_alibi=*/false, /*use_mask=*/false,
                                       window),
      {q_c, key_c, value_c, table_c, lens_c, no_alibi, no_mask});
}

array paged_attention_alibi(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& alibi_slopes,
    float scale,
    int window,
    StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_alibi: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("paged_attention_alibi: caches must have shape (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_alibi: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument("paged_attention_alibi: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (alibi_slopes.ndim() != 1 || alibi_slopes.shape(0) != q.shape(1)) {
    throw std::invalid_argument("paged_attention_alibi: alibi_slopes must be (num_heads,)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_alibi: head_size must be 64 or 128");
  }
  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument("paged_attention_alibi: dtype must be float32, float16, or bfloat16");
  }
  auto q_c = contiguous_cast(q, dtype, s);
  auto key_c = contiguous_cast(key_cache, dtype, s);
  auto value_c = contiguous_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  auto slopes_c = contiguous(astype(alibi_slopes, float32, s), false, s);
  auto no_mask = zeros({1}, int32, s);
  return array(
      q.shape(),
      dtype,
      std::make_shared<PagedAttention>(to_stream(s), scale, /*use_alibi=*/true, /*use_mask=*/false,
                                       window),
      {q_c, key_c, value_c, table_c, lens_c, slopes_c, no_mask});
}

array paged_attention_block_sparse(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& block_mask,
    float scale,
    int window,
    StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_block_sparse: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("paged_attention_block_sparse: caches must have shape (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_block_sparse: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument("paged_attention_block_sparse: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_block_sparse: block_table must have shape (batch, max_blocks)");
  }
  if (block_mask.shape() != block_table.shape()) {
    throw std::invalid_argument("paged_attention_block_sparse: block_mask must match block_table shape (batch, max_blocks)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_block_sparse: head_size must be 64 or 128");
  }
  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument("paged_attention_block_sparse: dtype must be float32, float16, or bfloat16");
  }
  auto q_c = contiguous_cast(q, dtype, s);
  auto key_c = contiguous_cast(key_cache, dtype, s);
  auto value_c = contiguous_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  auto no_alibi = zeros({1}, float32, s);
  auto mask_c = contiguous(astype(block_mask, int32, s), false, s);
  return array(
      q.shape(),
      dtype,
      std::make_shared<PagedAttention>(to_stream(s), scale, /*use_alibi=*/false, /*use_mask=*/true,
                                       window),
      {q_c, key_c, value_c, table_c, lens_c, no_alibi, mask_c});
}

array paged_attention_xcache(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale,
    StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_xcache: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 5) {
    throw std::invalid_argument("paged_attention_xcache: key_cache must be (num_blocks, num_kv_heads, head_size/x, block_size, x)");
  }
  if (value_cache.ndim() != 4) {
    throw std::invalid_argument("paged_attention_xcache: value_cache must be (num_blocks, num_kv_heads, head_size, block_size)");
  }
  const int D = q.shape(2);
  const int x = key_cache.shape(4);
  const int num_kv_heads = key_cache.shape(1);
  const int block_size = key_cache.shape(3);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_xcache: head_size must be 64 or 128");
  }
  if (x <= 0 || D % x != 0 || key_cache.shape(2) != D / x) {
    throw std::invalid_argument("paged_attention_xcache: key_cache head_size/x split is inconsistent with q head_size");
  }
  if (value_cache.shape(1) != num_kv_heads || value_cache.shape(2) != D || value_cache.shape(3) != block_size ||
      value_cache.shape(0) != key_cache.shape(0)) {
    throw std::invalid_argument("paged_attention_xcache: value_cache shape inconsistent with key_cache");
  }
  if (num_kv_heads <= 0 || q.shape(1) % num_kv_heads != 0) {
    throw std::invalid_argument("paged_attention_xcache: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument("paged_attention_xcache: dtype must be float32, float16, or bfloat16");
  }
  auto q_c = contiguous_cast(q, dtype, s);
  auto key_c = contiguous_cast(key_cache, dtype, s);
  auto value_c = contiguous_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  return array(
      q.shape(),
      dtype,
      std::make_shared<PagedAttentionXcache>(to_stream(s), scale),
      {q_c, key_c, value_c, table_c, lens_c});
}

void PagedAttentionXcache::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionXcache has no CPU implementation.");
}

void PagedAttentionXcache::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = q.shape(2);
  const int x = key_cache.shape(4);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_xcache(
      enc, q, key_cache, value_cache, block_table, context_lens, out,
      q.shape(0), q.shape(1), key_cache.shape(1), D, key_cache.shape(3),
      block_table.shape(1), scale, x, type_to_name(q));
}

array paged_attention_staged(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale,
    StreamOrDevice s) {
  // Same validation/contract as paged_attention; only the GPU kernel (KV-reuse staging) differs.
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_staged: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("paged_attention_staged: caches must have shape (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_staged: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument("paged_attention_staged: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_staged: block_table must have shape (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_staged: context_lens must have shape (batch,)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_staged: head_size must be 64 or 128");
  }
  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!is_supported_float(dtype)) {
    throw std::invalid_argument("paged_attention_staged: dtype must be float32, float16, or bfloat16");
  }
  auto q_c = contiguous_cast(q, dtype, s);
  auto key_c = contiguous_cast(key_cache, dtype, s);
  auto value_c = contiguous_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  return array(
      q.shape(),
      dtype,
      std::make_shared<PagedAttentionStaged>(to_stream(s), scale),
      {q_c, key_c, value_c, table_c, lens_c});
}

void KvCacheScatter::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KvCacheScatter has no CPU implementation.");
}

void KvCacheScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& key = inputs[0];
  auto& value = inputs[1];
  auto& slot = inputs[2];
  auto& key_cache = outputs[0];
  auto& value_cache = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_cache.set_data(allocator::malloc_or_wait(key_cache.nbytes()));
  value_cache.set_data(allocator::malloc_or_wait(value_cache.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const std::string tn = type_to_name(key_cache);
  const uint64_t total = static_cast<uint64_t>(key_cache.size());
  tk::launch_kv_cache_zero(enc, key_cache, value_cache, total, tn);
  tk::launch_kv_cache_scatter(
      enc,
      key,
      value,
      slot,
      key_cache,
      value_cache,
      key.shape(0),
      key.shape(1),
      key.shape(2),
      block_size_,
      tn);
}

void KvCacheGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KvCacheGather has no CPU implementation.");
}

void KvCacheGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& key_cache = inputs[0];
  auto& value_cache = inputs[1];
  auto& block_table = inputs[2];
  auto& cu_seq_lens = inputs[3];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kv_cache_gather(
      enc,
      key_cache,
      value_cache,
      key_out,
      value_out,
      block_table,
      cu_seq_lens,
      num_tokens_,
      cu_seq_lens.shape(0) - 1,
      key_cache.shape(1),
      block_table.shape(1),
      key_cache.shape(2),
      key_cache.shape(3),
      type_to_name(key_cache));
}

void KvCacheCopyBlocks::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KvCacheCopyBlocks has no CPU implementation.");
}

void KvCacheCopyBlocks::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& key_cache = inputs[0];
  auto& value_cache = inputs[1];
  auto& mapping = inputs[2];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const std::string tn = type_to_name(key_cache);
  const uint64_t total = static_cast<uint64_t>(key_cache.size());
  const int numel_per_block =
      key_cache.shape(1) * key_cache.shape(2) * key_cache.shape(3);
  tk::launch_kv_cache_clone(enc, key_cache, value_cache, key_out, value_out, total, tn);
  tk::launch_kv_cache_copy_blocks(
      enc, key_cache, value_cache, key_out, value_out, mapping, mapping.shape(0),
      numel_per_block, tn);
}

void BeamBuildCopyPairs::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BeamBuildCopyPairs has no CPU implementation.");
}

void BeamBuildCopyPairs::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& parent_beam = inputs[0];
  auto& block_table = inputs[1];
  auto& seq_lens = inputs[2];
  auto& pairs = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  pairs.set_data(allocator::malloc_or_wait(pairs.nbytes()));

  const int BM = parent_beam.shape(1);
  const int max_blocks = block_table.shape(1);
  const int n_slots = static_cast<int>(pairs.shape(0));   // B*BM*max_blocks
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_beam_build_copy_pairs(enc, parent_beam, block_table, seq_lens, pairs,
                                   BM, max_blocks, block_size_, n_slots);
}

void KvCacheScales::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KvCacheScales has no CPU implementation.");
}

void KvCacheScales::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& key = inputs[0];
  auto& value = inputs[1];
  auto& key_scale = outputs[0];
  auto& value_scale = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_scale.set_data(allocator::malloc_or_wait(key_scale.nbytes()));
  value_scale.set_data(allocator::malloc_or_wait(value_scale.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const uint64_t n = static_cast<uint64_t>(key.size());
  tk::launch_kv_cache_scales(enc, key, value, key_scale, value_scale, n, type_to_name(key));
}

void PagedAttention::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttention has no CPU implementation.");
}

void PagedAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& alibi_slopes = inputs[5];
  auto& block_mask = inputs[6];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = q.shape(2);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention(
      enc,
      q,
      key_cache,
      value_cache,
      block_table,
      context_lens,
      out,
      q.shape(0),
      q.shape(1),
      key_cache.shape(2),
      D,
      key_cache.shape(1),
      block_table.shape(1),
      scale,
      alibi_slopes,
      use_alibi_ ? 1 : 0,
      block_mask,
      use_mask_ ? 1 : 0,
      window_,
      type_to_name(q));
}

void PagedAttentionStaged::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionStaged has no CPU implementation.");
}

void PagedAttentionStaged::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = q.shape(2);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_gqa_staged(
      enc, q, key_cache, value_cache, block_table, context_lens, out,
      q.shape(0), q.shape(1), key_cache.shape(2), D, key_cache.shape(1),
      block_table.shape(1), scale, type_to_name(q));
}

// --------------------------- fp8 KV cache ---------------------------

std::vector<array> kv_cache_scatter_fp8(
    const array& key,
    const array& value,
    const array& slot_mapping,
    int num_blocks,
    int block_size,
    const array& k_scale,
    const array& v_scale,
    int fmt,
    StreamOrDevice s) {
  if (key.ndim() != 3 || value.ndim() != 3 || key.shape() != value.shape()) {
    throw std::invalid_argument("kv_cache_scatter_fp8: key/value must be (num_tokens, num_heads, head_size)");
  }
  if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != key.shape(0)) {
    throw std::invalid_argument("kv_cache_scatter_fp8: slot_mapping must be (num_tokens,)");
  }
  if (num_blocks <= 0 || block_size <= 0) {
    throw std::invalid_argument("kv_cache_scatter_fp8: num_blocks and block_size must be positive");
  }
  const int H = key.shape(1);
  const int D = key.shape(2);
  if (k_scale.ndim() != 1 || k_scale.shape(0) != H || v_scale.shape() != k_scale.shape()) {
    throw std::invalid_argument("kv_cache_scatter_fp8: k_scale/v_scale must be (num_heads,)");
  }
  const auto dtype = promoted_float_dtype(key, value, "kv_cache_scatter_fp8");
  auto key_c = contiguous_cast(key, dtype, s);
  auto value_c = contiguous_cast(value, dtype, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto ks_c = contiguous(astype(k_scale, float32, s), false, s);
  auto vs_c = contiguous(astype(v_scale, float32, s), false, s);
  std::vector<int> cache_shape = {num_blocks, block_size, H, D};
  return array::make_arrays(
      {cache_shape, cache_shape},
      {uint8, uint8},
      std::make_shared<KvCacheScatterFp8>(to_stream(s), block_size, fmt),
      {key_c, value_c, slot_c, ks_c, vs_c});
}

array paged_attention_fp8(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& k_scale,
    const array& v_scale,
    float scale,
    int fmt,
    int window,
    StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_fp8: q must be (batch, num_heads, head_size)");
  }
  if (!is_supported_float(q.dtype())) {
    throw std::invalid_argument("paged_attention_fp8: q must be float32, float16, or bfloat16");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument("paged_attention_fp8: caches must be (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.dtype() != uint8 || value_cache.dtype() != uint8) {
    throw std::invalid_argument("paged_attention_fp8: caches must be uint8 (e4m3 codes)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_fp8: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument("paged_attention_fp8: num_q_heads must be a positive multiple of num_kv_heads");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_fp8: head_size must be 64 or 128");
  }
  const int num_kv_heads = key_cache.shape(2);
  if (k_scale.ndim() != 1 || k_scale.shape(0) != num_kv_heads || v_scale.shape() != k_scale.shape()) {
    throw std::invalid_argument("paged_attention_fp8: k_scale/v_scale must be (num_kv_heads,)");
  }
  auto q_c = contiguous(q, false, s);
  auto kc = contiguous(astype(key_cache, uint8, s), false, s);
  auto vc = contiguous(astype(value_cache, uint8, s), false, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  auto ks_c = contiguous(astype(k_scale, float32, s), false, s);
  auto vs_c = contiguous(astype(v_scale, float32, s), false, s);
  return array(
      q.shape(),
      q.dtype(),
      std::make_shared<PagedAttentionFp8>(to_stream(s), scale, fmt, window),
      {q_c, kc, vc, table_c, lens_c, ks_c, vs_c});
}

void KvCacheScatterFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KvCacheScatterFp8 has no CPU implementation.");
}

void KvCacheScatterFp8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& key = inputs[0];
  auto& value = inputs[1];
  auto& slot = inputs[2];
  auto& k_scale = inputs[3];
  auto& v_scale = inputs[4];
  auto& key_cache = outputs[0];
  auto& value_cache = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_cache.set_data(allocator::malloc_or_wait(key_cache.nbytes()));
  value_cache.set_data(allocator::malloc_or_wait(value_cache.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const uint64_t total = static_cast<uint64_t>(key_cache.size());
  tk::launch_kv_cache_zero_u8(enc, key_cache, value_cache, total);
  tk::launch_kv_cache_scatter_fp8(
      enc, key, value, slot, key_cache, value_cache,
      key.shape(0), key.shape(1), key.shape(2), block_size_, k_scale, v_scale, fmt_,
      type_to_name(key));
}

void PagedAttentionFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionFp8 has no CPU implementation.");
}

void PagedAttentionFp8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& k_scale = inputs[5];
  auto& v_scale = inputs[6];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = q.shape(2);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_fp8(
      enc, q, key_cache, value_cache, block_table, context_lens, out,
      q.shape(0), q.shape(1), key_cache.shape(2), D, key_cache.shape(1),
      block_table.shape(1), scale, k_scale, v_scale, fmt_, window_, type_to_name(q));
}

#define TK_KV_NO_AUTODIFF(CLASS, LABEL)                                      \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&,                                             \
      const std::vector<array>&,                                             \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&,                                             \
      const std::vector<array>&,                                             \
      const std::vector<int>&,                                               \
      const std::vector<array>&) {                                           \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&,                                             \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_KV_NO_AUTODIFF(KvCacheScatter, "KvCacheScatter")
TK_KV_NO_AUTODIFF(KvCacheScatterFp8, "KvCacheScatterFp8")
TK_KV_NO_AUTODIFF(PagedAttentionFp8, "PagedAttentionFp8")
TK_KV_NO_AUTODIFF(KvCacheGather, "KvCacheGather")
TK_KV_NO_AUTODIFF(KvCacheCopyBlocks, "KvCacheCopyBlocks")
TK_KV_NO_AUTODIFF(BeamBuildCopyPairs, "BeamBuildCopyPairs")
TK_KV_NO_AUTODIFF(KvCacheScales, "KvCacheScales")
TK_KV_NO_AUTODIFF(PagedAttention, "PagedAttention")
TK_KV_NO_AUTODIFF(PagedAttentionStaged, "PagedAttentionStaged")
TK_KV_NO_AUTODIFF(PagedAttentionXcache, "PagedAttentionXcache")

} // namespace mlx::core
