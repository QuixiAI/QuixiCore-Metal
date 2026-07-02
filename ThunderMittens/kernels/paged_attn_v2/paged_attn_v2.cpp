// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "paged_attn_v2/paged_attn_v2.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool pav2_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

static array pav2_cast(const array& x, Dtype dtype, StreamOrDevice s) {
  return contiguous(astype(x, dtype, s), false, s);
}

array paged_attention_v2(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale /* = 0.0f */,
    int partition_size /* = 512 */,
    int window /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_v2: q must have shape (batch, num_heads, head_size)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument(
        "paged_attention_v2: caches must have shape (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_v2: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument(
        "paged_attention_v2: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_v2: block_table must have shape (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_v2: context_lens must have shape (batch,)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_v2: head_size must be 64 or 128");
  }
  const int block_size = key_cache.shape(1);
  if (partition_size <= 0 || partition_size % block_size != 0) {
    throw std::invalid_argument("paged_attention_v2: partition_size must be a positive multiple of block_size");
  }

  auto dtype = promote_types(q.dtype(), key_cache.dtype());
  dtype = promote_types(dtype, value_cache.dtype());
  if (!pav2_is_float(dtype)) {
    throw std::invalid_argument("paged_attention_v2: dtype must be float32, float16, or bfloat16");
  }

  auto q_c = pav2_cast(q, dtype, s);
  auto key_c = pav2_cast(key_cache, dtype, s);
  auto value_c = pav2_cast(value_cache, dtype, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int max_ctx = block_table.shape(1) * block_size;
  const int num_partitions = std::max(1, (max_ctx + partition_size - 1) / partition_size);

  auto parts = array::make_arrays(
      {{B, H, num_partitions, D}, {B, H, num_partitions}, {B, H, num_partitions}},
      {float32, float32, float32},
      std::make_shared<PagedAttentionV2Partition>(to_stream(s), scale, num_partitions,
                                                  partition_size, window),
      {q_c, key_c, value_c, table_c, lens_c});

  return array(
      {B, H, D},
      dtype,
      std::make_shared<PagedAttentionV2Reduce>(to_stream(s)),
      {parts[0], parts[1], parts[2]});
}

array paged_attention_v2_fp8(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& k_scale,
    const array& v_scale,
    float scale /* = 0.0f */,
    int partition_size /* = 512 */,
    int fmt /* = 0 */,
    int window /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("paged_attention_v2_fp8: q must have shape (batch, num_heads, head_size)");
  }
  if (!pav2_is_float(q.dtype())) {
    throw std::invalid_argument("paged_attention_v2_fp8: q must be float32, float16, or bfloat16");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument(
        "paged_attention_v2_fp8: caches must have shape (num_blocks, block_size, num_kv_heads, head_size)");
  }
  if (key_cache.dtype() != uint8 || value_cache.dtype() != uint8) {
    throw std::invalid_argument("paged_attention_v2_fp8: caches must be uint8 (fp8 codes)");
  }
  if (key_cache.shape(3) != q.shape(2)) {
    throw std::invalid_argument("paged_attention_v2_fp8: q head_size must match cache head_size");
  }
  if (key_cache.shape(2) <= 0 || q.shape(1) % key_cache.shape(2) != 0) {
    throw std::invalid_argument(
        "paged_attention_v2_fp8: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_v2_fp8: block_table must have shape (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("paged_attention_v2_fp8: context_lens must have shape (batch,)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("paged_attention_v2_fp8: head_size must be 64 or 128");
  }
  const int block_size = key_cache.shape(1);
  if (partition_size <= 0 || partition_size % block_size != 0) {
    throw std::invalid_argument("paged_attention_v2_fp8: partition_size must be a positive multiple of block_size");
  }
  const int num_kv_heads = key_cache.shape(2);
  if (k_scale.ndim() != 1 || k_scale.shape(0) != num_kv_heads || v_scale.shape() != k_scale.shape()) {
    throw std::invalid_argument("paged_attention_v2_fp8: k_scale/v_scale must be (num_kv_heads,)");
  }

  auto q_c = pav2_cast(q, q.dtype(), s);
  auto key_c = contiguous(astype(key_cache, uint8, s), false, s);
  auto value_c = contiguous(astype(value_cache, uint8, s), false, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  auto ks_c = contiguous(astype(k_scale, float32, s), false, s);
  auto vs_c = contiguous(astype(v_scale, float32, s), false, s);

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int max_ctx = block_table.shape(1) * block_size;
  const int num_partitions = std::max(1, (max_ctx + partition_size - 1) / partition_size);

  auto parts = array::make_arrays(
      {{B, H, num_partitions, D}, {B, H, num_partitions}, {B, H, num_partitions}},
      {float32, float32, float32},
      std::make_shared<PagedAttentionV2PartitionFp8>(
          to_stream(s), scale, num_partitions, partition_size, fmt, window),
      {q_c, key_c, value_c, table_c, lens_c, ks_c, vs_c});

  return array(
      {B, H, D},
      q.dtype(),
      std::make_shared<PagedAttentionV2Reduce>(to_stream(s)),
      {parts[0], parts[1], parts[2]});
}

void PagedAttentionV2PartitionFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionV2PartitionFp8 has no CPU implementation.");
}

void PagedAttentionV2PartitionFp8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& k_scale = inputs[5];
  auto& v_scale = inputs[6];
  auto& tmp_out = outputs[0];
  auto& max_logits = outputs[1];
  auto& exp_sums = outputs[2];

  auto& s = stream();
  auto& d = metal::device(s.device);
  tmp_out.set_data(allocator::malloc_or_wait(tmp_out.nbytes()));
  max_logits.set_data(allocator::malloc_or_wait(max_logits.nbytes()));
  exp_sums.set_data(allocator::malloc_or_wait(exp_sums.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int D = q.shape(2);
  const int num_kv_heads = key_cache.shape(2);
  const int block_size = key_cache.shape(1);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_partition_fp8(
      enc, q, key_cache, value_cache, block_table, context_lens,
      tmp_out, max_logits, exp_sums, B, H, num_kv_heads, D, block_size,
      block_table.shape(1), scale, num_partitions_, partition_size_,
      k_scale, v_scale, fmt_, window_, type_to_name(q));
}

void PagedAttentionV2Partition::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionV2Partition has no CPU implementation.");
}

void PagedAttentionV2Partition::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& tmp_out = outputs[0];
  auto& max_logits = outputs[1];
  auto& exp_sums = outputs[2];

  auto& s = stream();
  auto& d = metal::device(s.device);
  tmp_out.set_data(allocator::malloc_or_wait(tmp_out.nbytes()));
  max_logits.set_data(allocator::malloc_or_wait(max_logits.nbytes()));
  exp_sums.set_data(allocator::malloc_or_wait(exp_sums.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int D = q.shape(2);
  const int num_kv_heads = key_cache.shape(2);
  const int block_size = key_cache.shape(1);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_partition(
      enc, q, key_cache, value_cache, block_table, context_lens,
      tmp_out, max_logits, exp_sums, B, H, num_kv_heads, D, block_size,
      block_table.shape(1), scale, num_partitions_, partition_size_, window_, type_to_name(q));
}

void PagedAttentionV2Reduce::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PagedAttentionV2Reduce has no CPU implementation.");
}

void PagedAttentionV2Reduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& tmp_out = inputs[0];
  auto& max_logits = inputs[1];
  auto& exp_sums = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int B = out.shape(0);
  const int H = out.shape(1);
  const int D = out.shape(2);
  const int num_partitions = max_logits.shape(2);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_paged_attention_reduce(
      enc, tmp_out, max_logits, exp_sums, out, B, H, D, num_partitions, type_to_name(out));
}

#define TK_PAV2_NO_AUTODIFF(CLASS, LABEL)                                    \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_PAV2_NO_AUTODIFF(PagedAttentionV2Partition, "PagedAttentionV2Partition")
TK_PAV2_NO_AUTODIFF(PagedAttentionV2PartitionFp8, "PagedAttentionV2PartitionFp8")
TK_PAV2_NO_AUTODIFF(PagedAttentionV2Reduce, "PagedAttentionV2Reduce")

} // namespace mlx::core
