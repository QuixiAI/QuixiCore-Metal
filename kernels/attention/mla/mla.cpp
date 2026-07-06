// Copyright © 2024 Apple Inc.

#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mla/mla.h"
#include "paged_attn_v2/paged_attn_v2.h"   // PagedAttentionV2Reduce (combines P4v2 partials)

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static array mla_contig_bf16(const array& x, StreamOrDevice s) {
  return contiguous(astype(x, bfloat16, s), false, s);
}

array mla_q_norm_rope(
    const array& q,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& norm_weight,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int norm_mode,
    float eps,
    StreamOrDevice s) {
  if (q.ndim() < 2) {
    throw std::invalid_argument("mla_q_norm_rope: q must be at least 2-D (…, head_dim)");
  }
  const int head_dim = q.shape(-1);
  if (head_dim % 64 != 0 || head_dim != nope_dim + rope_dim) {
    throw std::invalid_argument("mla_q_norm_rope: head_dim must be nope_dim+rope_dim and %64==0");
  }
  if (nope_dim % 2 != 0 || rope_dim % 2 != 0) {
    throw std::invalid_argument("mla_q_norm_rope: nope_dim and rope_dim must be even");
  }
  if (cos.shape(-1) != rope_dim / 2 || sin.shape(-1) != rope_dim / 2) {
    throw std::invalid_argument("mla_q_norm_rope: cos/sin must be (max_pos, rope_dim/2)");
  }
  if (norm_mode < 0 || norm_mode > 2) {
    throw std::invalid_argument("mla_q_norm_rope: norm_mode must be 0, 1, or 2");
  }

  auto q_c = mla_contig_bf16(q, s);
  auto cos_c = mla_contig_bf16(cos, s);
  auto sin_c = mla_contig_bf16(sin, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto w_c = mla_contig_bf16(norm_weight, s);

  return array(
      q.shape(),
      bfloat16,
      std::make_shared<MlaQNormRope>(to_stream(s), num_heads, nope_dim, rope_dim, norm_mode, eps),
      {q_c, cos_c, sin_c, pos_c, w_c});
}

void MlaQNormRope::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaQNormRope has no CPU implementation.");
}

void MlaQNormRope::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& positions = inputs[3];
  auto& norm_weight = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int head_dim = q.shape(-1);
  const int M = static_cast<int>(q.size() / head_dim);   // num_tokens * num_heads
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_q_norm_rope(
      enc, q, cos, sin, positions, norm_weight, out, M, num_heads_, nope_dim_, rope_dim_,
      norm_mode_, eps_, head_dim);
}

array mla_kv_insert(
    const array& kv_c,
    const array& k_pe,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& kv_cache,
    const array& norm_weight,
    int rope_dim,
    int norm_mode,
    float eps,
    StreamOrDevice s) {
  if (kv_c.ndim() < 2 || k_pe.ndim() < 2) {
    throw std::invalid_argument("mla_kv_insert: kv_c and k_pe must be at least 2-D");
  }
  const int latent = kv_c.shape(-1);
  if (latent % 64 != 0) {
    throw std::invalid_argument("mla_kv_insert: LATENT (kv_c last dim) must be %64==0");
  }
  if (k_pe.shape(-1) != rope_dim || rope_dim / 2 > 32 || rope_dim % 2 != 0) {
    throw std::invalid_argument("mla_kv_insert: k_pe last dim must be rope_dim (even, /2<=32)");
  }
  if (kv_cache.ndim() != 3 || kv_cache.shape(2) != latent + rope_dim) {
    throw std::invalid_argument("mla_kv_insert: kv_cache must be (num_blocks, block_size, LATENT+rope_dim)");
  }
  if (cos.shape(-1) != rope_dim / 2 || sin.shape(-1) != rope_dim / 2) {
    throw std::invalid_argument("mla_kv_insert: cos/sin must be (max_pos, rope_dim/2)");
  }

  auto kv_c_c = mla_contig_bf16(kv_c, s);
  auto k_pe_c = mla_contig_bf16(k_pe, s);
  auto cos_c = mla_contig_bf16(cos, s);
  auto sin_c = mla_contig_bf16(sin, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto cache_c = mla_contig_bf16(kv_cache, s);
  auto w_c = mla_contig_bf16(norm_weight, s);

  return array(
      kv_cache.shape(),
      bfloat16,
      std::make_shared<MlaKvInsert>(to_stream(s), rope_dim, norm_mode, eps),
      {kv_c_c, k_pe_c, cos_c, sin_c, pos_c, slot_c, cache_c, w_c});
}

void MlaKvInsert::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaKvInsert has no CPU implementation.");
}

void MlaKvInsert::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& kv_c = inputs[0];
  auto& k_pe = inputs[1];
  auto& cos = inputs[2];
  auto& sin = inputs[3];
  auto& positions = inputs[4];
  auto& slot_mapping = inputs[5];
  auto& cache_in = inputs[6];
  auto& norm_weight = inputs[7];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int latent = kv_c.shape(-1);
  const int num_tokens = static_cast<int>(kv_c.size() / latent);
  const int block_size = cache_in.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // clone the existing cache, then overwrite the mapped slots.
  tk::launch_mla_cache_clone(enc, cache_in, out, static_cast<uint64_t>(out.size()));
  tk::launch_mla_kv_insert(enc, kv_c, k_pe, cos, sin, positions, slot_mapping, out, norm_weight,
                           num_tokens, block_size, rope_dim_, norm_mode_, eps_, latent);
}

array mla_decode(
    const array& q,
    const array& kv_cache,
    const array& block_table,
    const array& context_lens,
    float scale,
    StreamOrDevice s) {
  constexpr int ROPE = 64;
  if (q.ndim() != 3) {
    throw std::invalid_argument("mla_decode: q must be (batch, num_heads, LATENT+rope)");
  }
  if (kv_cache.ndim() != 3 || kv_cache.shape(2) != q.shape(2)) {
    throw std::invalid_argument("mla_decode: kv_cache must be (num_blocks, block_size, LATENT+rope)");
  }
  const int qk = q.shape(2);
  const int latent = qk - ROPE;
  if (latent != 512) {
    throw std::invalid_argument("mla_decode: only LATENT=512, rope=64 (QK=576) is instantiated");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode: block_table must be (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode: context_lens must be (batch,)");
  }
  auto q_c = mla_contig_bf16(q, s);
  auto cache_c = mla_contig_bf16(kv_cache, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  // P4v2 route: partitioned multi-head decode + paged-v2 reduce (see mla.metal for rationale).
  const int B = q.shape(0);
  const int N = q.shape(1);
  const int block_size = kv_cache.shape(1);
  const int max_ctx = block_table.shape(1) * block_size;
  int psize = 512;
  psize = ((psize + block_size - 1) / block_size) * block_size;   // multiple of block_size
  const int P = std::max(1, (max_ctx + psize - 1) / psize);
  auto parts = array::make_arrays(
      {{B, N, P, latent}, {B, N, P}, {B, N, P}},
      {float32, float32, float32},
      std::make_shared<MlaDecodePartition>(to_stream(s), scale, P, psize),
      {q_c, cache_c, table_c, lens_c});
  return array(
      {B, N, latent},
      bfloat16,
      std::make_shared<PagedAttentionV2Reduce>(to_stream(s)),
      {parts[0], parts[1], parts[2]});
}

void MlaDecodePartition::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaDecodePartition has no CPU implementation.");
}

void MlaDecodePartition::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& kv_cache = inputs[1];
  auto& block_table = inputs[2];
  auto& context_lens = inputs[3];
  auto& tmp_out = outputs[0];
  auto& max_logits = outputs[1];
  auto& exp_sums = outputs[2];

  auto& s = stream();
  auto& d = metal::device(s.device);
  tmp_out.set_data(allocator::malloc_or_wait(tmp_out.nbytes()));
  max_logits.set_data(allocator::malloc_or_wait(max_logits.nbytes()));
  exp_sums.set_data(allocator::malloc_or_wait(exp_sums.nbytes()));

  const int qk = q.shape(2);
  const int rope = 64;
  const int latent = qk - rope;
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(qk));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_decode_partition(enc, q, kv_cache, block_table, context_lens,
                                  tmp_out, max_logits, exp_sums, q.shape(0), q.shape(1),
                                  kv_cache.shape(1), block_table.shape(1), scale, latent, rope,
                                  num_partitions_, partition_size_);
}

std::vector<array> MlaDecodePartition::jvp(const std::vector<array>&, const std::vector<array>&,
                                           const std::vector<int>&) {
  throw std::runtime_error("MlaDecodePartition has no jvp implementation.");
}
std::vector<array> MlaDecodePartition::vjp(const std::vector<array>&, const std::vector<array>&,
                                           const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("MlaDecodePartition has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MlaDecodePartition::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MlaDecodePartition has no vmap implementation.");
}

void MlaDecode::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaDecode has no CPU implementation.");
}

void MlaDecode::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& kv_cache = inputs[1];
  auto& block_table = inputs[2];
  auto& context_lens = inputs[3];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int qk = q.shape(2);
  const int rope = 64;
  const int latent = qk - rope;
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(qk));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_decode(enc, q, kv_cache, block_table, context_lens, out, q.shape(0), q.shape(1),
                        kv_cache.shape(1), block_table.shape(1), scale, latent, rope);
}

array mla_decode_fp8(
    const array& q,
    const array& data_cache,
    const array& scale_cache,
    const array& block_table,
    const array& context_lens,
    float scale,
    StreamOrDevice s) {
  if (q.ndim() != 3 || q.shape(2) != 512) {
    throw std::invalid_argument("mla_decode_fp8: q must be (batch, num_heads, 512)");
  }
  if (data_cache.ndim() != 3 || data_cache.shape(2) != 576) {
    throw std::invalid_argument("mla_decode_fp8: data_cache must be (num_blocks, block_size, 576) uint8");
  }
  if (scale_cache.ndim() != 3 || scale_cache.shape(2) != 8 ||
      scale_cache.shape(0) != data_cache.shape(0) || scale_cache.shape(1) != data_cache.shape(1)) {
    throw std::invalid_argument("mla_decode_fp8: scale_cache must be (num_blocks, block_size, 8) uint8");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode_fp8: block_table must be (batch, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode_fp8: context_lens must be (batch,)");
  }
  auto q_c = mla_contig_bf16(q, s);
  auto data_c = contiguous(astype(data_cache, uint8, s), false, s);
  auto scale_c = contiguous(astype(scale_cache, uint8, s), false, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto lens_c = contiguous(astype(context_lens, int32, s), false, s);
  // P4a-v2 route: partitioned decode + paged-v2 reduce (same upgrade as bf16 mla_decode).
  const int B = q.shape(0);
  const int N = q.shape(1);
  const int block_size = data_cache.shape(1);
  const int max_ctx = block_table.shape(1) * block_size;
  int psize = ((512 + block_size - 1) / block_size) * block_size;
  const int P = std::max(1, (max_ctx + psize - 1) / psize);
  auto parts = array::make_arrays(
      {{B, N, P, 512}, {B, N, P}, {B, N, P}},
      {float32, float32, float32},
      std::make_shared<MlaDecodeFp8Partition>(to_stream(s), scale, P, psize),
      {q_c, data_c, scale_c, table_c, lens_c});
  return array(
      {B, N, 512},
      bfloat16,
      std::make_shared<PagedAttentionV2Reduce>(to_stream(s)),
      {parts[0], parts[1], parts[2]});
}

void MlaDecodeFp8Partition::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaDecodeFp8Partition has no CPU implementation.");
}

void MlaDecodeFp8Partition::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& data_cache = inputs[1];
  auto& scale_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& tmp_out = outputs[0];
  auto& max_logits = outputs[1];
  auto& exp_sums = outputs[2];

  auto& s = stream();
  auto& d = metal::device(s.device);
  tmp_out.set_data(allocator::malloc_or_wait(tmp_out.nbytes()));
  max_logits.set_data(allocator::malloc_or_wait(max_logits.nbytes()));
  exp_sums.set_data(allocator::malloc_or_wait(exp_sums.nbytes()));

  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(512.0f);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  if (inputs.size() == 5) {   // dense: (q, data, scale, table, context_lens)
    auto& context_lens = inputs[4];
    tk::launch_mla_decode_fp8_partition(
        enc, q, data_cache, scale_cache, block_table, context_lens, tmp_out, max_logits,
        exp_sums, q.shape(0), q.shape(1), data_cache.shape(1), block_table.shape(1), scale,
        num_partitions_, partition_size_);
  } else {                    // sparse: (q, data, scale, table, indices, topk_length)
    auto& indices = inputs[4];
    auto& topk_length = inputs[5];
    tk::launch_mla_decode_fp8_sparse_partition(
        enc, q, data_cache, scale_cache, block_table, indices, topk_length, tmp_out,
        max_logits, exp_sums, q.shape(0), q.shape(1), data_cache.shape(1),
        block_table.shape(1), scale, indices.shape(1), num_partitions_, partition_size_);
  }
}

std::vector<array> MlaDecodeFp8Partition::jvp(const std::vector<array>&, const std::vector<array>&,
                                              const std::vector<int>&) {
  throw std::runtime_error("MlaDecodeFp8Partition has no jvp implementation.");
}
std::vector<array> MlaDecodeFp8Partition::vjp(const std::vector<array>&, const std::vector<array>&,
                                              const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("MlaDecodeFp8Partition has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MlaDecodeFp8Partition::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MlaDecodeFp8Partition has no vmap implementation.");
}

void MlaDecodeFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaDecodeFp8 has no CPU implementation.");
}

void MlaDecodeFp8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& data_cache = inputs[1];
  auto& scale_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(512.0f);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_decode_fp8(enc, q, data_cache, scale_cache, block_table, context_lens, out,
                            q.shape(0), q.shape(1), data_cache.shape(1), block_table.shape(1), scale);
}

array mla_decode_fp8_sparse(
    const array& q,
    const array& data_cache,
    const array& scale_cache,
    const array& block_table,
    const array& indices,
    const array& topk_length,
    float scale,
    StreamOrDevice s) {
  if (q.ndim() != 3 || q.shape(2) != 512) {
    throw std::invalid_argument("mla_decode_fp8_sparse: q must be (batch, num_heads, 512)");
  }
  if (data_cache.ndim() != 3 || data_cache.shape(2) != 576) {
    throw std::invalid_argument("mla_decode_fp8_sparse: data_cache must be (nb, bs, 576) uint8");
  }
  if (scale_cache.ndim() != 3 || scale_cache.shape(2) != 8) {
    throw std::invalid_argument("mla_decode_fp8_sparse: scale_cache must be (nb, bs, 8) uint8");
  }
  if (block_table.ndim() != 2 || block_table.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode_fp8_sparse: block_table must be (batch, max_blocks)");
  }
  if (indices.ndim() != 2 || indices.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode_fp8_sparse: indices must be (batch, max_topk)");
  }
  if (topk_length.ndim() != 1 || topk_length.shape(0) != q.shape(0)) {
    throw std::invalid_argument("mla_decode_fp8_sparse: topk_length must be (batch,)");
  }
  auto q_c = mla_contig_bf16(q, s);
  auto data_c = contiguous(astype(data_cache, uint8, s), false, s);
  auto scale_c = contiguous(astype(scale_cache, uint8, s), false, s);
  auto table_c = contiguous(astype(block_table, int32, s), false, s);
  auto idx_c = contiguous(astype(indices, int32, s), false, s);
  auto len_c = contiguous(astype(topk_length, int32, s), false, s);
  // P4b-v2 route: partition the top-k index list + paged-v2 reduce.
  const int B = q.shape(0);
  const int N = q.shape(1);
  const int max_topk = indices.shape(1);
  const int psize = 512;
  const int P = std::max(1, (max_topk + psize - 1) / psize);
  auto parts = array::make_arrays(
      {{B, N, P, 512}, {B, N, P}, {B, N, P}},
      {float32, float32, float32},
      std::make_shared<MlaDecodeFp8Partition>(to_stream(s), scale, P, psize),
      {q_c, data_c, scale_c, table_c, idx_c, len_c});
  return array(
      {B, N, 512},
      bfloat16,
      std::make_shared<PagedAttentionV2Reduce>(to_stream(s)),
      {parts[0], parts[1], parts[2]});
}

void MlaDecodeFp8Sparse::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaDecodeFp8Sparse has no CPU implementation.");
}

void MlaDecodeFp8Sparse::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& data_cache = inputs[1];
  auto& scale_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& indices = inputs[4];
  auto& topk_length = inputs[5];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(512.0f);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_decode_fp8_sparse(enc, q, data_cache, scale_cache, block_table, indices,
                                   topk_length, out, q.shape(0), q.shape(1), data_cache.shape(1),
                                   block_table.shape(1), scale, indices.shape(1));
}

std::vector<array> mla_kv_insert_fp8(
    const array& kv,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& data_cache,
    const array& scale_cache,
    StreamOrDevice s) {
  if (kv.ndim() < 2 || kv.shape(-1) != 512) {
    throw std::invalid_argument("mla_kv_insert_fp8: kv must be (…, 512) [448 NoPE | 64 RoPE]");
  }
  if (data_cache.ndim() != 3 || data_cache.shape(2) != 576) {
    throw std::invalid_argument("mla_kv_insert_fp8: data_cache must be (num_blocks, block_size, 576) uint8");
  }
  if (scale_cache.ndim() != 3 || scale_cache.shape(2) != 8 ||
      scale_cache.shape(0) != data_cache.shape(0) || scale_cache.shape(1) != data_cache.shape(1)) {
    throw std::invalid_argument("mla_kv_insert_fp8: scale_cache must be (num_blocks, block_size, 8) uint8");
  }
  if (cos.shape(-1) != 32 || sin.shape(-1) != 32) {
    throw std::invalid_argument("mla_kv_insert_fp8: cos/sin must be (max_pos, 32)");
  }

  auto kv_c = mla_contig_bf16(kv, s);
  auto cos_c = mla_contig_bf16(cos, s);
  auto sin_c = mla_contig_bf16(sin, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto data_c = contiguous(astype(data_cache, uint8, s), false, s);
  auto scale_c = contiguous(astype(scale_cache, uint8, s), false, s);

  return array::make_arrays(
      {data_cache.shape(), scale_cache.shape()},
      {uint8, uint8},
      std::make_shared<MlaKvInsertFp8>(to_stream(s)),
      {kv_c, cos_c, sin_c, pos_c, slot_c, data_c, scale_c});
}

void MlaKvInsertFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaKvInsertFp8 has no CPU implementation.");
}

void MlaKvInsertFp8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& kv = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& positions = inputs[3];
  auto& slot_mapping = inputs[4];
  auto& data_in = inputs[5];
  auto& scale_in = inputs[6];
  auto& data_out = outputs[0];
  auto& scale_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  data_out.set_data(allocator::malloc_or_wait(data_out.nbytes()));
  scale_out.set_data(allocator::malloc_or_wait(scale_out.nbytes()));

  const int num_tokens = static_cast<int>(kv.size() / 512);
  const int block_size = data_in.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_cache_clone_u8(enc, data_in, data_out, static_cast<uint64_t>(data_out.size()));
  tk::launch_mla_cache_clone_u8(enc, scale_in, scale_out, static_cast<uint64_t>(scale_out.size()));
  tk::launch_mla_kv_insert_fp8(enc, kv, cos, sin, positions, slot_mapping, data_out, scale_out,
                               num_tokens, block_size);
}

#define TK_MLA_NO_AUTODIFF(CLASS, LABEL)                                     \
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

TK_MLA_NO_AUTODIFF(MlaQNormRope, "MlaQNormRope")
TK_MLA_NO_AUTODIFF(MlaKvInsert, "MlaKvInsert")
TK_MLA_NO_AUTODIFF(MlaDecode, "MlaDecode")
TK_MLA_NO_AUTODIFF(MlaDecodeFp8, "MlaDecodeFp8")
TK_MLA_NO_AUTODIFF(MlaDecodeFp8Sparse, "MlaDecodeFp8Sparse")
TK_MLA_NO_AUTODIFF(MlaKvInsertFp8, "MlaKvInsertFp8")

} // namespace mlx::core
