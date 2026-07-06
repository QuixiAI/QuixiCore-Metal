// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rope_kv/rope_kv.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

std::vector<array> rope_kv_insert(
    const array& k,
    const array& v,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& key_cache,
    const array& value_cache,
    StreamOrDevice s /* = {} */) {
  if (k.ndim() != 3 || v.ndim() != 3 || k.shape() != v.shape()) {
    throw std::invalid_argument("rope_kv_insert: k/v must have shape (num_tokens, num_kv_heads, D)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument(
        "rope_kv_insert: caches must have shape (num_blocks, block_size, num_kv_heads, D)");
  }
  const int D = k.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("rope_kv_insert: D must be 64 or 128");
  }
  if (key_cache.shape(2) != k.shape(1) || key_cache.shape(3) != D) {
    throw std::invalid_argument("rope_kv_insert: cache heads/head_size must match k");
  }
  if (cos.ndim() != 2 || cos.shape(1) != D / 2 || cos.shape() != sin.shape()) {
    throw std::invalid_argument("rope_kv_insert: cos/sin must have shape (P, D/2)");
  }
  if (positions.ndim() != 1 || positions.shape(0) != k.shape(0)) {
    throw std::invalid_argument("rope_kv_insert: positions must have shape (num_tokens,)");
  }
  if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != k.shape(0)) {
    throw std::invalid_argument("rope_kv_insert: slot_mapping must have shape (num_tokens,)");
  }
  auto dtype = k.dtype();
  if (!(dtype == float32 || dtype == float16 || dtype == bfloat16)) {
    throw std::invalid_argument("rope_kv_insert: k must be float32, float16, or bfloat16");
  }

  auto k_c = contiguous(astype(k, dtype, s), false, s);
  auto v_c = contiguous(astype(v, dtype, s), false, s);
  auto cos_c = contiguous(astype(cos, dtype, s), false, s);
  auto sin_c = contiguous(astype(sin, dtype, s), false, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto kc_c = contiguous(astype(key_cache, dtype, s), false, s);
  auto vc_c = contiguous(astype(value_cache, dtype, s), false, s);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()},
      {dtype, dtype},
      std::make_shared<RopeKvInsert>(to_stream(s)),
      {k_c, v_c, cos_c, sin_c, pos_c, slot_c, kc_c, vc_c});
}

void RopeKvInsert::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RopeKvInsert has no CPU implementation.");
}

void RopeKvInsert::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& k = inputs[0];
  auto& v = inputs[1];
  auto& cos = inputs[2];
  auto& sin = inputs[3];
  auto& positions = inputs[4];
  auto& slot_mapping = inputs[5];
  auto& key_cache_in = inputs[6];
  auto& value_cache_in = inputs[7];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  const int num_tokens = k.shape(0);
  const int num_kv_heads = k.shape(1);
  const int D = k.shape(2);
  const int block_size = key_cache_in.shape(1);
  const std::string tn = type_to_name(key_out);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // Copy the existing caches through, then overwrite the inserted slot rows.
  const uint64_t total = static_cast<uint64_t>(key_out.size());
  tk::launch_kv_cache_clone(enc, key_cache_in, value_cache_in, key_out, value_out, total, tn);
  tk::launch_rope_kv_insert(
      enc, k, v, cos, sin, positions, slot_mapping, key_out, value_out,
      num_tokens * num_kv_heads, num_kv_heads, block_size, D, tn);
}

std::vector<array> RopeKvInsert::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsert has no jvp implementation.");
}
std::vector<array> RopeKvInsert::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RopeKvInsert has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RopeKvInsert::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsert has no vmap implementation.");
}

// --------------------------- rope_kv_insert_norm ---------------------------

std::vector<array> rope_kv_insert_norm(
    const array& k, const array& v, const array& cos, const array& sin, const array& positions,
    const array& slot_mapping, const array& key_cache, const array& value_cache,
    const array& norm_weight, float eps, bool gemma, StreamOrDevice s /* = {} */) {
  if (k.ndim() != 3 || v.ndim() != 3 || k.shape() != v.shape()) {
    throw std::invalid_argument("rope_kv_insert_norm: k/v must be (num_tokens, num_kv_heads, D)");
  }
  const int D = k.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("rope_kv_insert_norm: D must be 64 or 128");
  }
  if (key_cache.ndim() != 4 || key_cache.shape() != value_cache.shape() ||
      key_cache.shape(2) != k.shape(1) || key_cache.shape(3) != D) {
    throw std::invalid_argument("rope_kv_insert_norm: cache must be (num_blocks, block_size, num_kv_heads, D)");
  }
  if (cos.ndim() != 2 || cos.shape(1) != D / 2 || cos.shape() != sin.shape()) {
    throw std::invalid_argument("rope_kv_insert_norm: cos/sin must be (P, D/2)");
  }
  if (norm_weight.ndim() != 1 || norm_weight.shape(0) != D) {
    throw std::invalid_argument("rope_kv_insert_norm: norm_weight must be (D,)");
  }
  auto dtype = k.dtype();
  if (!(dtype == float32 || dtype == float16 || dtype == bfloat16)) {
    throw std::invalid_argument("rope_kv_insert_norm: k must be float32, float16, or bfloat16");
  }

  auto k_c = contiguous(astype(k, dtype, s), false, s);
  auto v_c = contiguous(astype(v, dtype, s), false, s);
  auto cos_c = contiguous(astype(cos, dtype, s), false, s);
  auto sin_c = contiguous(astype(sin, dtype, s), false, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto kc_c = contiguous(astype(key_cache, dtype, s), false, s);
  auto vc_c = contiguous(astype(value_cache, dtype, s), false, s);
  auto w_c = contiguous(astype(norm_weight, dtype, s), false, s);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()}, {dtype, dtype},
      std::make_shared<RopeKvInsertNorm>(to_stream(s), eps, gemma),
      {k_c, v_c, cos_c, sin_c, pos_c, slot_c, kc_c, vc_c, w_c});
}

void RopeKvInsertNorm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RopeKvInsertNorm has no CPU implementation.");
}

void RopeKvInsertNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& k = inputs[0];
  auto& v = inputs[1];
  auto& cos = inputs[2];
  auto& sin = inputs[3];
  auto& positions = inputs[4];
  auto& slot_mapping = inputs[5];
  auto& key_cache_in = inputs[6];
  auto& value_cache_in = inputs[7];
  auto& norm_weight = inputs[8];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  const int num_tokens = k.shape(0);
  const int num_kv_heads = k.shape(1);
  const int D = k.shape(2);
  const int block_size = key_cache_in.shape(1);
  const std::string tn = type_to_name(key_out);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const uint64_t total = static_cast<uint64_t>(key_out.size());
  tk::launch_kv_cache_clone(enc, key_cache_in, value_cache_in, key_out, value_out, total, tn);
  tk::launch_rope_kv_insert_norm(
      enc, k, v, cos, sin, positions, slot_mapping, key_out, value_out, norm_weight,
      num_tokens * num_kv_heads, num_kv_heads, block_size, D, eps_, gemma_ ? 1 : 0, tn);
}

std::vector<array> RopeKvInsertNorm::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsertNorm has no jvp implementation.");
}
std::vector<array> RopeKvInsertNorm::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RopeKvInsertNorm has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RopeKvInsertNorm::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsertNorm has no vmap implementation.");
}

// ------------------------------- rope_q -------------------------------

array rope_q(
    const array& q, const array& cos, const array& sin, const array& positions,
    const array& norm_weight, bool do_norm, bool gemma, float eps, StreamOrDevice s) {
  if (q.ndim() != 3) {
    throw std::invalid_argument("rope_q: q must be (num_tokens, num_q_heads, D)");
  }
  const int D = q.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("rope_q: D must be 64 or 128");
  }
  if (cos.ndim() != 2 || cos.shape(1) != D / 2 || cos.shape() != sin.shape()) {
    throw std::invalid_argument("rope_q: cos/sin must be (P, D/2)");
  }
  if (positions.ndim() != 1 || positions.shape(0) != q.shape(0)) {
    throw std::invalid_argument("rope_q: positions must be (num_tokens,)");
  }
  auto dtype = q.dtype();
  if (!(dtype == float32 || dtype == float16 || dtype == bfloat16)) {
    throw std::invalid_argument("rope_q: q must be float32, float16, or bfloat16");
  }
  auto q_c = contiguous(astype(q, dtype, s), false, s);
  auto cos_c = contiguous(astype(cos, dtype, s), false, s);
  auto sin_c = contiguous(astype(sin, dtype, s), false, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto w_c = contiguous(astype(norm_weight, dtype, s), false, s);
  return array(
      q.shape(), dtype, std::make_shared<RopeQ>(to_stream(s), do_norm, gemma, eps),
      {q_c, cos_c, sin_c, pos_c, w_c});
}

void RopeQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RopeQ has no CPU implementation.");
}

void RopeQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& positions = inputs[3];
  auto& norm_weight = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int num_heads = q.shape(1);
  const int D = q.shape(2);
  const int M = static_cast<int>(q.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rope_q(enc, q, cos, sin, positions, out, norm_weight, M, num_heads,
                    do_norm_ ? 1 : 0, gemma_ ? 1 : 0, eps_, D, type_to_name(q));
}

std::vector<array> RopeQ::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeQ has no jvp implementation.");
}
std::vector<array> RopeQ::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RopeQ has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RopeQ::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeQ has no vmap implementation.");
}

} // namespace mlx::core
