// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "turboquant/turboquant.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
std::string tq_tname(const array& x) {
  if (x.dtype() == float32) return "float32";
  if (x.dtype() == float16) return "float16";
  if (x.dtype() == bfloat16) return "bfloat16";
  throw std::invalid_argument("turboquant: key/value dtype must be f32/f16/bf16");
}
void tq_check_hs(int hs) {
  if (!(hs == 64 || hs == 128 || hs == 256)) {
    throw std::invalid_argument("turboquant: head_size must be 64, 128, or 256");
  }
}
} // namespace

std::vector<array> tq_encode(
    const array& key, const array& value, const array& key_cache, const array& value_cache,
    const array& key_scale, const array& value_scale, const array& key_zero,
    const array& slot_mapping, const array& v_centroids, const array& signs,
    int block_size, int k_bits, bool k_signed, int v_bits, StreamOrDevice s) {
  if (key.ndim() != 3 || value.shape() != key.shape()) {
    throw std::invalid_argument("tq_encode: key/value must be (tokens, num_kv_heads, head_size)");
  }
  const int head_size = key.shape(2);
  tq_check_hs(head_size);
  if (k_bits < 2 || k_bits > 8 || v_bits < 2 || v_bits > 8) {
    throw std::invalid_argument("tq_encode: k_bits/v_bits must be in [2, 8]");
  }
  auto tn = tq_tname(key);
  return array::make_arrays(
      {key_cache.shape(), value_cache.shape(), key_scale.shape(), value_scale.shape(),
       key_zero.shape()},
      {uint8, uint8, float16, float16, float16},
      std::make_shared<TQEncode>(to_stream(s), block_size, k_bits, k_signed, v_bits),
      {contiguous(key, false, s), contiguous(astype(value, key.dtype(), s), false, s),
       contiguous(astype(key_cache, uint8, s), false, s),
       contiguous(astype(value_cache, uint8, s), false, s),
       contiguous(astype(key_scale, float16, s), false, s),
       contiguous(astype(value_scale, float16, s), false, s),
       contiguous(astype(key_zero, float16, s), false, s),
       contiguous(astype(slot_mapping, int32, s), false, s),
       contiguous(astype(v_centroids, float32, s), false, s),
       contiguous(astype(signs, float32, s), false, s)});
}

std::vector<array> tq_decode(
    const array& key_cache, const array& value_cache, const array& key_scale,
    const array& value_scale, const array& key_zero, const array& slots,
    const array& v_centroids, const array& signs, int num_kv_heads, int head_size,
    int block_size, int k_bits, bool k_signed, int v_bits, StreamOrDevice s) {
  tq_check_hs(head_size);
  const int n = slots.shape(0);
  auto tn = tq_tname(signs.dtype() == float32 ? v_centroids : v_centroids);  // dtype from arg
  // output dtype: caller passes float32 signs; produce float32 rows (highest fidelity).
  (void)tn;
  return array::make_arrays(
      {{n, num_kv_heads, head_size}, {n, num_kv_heads, head_size}},
      {float32, float32},
      std::make_shared<TQDecode>(to_stream(s), num_kv_heads, head_size, block_size, k_bits,
                                 k_signed, v_bits),
      {contiguous(astype(key_cache, uint8, s), false, s),
       contiguous(astype(value_cache, uint8, s), false, s),
       contiguous(astype(key_scale, float16, s), false, s),
       contiguous(astype(value_scale, float16, s), false, s),
       contiguous(astype(key_zero, float16, s), false, s),
       contiguous(astype(slots, int32, s), false, s),
       contiguous(astype(v_centroids, float32, s), false, s),
       contiguous(astype(signs, float32, s), false, s)});
}

void TQEncode::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TQEncode has no CPU implementation.");
}
void TQEncode::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& key = inputs[0];
  auto& value = inputs[1];
  auto& slot_mapping = inputs[7];
  auto& v_centroids = inputs[8];
  auto& signs = inputs[9];
  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto& o : outputs) o.set_data(allocator::malloc_or_wait(o.nbytes()));
  const int num_tokens = key.shape(0);
  const int num_kv_heads = key.shape(1);
  const int head_size = key.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // functional: clone the five caches, then update the clones in place at slot offsets.
  tk::launch_tq_clone_bytes(enc, inputs[2], outputs[0], (uint32_t)inputs[2].nbytes());
  tk::launch_tq_clone_bytes(enc, inputs[3], outputs[1], (uint32_t)inputs[3].nbytes());
  tk::launch_tq_clone_bytes(enc, inputs[4], outputs[2], (uint32_t)inputs[4].nbytes());
  tk::launch_tq_clone_bytes(enc, inputs[5], outputs[3], (uint32_t)inputs[5].nbytes());
  tk::launch_tq_clone_bytes(enc, inputs[6], outputs[4], (uint32_t)inputs[6].nbytes());
  tk::launch_tq_encode(enc, key, value, outputs[0], outputs[1], outputs[2], outputs[3],
                       outputs[4], slot_mapping, v_centroids, signs, num_tokens, num_kv_heads,
                       head_size, block_size_, k_bits_, k_signed_ ? 1 : 0, v_bits_,
                       tq_tname(key));
}

void TQDecode::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TQDecode has no CPU implementation.");
}
void TQDecode::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto& o : outputs) o.set_data(allocator::malloc_or_wait(o.nbytes()));
  const int n = inputs[5].shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_tq_decode(enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                       inputs[6], inputs[7], outputs[0], outputs[1], n, num_kv_heads_,
                       head_size_, block_size_, k_bits_, k_signed_ ? 1 : 0, v_bits_,
                       "float32");
}

std::vector<array> TQEncode::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TQEncode has no jvp."); }
std::vector<array> TQEncode::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("TQEncode has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> TQEncode::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TQEncode has no vmap."); }

std::vector<array> TQDecode::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TQDecode has no jvp."); }
std::vector<array> TQDecode::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("TQDecode has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> TQDecode::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TQDecode has no vmap."); }

} // namespace mlx::core
