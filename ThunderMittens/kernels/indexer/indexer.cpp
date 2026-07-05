// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "indexer/indexer.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
std::string idx_tname(const array& x) {
  if (x.dtype() == float32) return "float32";
  if (x.dtype() == float16) return "float16";
  if (x.dtype() == bfloat16) return "bfloat16";
  throw std::invalid_argument("indexer: k dtype must be f32/f16/bf16");
}
} // namespace

std::vector<array> indexer_k_quant_and_cache(
    const array& k, const array& slot_mapping, const array& code_cache,
    const array& scale_cache, int quant_block_size, bool ue8m0, StreamOrDevice s) {
  if (k.ndim() != 2) {
    throw std::invalid_argument("indexer_k_quant_and_cache: k must be (tokens, head_dim)");
  }
  const int head_dim = k.shape(1);
  if (quant_block_size <= 0) {
    throw std::invalid_argument("indexer_k_quant_and_cache: quant_block_size must be > 0");
  }
  if (code_cache.ndim() != 2 || code_cache.shape(1) != head_dim) {
    throw std::invalid_argument("indexer_k_quant_and_cache: code_cache must be (num_slots, head_dim)");
  }
  return array::make_arrays(
      {code_cache.shape(), scale_cache.shape()},
      {uint8, float32},
      std::make_shared<IndexerKQuant>(to_stream(s), quant_block_size, ue8m0),
      {contiguous(k, false, s), contiguous(astype(slot_mapping, int32, s), false, s),
       contiguous(astype(code_cache, uint8, s), false, s),
       contiguous(astype(scale_cache, float32, s), false, s)});
}

array indexer_k_gather(
    const array& code_cache, const array& scale_cache, const array& slots, int head_dim,
    int quant_block_size, StreamOrDevice s) {
  const int n = slots.shape(0);
  return array(
      {n, head_dim}, bfloat16,
      std::make_shared<IndexerKGather>(to_stream(s), head_dim, quant_block_size),
      {contiguous(astype(code_cache, uint8, s), false, s),
       contiguous(astype(scale_cache, float32, s), false, s),
       contiguous(astype(slots, int32, s), false, s)});
}

void IndexerKQuant::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("IndexerKQuant has no CPU implementation.");
}
void IndexerKQuant::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& k = inputs[0];
  auto& slot_mapping = inputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));
  outputs[1].set_data(allocator::malloc_or_wait(outputs[1].nbytes()));
  const int num_tokens = k.shape(0);
  const int head_dim = k.shape(1);
  const int nq = (head_dim + qbs_ - 1) / qbs_;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // functional: clone both caches, then update the clones at slot offsets.
  tk::launch_indexer_clone_bytes(enc, inputs[2], outputs[0], (uint32_t)inputs[2].nbytes());
  tk::launch_indexer_clone_bytes(enc, inputs[3], outputs[1], (uint32_t)inputs[3].nbytes());
  tk::launch_indexer_k_quant_and_cache(enc, k, slot_mapping, outputs[0], outputs[1],
                                       num_tokens, head_dim, nq, qbs_, ue8m0_ ? 1 : 0,
                                       idx_tname(k));
}

void IndexerKGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("IndexerKGather has no CPU implementation.");
}
void IndexerKGather::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));
  const int n = inputs[2].shape(0);
  const int nq = (head_dim_ + qbs_ - 1) / qbs_;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_indexer_k_gather(enc, inputs[0], inputs[1], inputs[2], outputs[0], n, head_dim_,
                              nq, qbs_, "bfloat16");
}

std::vector<array> IndexerKQuant::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("IndexerKQuant has no jvp."); }
std::vector<array> IndexerKQuant::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("IndexerKQuant has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> IndexerKQuant::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("IndexerKQuant has no vmap."); }

std::vector<array> IndexerKGather::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("IndexerKGather has no jvp."); }
std::vector<array> IndexerKGather::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("IndexerKGather has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> IndexerKGather::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("IndexerKGather has no vmap."); }

} // namespace mlx::core
