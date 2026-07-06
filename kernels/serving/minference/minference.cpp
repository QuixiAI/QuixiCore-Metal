// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "minference/minference.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array minference_block_mask(
    const array& vertical_indexes, const array& slash_indexes, const array& context_lens,
    int max_blocks, int block_size, int vertical_topk, int slash_topk, int last_n_blocks,
    StreamOrDevice s) {
  if (vertical_indexes.ndim() != 3 || slash_indexes.ndim() != 3 ||
      vertical_indexes.shape(0) != slash_indexes.shape(0) ||
      vertical_indexes.shape(1) != slash_indexes.shape(1)) {
    throw std::invalid_argument(
        "minference_block_mask: vertical/slash indexes must be (batch, num_heads, nnz)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != vertical_indexes.shape(0)) {
    throw std::invalid_argument("minference_block_mask: context_lens must be (batch,)");
  }
  if (max_blocks <= 0 || block_size <= 0) {
    throw std::invalid_argument("minference_block_mask: max_blocks/block_size must be > 0");
  }
  const int B = vertical_indexes.shape(0);
  const int H = vertical_indexes.shape(1);
  return array(
      {B, H, max_blocks}, int32,
      std::make_shared<MinferenceBlockMask>(to_stream(s), max_blocks, block_size,
                                            vertical_topk, slash_topk, last_n_blocks),
      {contiguous(astype(vertical_indexes, int32, s), false, s),
       contiguous(astype(slash_indexes, int32, s), false, s),
       contiguous(astype(context_lens, int32, s), false, s)});
}

void MinferenceBlockMask::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MinferenceBlockMask has no CPU implementation.");
}

void MinferenceBlockMask::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& vert = inputs[0];
  auto& slash = inputs[1];
  auto& lens = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = vert.shape(0);
  const int H = vert.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_minference_block_mask(enc, vert, slash, lens, out, B, H, vert.shape(2),
                                   slash.shape(2), vertical_topk_, slash_topk_, block_size_,
                                   max_blocks_, last_n_blocks_);
}

std::vector<array> MinferenceBlockMask::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MinferenceBlockMask has no jvp implementation.");
}
std::vector<array> MinferenceBlockMask::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("MinferenceBlockMask has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MinferenceBlockMask::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MinferenceBlockMask has no vmap implementation.");
}

} // namespace mlx::core
