// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** MInference decode block-mask builder: vertical column indexes + slash diagonal offsets
 *  (per (batch, head), -1 padded) -> per-head block mask (batch, num_heads, max_blocks)
 *  int32 0/1, directly consumable by paged_attention_block_sparse. vertical_topk /
 *  slash_topk cap how many entries of each list are used; last_n_blocks recent KV blocks
 *  are always attended. */
array minference_block_mask(
    const array& vertical_indexes, const array& slash_indexes, const array& context_lens,
    int max_blocks, int block_size, int vertical_topk = 1 << 30, int slash_topk = 1 << 30,
    int last_n_blocks = 1, StreamOrDevice s = {});

class MinferenceBlockMask : public Primitive {
 public:
  MinferenceBlockMask(Stream stream, int max_blocks, int block_size, int vertical_topk,
                      int slash_topk, int last_n_blocks)
      : Primitive(stream), max_blocks_(max_blocks), block_size_(block_size),
        vertical_topk_(vertical_topk), slash_topk_(slash_topk),
        last_n_blocks_(last_n_blocks) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MinferenceBlockMask"; }
  void print(std::ostream& os) override { os << "MinferenceBlockMask"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MinferenceBlockMask&>(other);
    return max_blocks_ == o.max_blocks_ && block_size_ == o.block_size_ &&
           vertical_topk_ == o.vertical_topk_ && slash_topk_ == o.slash_topk_ &&
           last_n_blocks_ == o.last_n_blocks_;
  }

 private:
  int max_blocks_;
  int block_size_;
  int vertical_topk_;
  int slash_topk_;
  int last_n_blocks_;
};

} // namespace mlx::core
