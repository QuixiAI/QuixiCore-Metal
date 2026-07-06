// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  GDN / GatedDeltaNet linear attention (delta-rule recurrence). Varlen packed inputs:
 *  q/k (total_tokens, Hk, Dk), v (total_tokens, Hv, Dv), g/beta (total_tokens, Hv)
 *  (g is the decay MULTIPLIER in (0,1], not a log); cu_seqlens (R+1,) int32;
 *  state_pool (num_slots, Hv, Dv, Dk) fp32 indexed by slot_mapping (R,) int32.
 *  Dk in {64,128}; Hv % Hk == 0 (GQA). load_initial: read the pool (decode/continuation)
 *  vs start at 0 (fresh prefill). Returns [y (total_tokens, Hv, Dv), new_state_pool]
 *  (functional: the pool is clone-updated, untouched slots preserved).
 **/
std::vector<array> gdn_recur(
    const array& q, const array& k, const array& v, const array& g, const array& beta,
    const array& state_pool, const array& cu_seqlens, const array& slot_mapping,
    bool load_initial, StreamOrDevice s = {});

class GdnRecur : public Primitive {
 public:
  GdnRecur(Stream stream, bool load_initial)
      : Primitive(stream), load_initial_(load_initial) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnRecur"; }
  void print(std::ostream& os) override { os << "GdnRecur"; }
  bool is_equivalent(const Primitive& other) const override {
    return load_initial_ == static_cast<const GdnRecur&>(other).load_initial_;
  }

 private:
  bool load_initial_;
};

} // namespace mlx::core
