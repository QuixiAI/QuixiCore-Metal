// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Multi-warp flash-attention forward (non-causal): N_WARPS=4 simdgroups per
 *  threadgroup share each K/V block via threadgroup memory. q,k,v are (B,H,N,D),
 *  bf16; D in {64,128}, N a multiple of 32. */
array attn_multiwarp(const array& q, const array& k, const array& v, StreamOrDevice s = {});

class AttnMultiwarp : public Primitive {
 public:
  explicit AttnMultiwarp(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnMultiwarp"; }

  void print(std::ostream& os) override { os << "AttnMultiwarp"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
