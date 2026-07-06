// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Decay / retention linear attention (RetNet / Lightning-Attention-2), materialized chunked form:
 *  out_i = sum_{j<=i} exp(decay_i - decay_j) * (q_i . k_j) * v_j.
 *  q,k,v are (B,H,N,D) bf16; cl (B,H,N) fp32 is the decay-log ramp = -slope*position (the tk wrapper
 *  builds it from a per-head slope); out (B,H,N,D). D=64, N a multiple of 8. */
array lin_attn_decay(const array& q, const array& k, const array& v, const array& cl,
                     StreamOrDevice s = {});

class LinAttnDecay : public Primitive {
 public:
  explicit LinAttnDecay(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LinAttnDecay"; }

  void print(std::ostream& os) override { os << "LinAttnDecay"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
