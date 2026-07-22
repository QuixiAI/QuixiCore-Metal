// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  attn_fwd_sg_d256: simdgroup_matrix (MMA) flash attention, head-dim 256, GQA, f16 KV.
 *
 *  Bidirectional (non-causal) attention with an optional symmetric sliding window, built on
 *  raw simdgroup_float8x8 tiles rather than TK register tiles (head_dim=256 is wider than the
 *  TK attn_fwd D in {64,128}). q/o are (T, Hq, 256) float32; k/v are (T, Hkv, 256) float16.
 *  Hq must be a multiple of Hkv (GQA group Hq/Hkv). Q is scaled by `scale` (<=0 -> 1/sqrt(256)).
 *  window == 0 is full attention; window > 0 keeps keys within window/2 of the query.
 **/
array attn_fwd_sg_d256(
    const array& q,
    const array& k,
    const array& v,
    float scale = 0.0f,
    int window = 0,
    StreamOrDevice s = {});

class AttnFwdSgD256 : public Primitive {
 public:
  AttnFwdSgD256(Stream stream, float scale, int window, int n_tokens)
      : Primitive(stream), scale_(scale), window_(window), n_tokens_(n_tokens) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "AttnFwdSgD256"; }
  void print(std::ostream& os) override { os << "AttnFwdSgD256"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const AttnFwdSgD256&>(other);
    return scale_ == o.scale_ && window_ == o.window_ && n_tokens_ == o.n_tokens_;
  }

 private:
  float scale_;
  int window_;
  int n_tokens_;
};

}  // namespace mlx::core
