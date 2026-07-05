// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Causal (lower-triangular) flash-attention forward.
 *  q, k, v are (B, H, N, D), bf16; D in {64, 128}, N a multiple of 8.
 **/
array attn_causal(
    const array& q,
    const array& k,
    const array& v,
    float softcap = 0.0f,
    const std::optional<array>& sinks = std::nullopt,
    StreamOrDevice s = {}
);

/**
 *  Sliding-window causal flash-attention forward: a query at position i attends keys
 *  j in [max(0, i-window+1), i] (the `window` most recent tokens including self).
 *  window <= 0 disables the window (== attn_causal). q, k, v (B,H,N,D) bf16; D in {64,128}.
 **/
array attn_window(
    const array& q,
    const array& k,
    const array& v,
    int window,
    float softcap = 0.0f,
    const std::optional<array>& sinks = std::nullopt,
    StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class AttnWindow : public Primitive {
 public:
  explicit AttnWindow(Stream stream, int window, float softcap = 0.0f, bool has_sink = false)
      : Primitive(stream), window_(window), softcap_(softcap), has_sink_(has_sink) {};

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnWindow"; }
  void print(std::ostream& os) override { os << "AttnWindow[" << window_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const AttnWindow&>(other);
    return window_ == o.window_ && softcap_ == o.softcap_ && has_sink_ == o.has_sink_;
  }

 private:
  int window_;
  float softcap_;
  bool has_sink_;
};

class AttnCausal : public Primitive {
 public:
  explicit AttnCausal(Stream stream, float softcap = 0.0f, bool has_sink = false)
      : Primitive(stream), softcap_(softcap), has_sink_(has_sink) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;
  const char* name() const { return "AttnCausal"; }


  void print(std::ostream& os) override {
    os << "AttnCausal";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float softcap_;
  bool has_sink_;
};

} // namespace mlx::core
