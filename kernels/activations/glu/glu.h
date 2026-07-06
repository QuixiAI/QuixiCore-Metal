// Copyright © 2023 Apple Inc.

#pragma once

#include <string>
#include <utility>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array glu(
    const array& x,
    const array& gate,
    const std::string& mode = "swiglu",
    float alpha = 1.0f,
    float limit = 1.0e20f,
    StreamOrDevice s = {});

// GLU backward: given upstream grad dc (= grad wrt out), returns {da, db} (grads wrt x and gate).
// da = dc*gate*act'(x), db = dc*act(x). x/gate/dc same shape; returns two arrays of that shape.
std::vector<array> glu_backward(
    const array& x,
    const array& gate,
    const array& dc,
    const std::string& mode = "swiglu",
    float alpha = 1.0f,
    float limit = 1.0e20f,
    StreamOrDevice s = {});

class GluBwd : public Primitive {
 public:
  GluBwd(Stream stream, std::string mode, float alpha, float limit)
      : Primitive(stream), mode_(std::move(mode)), alpha_(alpha), limit_(limit) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GluBwd"; }
  void print(std::ostream& os) override { os << "GluBwd[" << mode_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const GluBwd&>(other);
    return mode_ == o.mode_ && alpha_ == o.alpha_ && limit_ == o.limit_;
  }

 private:
  std::string mode_;
  float alpha_;
  float limit_;
};

class Glu : public Primitive {
 public:
  Glu(Stream stream, std::string mode, float alpha, float limit)
      : Primitive(stream), mode_(std::move(mode)), alpha_(alpha), limit_(limit) {};

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
  const char* name() const { return "Glu"; }


  void print(std::ostream& os) override {
    os << "Glu[" << mode_ << "]";
  }

  bool is_equivalent(const Primitive& other) const override;

  const std::string& mode() const { return mode_; }
  float alpha() const { return alpha_; }
  float limit() const { return limit_; }

 private:
  std::string mode_;
  float alpha_;
  float limit_;
};

} // namespace mlx::core
