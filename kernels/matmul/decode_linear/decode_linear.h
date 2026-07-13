#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array decode_linear(const array& x, const array& weight, const array& bias,
                    bool gelu = false, StreamOrDevice s = {});
array decode_linear_residual(const array& x, const array& weight, const array& bias,
                             const array& residual, StreamOrDevice s = {});
array decode_linear_q8(const array& x, const array& packed_weight, const array& bias,
                       const array& residual, bool gelu = false, bool use_residual = false,
                       StreamOrDevice s = {});

class DecodeLinear : public Primitive {
 public:
  DecodeLinear(Stream stream, bool gelu) : Primitive(stream), gelu_(gelu) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeLinear"; }
  void print(std::ostream& os) override { os << "DecodeLinear"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  bool gelu_;
};

class DecodeLinearResidual : public Primitive {
 public:
  explicit DecodeLinearResidual(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeLinearResidual"; }
  void print(std::ostream& os) override { os << "DecodeLinearResidual"; }
  bool is_equivalent(const Primitive& other) const override {
    return typeid(*this) == typeid(other);
  }
};

class DecodeLinearQ8 : public Primitive {
 public:
  DecodeLinearQ8(Stream stream, bool gelu, bool use_residual)
      : Primitive(stream), gelu_(gelu), use_residual_(use_residual) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeLinearQ8"; }
  void print(std::ostream& os) override { os << "DecodeLinearQ8"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  bool gelu_;
  bool use_residual_;
};

} // namespace mlx::core
