#pragma once

#include <string>
#include <utility>

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
array decode_linear_epilogue(
    const array& x, const array& weight, const array& bias,
    const array& residual, const std::string& format = "",
    int activation = 0, bool use_bias = false, bool use_residual = false,
    StreamOrDevice s = {});
array decode_swiglu(
    const array& x, const array& gate_weight, const array& up_weight,
    const array& gate_bias, const array& up_bias,
    const std::string& format = "", bool use_bias = false,
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

class DecodeLinearEpilogue : public Primitive {
 public:
  DecodeLinearEpilogue(Stream stream, std::string format, int activation,
                       bool use_bias, bool use_residual)
      : Primitive(stream), format_(std::move(format)), activation_(activation),
        use_bias_(use_bias), use_residual_(use_residual) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeLinearEpilogue"; }
  void print(std::ostream& os) override { os << "DecodeLinearEpilogue"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  std::string format_;
  int activation_;
  bool use_bias_;
  bool use_residual_;
};

class DecodeSwiGLU : public Primitive {
 public:
  DecodeSwiGLU(Stream stream, std::string format, bool use_bias)
      : Primitive(stream), format_(std::move(format)), use_bias_(use_bias) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeSwiGLU"; }
  void print(std::ostream& os) override { os << "DecodeSwiGLU"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  std::string format_;
  bool use_bias_;
};

} // namespace mlx::core
