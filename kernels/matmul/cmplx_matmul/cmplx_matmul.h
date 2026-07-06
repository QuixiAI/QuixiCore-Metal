// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Complex GEMM: D = A @ B. Operands carry a leading size-2 (real, imag) axis:
 *  a is (2,N,K), b is (2,K,M), out is (2,N,M). f32/bf16; N%32, M%32, K%16.
 *  Exercises the complex-multiply MMA primitive (complex_mma_AB). */
array cmplx_matmul(const array& a, const array& b, StreamOrDevice s = {});

class CmplxMatmul : public Primitive {
 public:
  explicit CmplxMatmul(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CmplxMatmul"; }

  void print(std::ostream& os) override { os << "CmplxMatmul"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
