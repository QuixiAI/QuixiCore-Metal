// Copyright © 2023 Apple Inc.

#pragma once

#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Quantized GEMM (Marlin's method): out = dequantize(wq) @ x.
 *  wq is packed weight blocks of shape (N, K/block_k, block_bytes) uint8 for the given
 *  `format` (e.g. "q8_0"); x is (K, M) float16; out is (N, M) float16. Dequant-to-shared
 *  then a standard simdgroup MMA. Shapes: N%32, M%32, K%block_k. */
array qgemm(const array& wq, const array& x, const std::string& format = "q8_0",
            StreamOrDevice s = {});

class QGemm : public Primitive {
 public:
  explicit QGemm(Stream stream, std::string format)
      : Primitive(stream), fmt_(std::move(format)) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "QGemm[" << fmt_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);

 private:
  std::string fmt_;
};

} // namespace mlx::core
