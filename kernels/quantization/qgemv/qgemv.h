// Copyright © 2023 Apple Inc.

#pragma once

#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Quantized GEMV (batch-1 decode): out = dequantize(wq) @ x. wq packed weight blocks
 *  (N, K/block_k, block_bytes) uint8 for `format`; x is (K, 1) float16; out is (N, 1) float16. */
array qgemv(const array& wq, const array& x, const std::string& format = "q8_0",
            StreamOrDevice s = {});

class QGemv : public Primitive {
 public:
  explicit QGemv(Stream stream, std::string format)
      : Primitive(stream), fmt_(std::move(format)) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemv"; }

  void print(std::ostream& os) override { os << "QGemv[" << fmt_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);

 private:
  std::string fmt_;
};

} // namespace mlx::core
