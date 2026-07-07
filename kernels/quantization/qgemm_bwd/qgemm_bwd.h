// Copyright © 2023 Apple Inc.

#pragma once

#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array qgemm_bwd(const array& grad_y, const array& wq, const std::string& format = "bitnet",
                StreamOrDevice s = {});

class QGemmBwd : public Primitive {
 public:
  QGemmBwd(Stream stream, std::string format) : Primitive(stream), fmt_(std::move(format)) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemmBwd"; }
  void print(std::ostream& os) override { os << "QGemmBwd[" << fmt_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    return fmt_ == static_cast<const QGemmBwd&>(other).fmt_;
  }

 private:
  std::string fmt_;
};

} // namespace mlx::core
