#pragma once

#include <string>
#include <utility>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array dequant_gather(
    const array& table,
    const array& ids,
    const std::string& format,
    float scale = 1.0f,
    StreamOrDevice s = {});

class DequantGather : public Primitive {
 public:
  DequantGather(Stream stream, std::string format, int rows, int columns, float scale)
      : Primitive(stream), format_(std::move(format)), rows_(rows), columns_(columns),
        scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DequantGather"; }
  void print(std::ostream& os) override { os << "DequantGather[" << format_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::string format_;
  int rows_;
  int columns_;
  float scale_;
};

} // namespace mlx::core
