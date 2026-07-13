#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array swin_attn_d32(const array& qkv, const array& relative_bias, const array& mask,
                    int windows_per_image = 0, StreamOrDevice s = {});

class SwinAttnD32 : public Primitive {
 public:
  SwinAttnD32(Stream stream, int windows_per_image, bool has_mask)
      : Primitive(stream), windows_per_image_(windows_per_image), has_mask_(has_mask) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SwinAttnD32"; }
  void print(std::ostream& os) override { os << "SwinAttnD32"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int windows_per_image_;
  bool has_mask_;
};

} // namespace mlx::core
