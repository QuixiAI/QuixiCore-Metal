#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "swin_attn/swin_attn.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool swin_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }
}

array swin_attn_d32(const array& qkv, const array& relative_bias, const array& mask,
                    int windows_per_image, StreamOrDevice s) {
  if (!swin_float(qkv.dtype()) || qkv.ndim() != 5 || qkv.shape(2) != 3 || qkv.shape(4) != 32) {
    throw std::invalid_argument("swin_attn_d32: qkv must be (BW,N,3,H,32) float");
  }
  const int BW = qkv.shape(0), N = qkv.shape(1), H = qkv.shape(3);
  if (BW <= 0 || N <= 0 || H <= 0 || windows_per_image < 0) {
    throw std::invalid_argument(
        "swin_attn_d32: BW, N, and H must be positive; windows_per_image cannot be negative");
  }
  if (relative_bias.dtype() != qkv.dtype() || relative_bias.ndim() != 3 ||
      relative_bias.shape(0) != H || relative_bias.shape(1) != N || relative_bias.shape(2) != N) {
    throw std::invalid_argument("swin_attn_d32: relative_bias must be (H,N,N) with qkv dtype");
  }
  const bool has_mask = mask.ndim() == 3;
  if ((!has_mask && mask.size() != 1) ||
      (has_mask && (windows_per_image == 0 || mask.shape(1) != N || mask.shape(2) != N ||
                    windows_per_image != mask.shape(0)))) {
    throw std::invalid_argument(
        "swin_attn_d32: mask must be (windows_per_image,N,N) or a scalar placeholder");
  }
  auto mask_f = contiguous(astype(mask, float32, s), false, s);
  return array(
      {BW, N, H, 32}, qkv.dtype(),
      std::make_shared<SwinAttnD32>(to_stream(s), windows_per_image, has_mask),
      {contiguous(qkv, false, s), contiguous(relative_bias, false, s), mask_f});
}

void SwinAttnD32::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SwinAttnD32 has no CPU implementation.");
}

void SwinAttnD32::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& qkv = inputs[0];
  auto& relative_bias = inputs[1];
  auto& mask = inputs[2];
  auto& output = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  const int BW = qkv.shape(0), N = qkv.shape(1), H = qkv.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_swin_attn_d32(enc, qkv, relative_bias, mask, output, BW, N, H,
                           windows_per_image_, has_mask_ ? 1 : 0, type_to_name(qkv));
}

#define TK_SWIN_NO_AUTODIFF(CLASS, LABEL)                                    \
  std::vector<array> CLASS::jvp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&) {                    \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                           \
  std::vector<array> CLASS::vjp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&, const std::vector<array>&) { \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                           \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                \
      const std::vector<array>&, const std::vector<int>&) {                   \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_SWIN_NO_AUTODIFF(SwinAttnD32, "SwinAttnD32")

bool SwinAttnD32::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const SwinAttnD32&>(other);
  return windows_per_image_ == o.windows_per_image_ && has_mask_ == o.has_mask_;
}

} // namespace mlx::core
