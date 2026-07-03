// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "dropout/dropout.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool drop_is_float(Dtype dt) {
  return dt == float32 || dt == float16 || dt == bfloat16;
}
static void check_p(float p) {
  if (!(p >= 0.0f && p < 1.0f)) {
    throw std::invalid_argument("dropout: p must be in [0, 1)");
  }
}

array dropout(const array& x, float p, uint32_t seed, StreamOrDevice s /* = {} */) {
  if (!drop_is_float(x.dtype())) {
    throw std::invalid_argument("dropout: x must be float (fp16/bf16/fp32)");
  }
  check_p(p);
  auto x_c = contiguous(x, false, s);
  return array(x.shape(), x.dtype(), std::make_shared<Dropout>(to_stream(s), p, seed), {x_c});
}

array dropout_backward(const array& dy, float p, uint32_t seed, StreamOrDevice s /* = {} */) {
  if (!drop_is_float(dy.dtype())) {
    throw std::invalid_argument("dropout_backward: dy must be float (fp16/bf16/fp32)");
  }
  check_p(p);
  auto dy_c = contiguous(dy, false, s);
  return array(dy.shape(), dy.dtype(), std::make_shared<DropoutBwd>(to_stream(s), p, seed), {dy_c});
}

void Dropout::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Dropout has no CPU implementation.");
}
void Dropout::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_dropout(enc, x, out, seed_, p_, static_cast<uint32_t>(x.size()), false,
                     type_to_name(x));
}

void DropoutBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DropoutBwd has no CPU implementation.");
}
void DropoutBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& dy = inputs[0];
  auto& dx = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_dropout(enc, dy, dx, seed_, p_, static_cast<uint32_t>(dy.size()), true,
                     type_to_name(dy));
}

#define TK_DROPOUT_NO_AD(CLASS)                                                    \
  std::vector<array> CLASS::jvp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&) {                         \
    throw std::runtime_error(#CLASS " has no jvp implementation.");                \
  }                                                                               \
  std::vector<array> CLASS::vjp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&, const std::vector<array>&) { \
    throw std::runtime_error(#CLASS " has no vjp implementation.");                \
  }                                                                               \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                     \
      const std::vector<array>&, const std::vector<int>&) {                       \
    throw std::runtime_error(#CLASS " has no vmap implementation.");              \
  }

TK_DROPOUT_NO_AD(Dropout)
TK_DROPOUT_NO_AD(DropoutBwd)

} // namespace mlx::core
