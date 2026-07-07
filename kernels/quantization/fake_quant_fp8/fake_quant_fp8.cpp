// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "fake_quant_fp8/fake_quant_fp8.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool fq8_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}
} // namespace

std::vector<array> fake_quant_fp8(const array& x, StreamOrDevice s) {
  if (!fq8_is_float(x.dtype())) {
    throw std::invalid_argument("fake_quant_fp8: x must be float32/float16/bfloat16");
  }
  return array::make_arrays(
      {x.shape(), {1}, {1}},
      {x.dtype(), float32, uint32},
      std::make_shared<FakeQuantFp8>(to_stream(s)),
      {contiguous(x, false, s)});
}

void FakeQuantFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("FakeQuantFp8 has no CPU implementation.");
}

void FakeQuantFp8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& x_fq = outputs[0];
  auto& scale = outputs[1];
  auto& scale_u = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  x_fq.set_data(allocator::malloc_or_wait(x_fq.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  scale_u.set_data(allocator::malloc_or_wait(scale_u.nbytes()));
  const int n = static_cast<int>(x.size());
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_zero_i32(enc, scale_u, 1);
  tk::launch_quant_tensor_absmax(enc, x, scale_u, n, type_to_name(x));
  tk::launch_fake_quant_fp8(enc, x, scale_u, x_fq, scale, n, type_to_name(x));
}

#define TK_FQ8_NO_AUTODIFF(CLASS, LABEL)                                      \
  std::vector<array> CLASS::jvp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&) {                                               \
    throw std::runtime_error(LABEL " has no jvp implementation.");             \
  }                                                                            \
  std::vector<array> CLASS::vjp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&, const std::vector<array>&) {                    \
    throw std::runtime_error(LABEL " has no vjp implementation.");             \
  }                                                                            \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                 \
      const std::vector<array>&, const std::vector<int>&) {                    \
    throw std::runtime_error(LABEL " has no vmap implementation.");            \
  }

TK_FQ8_NO_AUTODIFF(FakeQuantFp8, "FakeQuantFp8")

} // namespace mlx::core
