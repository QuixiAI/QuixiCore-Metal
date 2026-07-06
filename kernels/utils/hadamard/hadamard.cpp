// Copyright © 2023-2024 Apple Inc.

#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "hadamard/hadamard.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array hadamard(const array& x, float scale, StreamOrDevice s) {
  if (x.ndim() == 0 || x.size() == 0) {
    throw std::invalid_argument("hadamard: input must be non-empty with a final axis");
  }
  const int D = x.shape(-1);
  if (!(D == 64 || D == 128 || D == 256 || D == 512)) {
    throw std::invalid_argument("hadamard: final axis must be 64, 128, 256, or 512");
  }
  if (!(x.dtype() == float32 || x.dtype() == float16 || x.dtype() == bfloat16)) {
    throw std::invalid_argument("hadamard: dtype must be float32, float16, or bfloat16");
  }

  auto x_c = contiguous(x, false, s);
  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<HadamardTK>(to_stream(s), scale),
      {x_c});
}

void HadamardTK::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("HadamardTK has no CPU implementation.");
}

void HadamardTK::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  const float scale = scale_ > 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(D));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_hadamard(enc, x, out, rows, D, scale, type_to_name(x));
}

std::vector<array> HadamardTK::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("HadamardTK has no jvp implementation.");
}

std::vector<array> HadamardTK::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("HadamardTK has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> HadamardTK::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("HadamardTK has no vmap implementation.");
}

} // namespace mlx::core
