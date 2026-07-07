// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "gemm_v3/gemm_v3.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array gemm_v3(const array& x, const array& y, StreamOrDevice s) {
  if (x.dtype() != y.dtype() || (x.dtype() != float32 && x.dtype() != bfloat16)) {
    throw std::invalid_argument("gemm_v3: dtype must be float32 or bfloat16");
  }
  if (x.ndim() != 2 || y.ndim() != 2 || x.shape(1) != y.shape(0)) {
    throw std::invalid_argument("gemm_v3: expected x (N,K), y (K,M)");
  }
  const int N = x.shape(0), K = x.shape(1), M = y.shape(1);
  if (N % 64 != 0 || M % 64 != 0 || K % 32 != 0) {
    throw std::invalid_argument("gemm_v3: requires N%64, M%64, K%32");
  }
  return array({N, M}, x.dtype(), std::make_shared<GemmV3>(to_stream(s)),
               {contiguous(x, false, s), contiguous(y, false, s)});
}

void GemmV3::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GemmV3 has no CPU implementation.");
}

void GemmV3::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gemm_v3(enc, out, x, y, x.shape(0), x.shape(1), y.shape(1), type_to_name(x));
}

#define TK_GEMM_V3_NO_AUTODIFF(CLASS, LABEL)                                  \
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

TK_GEMM_V3_NO_AUTODIFF(GemmV3, "GemmV3")

} // namespace mlx::core
