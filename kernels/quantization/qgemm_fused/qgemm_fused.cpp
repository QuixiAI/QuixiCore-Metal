// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemm_fused/qgemm_fused.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool qgf_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}
} // namespace

array qgemm_w2a8_fused(const array& wq, const array& x, StreamOrDevice s) {
  if (wq.dtype() != uint8 || !qgf_is_float(x.dtype()) || x.ndim() != 2 || wq.ndim() < 2) {
    throw std::invalid_argument("qgemm_w2a8_fused: expected wq uint8 and x float (M,K)");
  }
  const int M = x.shape(0), K = x.shape(1), N = wq.shape(0);
  const long row_bytes = wq.size() / N;
  if (static_cast<int>(row_bytes / 10 * 32) != K || K % 32 != 0 || K > 8192) {
    throw std::invalid_argument("qgemm_w2a8_fused: wq K mismatch or unsupported K");
  }
  return array({M, N}, float16, std::make_shared<QGemmW2A8Fused>(to_stream(s)),
               {contiguous(wq, false, s), contiguous(x, false, s)});
}

void QGemmW2A8Fused::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QGemmW2A8Fused has no CPU implementation.");
}

void QGemmW2A8Fused::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0];
  auto& x = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_w2a8_fused(enc, out, wq, x, x.shape(0), wq.shape(0), x.shape(1),
                              type_to_name(x));
}

#define TK_QGEMM_FUSED_NO_AUTODIFF(CLASS, LABEL)                              \
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

TK_QGEMM_FUSED_NO_AUTODIFF(QGemmW2A8Fused, "QGemmW2A8Fused")

} // namespace mlx::core
