// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemm_bwd/qgemm_bwd.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array qgemm_bwd(const array& grad_y, const array& wq, const std::string& format,
                StreamOrDevice s) {
  if (format != "bitnet") {
    throw std::invalid_argument("qgemm_bwd: only bitnet format is instantiated");
  }
  if (grad_y.dtype() != float16 || wq.dtype() != uint8 || grad_y.ndim() != 2 ||
      wq.ndim() < 2) {
    throw std::invalid_argument("qgemm_bwd: expected grad_y float16 (M,N), wq uint8");
  }
  const int M = grad_y.shape(0);
  const int N = wq.shape(0);
  const long row_bytes = wq.size() / N;
  const int K = static_cast<int>(row_bytes / 10 * 32);
  if (grad_y.shape(1) != N || M % 32 != 0 || N % 32 != 0 || K % 32 != 0) {
    throw std::invalid_argument("qgemm_bwd: requires grad_y (M,N), M/N/K % 32 == 0");
  }
  return array({M, K}, float16, std::make_shared<QGemmBwd>(to_stream(s), format),
               {contiguous(grad_y, false, s), contiguous(wq, false, s)});
}

void QGemmBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QGemmBwd has no CPU implementation.");
}

void QGemmBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& g = inputs[0];
  auto& wq = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_bwd(enc, out, g, wq, g.shape(0), wq.shape(0), out.shape(1), fmt_);
}

#define TK_QGEMM_BWD_NO_AUTODIFF(CLASS, LABEL)                                \
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

TK_QGEMM_BWD_NO_AUTODIFF(QGemmBwd, "QGemmBwd")

} // namespace mlx::core
