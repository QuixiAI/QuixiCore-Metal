// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "ternary_stats/ternary_stats.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array ternary_stats(const array& wq, StreamOrDevice s) {
  if (wq.dtype() != uint8 || wq.ndim() < 2 || wq.shape(-1) != 10) {
    throw std::invalid_argument("ternary_stats: wq must be uint8 (..., nblocks, 10)");
  }
  const int nblocks = wq.shape(-2);
  const int rows = static_cast<int>(wq.size() / (nblocks * 10));
  return array({rows, 3}, int32, std::make_shared<TernaryStats>(to_stream(s)),
               {contiguous(wq, false, s)});
}

array code_flip_count(const array& a, const array& b, StreamOrDevice s) {
  if (a.dtype() != uint8 || b.dtype() != uint8 || a.shape() != b.shape() ||
      a.ndim() < 2 || a.shape(-1) != 10) {
    throw std::invalid_argument("code_flip_count: inputs must be same-shape uint8 (..., nblocks, 10)");
  }
  const int nblocks = a.shape(-2);
  const int rows = static_cast<int>(a.size() / (nblocks * 10));
  return array({rows}, int32, std::make_shared<CodeFlipCount>(to_stream(s)),
               {contiguous(a, false, s), contiguous(b, false, s)});
}

void TernaryStats::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TernaryStats has no CPU implementation.");
}
void CodeFlipCount::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("CodeFlipCount has no CPU implementation.");
}

void TernaryStats::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int nblocks = wq.shape(-2);
  const int rows = out.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ternary_stats(enc, wq, out, rows, nblocks);
}

void CodeFlipCount::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int nblocks = a.shape(-2);
  const int rows = out.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_code_flip_count(enc, a, b, out, rows, nblocks);
}

#define TK_TERNARY_NO_AUTODIFF(CLASS, LABEL)                                  \
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

TK_TERNARY_NO_AUTODIFF(TernaryStats, "TernaryStats")
TK_TERNARY_NO_AUTODIFF(CodeFlipCount, "CodeFlipCount")

} // namespace mlx::core
