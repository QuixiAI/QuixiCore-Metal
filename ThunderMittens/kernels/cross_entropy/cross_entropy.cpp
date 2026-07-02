// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "cross_entropy/cross_entropy.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool ce_is_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }

std::vector<array> cross_entropy_fwd(
    const array& logits,
    const array& targets,
    int ignore_index,
    float label_smoothing,
    float z_loss,
    float softcap,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("cross_entropy_fwd: logits must be (T, V)");
  }
  if (!ce_is_float(logits.dtype())) {
    throw std::invalid_argument("cross_entropy_fwd: logits must be float32/float16/bfloat16");
  }
  if (targets.ndim() != 1 || targets.shape(0) != logits.shape(0)) {
    throw std::invalid_argument("cross_entropy_fwd: targets must be (T,)");
  }
  const int T = logits.shape(0);
  auto logits_c = contiguous(logits, false, s);
  auto targets_c = contiguous(astype(targets, int32, s), false, s);
  return array::make_arrays(
      {{T}, {T}}, {float32, float32},
      std::make_shared<CrossEntropyFwd>(to_stream(s), ignore_index, label_smoothing, z_loss,
                                        softcap),
      {logits_c, targets_c});
}

array cross_entropy_bwd(
    const array& logits,
    const array& targets,
    const array& lse,
    const array& grad_out,
    int ignore_index,
    float label_smoothing,
    float z_loss,
    float softcap,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("cross_entropy_bwd: logits must be (T, V)");
  }
  const int T = logits.shape(0);
  auto logits_c = contiguous(logits, false, s);
  auto targets_c = contiguous(astype(targets, int32, s), false, s);
  auto lse_c = contiguous(astype(lse, float32, s), false, s);
  auto grad_c = contiguous(astype(grad_out, float32, s), false, s);
  return array(
      logits.shape(), logits.dtype(),
      std::make_shared<CrossEntropyBwd>(to_stream(s), ignore_index, label_smoothing, z_loss,
                                        softcap),
      {logits_c, targets_c, lse_c, grad_c});
}

void CrossEntropyFwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("CrossEntropyFwd has no CPU implementation.");
}
void CrossEntropyFwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& targets = inputs[1];
  auto& loss = outputs[0];
  auto& lse = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  loss.set_data(allocator::malloc_or_wait(loss.nbytes()));
  lse.set_data(allocator::malloc_or_wait(lse.nbytes()));

  const int T = logits.shape(0);
  const int V = logits.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_cross_entropy_fwd(enc, logits, targets, loss, lse, V, ignore_index_,
                               label_smoothing_, z_loss_, softcap_, T, type_to_name(logits));
}

void CrossEntropyBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("CrossEntropyBwd has no CPU implementation.");
}
void CrossEntropyBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& targets = inputs[1];
  auto& lse = inputs[2];
  auto& grad_out = inputs[3];
  auto& grad_logits = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  grad_logits.set_data(allocator::malloc_or_wait(grad_logits.nbytes()));

  const int T = logits.shape(0);
  const int V = logits.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_cross_entropy_bwd(enc, logits, targets, lse, grad_out, grad_logits, V, ignore_index_,
                               label_smoothing_, z_loss_, softcap_, T, type_to_name(logits));
}

#define TK_CE_NO_AUTODIFF(CLASS, LABEL)                                      \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_CE_NO_AUTODIFF(CrossEntropyFwd, "CrossEntropyFwd")
TK_CE_NO_AUTODIFF(CrossEntropyBwd, "CrossEntropyBwd")

} // namespace mlx::core
