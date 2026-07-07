// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "quantize_tq2_0/quantize_tq2_0.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool tq_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}
} // namespace

std::vector<array> quantize_tq2_0(const array& w, StreamOrDevice s) {
  if (!tq_is_float(w.dtype()) || (w.ndim() != 2 && w.ndim() != 3)) {
    throw std::invalid_argument("quantize_tq2_0: W must be float (N,K) or (E,N,K)");
  }
  const int K = w.shape(-1);
  if (K % 256 != 0) {
    throw std::invalid_argument("quantize_tq2_0: K must be a multiple of 256");
  }
  auto qshape = w.shape();
  qshape.pop_back();
  qshape.push_back(K / 256);
  qshape.push_back(66);
  return array::make_arrays(
      {qshape, w.shape()},
      {uint8, bfloat16},
      std::make_shared<QuantizeTQ20>(to_stream(s)),
      {contiguous(w, false, s)});
}

void QuantizeTQ20::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizeTQ20 has no CPU implementation.");
}

void QuantizeTQ20::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& w = inputs[0];
  auto& wq = outputs[0];
  auto& w_deq = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  wq.set_data(allocator::malloc_or_wait(wq.nbytes()));
  w_deq.set_data(allocator::malloc_or_wait(w_deq.nbytes()));
  const int E = w.ndim() == 3 ? w.shape(0) : 1;
  const int N = w.shape(-2);
  const int K = w.shape(-1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_tq2_0(enc, w, wq, w_deq, E, N, K, type_to_name(w));
}

#define TK_TQ_NO_AUTODIFF(CLASS, LABEL)                                       \
  std::vector<array> CLASS::jvp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&) {                                              \
    throw std::runtime_error(LABEL " has no jvp implementation.");            \
  }                                                                           \
  std::vector<array> CLASS::vjp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&, const std::vector<array>&) {                   \
    throw std::runtime_error(LABEL " has no vjp implementation.");            \
  }                                                                           \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                \
      const std::vector<array>&, const std::vector<int>&) {                   \
    throw std::runtime_error(LABEL " has no vmap implementation.");           \
  }

TK_TQ_NO_AUTODIFF(QuantizeTQ20, "QuantizeTQ20")

} // namespace mlx::core
