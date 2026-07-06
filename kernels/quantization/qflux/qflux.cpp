// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qflux/qflux.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static int qflux_block_k(const std::string& fmt) {
  if (fmt == "q4_K" || fmt == "iq4_xs" || fmt == "iq2_xxs" || fmt == "iq2_xs" || fmt == "iq3_xxs" || fmt == "iq1_s" || fmt == "q2_K" || fmt == "q3_K" || fmt == "q5_K" || fmt == "q6_K") return 256;
  if (fmt == "kU4B8" || fmt == "kU4" || fmt == "fp8_block") return 128;
  if (fmt == "hqq") return 64;
  if (fmt == "nvfp4") return 16;
  return 32;
}

array qflux_gelu(const array& wq, const array& x, const array& bias,
                 const std::string& format, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && x.dtype() == float16 && bias.dtype() == float16);
  assert(wq.ndim() == 3 && x.ndim() == 2 && bias.ndim() == 1);
  const int N = wq.shape(0);
  const int K = wq.shape(1) * qflux_block_k(format);
  const int M = x.shape(1);
  assert(x.shape(0) == K && bias.shape(0) == M && N % 32 == 0 && M % 32 == 0);
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QFluxGelu>(to_stream(s), format), {wq, x, bias});
}

void QFluxGelu::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QFluxGelu::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void QFluxGelu::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& wq = inputs[0]; auto& x = inputs[1]; auto& bias = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0);
  const int K = wq.shape(1) * qflux_block_k(fmt_);
  const int M = x.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qflux_gelu(enc, out, wq, x, bias, N, K, M, fmt_);
}

std::vector<array> QFluxGelu::jvp(const std::vector<array>&, const std::vector<array>&,
                                  const std::vector<int>&) {
  throw std::runtime_error("QFluxGelu has no jvp implementation.");
}
std::vector<array> QFluxGelu::vjp(const std::vector<array>&, const std::vector<array>&,
                                  const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QFluxGelu has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QFluxGelu::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QFluxGelu has no vmap implementation.");
}
bool QFluxGelu::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  return fmt_ == static_cast<const QFluxGelu&>(other).fmt_;
}

} // namespace mlx::core
