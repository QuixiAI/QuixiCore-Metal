// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemv/qgemv.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static int qgemv_block_k(const std::string& fmt) {
  if (fmt == "q4_K") return 256;
  return 32;  // q8_0, q4_0
}

array qgemv(const array& wq, const array& x, const std::string& format, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && "qgemv: wq must be a packed uint8 weight-block array");
  assert(x.dtype() == float16 && "qgemv: x must be float16");
  assert(wq.ndim() == 3 && x.ndim() == 2 && x.shape(1) == 1 && "qgemv: wq (N,K/bk,bytes), x (K,1)");
  const int N = wq.shape(0);
  const int K = wq.shape(1) * qgemv_block_k(format);
  assert(x.shape(0) == K && "qgemv: x rows must equal K");
  (void)N; (void)K;
  return array({N, 1}, float16, std::make_shared<QGemv>(to_stream(s), format), {wq, x});
}

void QGemv::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemv::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void QGemv::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& wq = inputs[0]; auto& x = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0);
  const int K = wq.shape(1) * qgemv_block_k(fmt_);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv(enc, out, wq, x, N, K, fmt_);
}

std::vector<array> QGemv::jvp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&) {
  throw std::runtime_error("QGemv has no jvp implementation.");
}
std::vector<array> QGemv::vjp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemv has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QGemv::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemv has no vmap implementation.");
}
bool QGemv::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  return fmt_ == static_cast<const QGemv&>(other).fmt_;
}

} // namespace mlx::core
