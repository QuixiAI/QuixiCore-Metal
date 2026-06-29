// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemm/qgemm.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

// block_k for each supported format (weights per quant block).
static int format_block_k(const std::string& fmt) {
  if (fmt == "q8_0" || fmt == "q4_0") return 32;
  if (fmt == "q4_K") return 256;
  return 32;
}

array qgemm(const array& wq, const array& x, const std::string& format, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && "qgemm: wq must be a packed uint8 weight-block array");
  assert(x.dtype() == float16 && "qgemm: x must be float16");
  assert(wq.ndim() == 3 && x.ndim() == 2 && "qgemm: wq (N,K/bk,bytes), x (K,M)");
  const int N = wq.shape(0);
  const int K = wq.shape(1) * format_block_k(format);
  const int M = x.shape(1);
  assert(x.shape(0) == K && "qgemm: x rows must equal K");
  assert(N % 32 == 0 && M % 32 == 0 && "qgemm: requires N%32==0, M%32==0");
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QGemm>(to_stream(s), format), {wq, x});
}

void QGemm::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemm::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void QGemm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& wq = inputs[0]; auto& x = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0);
  const int K = wq.shape(1) * format_block_k(fmt_);
  const int M = x.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm(enc, out, wq, x, N, K, M, fmt_);
}

std::vector<array> QGemm::jvp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&) {
  throw std::runtime_error("QGemm has no jvp implementation.");
}
std::vector<array> QGemm::vjp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemm has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QGemm::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemm has no vmap implementation.");
}
bool QGemm::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  return fmt_ == static_cast<const QGemm&>(other).fmt_;
}

} // namespace mlx::core
