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
  if (fmt == "q4_K" || fmt == "iq4_xs" || fmt == "iq2_xxs" || fmt == "iq2_xs" || fmt == "iq3_xxs" || fmt == "iq1_s" || fmt == "q2_K" || fmt == "q3_K" || fmt == "q5_K" || fmt == "q6_K") return 256;
  if (fmt == "kU4B8" || fmt == "kU4") return 128;
  if (fmt == "nvfp4") return 16;
  return 32;  // q8_0, q4_0, fp8_e4m3, fp4_e2m1, mxfp8
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
  // dequant-direct-to-fragment (Marlin zero-shuffle) — ~40% faster than dequant-to-shared and
  // bit-identical; it is the default. The staged path remains available via the frag flag = false.
  return array({N, M}, float16,
               std::make_shared<QGemm>(to_stream(s), format, true), {wq, x});
}

array qgemm_direct(const array& wq, const array& x, const std::string& format, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && x.dtype() == float16);
  assert(wq.ndim() == 3 && x.ndim() == 2);
  const int N = wq.shape(0);
  const int K = wq.shape(1) * format_block_k(format);
  const int M = x.shape(1);
  assert(x.shape(0) == K && N % 32 == 0 && M % 32 == 0);
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QGemm>(to_stream(s), format, true), {wq, x});
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
  if (direct_)
    tk::launch_qgemm_frag(enc, out, wq, x, N, K, M, fmt_);
  else
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
  auto& o = static_cast<const QGemm&>(other);
  return fmt_ == o.fmt_ && direct_ == o.direct_;
}

} // namespace mlx::core
