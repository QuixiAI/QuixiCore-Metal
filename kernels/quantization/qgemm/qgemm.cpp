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
  if (fmt == "kU4B8" || fmt == "kU4" || fmt == "fp8_block") return 128;
  if (fmt == "hqq") return 64;
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
  // k-quant (256-superblock) prefill route: the fragment-path in-GEMM dequant of these branchy
  // formats measured 2-2.3x slower than dequantize-then-mx.matmul at every M — dequantize the
  // whole weight with the span-decode kernel and use the framework GEMM instead.
  if (format_block_k(format) == 256 && M >= 64) {
    auto w = array({N, K}, float16,
                   std::make_shared<QDequant>(to_stream(s), format), {wq});
    return matmul(w, x, s);
  }
  // dequant-direct-to-fragment (Marlin zero-shuffle) — ~40% faster than dequant-to-shared and
  // bit-identical; it is the default. The staged path remains available via the frag flag = false.
  return array({N, M}, float16,
               std::make_shared<QGemm>(to_stream(s), format, true), {wq, x});
}

void QDequant::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QDequant has no CPU implementation.");
}

void QDequant::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = out.shape(0);
  const int K = out.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qdequant_fp16(enc, out, wq, N, K, fmt_);
}

std::vector<array> QDequant::jvp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&) {
  throw std::runtime_error("QDequant has no jvp implementation.");
}
std::vector<array> QDequant::vjp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QDequant has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QDequant::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QDequant has no vmap implementation.");
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

array qgemm_actorder_k(const array& wq, const array& x, const array& perm,
                       const std::string& format, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && x.dtype() == float16 && perm.dtype() == int32);
  assert(wq.ndim() == 3 && x.ndim() == 2 && perm.ndim() == 1);
  const int N = wq.shape(0);
  const int K = wq.shape(1) * format_block_k(format);
  const int M = x.shape(1);
  assert(x.shape(0) == K && perm.shape(0) == K && N % 32 == 0 && M % 32 == 0);
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QGemmActorder>(to_stream(s), format), {wq, x, perm});
}

array qgemm_blockscale(const array& wq, const array& x, const array& scale2d, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && x.dtype() == float16 && scale2d.dtype() == float16);
  assert(wq.ndim() == 3 && x.ndim() == 2);
  const int N = wq.shape(0), K = wq.shape(1) * 128, M = x.shape(1);
  assert(x.shape(0) == K && N % 32 == 0 && M % 32 == 0);
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QGemmBlockScale>(to_stream(s)), {wq, x, scale2d});
}

array qgemm_fp8_scaled(const array& wq, const array& xq, const array& w_scale, const array& a_scale,
                       StreamOrDevice s) {
  assert(wq.dtype() == uint8 && xq.dtype() == uint8);
  assert(w_scale.dtype() == float16 && a_scale.dtype() == float16);
  assert(wq.ndim() == 2 && xq.ndim() == 2);
  const int N = wq.shape(0), K = wq.shape(1), M = xq.shape(1);
  assert(xq.shape(0) == K && N % 32 == 0 && M % 32 == 0 && K % 32 == 0);
  (void)N; (void)K; (void)M;
  return array({N, M}, float16,
               std::make_shared<QGemmFp8Scaled>(to_stream(s)), {wq, xq, w_scale, a_scale});
}

void QGemmFp8Scaled::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmFp8Scaled::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmFp8Scaled::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& ws = inputs[2]; auto& as = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1), M = xq.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_fp8_scaled(enc, out, wq, xq, ws, as, N, K, M);
}
std::vector<array> QGemmFp8Scaled::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmFp8Scaled has no jvp."); }
std::vector<array> QGemmFp8Scaled::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmFp8Scaled has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmFp8Scaled::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmFp8Scaled has no vmap."); }

void QGemmBlockScale::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmBlockScale::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmBlockScale::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& x = inputs[1]; auto& sc = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1) * 128, M = x.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_blockscale(enc, out, wq, x, sc, N, K, M);
}
std::vector<array> QGemmBlockScale::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmBlockScale has no jvp."); }
std::vector<array> QGemmBlockScale::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmBlockScale has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmBlockScale::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmBlockScale has no vmap."); }

void QGemmActorder::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmActorder::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmActorder::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& x = inputs[1]; auto& perm = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1) * format_block_k(fmt_), M = x.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_actorder(enc, out, wq, x, perm, N, K, M, fmt_);
}
std::vector<array> QGemmActorder::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmActorder has no jvp."); }
std::vector<array> QGemmActorder::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmActorder has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmActorder::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmActorder has no vmap."); }

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
