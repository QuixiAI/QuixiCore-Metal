// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemm_int/qgemm_int.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array qgemm_w8a8_azp(const array& wq, const array& xq, const array& w_scale,
                     const array& a_scale, const array& w_rowsum, const array& azp,
                     StreamOrDevice s) {
  assert(wq.dtype() == int8 && xq.dtype() == int8 && "qgemm_w8a8_azp: wq, xq int8");
  assert(wq.ndim() == 2 && xq.ndim() == 2 && wq.shape(1) == xq.shape(1));
  const int M = xq.shape(0);
  return array({wq.shape(0), M}, float16,
               std::make_shared<QGemmW8A8Azp>(to_stream(s)),
               {wq, xq, astype(w_scale, float16, s), astype(a_scale, float32, s),
                astype(w_rowsum, int32, s), astype(azp, int32, s)});
}

array qgemm_w8a8(const array& wq, const array& xq, const array& w_scale, const array& a_scale,
                 StreamOrDevice s) {
  assert(wq.dtype() == int8 && xq.dtype() == int8 && "qgemm_w8a8: wq, xq int8");
  assert(w_scale.dtype() == float16 && a_scale.dtype() == float16);
  assert(wq.ndim() == 2 && xq.ndim() == 2);
  const int N = wq.shape(0), K = wq.shape(1), M = xq.shape(0);
  assert(K % 4 == 0 && xq.shape(1) == K);
  (void)N; (void)K;
  return array({wq.shape(0), M}, float16,
               std::make_shared<QGemmW8A8>(to_stream(s)), {wq, xq, w_scale, a_scale});
}

array qgemm_w2a8(const array& wq, const array& xq, const array& a_scale, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && xq.dtype() == int8 && a_scale.dtype() == float16);
  assert(wq.ndim() == 3 && xq.ndim() == 2);
  const int N = wq.shape(0), K = wq.shape(1) * 32, M = xq.shape(0);
  assert(xq.shape(1) == K);
  (void)N; (void)K;
  return array({wq.shape(0), M}, float16,
               std::make_shared<QGemmW2A8>(to_stream(s)), {wq, xq, a_scale});
}

void QGemmW8A8Azp::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmW8A8Azp::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmW8A8Azp::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& ws = inputs[2]; auto& as = inputs[3];
  auto& rsum = inputs[4]; auto& azp = inputs[5];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1), M = xq.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_w8a8_azp(enc, out, wq, xq, ws, as, rsum, azp, N, K, M);
}
std::vector<array> QGemmW8A8Azp::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW8A8Azp has no jvp."); }
std::vector<array> QGemmW8A8Azp::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmW8A8Azp has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmW8A8Azp::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW8A8Azp has no vmap."); }

void QGemmW8A8::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmW8A8::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmW8A8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& ws = inputs[2]; auto& as = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1), M = xq.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_w8a8(enc, out, wq, xq, ws, as, N, K, M);
}
std::vector<array> QGemmW8A8::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW8A8 has no jvp."); }
std::vector<array> QGemmW8A8::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmW8A8 has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmW8A8::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW8A8 has no vmap."); }

void QGemmW2A8::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemmW2A8::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemmW2A8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& as = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1) * 32, M = xq.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemm_w2a8(enc, out, wq, xq, as, N, K, M);
}
std::vector<array> QGemmW2A8::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW2A8 has no jvp."); }
std::vector<array> QGemmW2A8::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemmW2A8 has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemmW2A8::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemmW2A8 has no vmap."); }

} // namespace mlx::core
