// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemv_int/qgemv_int.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array qgemv_w8a8(const array& wq, const array& xq, const array& w_scale, const array& a_scale,
                 StreamOrDevice s) {
  assert(wq.dtype() == int8 && xq.dtype() == int8 && "qgemv_w8a8: wq, xq must be int8");
  assert(w_scale.dtype() == float16 && a_scale.dtype() == float16);
  assert(wq.ndim() == 2 && "qgemv_w8a8: wq (N,K) int8");
  const int N = wq.shape(0), K = wq.shape(1);
  assert(K % 4 == 0 && xq.shape(0) == K && "qgemv_w8a8: K%4==0, xq rows==K");
  (void)N; (void)K;
  return array({wq.shape(0), 1}, float16,
               std::make_shared<QGemvW8A8>(to_stream(s)), {wq, xq, w_scale, a_scale});
}

array qgemv_w2a8(const array& wq, const array& xq, const array& a_scale, StreamOrDevice s) {
  assert(wq.dtype() == uint8 && xq.dtype() == int8 && a_scale.dtype() == float16);
  assert(wq.ndim() == 3 && "qgemv_w2a8: wq (N, K/32, 10) bitnet blocks");
  const int N = wq.shape(0), K = wq.shape(1) * 32;
  assert(xq.shape(0) == K && "qgemv_w2a8: xq rows==K");
  (void)N; (void)K;
  return array({wq.shape(0), 1}, float16,
               std::make_shared<QGemvW2A8>(to_stream(s)), {wq, xq, a_scale});
}

void QGemvW8A8::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemvW8A8::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemvW8A8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& ws = inputs[2]; auto& as = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv_w8a8(enc, out, wq, xq, ws, as, N, K);
}
std::vector<array> QGemvW8A8::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemvW8A8 has no jvp."); }
std::vector<array> QGemvW8A8::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemvW8A8 has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemvW8A8::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemvW8A8 has no vmap."); }

void QGemvW2A8::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void QGemvW2A8::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }
void QGemvW2A8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& wq = inputs[0]; auto& xq = inputs[1]; auto& as = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = wq.shape(0), K = wq.shape(1) * 32;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv_w2a8(enc, out, wq, xq, as, N, K);
}
std::vector<array> QGemvW2A8::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemvW2A8 has no jvp."); }
std::vector<array> QGemvW2A8::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("QGemvW2A8 has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> QGemvW2A8::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QGemvW2A8 has no vmap."); }

} // namespace mlx::core
