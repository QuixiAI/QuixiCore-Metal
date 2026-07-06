// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "cmplx_matmul/cmplx_matmul.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array cmplx_matmul(const array& a, const array& b, StreamOrDevice s) {
  assert(a.dtype() == b.dtype() &&
         (a.dtype() == float32 || a.dtype() == bfloat16) &&
         "cmplx_matmul: dtype must be float32 or bfloat16");
  assert(a.ndim() == 3 && b.ndim() == 3 && a.shape(0) == 2 && b.shape(0) == 2 &&
         a.shape(2) == b.shape(1) && "cmplx_matmul: a (2,N,K), b (2,K,M)");
  const int N = a.shape(1), K = a.shape(2), M = b.shape(2);
  assert(N % 32 == 0 && M % 32 == 0 && K % 16 == 0 &&
         "cmplx_matmul: requires N%32==0, M%32==0, K%16==0");
  (void)N; (void)K; (void)M;
  return array({2, a.shape(1), b.shape(2)}, a.dtype(),
               std::make_shared<CmplxMatmul>(to_stream(s)), {a, b});
}

void CmplxMatmul::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void CmplxMatmul::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void CmplxMatmul::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& a = inputs[0]; auto& b = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = a.shape(1), K = a.shape(2), M = b.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_cmplx_matmul(enc, out, a, b, N, K, M, type_to_name(out));
}

std::vector<array> CmplxMatmul::jvp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&) {
  throw std::runtime_error("CmplxMatmul has no jvp implementation.");
}
std::vector<array> CmplxMatmul::vjp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("CmplxMatmul has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> CmplxMatmul::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("CmplxMatmul has no vmap implementation.");
}
bool CmplxMatmul::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
