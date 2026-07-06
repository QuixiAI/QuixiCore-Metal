// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "gemm_staged/gemm_staged.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array gemm_staged(const array& x, const array& y, StreamOrDevice s) {
  assert(x.dtype() == y.dtype() &&
         (x.dtype() == float32 || x.dtype() == bfloat16) &&
         "gemm_staged: dtype must be float32 or bfloat16");
  assert(x.ndim() == 2 && y.ndim() == 2 && x.shape(1) == y.shape(0));
  const int N = x.shape(0), K = x.shape(1), M = y.shape(1);
  assert(N % 32 == 0 && M % 32 == 0 && K % 16 == 0 &&
         "gemm_staged: requires N%32==0, M%32==0, K%16==0");
  (void)N; (void)K; (void)M;
  return array({x.shape(0), y.shape(1)}, x.dtype(),
               std::make_shared<GemmStaged>(to_stream(s)), {x, y});
}

void GemmStaged::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void GemmStaged::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void GemmStaged::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& x = inputs[0]; auto& y = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = x.shape(0), K = x.shape(1), M = y.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gemm_staged(enc, out, x, y, N, K, M, type_to_name(out));
}

std::vector<array> GemmStaged::jvp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&) {
  throw std::runtime_error("GemmStaged has no jvp implementation.");
}
std::vector<array> GemmStaged::vjp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("GemmStaged has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> GemmStaged::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GemmStaged has no vmap implementation.");
}
bool GemmStaged::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
