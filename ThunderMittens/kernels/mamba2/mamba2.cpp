// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mamba2/mamba2.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array mamba2(const array& C, const array& B, const array& X, const array& cumlog,
             StreamOrDevice s) {
  assert(C.dtype() == bfloat16 && B.dtype() == bfloat16 && X.dtype() == bfloat16);
  assert(cumlog.dtype() == float32);
  assert(C.shape() == B.shape() && B.shape() == X.shape());
  const int N = C.shape(2), D = C.shape(3);
  assert(D == 64 && "mamba2 currently supports D=64");
  assert(N % 8 == 0 && "mamba2: N must be a multiple of 8");
  (void)N;
  return array(C.shape(), bfloat16,
               std::make_shared<Mamba2>(to_stream(s)), {C, B, X, cumlog});
}

void Mamba2::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void Mamba2::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void Mamba2::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& C = inputs[0]; auto& B = inputs[1]; auto& X = inputs[2]; auto& cumlog = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mamba2(enc, C, B, X, cumlog, out, static_cast<unsigned>(N),
                    static_cast<unsigned>(H), Bsz, D);
}

std::vector<array> Mamba2::jvp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no jvp implementation.");
}
std::vector<array> Mamba2::vjp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("Mamba2 has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> Mamba2::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no vmap implementation.");
}
bool Mamba2::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
