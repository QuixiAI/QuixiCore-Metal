// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "fftconv/fftconv.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array fftconv(const array& x, const array& F, const array& twf, const array& finv,
              const array& twi, const array& kf, StreamOrDevice s) {
  assert(x.dtype() == float32 && F.dtype() == float32 && kf.dtype() == float32 &&
         "fftconv: all inputs must be float32");
  assert(x.ndim() == 5 && x.shape(0) == 2 && "fftconv: x must be (2,B,H,S,S)");
  const int B = x.shape(1), H = x.shape(2), S = x.shape(3);
  assert(x.shape(4) == S && (S == 16 || S == 32) && "fftconv: S must be 16 or 32");
  assert(F.shape(0) == 2 && F.shape(1) == S && F.shape(2) == S);
  assert(kf.shape(0) == 2 && kf.shape(1) == H && kf.shape(2) == S && kf.shape(3) == S);
  (void)B;
  return array({B, H, S, S}, float32,
               std::make_shared<FFTConv>(to_stream(s)), {x, F, twf, finv, twi, kf});
}

void FFTConv::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void FFTConv::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void FFTConv::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 6);
  auto& x = inputs[0]; auto& F = inputs[1]; auto& twf = inputs[2];
  auto& finv = inputs[3]; auto& twi = inputs[4]; auto& kf = inputs[5];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = x.shape(1), H = x.shape(2), S = x.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_fftconv(enc, out, x, F, twf, finv, twi, kf, B * H, H, S);
}

std::vector<array> FFTConv::jvp(const std::vector<array>&, const std::vector<array>&,
                                const std::vector<int>&) {
  throw std::runtime_error("FFTConv has no jvp implementation.");
}
std::vector<array> FFTConv::vjp(const std::vector<array>&, const std::vector<array>&,
                                const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("FFTConv has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> FFTConv::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("FFTConv has no vmap implementation.");
}
bool FFTConv::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
