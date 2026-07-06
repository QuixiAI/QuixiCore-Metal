// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "lin_attn_decay/lin_attn_decay.h"
#include "mamba2/mamba2.h"   // ssd_chunked: shared chunked linear-time SSD pipeline

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array lin_attn_decay(const array& q, const array& k, const array& v, const array& cl,
                     StreamOrDevice s) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  assert(cl.dtype() == float32);
  assert(q.shape() == k.shape() && k.shape() == v.shape());
  const int N = q.shape(2), D = q.shape(3);
  assert(D == 64 && "lin_attn_decay currently supports D=64");
  assert(N % 8 == 0 && "lin_attn_decay: N must be a multiple of 8");
  if (N % 64 == 0 && N >= 2048) {   // measured crossover, same gate as mamba2 at D=64
    // same math as mamba2 with cl = -slope*position: use the chunked linear-time pipeline
    return ssd_chunked(q, k, v, cl, s);
  }
  return array(q.shape(), bfloat16,
               std::make_shared<LinAttnDecay>(to_stream(s)), {q, k, v, cl});
}

void LinAttnDecay::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void LinAttnDecay::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void LinAttnDecay::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2]; auto& cl = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int Bsz = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lin_attn_decay(enc, q, k, v, cl, out, static_cast<unsigned>(N),
                            static_cast<unsigned>(H), Bsz, D);
}

std::vector<array> LinAttnDecay::jvp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("LinAttnDecay has no jvp implementation.");
}
std::vector<array> LinAttnDecay::vjp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LinAttnDecay has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LinAttnDecay::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LinAttnDecay has no vmap implementation.");
}

} // namespace mlx::core
