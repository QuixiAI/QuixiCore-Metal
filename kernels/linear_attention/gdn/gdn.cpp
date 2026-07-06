// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "gdn/gdn.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

std::vector<array> gdn_recur(
    const array& q, const array& k, const array& v, const array& g, const array& beta,
    const array& state_pool, const array& cu_seqlens, const array& slot_mapping,
    bool load_initial, StreamOrDevice s) {
  if (q.ndim() != 3 || k.shape() != q.shape() || v.ndim() != 3 ||
      v.shape(0) != q.shape(0)) {
    throw std::invalid_argument(
        "gdn_recur: q/k must be (total_tokens, Hk, Dk), v (total_tokens, Hv, Dv)");
  }
  const int Hk = q.shape(1), Dk = q.shape(2);
  const int Hv = v.shape(1), Dv = v.shape(2);
  if (!(Dk == 64 || Dk == 128)) {
    throw std::invalid_argument("gdn_recur: Dk must be 64 or 128");
  }
  if (Hv % Hk != 0) {
    throw std::invalid_argument("gdn_recur: Hv must be a multiple of Hk (GQA)");
  }
  if (g.ndim() != 2 || g.shape(0) != q.shape(0) || g.shape(1) != Hv ||
      beta.shape() != g.shape()) {
    throw std::invalid_argument("gdn_recur: g/beta must be (total_tokens, Hv)");
  }
  if (state_pool.ndim() != 4 || state_pool.shape(1) != Hv || state_pool.shape(2) != Dv ||
      state_pool.shape(3) != Dk) {
    throw std::invalid_argument("gdn_recur: state_pool must be (num_slots, Hv, Dv, Dk)");
  }
  if (cu_seqlens.ndim() != 1 || cu_seqlens.shape(0) < 2 || slot_mapping.ndim() != 1 ||
      slot_mapping.shape(0) != cu_seqlens.shape(0) - 1) {
    throw std::invalid_argument(
        "gdn_recur: cu_seqlens must be (R+1,), slot_mapping (R,)");
  }
  auto dt = q.dtype();
  if (!(dt == float32 || dt == float16 || dt == bfloat16)) {
    throw std::invalid_argument("gdn_recur: dtype must be float32/float16/bfloat16");
  }
  return array::make_arrays(
      {v.shape(), state_pool.shape()},
      {dt, float32},
      std::make_shared<GdnRecur>(to_stream(s), load_initial),
      {contiguous(q, false, s), contiguous(astype(k, dt, s), false, s),
       contiguous(astype(v, dt, s), false, s), contiguous(astype(g, dt, s), false, s),
       contiguous(astype(beta, dt, s), false, s),
       contiguous(astype(state_pool, float32, s), false, s),
       contiguous(astype(cu_seqlens, int32, s), false, s),
       contiguous(astype(slot_mapping, int32, s), false, s)});
}

void GdnRecur::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnRecur has no CPU implementation.");
}

void GdnRecur::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& g = inputs[3];
  auto& beta = inputs[4];
  auto& pool_in = inputs[5];
  auto& cu = inputs[6];
  auto& slots = inputs[7];
  auto& y = outputs[0];
  auto& pool_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  pool_out.set_data(allocator::malloc_or_wait(pool_out.nbytes()));
  const int R = cu.shape(0) - 1;
  const int Hk = q.shape(1), Dk = q.shape(2);
  const int Hv = v.shape(1), Dv = v.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // functional pool: clone, then the kernel updates its request slots in place
  tk::launch_sscan_pool_clone(enc, pool_in, pool_out,
                              static_cast<uint32_t>(pool_in.size()));
  tk::launch_gdn_recur(enc, q, k, v, g, beta, pool_out, cu, slots, y, R, Hk, Hv, Dv, Dk,
                       load_initial_ ? 1 : 0, type_to_name(q));
}

std::vector<array> GdnRecur::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GdnRecur has no jvp implementation.");
}
std::vector<array> GdnRecur::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("GdnRecur has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> GdnRecur::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GdnRecur has no vmap implementation.");
}

} // namespace mlx::core
