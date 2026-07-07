// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rms_norm/rms_norm.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  RMSNorm over the last axis:
 *      y = x * rsqrt(mean(x^2) + eps) * weight
 *
 *  Inputs are assumed bf16 and row-contiguous. Common widths use dedicated
 *  kernels; other widths use the dynamic-D kernel when D is a multiple of 4.
 **/
array rms_norm(
    const array& x,
    const array& weight,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && weight.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert(weight.ndim() == 1 && weight.shape(0) == D);
  if (D % 4 != 0) {
    throw std::invalid_argument("rms_norm: last dim must be a multiple of 4");
  }

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<RMSNorm>(to_stream(s), eps),
      /* const std::vector<array>& inputs = */ {x, weight});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void RMSNorm::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.rms_norm for a reference.
  assert(false);
}

void RMSNorm::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // rows = prod(shape[:-1])

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  if (D == 256 || D == 512 || D == 768 || D == 1024) {
    tk::launch_rms_norm(enc, x, weight, out, M, D, eps_);
  } else {
    tk::launch_rms_norm_dyn(enc, x, weight, out, M, D, eps_);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> RMSNorm::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("RMSNorm has no jvp implementation.");
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("RMSNorm has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> RMSNorm::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("RMSNorm has no vmap implementation.");
}

// ---- RMSNorm backward (dX kernel; dW/rstd via the framework in the router) ----
array rms_norm_bwd_dx(
    const array& x,
    const array& weight,
    const array& dy,
    const array& rstd,
    StreamOrDevice s /* = {} */) {
  const Dtype dt = x.dtype();
  auto x_c = contiguous(x, false, s);
  auto w_c = contiguous(astype(weight, dt, s), false, s);
  auto dy_c = contiguous(astype(dy, dt, s), false, s);
  auto rstd_c = contiguous(astype(rstd, float32, s), false, s);
  return array(x.shape(), dt, std::make_shared<RMSNormBwdDx>(to_stream(s)),
               {x_c, w_c, dy_c, rstd_c});
}

void RMSNormBwdDx::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RMSNormBwdDx has no CPU implementation.");
}
void RMSNormBwdDx::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& dy = inputs[2];
  auto& rstd = inputs[3];
  auto& dx = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rms_norm_bwd_dx(enc, x, w, dy, rstd, dx, rows, D, type_to_name(x));
}
std::vector<array> RMSNormBwdDx::jvp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("RMSNormBwdDx has no jvp implementation.");
}
std::vector<array> RMSNormBwdDx::vjp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("RMSNormBwdDx has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RMSNormBwdDx::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RMSNormBwdDx has no vmap implementation.");
}

std::vector<array> rms_norm_bwd_fused(
    const array& x, const array& weight, const array& dy, float eps, StreamOrDevice s /* = {} */) {
  const Dtype dt = x.dtype();
  const int D = x.shape(-1);
  auto x_c = contiguous(x, false, s);
  auto w_c = contiguous(astype(weight, dt, s), false, s);
  auto dy_c = contiguous(astype(dy, dt, s), false, s);
  return array::make_arrays(
      {x.shape(), {D}}, {dt, float32},
      std::make_shared<RMSNormBwdFused>(to_stream(s), eps), {x_c, w_c, dy_c});
}

void RMSNormBwdFused::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RMSNormBwdFused has no CPU implementation.");
}
void RMSNormBwdFused::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& dy = inputs[2];
  auto& dx = outputs[0];
  auto& dweight = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  dweight.set_data(allocator::malloc_or_wait(dweight.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_embedding_zero_f32(enc, dweight, D);      // zero the atomic dweight accumulator first
  tk::launch_rms_norm_bwd_fused(enc, x, w, dy, dx, dweight, rows, D, eps_, type_to_name(x));
}
std::vector<array> RMSNormBwdFused::jvp(const std::vector<array>&, const std::vector<array>&,
                                        const std::vector<int>&) {
  throw std::runtime_error("RMSNormBwdFused has no jvp implementation.");
}
std::vector<array> RMSNormBwdFused::vjp(const std::vector<array>&, const std::vector<array>&,
                                        const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("RMSNormBwdFused has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RMSNormBwdFused::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RMSNormBwdFused has no vmap implementation.");
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& r_other = static_cast<const RMSNorm&>(other);
  return eps_ == r_other.eps_;
}

} // namespace mlx::core
