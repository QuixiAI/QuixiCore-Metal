// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "layernorm/layernorm.h"

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
 *  LayerNorm over the last axis:
 *      y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
 *
 *  Inputs are assumed bf16 and row-contiguous. Common widths use fixed
 *  instantiations; other widths divisible by four use the dynamic kernel.
 **/
array layernorm(
    const array& x,
    const array& weight,
    const array& bias,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && weight.dtype() == bfloat16 &&
         bias.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert(weight.ndim() == 1 && weight.shape(0) == D);
  assert(bias.ndim() == 1 && bias.shape(0) == D);
  assert(D > 0 && D % 4 == 0 &&
         "layernorm: last dim must be positive and divisible by 4");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<LayerNorm>(to_stream(s), eps),
      /* const std::vector<array>& inputs = */ {x, weight, bias});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Fall back implementation for evaluation on CPU */
void LayerNorm::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.layer_norm for a reference.
  assert(false);
}

void LayerNorm::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on GPU */
void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // rows = prod(shape[:-1])

  // Dispatch via the shared host ABI (one simdgroup per row; M threadgroups).
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  if (D == 256 || D == 512 || D == 768 || D == 1024) {
    tk::launch_layernorm(enc, x, weight, bias, out, M, D, eps_);
  } else {
    tk::launch_layernorm_dyn(enc, x, weight, bias, out, M, D, eps_);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> LayerNorm::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("LayerNorm has no jvp implementation.");
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("LayerNorm has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> LayerNorm::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("LayerNorm has no vmap implementation.");
}

/** Equivalence check **/
// ---- LayerNorm backward (dX kernel; dW/dbias/mean/rstd via the framework in the router) ----
array layernorm_bwd_dx(
    const array& x,
    const array& weight,
    const array& dy,
    const array& mean,
    const array& rstd,
    StreamOrDevice s /* = {} */) {
  const Dtype dt = x.dtype();
  auto x_c = contiguous(x, false, s);
  auto w_c = contiguous(astype(weight, dt, s), false, s);
  auto dy_c = contiguous(astype(dy, dt, s), false, s);
  auto mean_c = contiguous(astype(mean, float32, s), false, s);
  auto rstd_c = contiguous(astype(rstd, float32, s), false, s);
  return array(x.shape(), dt, std::make_shared<LayerNormBwdDx>(to_stream(s)),
               {x_c, w_c, dy_c, mean_c, rstd_c});
}

void LayerNormBwdDx::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LayerNormBwdDx has no CPU implementation.");
}
void LayerNormBwdDx::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& dy = inputs[2];
  auto& mean = inputs[3];
  auto& rstd = inputs[4];
  auto& dx = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_layernorm_bwd_dx(enc, x, w, dy, mean, rstd, dx, rows, D, type_to_name(x));
}
std::vector<array> LayerNormBwdDx::jvp(const std::vector<array>&, const std::vector<array>&,
                                       const std::vector<int>&) {
  throw std::runtime_error("LayerNormBwdDx has no jvp implementation.");
}
std::vector<array> LayerNormBwdDx::vjp(const std::vector<array>&, const std::vector<array>&,
                                       const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LayerNormBwdDx has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LayerNormBwdDx::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LayerNormBwdDx has no vmap implementation.");
}

std::vector<array> layernorm_bwd_fused(
    const array& x, const array& weight, const array& dy, float eps, StreamOrDevice s /* = {} */) {
  const Dtype dt = x.dtype();
  const int D = x.shape(-1);
  auto x_c = contiguous(x, false, s);
  auto w_c = contiguous(astype(weight, dt, s), false, s);
  auto dy_c = contiguous(astype(dy, dt, s), false, s);
  return array::make_arrays(
      {x.shape(), {D}, {D}}, {dt, float32, float32},
      std::make_shared<LayerNormBwdFused>(to_stream(s), eps), {x_c, w_c, dy_c});
}

void LayerNormBwdFused::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LayerNormBwdFused has no CPU implementation.");
}
void LayerNormBwdFused::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& dy = inputs[2];
  auto& dx = outputs[0];
  auto& dweight = outputs[1];
  auto& dbias = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  dweight.set_data(allocator::malloc_or_wait(dweight.nbytes()));
  dbias.set_data(allocator::malloc_or_wait(dbias.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_embedding_zero_f32(enc, dweight, D);      // zero the atomic accumulators first
  tk::launch_embedding_zero_f32(enc, dbias, D);
  tk::launch_layernorm_bwd_fused(enc, x, w, dy, dx, dweight, dbias, rows, D, eps_, type_to_name(x));
}
std::vector<array> LayerNormBwdFused::jvp(const std::vector<array>&, const std::vector<array>&,
                                          const std::vector<int>&) {
  throw std::runtime_error("LayerNormBwdFused has no jvp implementation.");
}
std::vector<array> LayerNormBwdFused::vjp(const std::vector<array>&, const std::vector<array>&,
                                          const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LayerNormBwdFused has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LayerNormBwdFused::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LayerNormBwdFused has no vmap implementation.");
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& r_other = static_cast<const LayerNorm&>(other);
  return eps_ == r_other.eps_;
}

} // namespace mlx::core
