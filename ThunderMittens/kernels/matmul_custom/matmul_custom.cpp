// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "matmul_custom/matmul_custom.h"

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
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
array matmul_custom(
    const array& x, // Input array x
    const array& y, // Input array y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    // x is (N, K), y is (K, M), out is (N, M).
    // The compiled kernel uses the <N_BLOCK=4, K_BLOCK=2, M_BLOCK=4> tiling with
    // TILE_DIM=8, i.e. 32x16x32 element blocks and a grid of (M/32, N/32). The
    // inputs must therefore satisfy N%32==0, M%32==0, K%16==0, or the grid would
    // silently truncate and drop output tiles.
    assert(x.shape(1) == y.shape(0) && "matmul_custom: inner dims must match");
    assert(x.shape(0) % 32 == 0 && y.shape(1) % 32 == 0 && x.shape(1) % 16 == 0 &&
           "matmul_custom requires N%32==0, M%32==0, K%16==0 for the <4,2,4> block config");
    assert(x.dtype() == y.dtype() &&
           (x.dtype() == float32 || x.dtype() == bfloat16) &&
           "matmul_custom only supports float32 and bfloat16");
    auto out_dtype = x.dtype();
    auto x_casted = astype(x, out_dtype, s);
    auto y_casted = astype(y, out_dtype, s);

  // Broadcast the shapes of x and y (on the same stream s)

//   // Broadcast the shapes of x and y (on the same stream s)
//   auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
//   auto out_shape = broadcasted_inputs[0].shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return array(
      /* const std::vector<int>& shape = */ {x.shape(0), y.shape(1)},
      /* Dtype dtype = */ out_dtype,
      /* std::unique_ptr<Primitive> primitive = */
      std::make_shared<MatmulCustom>(to_stream(s)),
      /* const std::vector<array>& inputs = */ {x_casted, y_casted});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void matmul_custom_impl(
    const array& x,
    const array& y,
    array& out) {
  assert(false);
}

/** Fall back implementation for evaluation on CPU */
void MatmulCustom::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
    assert(false);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Accelerate Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on CPU falling back to common backend */
void MatmulCustom::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
    assert(false);
}


///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

// #ifdef _METAL_

/** Evaluate primitive on GPU */
void MatmulCustom::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Prepare inputs
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Each primitive carries the stream it should execute on
  // and each stream carries its device identifiers
  auto& s = stream();
  // We get the needed metal device using the stream
  auto& d = metal::device(s.device);
  
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // out is (N, M) = x (N, K) @ y (K, M). Dispatch via the shared host ABI.
  int N = x.shape(0);
  int K = x.shape(1);
  int M = y.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_matmul_custom(enc, out, x, y, N, K, M, type_to_name(out));
}

// #else // Metal is not available

// /** Fail evaluation on GPU */
// void Axpby::eval_gpu(
//     const std::vector<array>& inputs,
//     std::vector<array>& out) {
//   throw std::runtime_error("Axpby has no GPU implementation.");
// }

// #endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<array> MatmulCustom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
//   // Forward mode diff that pushes along the tangents
//   // The jvp transform on the primitive can built with ops
//   // that are scheduled on the same stream as the primitive

//   // If argnums = {0}, we only push along x in which case the
//   // jvp is just the tangent scaled by alpha
//   // Similarly, if argnums = {1}, the jvp is just the tangent
//   // scaled by beta
//   if (argnums.size() > 1) {
//     auto scale = argnums[0] == 0 ? alpha_ : beta_;
//     auto scale_arr = array(scale, tangents[0].dtype());
//     return {multiply(scale_arr, tangents[0], stream())};
//   }
//   // If, argnums = {0, 1}, we take contributions from both
//   // which gives us jvp = tangent_x * alpha + tangent_y * beta
//   else {
//     return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
//   }
    throw std::runtime_error("MatmulCustom has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<array> MatmulCustom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
//   // Reverse mode diff
//   std::vector<array> vjps;
//   for (auto arg : argnums) {
//     auto scale = arg == 0 ? alpha_ : beta_;
//     auto scale_arr = array(scale, cotangents[0].dtype());
//     vjps.push_back(multiply(scale_arr, cotangents[0], stream()));
//   }
//   return vjps;
throw std::runtime_error("MatmulCustom has no vjp implementation.");
}

/** Vectorize primitive along given axis */
std::pair<std::vector<array>, std::vector<int>> MatmulCustom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("MatmulCustom has no vmap implementation.");
}

/** Equivalence check **/
bool MatmulCustom::is_equivalent(const Primitive& other) const {
  const MatmulCustom& r_other = static_cast<const MatmulCustom&>(other);
  return true;
}

} // namespace mlx::core
