// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "add_rt/add_rt.h"

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
array add_rt(
    const array& x, // Input array x
    const array& y, // Input array y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  // Promote dtypes between x and y as needed
  auto promoted_dtype = promote_types(x.dtype(), y.dtype());
  assert((promoted_dtype == float32 || promoted_dtype == float16 ||
          promoted_dtype == bfloat16) &&
         "add_rt only supports float32, float16, bfloat16");

  // Preserve the (promoted) floating dtype; the kernel is instantiated for all three.
  auto out_dtype = promoted_dtype;

  // Cast x and y up to the determined dtype (on the same stream s)
    auto x_casted = astype(x, out_dtype, s);
    auto y_casted = astype(y, out_dtype, s);


  // Broadcast the shapes of x and y (on the same stream s)
  auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
  auto out_shape = broadcasted_inputs[0].shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return array(
      /* const std::vector<int>& shape = */ out_shape,
      /* Dtype dtype = */ out_dtype,
      /* std::unique_ptr<Primitive> primitive = */
      std::make_shared<AddRT>(to_stream(s)),
      /* const std::vector<array>& inputs = */ broadcasted_inputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void add_rt_impl(
    const array& x,
    const array& y,
    array& out) {
  // We only allocate memory when we are ready to fill the output
  // malloc_or_wait synchronously allocates available memory
  // There may be a wait executed here if the allocation is requested
  // under memory-pressured conditions
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Collect input and output data pointers
  const T* x_ptr = x.data<T>();
  const T* y_ptr = y.data<T>();
  T* out_ptr = out.data<T>();

  // Cast alpha and beta to the relevant types

  // Do the element-wise operation for each output
  for (size_t out_idx = 0; out_idx < out.size(); out_idx++) {
    // Map linear indices to offsets in x and y
    auto x_offset = elem_to_loc(out_idx, x.shape(), x.strides());
    auto y_offset = elem_to_loc(out_idx, y.shape(), y.strides());

    // We allocate the output to be contiguous and regularly strided
    // (defaults to row major) and hence it doesn't need additional mapping
    out_ptr[out_idx] =  x_ptr[x_offset] + y_ptr[y_offset];
  }
}

/** Fall back implementation for evaluation on CPU */
void AddRT::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {

  // Check the inputs (registered in the op while constructing the out array)
  assert(false);
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Dispatch to the correct dtype
  if (out.dtype() == float32) {
    return add_rt_impl<float>(x, y, out);
  } else if (out.dtype() == float16) {
    return add_rt_impl<float16_t>(x, y, out);
  } else if (out.dtype() == bfloat16) {
    return add_rt_impl<bfloat16_t>(x, y, out);
  } else {
    throw std::runtime_error(
        "Axpby is only supported for floating point types.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Accelerate Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on CPU falling back to common backend */
void AddRT::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}


///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

// #ifdef _METAL_

/** Evaluate primitive on GPU */
void AddRT::eval_gpu(
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

  // Prepare to specialize based on contiguity
  bool contiguous_kernel =
      (x.flags().row_contiguous && y.flags().row_contiguous) ||
      (x.flags().col_contiguous && y.flags().col_contiguous);

  if (contiguous_kernel) {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * out.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }


  // Elementwise add over a 2D (rows, cols) tile grid. Dispatch via the shared host ABI.
  int rows = out.shape()[0];
  int cols = out.shape()[1];
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_add_rt(enc, x, y, out, rows, cols, type_to_name(out));
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
std::vector<array> AddRT::jvp(
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
    throw std::runtime_error("AddRT has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<array> AddRT::vjp(
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
throw std::runtime_error("AddCustom has no vjp implementation.");
}

/** Vectorize primitive along given axis */
std::pair<std::vector<array>, std::vector<int>> AddRT::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("AddCustom has no vmap implementation.");
}

/** Equivalence check **/
bool AddRT::is_equivalent(const Primitive& other) const {
  const AddRT& r_other = static_cast<const AddRT&>(other);
  return true;
}

} // namespace mlx::core
