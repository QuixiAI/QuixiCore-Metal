// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "marginal/marginal.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
std::string marg_tname(const array& x) {
  if (x.dtype() == float32) return "float32";
  if (x.dtype() == float16) return "float16";
  if (x.dtype() == bfloat16) return "bfloat16";
  throw std::invalid_argument("tau_tail: dtype must be f32/f16/bf16");
}
} // namespace

array tau_tail(const array& qkv, const array& tok_qv_lin, const array& tau_pos_table,
               const array& positions, int n_heads, int head_dim, StreamOrDevice s) {
  if (qkv.ndim() != 2 || qkv.shape(1) % 3 != 0) {
    throw std::invalid_argument("tau_tail: qkv must be (T, 3*q_dim)");
  }
  const int q_dim = qkv.shape(1) / 3;
  if (n_heads * head_dim != q_dim) {
    throw std::invalid_argument("tau_tail: n_heads*head_dim must equal q_dim");
  }
  // functional: operate on a fresh copy of qkv (K slice copies through unchanged).
  return array(
      qkv.shape(), qkv.dtype(),
      std::make_shared<Marginal>(to_stream(s), 0, n_heads, head_dim, q_dim),
      {contiguous(qkv, false, s),
       contiguous(astype(tok_qv_lin, qkv.dtype(), s), false, s),
       contiguous(astype(tau_pos_table, qkv.dtype(), s), false, s),
       contiguous(astype(positions, int32, s), false, s)});
}

array packbits(const array& x, bool bit_order_big, StreamOrDevice s) {
  const int n = static_cast<int>(x.size());
  const int nbytes = (n + 7) / 8;
  return array(
      {nbytes}, uint8,
      std::make_shared<Marginal>(to_stream(s), 1, n, bit_order_big ? 1 : 0, 0),
      {contiguous(astype(reshape(x, {n}, s), uint8, s), false, s)});
}

array segment_packbits(const array& x, const array& input_indptr, const array& output_indptr,
                       int total_output_bytes, bool bit_order_big, StreamOrDevice s) {
  if (input_indptr.ndim() != 1 || output_indptr.ndim() != 1 ||
      input_indptr.shape(0) != output_indptr.shape(0)) {
    throw std::invalid_argument("segment_packbits: indptrs must be matching (S+1,) arrays");
  }
  const int num_segments = input_indptr.shape(0) - 1;
  return array(
      {total_output_bytes}, uint8,
      std::make_shared<Marginal>(to_stream(s), 2, num_segments, total_output_bytes,
                                 bit_order_big ? 1 : 0),
      {contiguous(astype(reshape(x, {static_cast<int>(x.size())}, s), uint8, s), false, s),
       contiguous(astype(input_indptr, int32, s), false, s),
       contiguous(astype(output_indptr, int32, s), false, s)});
}

array permute_cols(const array& x, const array& perm, StreamOrDevice s) {
  if (x.ndim() != 2 || perm.ndim() != 1 || perm.shape(0) != x.shape(1)) {
    throw std::invalid_argument("permute_cols: x (rows, cols), perm (cols,)");
  }
  if (x.itemsize() != 2) {
    throw std::invalid_argument("permute_cols: x must be a 16-bit dtype (f16/bf16/int16/uint16)");
  }
  return array(
      x.shape(), x.dtype(),
      std::make_shared<Marginal>(to_stream(s), 3, x.shape(0), x.shape(1), 0),
      {contiguous(x, false, s), contiguous(astype(perm, int32, s), false, s)});
}

void Marginal::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Marginal has no CPU implementation.");
}

void Marginal::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  switch (kind_) {
    case 0: {
      auto& qkv = inputs[0];
      const int T = qkv.shape(0);
      const int elements = T * i0_ * i1_;   // T * n_heads * head_dim
      // out is a fresh copy; encode from the copied qkv (input[0] already contiguous copy).
      tk::launch_marginal_copy(enc, qkv, out, (uint32_t)qkv.nbytes());
      tk::launch_tau_tail(enc, out, inputs[1], inputs[2], inputs[3], elements, i0_, i1_, i2_,
                          marg_tname(qkv));
      break;
    }
    case 1:
      tk::launch_packbits(enc, inputs[0], out, i0_, i1_);
      break;
    case 2:
      tk::launch_segment_packbits(enc, inputs[0], inputs[1], inputs[2], out, i0_, i1_, i2_);
      break;
    case 3:
      tk::launch_permute_cols(enc, inputs[0], inputs[1], out, i0_, i1_);
      break;
    default:
      throw std::runtime_error("Marginal: unknown kind");
  }
}

std::vector<array> Marginal::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Marginal has no jvp."); }
std::vector<array> Marginal::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("Marginal has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> Marginal::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Marginal has no vmap."); }

} // namespace mlx::core
