// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "selective_scan/selective_scan.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool sscan_is_io(Dtype t) { return t == float32 || t == float16 || t == bfloat16; }
} // namespace

std::vector<array> selective_scan(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& state, const std::optional<array>& D,
    const std::optional<array>& delta_bias, const std::optional<array>& z,
    bool delta_softplus, StreamOrDevice s) {
  if (u.ndim() != 3 || !sscan_is_io(u.dtype())) {
    throw std::invalid_argument("selective_scan: u must be (batch, dim, seqlen) f32/f16/bf16");
  }
  const int batch = u.shape(0), dim = u.shape(1);
  if (B.ndim() != 4 || B.shape(0) != batch || C.shape() != B.shape()) {
    throw std::invalid_argument("selective_scan: B/C must be (batch, n_groups, dstate, seqlen)");
  }
  const int n_groups = B.shape(1), dstate = B.shape(2);
  if (dstate > 256 || dim % n_groups != 0) {
    throw std::invalid_argument("selective_scan: dstate <= 256 and dim % n_groups == 0");
  }
  if (A.ndim() != 2 || A.shape(0) != dim || A.shape(1) != dstate) {
    throw std::invalid_argument("selective_scan: A must be (dim, dstate)");
  }
  if (state.ndim() != 3 || state.shape(0) != batch || state.shape(1) != dim ||
      state.shape(2) != dstate) {
    throw std::invalid_argument("selective_scan: state must be (batch, dim, dstate)");
  }
  auto dt = u.dtype();
  auto dummy_io = astype(zeros({1}, s), dt, s);
  auto dummy_f = zeros({1}, float32, s);
  return array::make_arrays(
      {u.shape(), state.shape()},
      {dt, float32},
      std::make_shared<SelectiveScan>(to_stream(s), false, D.has_value(),
                                      delta_bias.has_value(), z.has_value(), delta_softplus,
                                      false, false, -1, batch),
      {contiguous(u, false, s), contiguous(astype(delta, dt, s), false, s),
       contiguous(astype(A, float32, s), false, s),
       contiguous(astype(B, dt, s), false, s), contiguous(astype(C, dt, s), false, s),
       D ? contiguous(astype(*D, float32, s), false, s) : dummy_f,
       delta_bias ? contiguous(astype(*delta_bias, float32, s), false, s) : dummy_f,
       z ? contiguous(astype(*z, dt, s), false, s) : dummy_io,
       contiguous(astype(state, float32, s), false, s)});
}

std::vector<array> selective_scan_varlen(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& query_start_loc, const array& state, const std::optional<array>& D,
    const std::optional<array>& delta_bias, const std::optional<array>& z,
    const std::optional<array>& cache_indices, const std::optional<array>& has_initial_state,
    bool delta_softplus, int null_block_id, StreamOrDevice s) {
  if (u.ndim() != 2 || !sscan_is_io(u.dtype())) {
    throw std::invalid_argument("selective_scan_varlen: u must be (dim, total_tokens)");
  }
  const int dim = u.shape(0);
  if (B.ndim() != 3 || C.shape() != B.shape() || B.shape(2) != u.shape(1)) {
    throw std::invalid_argument(
        "selective_scan_varlen: B/C must be (n_groups, dstate, total_tokens)");
  }
  const int n_groups = B.shape(0), dstate = B.shape(1);
  if (dstate > 256 || dim % n_groups != 0) {
    throw std::invalid_argument("selective_scan_varlen: dstate <= 256, dim % n_groups == 0");
  }
  if (query_start_loc.ndim() != 1 || query_start_loc.shape(0) < 2) {
    throw std::invalid_argument("selective_scan_varlen: query_start_loc must be (B+1,)");
  }
  const int batch = query_start_loc.shape(0) - 1;
  if (state.ndim() != 3 || state.shape(1) != dim || state.shape(2) != dstate) {
    throw std::invalid_argument("selective_scan_varlen: state must be (num_slots, dim, dstate)");
  }
  auto dt = u.dtype();
  auto dummy_io = astype(zeros({1}, s), dt, s);
  auto dummy_f = zeros({1}, float32, s);
  auto dummy_i = zeros({1}, int32, s);
  auto dummy_u8 = astype(zeros({1}, s), uint8, s);
  return array::make_arrays(
      {u.shape(), state.shape()},
      {dt, float32},
      std::make_shared<SelectiveScan>(to_stream(s), true, D.has_value(),
                                      delta_bias.has_value(), z.has_value(), delta_softplus,
                                      cache_indices.has_value(), has_initial_state.has_value(),
                                      null_block_id, batch),
      {contiguous(u, false, s), contiguous(astype(delta, dt, s), false, s),
       contiguous(astype(A, float32, s), false, s),
       contiguous(astype(B, dt, s), false, s), contiguous(astype(C, dt, s), false, s),
       D ? contiguous(astype(*D, float32, s), false, s) : dummy_f,
       delta_bias ? contiguous(astype(*delta_bias, float32, s), false, s) : dummy_f,
       z ? contiguous(astype(*z, dt, s), false, s) : dummy_io,
       contiguous(astype(query_start_loc, int32, s), false, s),
       cache_indices ? contiguous(astype(*cache_indices, int32, s), false, s) : dummy_i,
       has_initial_state ? contiguous(astype(*has_initial_state, uint8, s), false, s)
                         : dummy_u8,
       contiguous(astype(state, float32, s), false, s)});
}

std::vector<array> selective_scan_varlen_apc(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& query_start_loc, const array& cache_indices, const array& has_initial_state,
    const array& state, const array& block_idx_first_scheduled_token,
    const array& block_idx_last_scheduled_token, const array& initial_state_idx,
    const array& cu_chunk_seqlen, const array& last_chunk_indices, int block_size,
    int cache_indices_stride, bool use_chunk_metadata, const std::optional<array>& D,
    const std::optional<array>& delta_bias, const std::optional<array>& z, bool delta_softplus,
    int null_block_id, StreamOrDevice s) {
  if (u.ndim() != 2 || !sscan_is_io(u.dtype())) {
    throw std::invalid_argument("selective_scan_varlen_apc: u must be (dim, total_tokens)");
  }
  const int dim = u.shape(0);
  if (B.ndim() != 3 || C.shape() != B.shape() || B.shape(2) != u.shape(1)) {
    throw std::invalid_argument(
        "selective_scan_varlen_apc: B/C must be (n_groups, dstate, total_tokens)");
  }
  const int n_groups = B.shape(0), dstate = B.shape(1);
  if (dstate > 256 || dim % n_groups != 0) {
    throw std::invalid_argument("selective_scan_varlen_apc: dstate <= 256, dim % n_groups == 0");
  }
  if (query_start_loc.ndim() != 1 || query_start_loc.shape(0) < 2) {
    throw std::invalid_argument("selective_scan_varlen_apc: query_start_loc must be (B+1,)");
  }
  const int batch = query_start_loc.shape(0) - 1;
  if (state.ndim() != 3 || state.shape(1) != dim || state.shape(2) != dstate) {
    throw std::invalid_argument(
        "selective_scan_varlen_apc: state must be (num_slots, dim, dstate)");
  }
  auto dt = u.dtype();
  auto dummy_io = astype(zeros({1}, s), dt, s);
  auto dummy_f = zeros({1}, float32, s);
  return array::make_arrays(
      {u.shape(), state.shape()},
      {dt, float32},
      std::make_shared<SelectiveScanApc>(to_stream(s), D.has_value(), delta_bias.has_value(),
                                         z.has_value(), delta_softplus, null_block_id, batch,
                                         block_size, cache_indices_stride, use_chunk_metadata),
      {contiguous(u, false, s), contiguous(astype(delta, dt, s), false, s),
       contiguous(astype(A, float32, s), false, s),
       contiguous(astype(B, dt, s), false, s), contiguous(astype(C, dt, s), false, s),
       D ? contiguous(astype(*D, float32, s), false, s) : dummy_f,
       delta_bias ? contiguous(astype(*delta_bias, float32, s), false, s) : dummy_f,
       z ? contiguous(astype(*z, dt, s), false, s) : dummy_io,
       contiguous(astype(query_start_loc, int32, s), false, s),
       contiguous(astype(cache_indices, int32, s), false, s),
       contiguous(astype(has_initial_state, uint8, s), false, s),
       contiguous(astype(state, float32, s), false, s),
       contiguous(astype(block_idx_first_scheduled_token, int32, s), false, s),
       contiguous(astype(block_idx_last_scheduled_token, int32, s), false, s),
       contiguous(astype(initial_state_idx, int32, s), false, s),
       contiguous(astype(cu_chunk_seqlen, int32, s), false, s),
       contiguous(astype(last_chunk_indices, int32, s), false, s)});
}

void SelectiveScanApc::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SelectiveScanApc has no CPU implementation.");
}

void SelectiveScanApc::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  auto& state_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  state_out.set_data(allocator::malloc_or_wait(state_out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  auto& u = inputs[0];
  const int dim = u.shape(0);
  const int total_tokens = u.shape(1);
  const int n_groups = inputs[3].shape(0);
  const int dstate = inputs[3].shape(1);
  auto& state_in = inputs[11];
  tk::launch_sscan_pool_clone(enc, state_in, state_out,
                              static_cast<uint32_t>(state_in.size()));
  tk::launch_selective_scan_varlen_apc(
      enc, u, inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
      inputs[8], inputs[9], inputs[10], out, state_out, inputs[12], inputs[13], inputs[14],
      inputs[15], inputs[16], batch_, dim, total_tokens, dstate, n_groups, has_d_ ? 1 : 0,
      has_bias_ ? 1 : 0, has_z_ ? 1 : 0, delta_softplus_ ? 1 : 0, null_block_id_, block_size_,
      cache_indices_stride_, use_chunk_metadata_ ? 1 : 0, type_to_name(u));
}

std::vector<array> SelectiveScanApc::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SelectiveScanApc has no jvp."); }
std::vector<array> SelectiveScanApc::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("SelectiveScanApc has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> SelectiveScanApc::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SelectiveScanApc has no vmap."); }

void SelectiveScan::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SelectiveScan has no CPU implementation.");
}

void SelectiveScan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  auto& state_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  state_out.set_data(allocator::malloc_or_wait(state_out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);

  auto& u = inputs[0];
  auto& delta = inputs[1];
  auto& A = inputs[2];
  auto& B = inputs[3];
  auto& C = inputs[4];
  auto& D = inputs[5];
  auto& bias = inputs[6];
  auto& z = inputs[7];
  const int dim = varlen_ ? u.shape(0) : u.shape(1);
  const int n_groups = varlen_ ? B.shape(0) : B.shape(1);
  const int dstate = varlen_ ? B.shape(1) : B.shape(2);

  if (!varlen_) {
    auto& state_in = inputs[8];
    const int seqlen = u.shape(2);
    // clone the incoming state, then the kernel updates it in place (functional output)
    tk::launch_sscan_pool_clone(enc, state_in,
                                state_out, static_cast<uint32_t>(state_in.size()));
    tk::launch_selective_scan_dense(enc, u, delta, A, B, C, D, bias, z, out, state_out,
                                    batch_, dim, seqlen, dstate, n_groups,
                                    has_d_ ? 1 : 0, has_bias_ ? 1 : 0, has_z_ ? 1 : 0,
                                    delta_softplus_ ? 1 : 0, type_to_name(u));
  } else {
    auto& qsl = inputs[8];
    auto& cidx = inputs[9];
    auto& his = inputs[10];
    auto& state_in = inputs[11];
    const int total_tokens = u.shape(1);
    tk::launch_sscan_pool_clone(enc, state_in,
                                state_out, static_cast<uint32_t>(state_in.size()));
    tk::launch_selective_scan_varlen(enc, u, delta, A, B, C, D, bias, z, qsl, cidx, his,
                                     out, state_out, batch_, dim, total_tokens, dstate,
                                     n_groups, has_d_ ? 1 : 0, has_bias_ ? 1 : 0,
                                     has_z_ ? 1 : 0, delta_softplus_ ? 1 : 0,
                                     use_cache_indices_ ? 1 : 0,
                                     use_has_initial_state_ ? 1 : 0, null_block_id_,
                                     type_to_name(u));
  }
}

std::vector<array> SelectiveScan::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SelectiveScan has no jvp implementation.");
}
std::vector<array> SelectiveScan::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("SelectiveScan has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SelectiveScan::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SelectiveScan has no vmap implementation.");
}

} // namespace mlx::core
