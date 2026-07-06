// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_varlen/attn_varlen.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array attn_varlen_prefill(
    const array& q_hm,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& tile_seq,
    const array& tile_local0,
    const array& seq_qlen,
    float scale,
    float softcap /* = 0.0f */,
    const std::optional<array>& sinks /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  assert(q_hm.dtype() == bfloat16 && key_cache.dtype() == bfloat16 &&
         value_cache.dtype() == bfloat16);
  assert(q_hm.ndim() == 3);
  const int H = q_hm.shape(0);
  const int D = q_hm.shape(2);
  assert(D == 64 || D == 128);
  const bool has_sink = sinks.has_value();
  if (has_sink && (sinks->ndim() != 1 || sinks->shape(0) != H)) {
    throw std::invalid_argument("attn_varlen_prefill: sinks must be (H,)");
  }
  // The kernel assumes contiguous head-major q/o and contiguous index arrays; the Python wrapper
  // builds q_hm by transpose (a strided view), so force row-contiguity here.
  auto q_c = contiguous(q_hm, false, s);
  // 9th input: sink buffer (q as the placeholder binding when absent — never read then).
  auto sink_arr = has_sink ? contiguous(astype(*sinks, float32, s), false, s) : q_c;
  return array(
      q_hm.shape(), bfloat16,
      std::make_shared<AttnVarlenPrefill>(to_stream(s), scale, softcap, has_sink),
      {q_c, contiguous(key_cache, false, s),
       contiguous(value_cache, false, s), contiguous(astype(block_table, int32, s), false, s),
       contiguous(astype(context_lens, int32, s), false, s),
       contiguous(astype(tile_seq, int32, s), false, s),
       contiguous(astype(tile_local0, int32, s), false, s),
       contiguous(astype(seq_qlen, int32, s), false, s), sink_arr});
}

std::vector<array> varlen_build_worklist(
    const array& cu_seqlens,
    int max_tiles,
    StreamOrDevice s /* = {} */) {
  assert(cu_seqlens.ndim() == 1);
  const int B = cu_seqlens.shape(0) - 1;
  assert(B >= 1 && "varlen_build_worklist: B must be >= 1");   // chunked scan: any B
  auto cu_c = contiguous(astype(cu_seqlens, int32, s), false, s);
  auto out = array::make_arrays(
      {{B}, {B + 1}, {max_tiles}, {max_tiles}, {1}},
      {int32, int32, int32, int32, int32},
      std::make_shared<VarlenBuildWorklist>(to_stream(s), max_tiles),
      {cu_c});
  return out;
}

void VarlenBuildWorklist::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("VarlenBuildWorklist has no CPU implementation.");
}

void VarlenBuildWorklist::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& cu_seqlens = inputs[0];
  auto& qlens = outputs[0];
  auto& pad_off = outputs[1];
  auto& tile_seq = outputs[2];
  auto& tile_local0 = outputs[3];
  auto& n_tiles = outputs[4];
  for (auto* o : {&qlens, &pad_off, &tile_seq, &tile_local0, &n_tiles}) {
    o->set_data(allocator::malloc_or_wait(o->nbytes()));
  }
  const int B = cu_seqlens.shape(0) - 1;
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_varlen_build_worklist(enc, cu_seqlens, qlens, pad_off, tile_seq, tile_local0, n_tiles,
                                   B, max_tiles_);
}

std::vector<array> VarlenBuildWorklist::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("VarlenBuildWorklist has no jvp implementation.");
}
std::vector<array> VarlenBuildWorklist::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("VarlenBuildWorklist has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> VarlenBuildWorklist::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("VarlenBuildWorklist has no vmap implementation.");
}

array varlen_pad_q(
    const array& q_packed, const array& cu_seqlens, const array& pad_off, int total_padded,
    StreamOrDevice s /* = {} */) {
  if (q_packed.ndim() != 3 || q_packed.dtype() != bfloat16) {
    throw std::invalid_argument("varlen_pad_q: q_packed must be (total_q, H, D) bf16");
  }
  const int H = q_packed.shape(1), D = q_packed.shape(2);
  return array({H, total_padded, D}, bfloat16, std::make_shared<VarlenPackGather>(to_stream(s), false),
               {contiguous(q_packed, false, s), contiguous(astype(cu_seqlens, int32, s), false, s),
                contiguous(astype(pad_off, int32, s), false, s)});
}

array varlen_regather_o(
    const array& o_hm, const array& cu_seqlens, const array& pad_off, int total_q,
    StreamOrDevice s /* = {} */) {
  if (o_hm.ndim() != 3 || o_hm.dtype() != bfloat16) {
    throw std::invalid_argument("varlen_regather_o: o_hm must be (H, total_padded, D) bf16");
  }
  const int H = o_hm.shape(0), D = o_hm.shape(2);
  return array({total_q, H, D}, bfloat16, std::make_shared<VarlenPackGather>(to_stream(s), true),
               {contiguous(o_hm, false, s), contiguous(astype(cu_seqlens, int32, s), false, s),
                contiguous(astype(pad_off, int32, s), false, s)});
}

void VarlenPackGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("VarlenPackGather has no CPU implementation.");
}
void VarlenPackGather::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& src = inputs[0];
  auto& cu = inputs[1];
  auto& po = inputs[2];
  auto& dst = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dst.set_data(allocator::malloc_or_wait(dst.nbytes()));
  const int B = cu.shape(0) - 1;
  int total_q, H, D, total_padded;
  if (regather_) {          // src o_hm (H, tp, D) -> dst o_packed (total_q, H, D)
    H = src.shape(0); total_padded = src.shape(1); D = src.shape(2); total_q = dst.shape(0);
  } else {                  // src q_packed (total_q, H, D) -> dst q_hm (H, tp, D)
    total_q = src.shape(0); H = src.shape(1); D = src.shape(2); total_padded = dst.shape(1);
  }
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_varlen_pack_gather(enc, src, cu, po, dst, total_q, B, H, D, total_padded, regather_);
}
std::vector<array> VarlenPackGather::jvp(const std::vector<array>&, const std::vector<array>&,
                                         const std::vector<int>&) {
  throw std::runtime_error("VarlenPackGather has no jvp implementation.");
}
std::vector<array> VarlenPackGather::vjp(const std::vector<array>&, const std::vector<array>&,
                                         const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("VarlenPackGather has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> VarlenPackGather::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("VarlenPackGather has no vmap implementation.");
}

void AttnVarlenPrefill::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnVarlenPrefill has no CPU implementation.");
}

void AttnVarlenPrefill::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 9);
  auto& q_hm = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& tile_seq = inputs[5];
  auto& tile_local0 = inputs[6];
  auto& seq_qlen = inputs[7];
  auto& sinks = inputs[8];   // == q_hm (placeholder) when has_sink_ is false
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int H = q_hm.shape(0);
  const int total_padded = q_hm.shape(1);
  const int D = q_hm.shape(2);
  const int H_KV = key_cache.shape(2);
  const int block_size = key_cache.shape(1);
  const int bt_stride = block_table.shape(1);
  const int n_tiles = tile_seq.shape(0);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_varlen_prefill(enc, q_hm, key_cache, value_cache, block_table, context_lens,
                                 tile_seq, tile_local0, seq_qlen, out, n_tiles, total_padded, H,
                                 H_KV, block_size, bt_stride, scale_, D, softcap_, sinks,
                                 has_sink_ ? 1 : 0);
}

std::vector<array> AttnVarlenPrefill::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnVarlenPrefill has no jvp implementation.");
}
std::vector<array> AttnVarlenPrefill::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("AttnVarlenPrefill has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AttnVarlenPrefill::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnVarlenPrefill has no vmap implementation.");
}

} // namespace mlx::core
