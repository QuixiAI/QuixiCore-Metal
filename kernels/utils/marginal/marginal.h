// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** tau_tail (Kimi-style tail scaling): in-place-semantics functional scale of the Q and V
 *  slices of a packed (T, 3*q_dim) QKV by tanh(tok_qv_lin) + tau_pos_table[pos, head].
 *  tok_qv_lin (T, 2*n_heads); tau_pos_table (max_pos, n_heads); positions (T,). Returns a
 *  new (T, 3*q_dim) array (K slice copied through). */
array tau_tail(const array& qkv, const array& tok_qv_lin, const array& tau_pos_table,
               const array& positions, int n_heads, int head_dim, StreamOrDevice s = {});

/** packbits: pack a bool/uint8 array's last-axis (row-major flattened) into bits
 *  (np.packbits). bitorder "big" (default) or "little". Returns uint8 (ceil(N/8),). */
array packbits(const array& x, bool bit_order_big = true, StreamOrDevice s = {});

/** segment_packbits: ragged per-row packbits. input flat uint8, input_indptr (S+1,) row
 *  offsets, output_indptr (S+1,) host-computed byte offsets (cumsum of ceil(len/8)).
 *  Returns uint8 (output_indptr[-1],). */
array segment_packbits(const array& x, const array& input_indptr, const array& output_indptr,
                       int total_output_bytes, bool bit_order_big = true,
                       StreamOrDevice s = {});

/** permute_cols: 16-bit column gather output[:, c] = x[:, perm[c]] (dtype-agnostic on
 *  2-byte elements). x is (rows, cols) of a 16-bit dtype; perm (cols,) int32. */
array permute_cols(const array& x, const array& perm, StreamOrDevice s = {});

class Marginal : public Primitive {
 public:
  // kind: 0 tau_tail, 1 packbits, 2 segment_packbits, 3 permute_cols.
  Marginal(Stream stream, int kind, int i0, int i1, int i2)
      : Primitive(stream), kind_(kind), i0_(i0), i1_(i1), i2_(i2) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Marginal"; }
  void print(std::ostream& os) override { os << "Marginal[" << kind_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const Marginal&>(other);
    return kind_ == o.kind_ && i0_ == o.i0_ && i1_ == o.i1_ && i2_ == o.i2_;
  }

 private:
  int kind_, i0_, i1_, i2_;
};

} // namespace mlx::core
