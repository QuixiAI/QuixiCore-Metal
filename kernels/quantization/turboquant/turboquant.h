// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** TurboQuant KV-cache codec. tq_encode: quantize K/V rows into the paged caches at
 *  slot_mapping offsets, returning [key_cache, value_cache, key_scale, value_scale,
 *  key_zero] (functional — untouched slots preserved). tq_decode: gather + dequantize a
 *  slot list back to [k_out, v_out] float rows. head_size in {64,128,256}; K/V bits 2..8. */
std::vector<array> tq_encode(
    const array& key, const array& value, const array& key_cache, const array& value_cache,
    const array& key_scale, const array& value_scale, const array& key_zero,
    const array& slot_mapping, const array& v_centroids, const array& signs,
    int block_size, int k_bits, bool k_signed, int v_bits, StreamOrDevice s = {});

std::vector<array> tq_decode(
    const array& key_cache, const array& value_cache, const array& key_scale,
    const array& value_scale, const array& key_zero, const array& slots,
    const array& v_centroids, const array& signs, int num_kv_heads, int head_size,
    int block_size, int k_bits, bool k_signed, int v_bits, StreamOrDevice s = {});

class TQEncode : public Primitive {
 public:
  TQEncode(Stream stream, int block_size, int k_bits, bool k_signed, int v_bits)
      : Primitive(stream), block_size_(block_size), k_bits_(k_bits), k_signed_(k_signed),
        v_bits_(v_bits) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TQEncode"; }
  void print(std::ostream& os) override { os << "TQEncode"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const TQEncode&>(other);
    return block_size_ == o.block_size_ && k_bits_ == o.k_bits_ &&
           k_signed_ == o.k_signed_ && v_bits_ == o.v_bits_;
  }

 private:
  int block_size_, k_bits_;
  bool k_signed_;
  int v_bits_;
};

class TQDecode : public Primitive {
 public:
  TQDecode(Stream stream, int num_kv_heads, int head_size, int block_size, int k_bits,
           bool k_signed, int v_bits)
      : Primitive(stream), num_kv_heads_(num_kv_heads), head_size_(head_size),
        block_size_(block_size), k_bits_(k_bits), k_signed_(k_signed), v_bits_(v_bits) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TQDecode"; }
  void print(std::ostream& os) override { os << "TQDecode"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const TQDecode&>(other);
    return num_kv_heads_ == o.num_kv_heads_ && head_size_ == o.head_size_ &&
           block_size_ == o.block_size_ && k_bits_ == o.k_bits_ &&
           k_signed_ == o.k_signed_ && v_bits_ == o.v_bits_;
  }

 private:
  int num_kv_heads_, head_size_, block_size_, k_bits_;
  bool k_signed_;
  int v_bits_;
};

} // namespace mlx::core
