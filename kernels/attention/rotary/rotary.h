// Copyright © 2023 Apple Inc.

#pragma once

#include <utility>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Rotary positional embedding (RoPE), split-half / GPT-NeoX convention
 *  (matches mx.fast.rope(..., traditional=False)).
 *
 *  x is (B, H, N, D); cos and sin are precomputed (N, D/2). bf16 in/out.
 *  D in {64, 128}.
 **/
array rotary(
    const array& x,   // (B, H, N, D)
    const array& cos, // (N, D/2)
    const array& sin, // (N, D/2)
    bool interleaved = false,  // false = split-half (NeoX); true = GPT-J interleaved
    StreamOrDevice s = {}
);

/** Positioned RoPE with explicit positions and partial rotary dimensions.
 *
 * x is (B,H,N,D), D in {64,128,256,512}; positions is (N,) or (B,N).
 * cos/sin are (max_pos, rotary_dim/2).  A rotary_dim of 0 means D.
 */
array rotary_positioned(
    const array& x,
    const array& cos,
    const array& sin,
    const array& positions,
    int rotary_dim = 0,
    bool interleaved = false,
    StreamOrDevice s = {});

/** Three-axis temporal/height/width M-RoPE using split-half pairing.
 *
 * positions is (3,N) or (B,3,N). sections contains three rotary-pair
 * counts summing to rotary_dim/2. section_interleaved=False uses contiguous
 * T/H/W sections; true uses the Qwen THWTHW... map.
 */
array mrope(
    const array& x,
    const array& cos,
    const array& sin,
    const array& positions,
    const std::vector<int>& sections,
    int rotary_dim = 0,
    bool section_interleaved = false,
    StreamOrDevice s = {});

/** Two-axis vision RoPE with explicit channel layout.
 *
 * x is (B,H,N,D) bf16, positions is (B,N,2), and cos/sin are
 * (max_pos,D/4). global_split=false uses Gemma's independent D/2 x/y blocks,
 * each with local split-half pairing. global_split=true uses Qwen's global
 * split-half pairing with x/y frequency sections repeated across both halves.
 */
array vision_rope_2d(
    const array& x, const array& cos, const array& sin,
    const array& positions, bool global_split = false, StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Rotary : public Primitive {
 public:
  explicit Rotary(Stream stream, bool interleaved = false)
      : Primitive(stream), interleaved_(interleaved) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;
  const char* name() const { return "Rotary"; }


  void print(std::ostream& os) override {
    os << "Rotary";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  bool interleaved_;
};

class RotaryPositioned : public Primitive {
 public:
  RotaryPositioned(Stream stream, int rotary_dim, bool interleaved, bool batched_positions)
      : Primitive(stream), rotary_dim_(rotary_dim), interleaved_(interleaved),
        batched_positions_(batched_positions) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "RotaryPositioned"; }
  void print(std::ostream& os) override { os << "RotaryPositioned"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int rotary_dim_;
  bool interleaved_;
  bool batched_positions_;
};

class MRope : public Primitive {
 public:
  MRope(Stream stream, std::vector<int> sections, int rotary_dim,
        bool section_interleaved, bool batched_positions)
      : Primitive(stream), sections_(std::move(sections)), rotary_dim_(rotary_dim),
        section_interleaved_(section_interleaved), batched_positions_(batched_positions) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MRope"; }
  void print(std::ostream& os) override { os << "MRope"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> sections_;
  int rotary_dim_;
  bool section_interleaved_;
  bool batched_positions_;
};

class VisionRoPE2D : public Primitive {
 public:
  VisionRoPE2D(Stream stream, bool global_split)
      : Primitive(stream), global_split_(global_split) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "VisionRoPE2D"; }
  void print(std::ostream& os) override {
    os << "VisionRoPE2D[" << (global_split_ ? "global_split" : "axis_blocks") << "]";
  }
  bool is_equivalent(const Primitive& other) const override {
    return global_split_ == static_cast<const VisionRoPE2D&>(other).global_split_;
  }

 private:
  bool global_split_;
};

} // namespace mlx::core
