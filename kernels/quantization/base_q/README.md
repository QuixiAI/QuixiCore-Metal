# Canonical BaseQN operations

This directory implements QuixiCore's framework-neutral operation contract for
canonical `base_q2`, `base_q3`, `base_q4`, `base_q5`, `base_q6`, and `base_q8`
weights. It does not load `.base` bundles and does not reproduce the BaseRT
engine API.

## Tensor contract

- `codes`: `uint8 (rows, K * bits / 8)`, with little-endian lane bits.
- `scales`: `(rows, K / group_size)` in the declared scale storage type.
- `biases`: the same shape and dtype as `scales` for asymmetric tensors; absent
  for symmetric tensors.
- `group_size`: one of 32, 64, or 128.
- `scale_dtype`: BF16, F16, E8M0, or E4M3. E4M3 is accepted only for Q8.
- `layout`: `metal`, `metal_lane_strided`, or the width-qualified
  `metal_lane_strided_qN` spelling.
- Expert stacks add a leading dimension: codes are
  `(experts, output_rows, K * bits / 8)` and scales/biases are
  `(experts, output_rows, K / group_size)`.

Asymmetric reconstruction is `code * scale + bias`. Symmetric reconstruction
is `(code - 2^(bits - 1)) * scale`. F16, BF16, and F32 activation/output types
are supported by the operation bindings. The embedding operation returns a
zero row for a negative or out-of-range id, matching QuixiCore's existing
packed-embedding contract.

`base_qgemv` decodes directly in Metal. Each simdgroup owns one output row and
each lane amortizes one scale/bias decode over eight adjacent values.
`base_qgemm` routes M=1 to that specialization. For M>1 it materializes the
weight once with `base_qdequant` and uses the framework GEMM; direct decode was
measured 6.8–7.0x slower at M=2 and M=8 on the focused shape.

`base_qgemv_qkv` accepts three independent BaseQN planes with a shared
descriptor and inner dimension. A measured crossover uses one combined grid
through K=1024 and composes three direct BaseQN GEMV operations for longer
bandwidth-bound projections. `base_qgemv_swiglu`
decodes gate and up planes together, shares activation loads, and applies
`silu(gate) * up` before the final output rounding. Neither operation owns a
model graph, cache, loader, or scheduler.

`base_qlm_head_argmax` accepts vocabulary weights in the same BaseQN planes and
an activation matrix shaped `(K, batch)`. It returns one int32 token id per
column, with lower token ids winning exact ties. The measured route uses one
direct GEMV per batch column followed by argmax. Parallel and serial two-pass
packed-reduction candidates were slower and are not retained.

`base_qmoe_gemm` and `base_qmoe_swiglu` consume QuixiCore's existing padded
MoE schedule: `expert_of_tile[i]` selects the expert for one 32-row input tile.
Both operations decode 32x32 weight tiles directly into MMA registers.
Rectangular projection retains one simdgroup per tile; four-way split-K
regressed that path but improved fused expert SwiGLU by 2.31x on the focused
shape, so split-K is retained only for SwiGLU. The gate/up row axis is laid out
as `[gate(intermediate), up(intermediate)]`. Routing, alignment, finalize, and
shared-expert policy remain separate existing tensor operations.

## Public operations

- `tk.base_qdequant`
- `tk.base_qgemv`
- `tk.base_qgemm`
- `tk.base_qembedding`
- `tk.base_qgemv_qkv`
- `tk.base_qgemv_swiglu`
- `tk.base_qlm_head_argmax`
- `tk.base_qmoe_gemm`
- `tk.base_qmoe_swiglu`

The same calls accept MLX arrays and PyTorch MPS tensors. NumPy pack/unpack and
dequantization oracles live in `bindings/python/tk/base_q.py` for fixtures and
tests, not runtime conversion.

## Provenance

The contract was independently implemented from the public mathematical and
format descriptions in the Apache-2.0 BaseRT ecosystem checkout:

- `.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md`
- `.reference/baseRT/base-convert/crates/base-quant/src/base_qn.rs`
- `.reference/baseRT/base-convert/crates/base-quant/src/base_q4.rs`
- `.reference/baseRT/base-convert/crates/base-quant/src/base_q8.rs`

No proprietary BaseRT engine source, metallib extraction, or reverse-engineered
pipeline ABI was used. The local reference contains no engine Metal source.
Consequently these operations establish canonical format semantics, not a claim
of complete BaseRT engine-kernel parity.

The fused scheduling was independently designed from QuixiCore's existing
`qgemv_fused` kernels and general decode/dequant patterns inspected in the
local MIT-licensed `~/llama.cpp` checkout at revision `2beefef68`. No llama.cpp
model runtime or inference-engine code is included in this operation layer.
