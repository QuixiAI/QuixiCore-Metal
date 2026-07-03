# ThunderMittens

ThunderMittens is an Apple Metal Shading Language (MSL) port for the [ThunderKittens](github.com/HazyResearch/ThunderKittens/) framework.

<div align="center" >
    <img src="assets/mittens.jpeg" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

## Performance highlights

Hand-tuned tile kernels that **beat Apple's own optimized primitives** — and, where the algorithm
allows, change the complexity class entirely. All numbers are measured median per-call latency on an
**Apple M4 Max** (40-core GPU, ~546 GB/s), MLX backend, `speedup = baseline_ms / tk_ms`, reproducible
via `perf/bench_kernels.py`. ~40 kernel families ship on **two backends** (MLX + PyTorch/MPS) from one
metallib, cross-checked by 1,798 correctness + parity tests.

### Faster than the framework's own tuned kernels

| kernel | shape | vs | speedup |
|---|---|---|--:|
| **cross-entropy** (fused-linear) | T2048 × V128256 | framework CE composition | **13.6×** |
| **layernorm** | 16384 × 768 | `mx.fast.layer_norm` | **3.1×** |
| **rms_norm** | 16384 × 768 | `mx.fast.rms_norm` | **3.1×** |
| **gelu** (tanh) | 16384 × 256 | `mx.nn.gelu_approx` | **3.1×** |
| **softmax** | 16384 × 768 | `mx.softmax` | **2.9×** |
| **causal attention** | 1×8×2048×128 | `scaled_dot_product_attention` | **3.9×** |
| **fused add + rms_norm** | 4096 × 1024 | `add` + `mx.fast.rms_norm` | **1.9×** |
| **matmul** | 4096×4096×32 (thin-K) | `mx.matmul` | **1.8×** |

### Algorithmic & serving wins

| kernel | shape | vs | speedup |
|---|---|---|--:|
| **linear attention** (chunked, causal) | 2×8×4096×64 | masked O(N²) reference | **6.4×** |
| **Mamba-2 / SSD backward** (linear-time chunked) | 1×8×8192×64 | O(N²) quadratic backward | **15.9×** |
| **MLA decode** (paged, partitioned) | 8×16, ctx 8192 | single-partition v1 | **6.3×** |
| **attention backward** (FlashAttention-2) | 1×8×2048×64 | naive VJP | **5.4×** |
| **beam-search advance** (single-pass top-M) | 16 beams × V128256 | framework top-k | **5.2×** |
| **paged attention v2** (partitioned decode) | 8×32, ctx 2048 | v1 decode | **4.6×** |
| **Mamba-2 / SSD forward** (chunked) | 2×16×4096×64 | O(N²) quadratic | **4.7×** |
| **Hadamard transform** | 16384 × 512 | matmul-with-H | **9.4×** |
| **weight-only quantized GEMV** | 32000 × 4096 (BitNet) | fp16 matmul | **3.8×** |
| **grouped MoE** (schedule + gather) | E8, H2048 | per-expert loop | **1.5×** |

Weight-only quant covers **~29 formats** (GGUF k-/i-quants, fp8, NVFP4, AWQ, GPTQ act-order, BitNet
ternary, …), and **every format now beats the fp16 GEMV baseline**. Sliding-window, ALiBi, and
block-sparse masks compose on the paged-decode path; the Mamba-2/linear-attention families auto-route
between an O(N²) kernel (small N) and a linear-time chunked pipeline (D∈{64,128}) at measured
crossovers.

### Serving & training kernels (Waves 5–6)

A full serving + training surface, all dual-backend (MLX + PyTorch-MPS from one metallib) and
cross-backend parity-tested:

- **Sampling / decoding** — top-k, top-p, min-p, **typical-p**, categorical, argmax (fused Gumbel-max,
  seed-reproducible on host); grammar **allow-bitmask** + **bad/stop-word** masking; repetition /
  presence / frequency penalties; fused quantized LM-head sampling (argmax / categorical / top-k /
  **top-p nucleus**, no `(T,V)` logits materialization).
- **Speculative decoding** — linear (vLLM) **and tree** rejection verification, plus device-resident
  `spec_compact` / `spec_update_kv_meta` post-processing (no host readback between verify and KV append).
- **Beam search** — single-pass advance, on-device KV reorder (copy-path **and** zero-copy
  block-table remap), device copy-pair builder.
- **Attention** — cascade / shared-prefix decode, **N-level** (multi shared prefix) with an optional
  **fp8 prefix**, over the paged-v2 partition/reduce; **fully device-resident varlen prefill** (device
  Q pad/gather + output re-gather, no host pad/transpose loop).
- **Multimodal / embeddings** — token embedding lookup **+ backward** (atomic scatter-add), on-device
  multimodal `src` builder + span merge.
- **Training** — RMSNorm / LayerNorm / GELU **backward**, fused-add-RMSNorm backward, GLU-family
  backward (all 6 modes), **dropout** (mask-free, seed-reproducible), and a fused **AdamW** step.

Measured Wave-6 perf wins (MLX, M-series): **embedding_lookup 8.0×** and **merge_multimodal_spans
11.9×** (threadgroup-per-token + vec4; embedding is now 2.3× *faster* than the framework gather),
**beam KV reorder ~1.45×** (vec4 cache clone/copy). See `perf/optimization_status.md` for the full
sweep + the honest rejects.

## Prerequisites (all paths)

- Apple Silicon Mac.
- **Xcode with the Metal Toolchain installed.** Recent Xcode ships the Metal compiler as a
  separate component — `xcrun --find metal` resolving a path is not enough. Install it once with:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  Without this, both the MLX build and the PyTorch metallib build fail with
  `cannot execute tool 'metal' due to missing Metal Toolchain`.

## Project Structure

The same framework-agnostic `.metal` kernels (under `ThunderMittens/include` + `ThunderMittens/kernels`)
power three use cases:

### 1. MSL Kernel Development

For writing Metal Shading Language (MSL) kernels:
- Open the project in Xcode
- Xcode will handle all build processes
- Primitive unit tests live in `ThunderMittens/tests/unit` (gated by `ENABLE_TESTS` in
  `tests/unit/testing_commons/testing_flags.hpp`); build/run the `ThunderMittens` scheme.

### 2. MLX Kernel Integration with Python

For using ThunderKittens kernels within MLX in Python:

#### Prerequisites
- Python 3.8+
- CMake
- Xcode Command Line Tools

#### Installation Steps

1. Navigate to ThunderMittens/mlx directory
2. Install MLX with parallel build:
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"
   ```

3. Navigate to ThunderMittens/kernels directory
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Build kernels and bindings:
   ```bash
   python setup.py build_ext -j8 --inplace
   ```

> Build artifacts are written to a repo-root `build/` dir (via `kernels/setup.cfg`) so they stay
> out of the Xcode-synchronized source tree. Validate with `python -m pytest */correctness/`.

### 3. PyTorch (Apple MPS) Integration with Python

For using ThunderKittens kernels from PyTorch on the `mps` device. This path is **independent of
MLX** — it needs only PyTorch and the Metal toolchain (no CMake/nanobind, no MLX build). The same
`.metal` kernels are compiled into a standalone metallib with `xcrun metal` and dispatched onto
PyTorch's MPS stream.

#### Prerequisites
- Python 3.9+
- PyTorch with MPS (`torch>=2.1`; nightlies work). MPS custom-kernel support uses
  `torch::mps::get_command_buffer()`.
- The Metal toolchain (see [Prerequisites](#prerequisites-all-paths)).

#### Installation Steps

1. Navigate to the PyTorch backend and install it (pulls in PyTorch, leaves an existing install
   such as a nightly untouched):
   ```bash
   pip install -e ThunderMittens/kernels/tk_torch
   ```
2. Use it — the metallib and the ObjC++ extension build automatically on first import:
   ```python
   import torch
   import tk_torch

   x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device="mps")
   w = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   b = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   y = tk_torch.layernorm(x, w, b)   # matches torch.nn.functional.layer_norm
   ```

## Testing

Run from `ThunderMittens/kernels/`:

```bash
# MLX correctness (each kernel vs an MLX oracle, e.g. mx.fast.layer_norm)
python -m pytest */correctness/

# PyTorch MPS correctness (each kernel vs a torch reference, e.g. F.layer_norm)
python -m pytest tk_torch/tests/

# Cross-backend parity: the SAME metallib kernel on MLX vs MPS must agree for
# identical inputs (catches host-ABI drift between the two backends)
python -m pytest tests_parity/

# ...or everything at once (each suite skips cleanly if its framework is absent)
python -m pytest */correctness/ tk_torch/tests/ tests_parity/
```

Primitive-level MSL unit tests (register/shared tile ops) build and run through Xcode — see
[1. MSL Kernel Development](#1-msl-kernel-development).

## References

Please see [our blog post](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx) to learn more about this work. Please checkout [our paper](https://arxiv.org/abs/2410.20399) to learn more about the ThunderKittens project. 



