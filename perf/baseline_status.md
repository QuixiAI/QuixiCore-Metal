# QuixiCore Metal Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw benchmark
output should live under `perf/results/`; stable conclusions should be copied
into `perf/optimization_status.md`.

## Environment

Date: 2026-07-13.

Each real baseline entry should record:

- Apple Silicon model and memory configuration.
- macOS version and Xcode/Metal toolchain version.
- Integration path: MLX, PyTorch MPS, native Metal harness, or Xcode test.
- Git commit or working-tree label.
- Command line and benchmark script path.
- Warmups, measured iterations, median, variance, and raw result path.
- Correctness tolerance and observed error.

## Current Harness Index

| Area | Source | Notes |
|---|---|---|
| Attention timing | `perf/harness/time_attn.py` | Focused timing for attention kernels |
| GEMM timing | `perf/harness/time_gemm.py` | Focused timing for GEMM kernels |
| LayerNorm timing | `perf/harness/time_layernorm.py` | Focused timing for row-reduction kernels |
| General timing helpers | `perf/harness/time_perf.py` | Shared harness utilities |
| Shared kernel harness | `perf/bench_kernels.py` | Schema-v1 correctness and timing cases for every active family, including specialized composed operations |
| Kernel notebook | `perf/optimization_status.md` | Detailed historical optimization entries |

## 2026-07-13 Specialized Operation Baseline

The focused MLX baseline and final optimized runs for packed embeddings, decode
epilogues/SwiGLU, masked and candidate output projection, spatial projection,
and functional cache attention are indexed below. Exact per-case p20/p80, CV,
correctness error, and framework baseline statistics are in the JSONL files;
durable decisions are in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Initial geometry | `294f8bd-dirty` | MLX / quick | 5 / 20 | `perf/results/2026-07-13/new-kernels-baseline-quick/` |
| Candidate geometry | `294f8bd-dirty` | MLX / quick | 5 / 20 | `perf/results/2026-07-13/new-kernels-candidate-quick/` |
| Final edge shapes | `294f8bd-dirty` | MLX / smoke | 3 / 5 | `perf/results/2026-07-13/new-kernels-final-smoke/` |
| Final priority shapes | `294f8bd-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-final-quick/` |
| Second-pass baseline | `294f8bd-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-baseline-quick/` |
| Second-pass final | `294f8bd-dirty` | MLX / quick | 15 / 50 | `perf/results/2026-07-13/new-kernels-second-pass-final-quick/` |
| Cache edge | `294f8bd-dirty` | MLX / smoke | 10 / 40 | `perf/results/2026-07-13/new-kernels-second-pass-cache-smoke/` |
| Cache comprehensive | `294f8bd-dirty` | MLX / comprehensive | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-comprehensive/` |
| Cache MPS edge | `294f8bd-dirty` | PyTorch MPS / smoke | 10 / 40 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-smoke/` |
| Cache MPS priority | `294f8bd-dirty` | PyTorch MPS / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-quick/` |
| Cache MPS comprehensive | `294f8bd-dirty` | PyTorch MPS / comprehensive | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-comprehensive/` |
| Cross-kernel follow-up baseline | `bc90717` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-followups-baseline/` |
| Cross-kernel follow-up final | `bc90717-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-final-mlx/` |
| Cross-kernel follow-up MPS | `bc90717-dirty` | PyTorch MPS / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-final-mps/` |
| NVFP4 inference baseline | `c880769-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/nvfp4-experiments-baseline/` |
| NVFP4 inference final | `c880769-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/nvfp4-experiments-final/` |

## Migration Tasks

- Promote stable benchmark runs into compact per-kernel baseline tables.
- Keep large profiler traces out of git; record trace paths and summaries only.
- Store normalized raw output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
