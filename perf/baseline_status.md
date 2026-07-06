# QuixiCore Metal Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw benchmark
output should live under `perf/results/`; stable conclusions should be copied
into `perf/optimization_status.md`.

## Environment

Date: 2026-07-06.

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
| Kernel notebook | `perf/optimization_status.md` | Detailed historical optimization entries |

## Migration Tasks

- Promote stable benchmark runs into compact per-kernel baseline tables.
- Keep large profiler traces out of git; record trace paths and summaries only.
- Store normalized raw output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
