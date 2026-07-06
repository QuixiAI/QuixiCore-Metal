# Performance

Metal performance work is tracked in `perf/`.

- `perf/perf.md` is the performance handbook and result summary.
- `perf/optimization_status.md` records optimization passes and measured
  accepts/rejects.
- `perf/bench_kernels.py` runs local benchmark sweeps.
- Raw results should stay under ignored result directories.

Benchmark reports should include Apple Silicon model, macOS version, Xcode/Metal
toolchain version, integration path, input shape, dtype, quant format, and
relevant kernel variant.
