#!/usr/bin/env bash
#
# QuixiCore Metal bench entrypoint. Thin wrapper around the shared core
# (run_bench_core.sh, synced from the umbrella); this file is hand-written
# and backend-owned.
#
#   perf/harness/run_bench.sh --preset quick --kernel qgemv --label qgemv-ab
#   perf/harness/run_bench.sh --preset smoke --kernel layernorm --label smoke
#   perf/harness/run_bench.sh --dry-run
#
# Wraps perf/bench_kernels.py (MLX by default; set QC_METAL_BACKEND=torch for
# the PyTorch-MPS side). Uses the repo .venv python when present.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QC_BACKEND="metal"
QC_PYTHON="${QC_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$QC_PYTHON" ] || QC_PYTHON="python3"

qc_bench_cmd() {
    qc_exec "$QC_PYTHON" "$REPO_ROOT/perf/bench_kernels.py" \
        --backend "${QC_METAL_BACKEND:-mlx}" \
        --preset "${QC_PRESET:-quick}" \
        --kernel "${QC_KERNELS:-all}" \
        --out-dir "$OUT_DIR" \
        ${QC_PASSTHROUGH[@]+"${QC_PASSTHROUGH[@]}"}
}

qc_device_info() {
    echo "device=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "macos=$(sw_vers -productVersion 2>/dev/null)"
    echo "uname=$(uname -srm)"
}

source "$(dirname "${BASH_SOURCE[0]}")/run_bench_core.sh"
