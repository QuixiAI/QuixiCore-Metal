# Tests

Use `scripts/test` as the common entrypoint when possible.

Current Metal tests live under `ThunderMittens/tests/`,
`ThunderMittens/kernels/*/correctness/`, `ThunderMittens/kernels/tk_torch/tests/`,
and `ThunderMittens/kernels/tests_parity/`. New contract tests should mirror the
kernel taxonomy under `tests/correctness/<family>/<operation>/`.
