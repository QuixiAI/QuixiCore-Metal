# Tests

Use `scripts/test` as the common entrypoint when possible.

Current Metal tests live under `tests/unit/`, `tests/correctness/`,
`bindings/pytorch_mps/tests/`, and `bindings/python/tk/tests/`.
New contract tests should mirror the kernel taxonomy under
`tests/correctness/<family>/<operation>/`.
