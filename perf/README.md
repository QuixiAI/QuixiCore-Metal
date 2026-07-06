# Performance

Use `scripts/bench` as the common entrypoint when possible. Existing Metal
benchmark harnesses remain under this directory.

Record results with enough context to reproduce them: Apple Silicon model,
macOS version, Xcode/Metal toolchain version, integration path, kernel variant,
shape, dtype, quant format, and git commit.
