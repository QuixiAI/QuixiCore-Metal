# Security Policy

QuixiCore Metal is a native GPU backend. Security issues may involve host-side
bindings, Metal dispatch validation, memory bounds, Xcode/build tooling, or
packaged artifacts.

## Reporting

Do not open a public issue for a suspected vulnerability. Report security
issues through the QuixiAI GitHub security advisory flow or contact the
maintainers privately.

When reporting, include:

- Affected repository and commit.
- Affected Apple Silicon model, macOS version, and Xcode version.
- Minimal reproduction steps.
- Whether the issue affects public APIs, MLX bindings, PyTorch MPS bindings,
  generated metallibs, or kernel execution.

## Scope

Issues in shared QuixiCore semantics should be reported against
QuixiAI/QuixiCore. Issues in Metal implementation code should be reported
against this repository.
