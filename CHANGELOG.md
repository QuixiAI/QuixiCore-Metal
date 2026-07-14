# Changelog

All notable QuixiCore Metal changes should be recorded here.

## Unreleased

- Added packed quantized embedding lookup/bag, fused dense and packed decode
  epilogues/SwiGLU, packed-mask and CSR-candidate output projection,
  space-to-depth/norm/projection, and functional decode-cache attention for MLX
  and PyTorch MPS.
- Added measured routing for dense decode, spatial projection, and functional
  cache attention, plus focused benchmark registrations for all new operations.
- Added QuixiCore-standard repository structure documentation, including
  Metal/Xcode-specific layout rules.
- Added QuixiCore metadata manifests for backend identity, kernel family
  coverage, and quant format coverage.
- Added standardized contributor, security, changelog, formatting, and script
  entrypoint files.
