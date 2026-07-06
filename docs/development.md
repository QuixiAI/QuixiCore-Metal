# Development

QuixiCore Metal uses Xcode, Metal Shading Language, Objective-C++, MLX, and
PyTorch MPS while tracking the shared QuixiCore contract.

Use the common scripts first:

```bash
scripts/configure
scripts/build help
scripts/test help
scripts/bench --help
```

Xcode project groups should mirror filesystem directories. Keep shared project
settings committed, but do not commit user-specific `xcuserdata/`.
