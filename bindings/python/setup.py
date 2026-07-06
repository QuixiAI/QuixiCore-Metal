# Copyright © 2023-2024 Apple Inc.

import os
import sys

from setuptools import setup

from mlx import extension

python_arg = f"-DPython_EXECUTABLE={sys.executable}"
cmake_args = os.environ.get("CMAKE_ARGS", "")
if python_arg not in cmake_args.split():
    os.environ["CMAKE_ARGS"] = f"{cmake_args} {python_arg}".strip()

if __name__ == "__main__":
    setup(
        name="tk",
        version="0.0.0",
        description="QuixiCore Metal Python bindings for MLX and PyTorch MPS dispatch.",
        ext_modules=[extension.CMakeExtension("tk._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["tk"],
        package_data={"tk": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
