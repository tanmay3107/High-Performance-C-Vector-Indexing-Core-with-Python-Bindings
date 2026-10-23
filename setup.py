from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "vector_core",
        ["src/vector_core.cpp"],
        cxx_std=20,
    ),
]

setup(
    name="vector_core",
    version="0.1.0",
    author="Tanmay Janak",
    description="High-performance C++ vector index core with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)