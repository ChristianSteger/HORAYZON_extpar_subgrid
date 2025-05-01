# Compile with: python setup.py build_ext --inplace

import sys
import os
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

path_lib_conda = os.environ["CONDA_PREFIX"] + "/lib/"
print(os.environ["CONDA_PREFIX"])
if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
    compiler = "gcc"
    extra_compile_args = ["-O3"]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
    compiler = "clang"
    extra_compile_args = ["-O3", "-std=c++11"]
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")
include_dirs_cpp = [np.get_include()]
extra_objects_cpp = [path_lib_conda + i + lib_end for i in ["libembree4"]]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

os.environ["CC"] = compiler

setup(ext_modules=cythonize(Extension(
           "horizon_svf",
           sources=["horizon_svf.pyx", "mo_lradtopo_horayzon.cpp"],
           include_dirs=include_dirs_cpp,
           extra_objects=extra_objects_cpp,
           extra_compile_args=extra_compile_args,
           language="c++",
      )))
