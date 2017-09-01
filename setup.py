import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

"""
The module uses Eigen C++ library and therefore
setup.py needs to be able to locate it.
Change the last directory in include_dirs
for your local directory containing Eigen
header files.
"""

cur_dir = os.path.dirname(__file__)
lib_dir = os.path.join(cur_dir, r'lib')
os.chdir(cur_dir)

sourcefiles = ["pyhawkes.pyx", "lib\exp_hawkes.cpp",
               "lib\power_hawkes.cpp", "lib\general_hawkes.cpp"]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pyhawkes", sourcefiles, language="c++",
                             include_dirs=[".", np.get_include(), lib_dir,
                                           r'D:\Materials\Programming\C++\Libraries\include'])])

