# py-hawkes

Python library with C++ extensions for simulation, compensator, log-likelihood and intensity function computation for a multivariate Hawkes processes with exponential
and approximated power-law kernels with a possibility of an arbitrary base intensity function specification. 

## Getting Started

The library provides functions for simulation and computation of compensator, negative log-likelihood and conditional intensity function for a multivariate Hawkes process
with exponential, power-law or general kernel. The last one also allows specification of an arbitrary base intensity function and subsumes both exponential and power-law kernels. 
The reason why also separate exponential and power-law kernel functionalities are present is due to efficiency reasons, as general kernel procedures require Python function calls 
from C++ that are very performance-expensive. See also the Benchmarks section below for a performance comparison of different simulation routines.

For the specific usage of library functionalities see the Jupyter Notebooks in examples subfolder.

### Prerequisites

Python (together with numpy and Cython)

Eigen C++ library

### Installing

There is a dynamic library built on Windows 10, Python 3.6 and MSVC 14.0 in the main directory. Just by downloading this .pyd file in your module directory,
one can simply use "import pyhawkes" to import all the functionality of the library.

For usage on other systems and Python and compiler versions one should build the library by downloading it and running from the terminal/command prompt "python [your/pyhawkes/dir]setup.py build_ext --inplace".
In this case, Cython and numpy needs to be installed, Eigen C++ library has to be downloaded and setup.py needs to be updated in include_dirs argument by the directory where Eigen C++ library is located, i.e. 
replace the existing setup part by this snippet with updated Eigen C++ directory:

```python
setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pyhawkes", sourcefiles, language="c++",
                             include_dirs=[".", np.get_include(), lib_dir, "YOUR/EIGEN/C++/DIRECTORY"])])
```

## Running the tests

Tests are build with py.test 3.0.7. Run by py.test [your/pyhawkes/dir] > [test/output/dir]

## Built With

Python 3.6 (+ numpy 1.13.1, py.test 3.0.7, Cython 0.25.2)

MSVC 14.0

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Benchmarks

Here I present the results of simulations for different types of kernels, different base intensities and dimensions.
Exponential simulations with constant base intensity are run with sim_exp_hawkes function and 
power-law simulations with constant base intensity are run with sim_power_hawkes function.
All quadratic routines are run with the general procedure (i.e. sim_gen_hawkes).
Benchmarking was run with [benchmarks.py](https://github.com/ragoragino/py-hawkes/tree/master/examples/benchmarks.py) on a single core 
on CPU Intel® Core i7-7500U 2.7Ghz with Turbo Boost up to 3.5GHz.

| Type          | Dimensionality | Base Intensity | No. of Jumps  | Time (s)  |
| ------------- |:--------------:| :-------------:|:-------------:| ---------:| 
| Exponential   | 1              | C              | 5,000,382     | 0.990     |
| Exponential   | 1              | Q              | 4,833,340     | 14.604    |
| Power-Law     | 1              | C              | 5,004,875     | 6.279     |
| Power-Law     | 1              | Q              | 4,839,030     | 23.644    |
| Exponential   | 2              | C              | 4,897,864     | 2.270     |
| Exponential   | 2              | Q              | 5,222,768     | 37.234    |
| Power-Law     | 2              | C              | 5,004,875     | 22.231    |
| Power-Law     | 2              | Q              | 5,338,854     | 63.223    |
| Exponential   | 3              | C              | 5,162,254     | 4.316     |  
| Exponential   | 3              | Q              | 4,961,617     | 57.054    |
| Power-Law     | 3              | C              | 4,897,456     | 48.083    |
| Power-Law     | 3              | Q              | 4,706,320     | 100.749   |

where C means constant base intensity and Q quadratic base intensity. 