from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("cython_func.pyx",compiler_directives={"language_level": "3"}, annotate=True
    ),
    include_dirs=[numpy.get_include()]
    )