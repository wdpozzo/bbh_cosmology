from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules=[
             Extension("cosmology",
                       sources=["cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = ["/Users/wdp/opt/lalonference_o2/lib"],
                       include_dirs=[numpy.get_include(),"/Users/wdp/opt/lalonference_o2/include"]
                       )
             ]

setup(
      name = "cosmology",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),"/Users/wdp/opt/lalonference_o2/include"]
      )
