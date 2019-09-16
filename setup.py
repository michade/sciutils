import os
from itertools import chain
from Cython.Build import cythonize

from setuptools import setup, Extension, find_packages


ext_modules = [
    Extension(
        'chiapet.points_in_regions',
        ['chiapet/points_in_regions.pyx'],
        extra_compile_args=['-fopenmp'],  # TODO: Check if this is still required
        extra_link_args=['-fopenmp']
    )
]


setup(
   name='sciutils',
   version='0.8',
   description='Various utilities for scientific and bioinformatics computing.',
   author='Michał Denkiewicz',
   author_email='michal.denkiewicz@gmail.com',
   packages=find_packages(),
   install_requires=['numpy', 'pandas', 'networkx', 'cython'],
   scripts=[
      entry.path for entry in
      chain(os.scandir('examples'), os.scandir('scripts'))
      if entry.is_file() and entry.name.endswith('.py')
   ],
   ext_modules=cythonize(
        ext_modules, annotate=False
   )
)
