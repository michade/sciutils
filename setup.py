# -*- coding: utf-8 -*-

import os
from itertools import chain
from Cython.Build import cythonize

from setuptools import setup, Extension, find_packages


ext_modules = [
    Extension(
        'chiapet.points_in_regions',
        ['chiapet/points_in_regions.pyx']
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp']
    )
]


def _get_version(dir_):
    with open(os.path.join(dir_, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


setup(
    name='sciutils',
    version=_get_version('.'),
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
         ext_modules, annotate=True
    )
)
