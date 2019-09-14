import setuptools
from itertools import chain
import os

from setuptools import setup


setup(
   name='sciutils',
   version='0.8',
   description='Various utilities for scientific and bioinformatics computing.',
   author='Michał Denkiewicz',
   author_email='michal.denkiewicz@gmail.com',
   packages=setuptools.find_packages(),
   install_requires=['numpy', 'pandas', 'networkx'],
   scripts=[
      entry.path for entry in
      chain(os.scandir('examples'), os.scandir('scripts'))
      if entry.is_file() and entry.name.endswith('.py')
   ]
)