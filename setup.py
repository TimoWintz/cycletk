#!/usr/bin/env python

from distutils.core import setup

setup(name='Cycle Toolkit',
      version='0.0',
      description='Cycle Toolkit',
      author='Timothée Wintz',
      author_email='timothee@timwin.fr',
      url='https://power.timwin.fr',
        packages=['cycletk'],
        package_dir = {'cycletk': 'cycletk'}
     )