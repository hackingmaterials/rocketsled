from __future__ import unicode_literals, print_function, division


try:
    import os
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='rocketsled',
      version='1.0',
      author = 'Alex Dunn',
      description = 'Black box optimization with Fireworks workflows, on rails',
      packages=['rocketsled'],
      test_suite="nose.collector",
      install_requires=['fireworks', 'scikit-learn', 'scipy', 'numpy', 'pymongo'
                        , 'nose', 'matplotlib'])
