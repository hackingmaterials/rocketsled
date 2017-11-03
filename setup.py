from __future__ import unicode_literals, print_function, division


try:
    import os
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='rocketsled',
      version='0.1',
      author = 'Alex Dunn',
      description = 'Machine learning integration with Fireworks workflows',
      packages=['rocketsled', 'rs_examples', 'rs_benchmarks'],
      test_suite="rocketsled")
