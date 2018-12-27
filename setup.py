from setuptools import setup, find_packages

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='rocketsled',
    version='2018.12.26',
    description='Black box optimization with Fireworks workflows, on rails',
    url='https://github.com/hackingmaterials/rocktsled',
    author='Alex Dunn',
    author_email='ardunn@lbl.gov',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='modified BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'Topic :: Other/Nonlisted Topic',
        'Operating System :: OS Independent',
        ],
    keywords='black-box-optimization optimization workflows',
    test_suite='nose.collector',
    packages=find_packages(),
    install_requires=['fireworks', 'scikit-learn', 'scipy', 'numpy',
                      'pymongo==3.7.1', 'nose', 'matplotlib'],
    data_files=['LICENSE'])