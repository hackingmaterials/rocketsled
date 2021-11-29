import os
from setuptools import setup, find_packages

# Version is MAJOR.MINOR.PATCH.YYYYMMDD
version = "1.1.0.20211129"

module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, "requirements.txt"), "r") as f:
    requirements = f.read().replace(" ", "").split("\n")

long_description = \
    """
    rocketsled is a black-box optimization framework "on rails" for high-throughput computation with FireWorks.

    - **Website (including documentation):** https://hackingmaterials.github.io/rocketsled/
    - **Help/Support:** https://groups.google.com/forum/#!forum/fireworkflows
    - **Source:** https://github.com/hackingmaterials/rocketsled
    - **FireWorks website:** https://materialsproject.github.io/fireworks
    
    If you find rocketsled useful, please encourage its development by citing the [following paper](http://doi.org//10.1088/2515-7639/ab0c3d) in your research:
    
    ```
    Dunn, A., Brenneck, J., Jain, A., Rocketsled: a software library for optimizing
    high-throughput computational searches. J. Phys. Mater. 2, 034002 (2019).
    ```
    
    If you find FireWorks useful, please consider citing [its paper](http://dx.doi.org/10.1002/cpe.3505) as well:
    
    ```
    Jain, A., Ong, S. P., Chen, W., Medasani, B., Qu, X., Kocher, M., Brafman, M.,
    Petretto, G., Rignanese, G.-M., Hautier, G., Gunter, D., and Persson, K. A.
    FireWorks: a dynamic workflow system designed for high-throughput applications.
    Concurrency Computat.: Pract. Exper., 27: 5037–5059. (2015)
    ```
    """

setup(
    name='rocketsled',
    version=str(version),
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
    test_suite='rocketsled',
    tests_require='tests',
    packages=find_packages(),
    # package_data={'rocketsled': ['defaults.yaml']},
    install_requires=requirements,
    # data_files=['LICENSE', 'README.md', 'VERSION'],
    include_package_data=True
)
