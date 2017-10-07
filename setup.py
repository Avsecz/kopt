#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

# TODO - update

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "future",
    "pyyaml",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn>=0.18",
    "matplotlib",
    # "tensorflow", # - not per-se required
    # "glmnet",
    "keras>=2.0.4",
    'hyperopt',
]

test_requirements = [
    "pytest",
]

setup(
    name='kopt',
    version='0.1.0',
    description="Keras-hyperopt (kopt); Hyper-parameter tuning for Keras using hyperopt.",
    long_description=readme,
    author="Žiga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/avsecz/keras-hyperopt',
    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=requirements,
    # dependency_links=dependency_links,
    license="MIT license",
    zip_safe=False,
    keywords=["hyper-parameter tuning", "keras",
              "deep learning", "tensorflow", ],
    extras_require={
        'tensorflow': ['tensorflow>=1.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.0'],
        "develop": ["bumpversion",
                    "wheel",
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    test_suite='tests',
    tests_require=test_requirements
)
