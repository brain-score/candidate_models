#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "brain-score",  # (see dependency_links for error handling)
    "keras",
    "tensorflow>=1.4",
    "pytorch",
    "torchvision",
    "h5py",
    "scikit-learn",
    "scikit-image",
    "caching",
    "keras-squeezenet",
    "pillow",
    "llist",
    "networkx",
    "boto3",
    "matplotlib",
    "seaborn",
    "jupyter",
]

test_requirements = [
    "pytest",
    "pytest-mock",
]

dependency_links = [
    # if you get an error here (due to the repository being private,
    # install it by hand: pip install https://github.com/dicarlolab/brain-score.git
    "https://github.com/dicarlolab/brain-score/master/tarball",
    "https://github.com/mschrimpf/caching/master/tarball",
    "https://github.com/rcmalli/keras-squeezenet/master/tarball",
]

setup(
    name='candidate-models',
    version='0.1.0',
    description="A framework of candidate models tested on brain data",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='msch@mit.edu',
    url='https://github.com/mschrimpf/candidate-models',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    dependency_links=dependency_links,
    license="MIT license",
    zip_safe=False,
    keywords='candidate-models brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
