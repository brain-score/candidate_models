#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "brain-score @ git+https://github.com/brain-score/brain-score",
    "model-tools @ git+https://github.com/brain-score/model-tools",
    "numpy",
    "result_caching @ git+https://github.com/mschrimpf/result_caching",
    "networkx",
    "tqdm",

    "cornet @ git+https://github.com/dicarlolab/CORnet",
    "bagnet @ git+https://github.com/mschrimpf/bag-of-local-features-models.git",
]

setup(
    name='candidate-models',
    version='0.1.0',
    description="A framework of candidate models to test on brain data",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='mschrimpf@mit.edu',
    url='https://github.com/brain-score/candidate-models',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='candidate-models brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
