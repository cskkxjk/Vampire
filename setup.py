import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='Vampire',
    version='0.0.1',
    author='Junkai Xu',
    author_email='xujunkai@zju.edu.cn',
    description='Code for Vampire',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=None,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    cmdclass={'build_ext': BuildExtension},
)
