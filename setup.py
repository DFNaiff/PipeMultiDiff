# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
# import os
# import sys
# import zipfile
# import urllib
packages = ['tmcdiff'] + \
           ['tmcdiff.' + subpack for subpack in find_packages('tmcdiff')]

setup(
    name="tmcdiff",
    version="0.0.0.1",
    description="Comprehensive model for scaling of inorganic solids in pipe flows.",
    packages=packages,
    author="Danilo Naif",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    url="First principles model of multi-ion scaling in pipe flows.",
    python_requires=">=3.6",
    install_requires=['numpy', 'scipy', 'matplotlib', 'torch']
)
