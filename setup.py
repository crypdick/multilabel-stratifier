# -*- coding: utf-8 -*-
# import sphinx_pypi_upload
import io

from setuptools import find_packages, setup

with io.open("README.md", "r") as f:
    readme = f.read()

setup(
    name="multilabel-stratifier",
    version="0.3.0",
    packages=find_packages(exclude=["docs", "tests"]),
    author="Richard Decal",
    author_email="public@richarddecal.com",
    license="BSD",
    long_description=readme,
    url="https://github.com/crypdick/multilabel-stratifier",
    description="Maintained iterative stratification algorithm from the abandoned scikit-multilearn library.",
    classifiers=[
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
