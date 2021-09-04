# -*- coding: utf-8 -*-
# import sphinx_pypi_upload
import io

from setuptools import find_packages, setup

with io.open("README.md", "r") as f:
    readme = f.read()

setup(
    name="multilabel-stratifier",
    version="0.2.0",
    packages=find_packages(exclude=["docs", "tests"]),
    author=u"Piotr Szymański",
    author_email=u"niedakh@gmail.com",
    license=u"BSD",
    long_description=readme,
    url=u"http://scikit.ml/",
    description=u"Scikit-multilearn is a BSD-licensed library for multi-label classification that is built on top of the well-known scikit-learn ecosystem.",
    classifiers=[
        u"Development Status :: 5 - Production/Stable",
        u"Environment :: Console",
        u"Environment :: Web Environment",
        u"Intended Audience :: Developers",
        u"Intended Audience :: Education",
        u"Intended Audience :: Science/Research",
        u"License :: OSI Approved :: BSD License",
        u"Operating System :: MacOS :: MacOS X",
        u"Operating System :: Microsoft :: Windows",
        u"Operating System :: POSIX",
        u"Programming Language :: Python",
        u"Topic :: Scientific/Engineering",
        u"Topic :: Scientific/Engineering :: Information Analysis",
        u"Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
