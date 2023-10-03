#! /usr/bin/env python

# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

import codecs
import os
import re
from distutils.core import Extension, setup

# Package meta-data.
NAME = "pysurvival"
DESCRIPTION = "Open source package for Survival Analysis modeling"
URL = "https://www.pysurvival.io"
EMAIL = "stephane@squareup.com"
AUTHOR = "steph-likes-git"
LICENSE = "Apache Software License (Apache 2.0)"

# Current Directory
try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath("__file__")) + "/"


# Utility functions
def read_long_description():
    with open(CURRENT_DIR + "LONG_DESCRIPTION.txt", "r") as f:
        return f.read()


def install_requires():
    with open(CURRENT_DIR + "requirements.txt", "r") as requirements_file:
        requirements = requirements_file.readlines()
    return requirements


def read_version(*file_paths):
    with codecs.open(os.path.join(CURRENT_DIR, *file_paths), "r") as fp:
        version_file = fp.read()

    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Extensions Compilation arguments #
extra_compile_args = ["-std=c++11", "-O3"]

# Extensions info #
ext_modules = [
    Extension(
        name="pysurvival.utils._functions",
        sources=[
            "pysurvival/cpp_extensions/_functions.cpp",
            "pysurvival/cpp_extensions/functions.cpp",
        ],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.utils._metrics",
        sources=[
            "pysurvival/cpp_extensions/_metrics.cpp",
            "pysurvival/cpp_extensions/non_parametric.cpp",
            "pysurvival/cpp_extensions/metrics.cpp",
            "pysurvival/cpp_extensions/functions.cpp",
        ],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.models._survival_forest",
        sources=[
            "pysurvival/cpp_extensions/_survival_forest.cpp",
            "pysurvival/cpp_extensions/survival_forest_data.cpp",
            "pysurvival/cpp_extensions/survival_forest_utility.cpp",
            "pysurvival/cpp_extensions/survival_forest_tree.cpp",
            "pysurvival/cpp_extensions/survival_forest.cpp",
        ],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

# Setup
setup(
    name=NAME,
    version=read_version("pysurvival", "__init__.py"),
    description=DESCRIPTION,
    long_description=read_long_description(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    install_requires=install_requires(),
    include_package_data=True,
    package_data={
        "": ["*.csv"],
    },
    extras_require={"dev": ["pytest", "black", "isort", "pylint"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    ext_modules=ext_modules,
)
