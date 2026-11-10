#!/usr/bin/env python

# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="mpixel",
    version="0.1.0",
    author="MPIXEL Authors",
    author_email="ilker.kesen@hu-berlin.de",
    url="https://github.com/ilkerkesen/mpixel",
    description="Research code for MPIXEL",
    license="Apache",
    packages=find_packages(include=["mpixel", "mpixel.*"]),
    zip_safe=True,
)
