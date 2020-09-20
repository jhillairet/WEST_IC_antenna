# -*- coding: utf-8 -*-
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# retrieve version
with open('west-ic-antenna/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break
        
# This call to setup() does all the work
setup(
    name="west-ic-antenna",
    version=VERSION,
    description="WEST ICRF Antenna RF Model",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jhillairet/WEST_IC_antenna",
    author="Julien Hillairet",
    author_email="julien.hillairet@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
	install_requires = [
		'numpy',
		'scipy',
        'scikit-rf',
        'matplotlib',
		],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)

