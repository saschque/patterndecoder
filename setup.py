"""
Setup script for the patterndecoder package.

This script uses setuptools to define the package configuration and dependencies
for the patterndecoder package. It specifies the package name, version,
and required dependencies for installation.

The package includes various transformer-based models and utilities for
time series analysis and prediction. It relies on data science and machine 
learning libraries such as TensorFlow, NumPy, pandas, and scikit-learn.

To install the package, run:
pip install .

Dependencies:

- tensorflow: Deep learning framework
- numpy: Numerical computing library
- pandas: Data manipulation and analysis
- matplotlib: Data visualization
- prettytable: ASCII table creation
- yfinance: Yahoo Finance market data downloader
- scikit-learn: Machine learning library

For more information about the package and its usage, refer to the README
or documentation.
"""

from setuptools import setup, find_packages

with open('requirements.txt', encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="patterndecoder",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements
)
