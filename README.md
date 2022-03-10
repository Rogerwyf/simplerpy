# simplerpy
[![Build Status](https://app.travis-ci.com/Rogerwyf/simplerpy.svg?branch=main)](https://app.travis-ci.com/Rogerwyf/simplerpy)
[![Coverage Status](https://coveralls.io/repos/github/Rogerwyf/simplerpy/badge.svg?branch=main)](https://coveralls.io/github/Rogerwyf/simplerpy?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
rpy2 is a well-known bridging package for Python programmers to use R in Python. This package, *simplerpy*, will focus on looking and building upon its interface to make it simplier and easier to use, focusing on functions related to regression and hypothesis testing.

## What's inside simplerpy?
The objects/functions in simplerpy are listed below:
* `LM` - a linear model class corresponding to `lm()` in R
* `tTest` - a T-test class corresonding to `t.test()` in R
* `aov` - an Analysis of Variance Model class corresponding to `aov()` in R

## Project Structure
```
simplerpy
    |   LICENSE
    |   README.md
    |   requirements.txt
    |   setup.py
    |   
    +---simplerpy
    |   |   aov.py
    |   |   linear_model.py
    |   |   t_test.py
    |   |   __init__.py
    |           
    +---docs
    |       design_specifications.md
    |       functional_specifications.md
    |       simplerpy presentation.pptx
    |       
    +---examples
    |       demo_aov.py
    |       demo_linear_model.py
    |       demo_t_test.py
    |       
    |           
    \---test
            test_aov.py
            test_linear_model.py
            test_t_test.py
```
## Code Style
The source codes for this package under directory `simplerpy` have passed Flake8 stylecheck.

## Installation
Clone the repo and create a virtual environment in the root of the repo:
```
python -m venv venv
source venv/bin/activate
```
If you're using Anaconda/Miniconda, create and activate a new Conda environment with Python 3.7:
```
conda create --name simplerpy python=3.7
conda activate simplerpy
```
Install the all the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```
Generate the installation package using the following command with `setuptools`:
```
python setup.py sdist bdist_wheel
```
which will generate `simplerpy-1.0.0.tar.gz` in the `dist` directory. Now the package can be
installed using:
```
cd dist
pip install simplerpy-1.0.0.tar.gz
```

Installation of R is also required for running `rpy2`, and the latest version can be found [here](https://cran.r-project.org/).
## Usage
For details of usage on each object in this package, refer to their corresponding demo python script under `example` dirctory.
For example, see `demo_lienar_model.py` for the usage of `LM` class.



