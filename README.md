# L3SVMs
## Landmarks-based Linear Local Support Vectors Machines

**python3 project**

L3SVMs is a new **local SVM method** which clusters the input space, carries out dimensionality reduction by projecting the data on *landmarks*, and jointly learns a linear combination of local models.

Main Features:

* it captures non-linearities while scaling to large datasets
* it's customizable: projection function, landmark selection procedure, linear or kernelized

## Installation

1. Install [liblinear](https://github.com/cjlin1/liblinear)

2. Add to your PYTHONPATH the paths to *liblinear/python/*

3. Install required python modules:

 > pip install -r requirements.txt

## Usage

### Example Scripts

1. *validation.py* trains a L3SVM on a training set and tests it on a testing set. For help, run

 > python validation.py -h

2. *cross_validation.py* performs a cross-validation on a dataset. For help, run

 > python cross_validation.py -h