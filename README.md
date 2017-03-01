# L3SVMs
## Landmarks-based Linear Local Support Vectors Machines
----

L3SVMs is a new **local SVM method** which clusters the input space, carries out dimensionality reduction by projecting the data on *landmarks*, and jointly learns a linear combination of local models.

Main Features:

* it captures non-linearities while scaling to large datasets
* it's customizable: projection function, landmark selection procedure, linear or kernelized

## Installation

1. Install [liblinear](https://github.com/cjlin1/liblinear) and [libsvm](https://github.com/arnaudsj/libsvm)

2. Add to your PYTHONPATH the paths to *liblinear/python/* and to *libsvm/python/*

3. Install required python modules:

 > pip install -r requirements.txt

## Usage


