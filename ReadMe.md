Supplementary material for article "A.N. Chernodub, D.V. Nowicki, Norm-preserving Orthogonal Permutation Linear Unit Activation Functions", 2016.

There are two options to add the OPLU activation functions to your Caffe project:

1) genoplu.py - this script generates OPLU functions as a .prototxt file. It contains a collection of built-in Caffe layers such as split, eltwise, conv, concat. This solution could be time-demanding for blobs with many channels or fully-connected layers.

2) oplu.cpp, oplu.cu and oplu.hpp are source files to be included to Caffe's sources. Then you have to rebuild Caffe. Please, note that the current version works for fully connected layers only (will be fixed soon)!