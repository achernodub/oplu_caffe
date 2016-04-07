# This script generates OPLU activation function for Caffe framework.
#
# Supplementary material for article "A.N. Chernodub, D.V. Nowicki,
# Norm-preserving Orthogonal Permutation Linear Unit Activation Functions", 2016.
#

import sys
import numpy as np

if (len(sys.argv) != 4):
    print "\nGenerates .prototxt file with OPLU layer."
    print "\nUsage: python genoplu.py oplu_layer_name bottom_layer_name bottom_layer_num_output"
    print "where: oplu_layer_name - original name of new OPLU layer;"
    print "       bottom_layer_name - name of previous layer;"
    print "       bottom_layer_num_output - number of previous layer\'s outputs.\n"
    print "Example: python genoplu.py OPLU1 conv1 50\n"
    sys.exit(1)

argument_list = str(sys.argv)

oplu_layer_name = sys.argv[1]
bottom_layer_name = sys.argv[2]
bottom_layer_num_output = int(sys.argv[3])

if (bottom_layer_num_output % 2 != 0):
    print "Error: number of previous layer's outputs must be even!"
    sys.exit(2)

indexes = range(0, bottom_layer_num_output)

indexes_A = []
indexes_B = []
for i in range(0, bottom_layer_num_output):
    if (i % 2 == 0):
        indexes_A.append(i)
    else:
        indexes_B.append(i)

#print indexes
#print indexes_A
#print indexes_B
#sys.exit(3)

f = open('out.prototxt', 'w')

f.write("# ----------- OPLU function ----------- \n\n")
f.write("# oplu_layer_name = \"%s\"\n" % (oplu_layer_name))
f.write("# bottom_layer_name = \"%s\"\n" % (bottom_layer_name))
f.write("# bottom_layer_num_output = %d\n\n" % (bottom_layer_num_output))

f.write("# 1) Slice blob to slices by channels\n")

f.write("layer {\n")
f.write("name: \"%s_slicer\"\n" % oplu_layer_name)
f.write("type: \"Slice\"\n")
f.write("bottom: \"%s\"\n" % bottom_layer_name)

for i in indexes:
    f.write("top: \"%s_sl_%d\"\n" % (oplu_layer_name, i))

f.write("slice_param {\n")
f.write("axis: 1\n")

for i in range(1, bottom_layer_num_output):
    f.write("  slice_point: %d\n" % i)

f.write("  }\n")
f.write("}\n")

f.write("# 2) Produce element-wise MAX\n");

for k in range(0, bottom_layer_num_output / 2):
    i = indexes_A[k]
    j = indexes_B[k]

    f.write("layer {\n")
    f.write("  name: \"%s_sl_%d_sl_%d_max\"\n" % (oplu_layer_name, i, j))
    f.write("  type: \"Eltwise\"\n")
    f.write("  bottom: \"%s_sl_%d\"\n" % (oplu_layer_name, i))
    f.write("  bottom: \"%s_sl_%d\"\n" % (oplu_layer_name, j))
    f.write("  top: \"%s_sl_%d_sl_%d_max\"\n" % (oplu_layer_name, i, j))
    f.write("  eltwise_param { operation: MAX }\n")
    f.write("}\n")

f.write("# MIN is not available in Eltwise; so we need to find MAX of negative values.\n")

f.write("# 3) Making negative values\n")

f.write("layer {\n")
f.write("  name: \"%s_make_negative\"\n" % (oplu_layer_name))
f.write("  type: \"Convolution\"\n")

for k in range(0, bottom_layer_num_output):
    f.write("  bottom: \"%s_sl_%d\"\n" % (oplu_layer_name, k))

for k in range(0, bottom_layer_num_output):
    f.write("  top: \"%s_sl_%d_neg\"\n" % (oplu_layer_name, k))

f.write("  param { lr_mult: 0 decay_mult: 0 }\n")
f.write("  param { lr_mult: 0 decay_mult: 0 }\n")
f.write("  convolution_param {\n")
f.write("    num_output: 1\n")
f.write("    kernel_size: 1\n")
f.write("    stride: 1\n")
f.write("    weight_filler {\n")
f.write("    type: \"constant\"\n")
f.write("    value: -1\n")
f.write("  }\n")
f.write("  bias_filler {\n")
f.write("    type: \"constant\"\n")
f.write("  }\n")
f.write(" }\n")
f.write("}\n")

f.write("# 4) Produce element-wise MAX of negative elements\n");

for k in range(0, bottom_layer_num_output / 2):
    i = indexes_A[k]
    j = indexes_B[k]

    f.write("layer {\n")
    f.write("  name: \"%s_sl_%d_sl_%d_max_neg\"\n" % (oplu_layer_name, i, j))
    f.write("  type: \"Eltwise\"\n")
    f.write("  bottom: \"%s_sl_%d_neg\"\n" % (oplu_layer_name, i))
    f.write("  bottom: \"%s_sl_%d_neg\"\n" % (oplu_layer_name, j))
    f.write("  top: \"%s_sl_%d_sl_%d_max_neg\"\n" % (oplu_layer_name, i, j))
    f.write("  eltwise_param { operation: MAX }\n")
    f.write("}\n")

f.write("# 5) Making back positive values.\n");

f.write("layer {\n")
f.write("  name: \"%s_make_positive_back\"\n" % (oplu_layer_name))
f.write("  type: \"Convolution\"\n")

for k in range(0, bottom_layer_num_output / 2):
    i = indexes_A[k]
    j = indexes_B[k]
    f.write("  bottom: \"%s_sl_%d_sl_%d_max_neg\"\n" % (oplu_layer_name, i, j))

for k in range(0, bottom_layer_num_output / 2):
    i = indexes_A[k]
    j = indexes_B[k]
    f.write("  top: \"%s_sl_%d_sl_%d_min\"\n" % (oplu_layer_name, i, j))

f.write("  param { lr_mult: 0 decay_mult: 0 }\n")
f.write("  param { lr_mult: 0 decay_mult: 0 }\n")
f.write("  convolution_param {\n")
f.write("    num_output: 1\n")
f.write("    kernel_size: 1\n")
f.write("    stride: 1\n")
f.write("    weight_filler {\n")
f.write("    type: \"constant\"\n")
f.write("    value: -1\n")
f.write("  }\n")
f.write("  bias_filler {\n")
f.write("    type: \"constant\"\n")
f.write("  }\n")
f.write(" }\n")
f.write("}\n")

f.write("# 6) Concatenate MAX-MIN slices\n");

f.write("layer {\n")
f.write("  name: \"%s\"\n" % (oplu_layer_name))
f.write("  type: \"Concat\"\n")

for k in range(0, bottom_layer_num_output / 2):
    i = indexes_A[k]
    j = indexes_B[k]
    f.write("  bottom: \"%s_sl_%d_sl_%d_max\"\n" % (oplu_layer_name, i, j))
    f.write("  bottom: \"%s_sl_%d_sl_%d_min\"\n" % (oplu_layer_name, i, j))

f.write("  top: \"%s\"\n" % oplu_layer_name)
f.write("  concat_param {\n")
f.write("    axis: 1\n")
f.write("  }\n")
f.write("}\n")

f.write("# ------------------------------------- \n\n")

f.close()

print "Please, check for file \"out.prototxt\"."
