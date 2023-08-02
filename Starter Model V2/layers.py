# Calculator for image size after conv layers
import math

# imagesize assumes square image
# sum = ((imagesize - kernelsize + 2 * padding)) + 1

# The second sum function is used for max pooling using sum
# sum2 = ((sum - kernelsize + 2 * 0)) + 1

# Each sum and sum2 function set is for one conv layer
# add more sets for more layers
# Initial conv layer

size = 360
layers = 3
padding = 1
poolPadding = 0
kernelsize = 3
poolKernelSize = 2
stride = 1
poolStride = 2
# Calculating image dimensions after the first conv layer
# sum = ((size - 3 + 2 * 1 ) / 1)+1
# sum2 = ((sum - 2 + 2 * 0 ) / 2)+1

# Calculating image dimensions after the second conv layer
# sum = ((sum2 - 3 + 2 * 1 ) / 1)+1
# sum2 = ((sum - 2 + 2 * 0 ) / 2)+1

# Calculating image dimensions after the third conv layer
# sum = ((sum2 - 3 + 2 * 1 ) / 1)+1
# sum2 = ((sum - 2 + 2 * 0 ) / 2)+1

# Calculating image dimensions after the fourth conv layer
# sum = ((sum2 - 3 + 2 * 1 ) / 1)+1
# sum2 = ((sum - 2 + 2 * 0 ) / 2)+1

def layerCalc(size, layers, padding, poolPadding, kernelsize, poolKernelSize, stride, poolStride):
    sum = ((size - kernelsize + 2 * padding ) / stride) + 1
    sum2 = ((sum - poolKernelSize + 2 * poolPadding) / poolStride) + 1
    size = sum2
    layers = layers - 1
    if layers == 0:
        print(math.floor(size))
    else:
        layerCalc(size, layers, padding, poolPadding, kernelsize, poolKernelSize, stride, poolStride)

layerCalc(size, layers, padding, poolPadding, kernelsize, poolKernelSize, stride, poolStride)