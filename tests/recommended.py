"""
A compendium of useful get-parameters-functions base on observations and investigation not intended
for the main distribution.

These functions are found here because they cannot be included in the main distribution of
the module but are valuable to for investigations and development of new algorithms
"""
from __future__ import absolute_import
from .tesisfunctions import circularKernel
import numpy as np

def getKernel(imsize):
    """
    From size of image get recommended kernel

    :param imsize: image.size from numpy arrays
    :return: numpy circular kernel
    """
    """
    some considerations:
      - it seems that if the object is near the borders, then a big ksize could cause it to
      expand to the borders in the case of closing for examples. One solution could be to get the
      object contours and then perform a calculation of the nearest points to the border, i think
      that kernel size should not have more than the smallest distance to a border. the iterations
      also cause a behaviour similar to ksize, if iterations is big, object is extended to the borders.

      - kernel sizes bigger than 79 will act more quickly but will cause bulky or pixel-like objects

    Therefore, i should be considered to accept the image as input, and not only return the kernel
    but iterations for the morphological operation.
    """
    ksize = int(np.clip(imsize * 0.00009, 3, 50)) # kernel size recommended from 3 to 80
    return circularKernel((ksize,ksize),np.uint8)