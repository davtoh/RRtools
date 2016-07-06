__author__ = 'Davtoh'

import numpy as np

from deprecated.common import Timer


def test(x,max=255,min=0):
    #http://www.itk.org/Doxygen/html/classitk_1_1SigmoidImageFilter.html
    if min==0:
        return max*x
    else:
        return (max-min)*x+min

x = np.ones((5000,5000),dtype=np.float)

with Timer("here"):
    for i in xrange(100):
        test(x,max=60000,min=100)