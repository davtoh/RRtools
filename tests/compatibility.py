"""
It is a know issue that matplotlib and cv2.imshow (pyQT backend in general it seems) have serious
compatibilities when used both combined.
"""

import pylab as plt
import cv2

#cv2.namedWindow("toro") ## THIS WORKS
f = plt.figure()
ax = plt.subplot()
ax.plot([1,2,3],[1,2,3])
ax.hold(True)

cv2.namedWindow("toro") ## THIS DOES NOT WORKS

plt.show()