from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Davtoh'
from .tesisfunctions import Plotim,overlay,padVH
import cv2
import numpy as np
#from invariantMoments import centroid,invmoments,normalizedinvariantmoment,bwmoment
from .tesisfunctions import sigmoid,histogram,brightness,getthresh,threshold,pad,graphpolygontest, polygontest

#http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

fn1 = r'im1_2.jpg'
#fn1 = r"asift2Result_with_alfa1.png"
#fn1 = r"im_completer_Result2.png"
fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300))
name = fn1.split('\\')[-1].split(".")[0]
fore2 = fore.copy()
"""
fore = fore.astype("float")
fb = fore[:,:,0]
fg = fore[:,:,1]
fr = fore[:,:,2]

# threshold retinal area
alfa = -1
beta = 50 # if alfa >0 :if beta = 50 with noise, if beta = 200 without noise
th = 1
kernel = np.ones((100,100),np.uint8)
enhanced = sigmoid(fr,alfa,beta)
thresh = cv2.threshold(enhanced.astype("uint8"),th,1,cv2.THRESH_BINARY_INV)[1]
#dilation = cv2.dilate(thresh,kernel,iterations = 1)
#erosion = cv2.erode(dilation,kernel,iterations = 1)
#closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#dilation = cv2.dilate(opening,kernel,iterations = 1)
lastthresh = opening
"""
from .recommended import getKernel
P = brightness(fore)
shape = P.shape
kernel = getKernel(shape[0]*shape[1])

thresh = getthresh(cv2.resize(P,(300,300)))
print(thresh)
### METHOD 1
#lastthresh=threshold(P,thresh,1,0)

### METHOD 2
#thresh,lastthresh = cv2.threshold(P,thresh,1,cv2.THRESH_BINARY)
#lastthresh = cv2.morphologyEx(lastthresh, cv2.MORPH_CLOSE, kernel,iterations=3)

### METHOD 3
thresh,lastthresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#lastthresh = cv2.erode(lastthresh,kernel,iterations = 3)
#lastthresh = pad(lastthresh,1)
plotc = Plotim(name + " overlayed lastthresh", overlay(fore.copy(), lastthresh * 255, alpha=lastthresh))
plotc.show()
