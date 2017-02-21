from __future__ import absolute_import


import cv2
import numpy as np
import glob
from .tesisfunctions import brightness,sigmoid,Plotim,overlay,IMAGEPATH, circularKernel
from .recommended import getKernel

rootpath = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/"
imlist= glob.glob(rootpath+"IMG*.jpg")
fn1 = imlist[25]
#fn1 = r"C:\Users\Davtoh\Documents\2015_01\Tesis tests\retinal photos\ALCATEL ONE TOUCH IDOL X\left_DAVID\IMG_20150730_115534_1.jpg"
#fn1 = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\tesis\im4_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(400,400))

P = brightness(fore) # get gray image
#thresh = getthresh(cv2.resize(P,shape)) # obtain threshold value
#bImage=threshold(P, thresh, 1, 0) #binary image
thresh, ROI = cv2.threshold(P, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # threshold value, and binary image
level = int(np.clip(ROI.size*0.000009,2,10))
# use Morphological Gradient from http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
ROI = cv2.morphologyEx(ROI,4,circularKernel((level,level),np.uint8),iterations = 2)
plot = Plotim("borders", ROI).show()