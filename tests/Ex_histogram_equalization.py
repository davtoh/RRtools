from __future__ import absolute_import

# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
# http://opencvpython.blogspot.com.co/2013/03/histograms-2-histogram-equalization.html
# http://stackoverflow.com/a/31493356/5288758
import cv2
import numpy as np
from .tesisfunctions import graphHistogram, Plotim, equalization
from matplotlib import pyplot as plt

img = cv2.imread(r'im1_1.jpg',0)
hist_img = graphHistogram(img, False)
# normal equalization
equ = cv2.equalizeHist(img)
hist_equ = graphHistogram(equ, False)
# create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
hist_cl1 = graphHistogram(cl1, False)

res = cv2.resize(np.hstack((img,equ,cl1)),(1000,1000)) #stacking images side-by-side
hists = np.hstack((hist_img,hist_equ,hist_cl1)) #stacking images side-by-side
plt.show()
Plotim("equalizaiton", res).show()