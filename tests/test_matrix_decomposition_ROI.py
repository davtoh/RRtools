import numpy as np
import cv2
from RRtoolbox.lib.arrayops import brightness,normalize
from RRtoolbox.lib.plotter import fastplt,plotim,plt
from tesisfunctions import graphHistogram,thresh_biggestCnt

shape = (300,300)
im = cv2.imread("im1_1.jpg")
if shape is not None:
    im = cv2.resize(im,shape)

P = brightness(im)

U, s, V = np.linalg.svd(P, full_matrices=False)
s_1 = s.copy()
s_1[1:] = 0
gray1 = np.abs(np.dot(U, np.dot(np.diag(s_1), V))).astype(np.uint8)
th, ROI = cv2.threshold(gray1,0,1,cv2.THRESH_OTSU)
#plotim("thresh ROI",ROI).show()
plt.figure("histograms")
plt.subplot(121)
plt.title("original hist")
graphHistogram(P,show=False)

x0,y0,w,h = cv2.boundingRect(thresh_biggestCnt(ROI))
x,y = x0+w,y0+h
P_ROI = P[x0:x,y0:y]

#plotim("P ROI",P_ROI).show()
plt.subplot(122)
plt.title("ROI hist")
graphHistogram(P_ROI)
#plt.show()

s_2 = s.copy()
s_2[:1] = 0
gray2 = np.abs(np.dot(U, np.dot(np.diag(s_2), V))).astype(np.uint8)
#plotim("adjusted",gray2).show()


s_3 = s.copy()
s_3[50:] = 0
gray3 = np.abs(np.dot(U, np.dot(np.diag(s_3), V))).astype(np.uint8)

plotim("difference",gray3).show()


