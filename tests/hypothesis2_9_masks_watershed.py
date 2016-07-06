__author__ = 'Davtoh'

from tesisfunctions import plotim,overlay
import cv2
import numpy as np
import tesisfunctions as tf
from tesisfunctions import sigmoid,histogram,brightness,getthresh,threshold,pad,filterFactory,graphpolygontest

#http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

pallet = [[0,0,0],[255,255,255],[0,0,255],[255,0,255]]
pallet = np.array(pallet,np.uint8)

#fn1 = r'im1_2.jpg'
fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300))

fore2 = fore.copy()


#paremters = 20,20,250
paremters = 10,30,None
myfilter = filterFactory(*paremters) # alfa,beta1,beta2
fore2=myfilter(fore2.astype("float"))*255#*fore.astype("float")
fore2 = fore2.astype("uint8")

P = brightness(fore2)
#P = sigmoid(P,20,20).astype("uint8")
#P = cv2.equalizeHist(P)
#P = imEqualization(P)[0]
plotc = plotim(name+" brightness",P)
plotc.show()

thresh,sure_bg = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # obtain over threshold
print thresh
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_LABEL_PIXEL,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),1,0)  # obtain under threshold

markers = np.ones_like(sure_fg).astype("int32")
markers[sure_bg==1]=0
markers[sure_fg==1]=2

plotc = plotim(name+" markers",pallet[markers])
plotc.show()

cv2.watershed(fore,markers)

#markers = pad(markers,1,2)
water = pallet[markers]
plotc = plotim(name+" watershed",water)
plotc.show()
markers[markers==-1]=1

kernel = np.ones((100,100),np.uint8)
thresh,lastthresh = cv2.threshold(markers.astype("uint8"),1,1,cv2.THRESH_BINARY)
#dilation = cv2.dilate(thresh,kernel,iterations = 1)
#erosion = cv2.erode(dilation,kernel,iterations = 1)

plotc = plotim(name +" overlayed lastthresh", overlay(fore.copy(), lastthresh * 255, alpha=lastthresh))
plotc.show()

# find biggest area
contours,hierarchy = cv2.findContours(lastthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print "objects: ",len(contours)
index = 0
maxarea = 0
#objectarea = np.sum(lastthresh)
for i in xrange(len(contours)):
    area = cv2.contourArea(contours[i])
    if area>maxarea:
        index = i
        maxarea = area

print "area contour:",maxarea,"index: ",index
cnt = contours[index]

"""
# FIND ROI
ROI = np.zeros(P.shape,dtype=np.uint8)
cv2.drawContours(ROI,[cnt],0,1,-1)
plotc = plotim(name+" ROI",ROI)
plotc.show()"""

ellipse = cv2.fitEllipse(cnt)
mask = np.ones(P.shape,dtype=np.uint8)
cv2.ellipse(mask,ellipse,0,-1)
fore[mask>0]=0
plotc = plotim(name+" result",fore)
plotc.show()
