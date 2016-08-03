__author__ = 'Davtoh'
from tesisfunctions import Plotim,overlay
import cv2
import numpy as np
import tesisfunctions as tf


#from invariantMoments import centroid,invmoments,normalizedinvariantmoment,bwmoment
from tesisfunctions import sigmoid,histogram,brightness,getthresh,threshold,pad,circularKernel

#http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
fn1 = r'im1_2.jpg'
#fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
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
P = brightness(fore)
thresh = getthresh(cv2.resize(P,(300,300)))
print thresh
lastthresh=threshold(P,thresh,1,0)
thresh,lastthresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#lastthresh = pad(lastthresh,1)
plotc = Plotim(name + " overlayed lastthresh", overlay(fore.copy(), lastthresh * 255, alpha=lastthresh)).show()

# find biggest area
contours,hierarchy = cv2.findContours(lastthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print "objects: ",len(contours)
index,maxarea = tf.biggestCntData(contours)
print "area contour:",maxarea,"index: ",index
cnt = contours[index]
print "optaining polygon test..."

#POLYGON TEST
polygontest = tf.polygontest(P.copy().astype(np.int32),cnt)
plotpt = tf.graphpolygontest(polygontest,name+" polygon test")
plotpt.show()
test = np.zeros_like(lastthresh,np.uint8)
level = 75
test[polygontest>level] = 1
plotc = Plotim(name + " test", test)
plotc.show()
# TODO this can be collected in a function and used with the algorithm defect_lines
lastthresh = test
# find biggest area
contours,hierarchy = cv2.findContours(lastthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print "objects: ",len(contours)
index,maxarea = tf.biggestCntData(contours)
print "area contour:",maxarea,"index: ",index
cnt = contours[index]

#DEFECTS
pallet = [[0,0,0],[255,255,255]]
pallet = np.array(pallet,np.uint8)
imdefects = pallet[lastthresh]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
tf.graphDeffects(imdefects,cnt,defects)

#SEPARATING LINE
start,end = tf.extendedSeparatingLine(imdefects.shape, cnt, defects)
cv2.line(imdefects,start,end,[0,0,100],2)
Plotim(name + " defects", imdefects).show()

cv2.line(lastthresh,start,end,0,2)

# find biggest cnt
cnt = tf.thresh_biggestCnt(lastthresh)

#dilating ROI
ROI = np.zeros(P.shape,dtype=np.uint8)
cv2.drawContours(ROI,[cnt],0,1,-1)
#kernel = np.ones((level,level),np.uint8)
kernel = circularKernel((level,level),np.uint8)
lastthresh = cv2.dilate(ROI,kernel,iterations = 2)
plotc = Plotim(name + " lastthresh", lastthresh)
plotc.show()

# find biggest cnt
cnt = tf.thresh_biggestCnt(lastthresh)

#ROI
ROI = np.zeros(P.shape,dtype=np.uint8)
cv2.drawContours(ROI,[cnt],0,1,-1)
plotc = Plotim(name + " ROI", ROI)
plotc.show()


M = cv2.moments(cnt) # find moments
#M2 = invmoments(ROI,Area=None,center=None)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#x,y = centroid(ROI,maxarea)
#normalizedinvariantmoment(ROI,maxarea,0,0,x,y)
#n00 = bwmoment(ROI,0,0,cx,cy)
print "(cx,cy)",(cx,cy)
#print "x,y",x,y
cv2.circle(fore, (cx,cy), 10, (0, 0, 255), -1, 8)
#cv2.circle(fore, (int(x),int(y)), 10, (0, 255, 255), -1, 8)
cv2.drawContours(fore,[cnt],0,(0, 0, 255),2)
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(fore,ellipse,(0,255,0),2)

plotc = Plotim(name + " description", fore)
plotc.show()

mask = np.ones(P.shape,dtype=np.uint8)
cv2.ellipse(mask,ellipse,0,-1)
fore2[mask>0]=0
plotc = Plotim(name + " result", fore2)
plotc.show()
cv2.imwrite("mask_"+name+".png",fore2)
"""
# Saving the objects:
import pickle

data = {"thresh":thresh,"lastthresh":lastthresh,"cnt":cnt,"ellipse":ellipse,"polygontest":polygontest}
with open("masks_"+name+'.pickle', 'w') as f:
    pickle.dump(data, f)"""