from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from past.utils import old_div

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
P = brightness(fore)
thresh = getthresh(cv2.resize(P,(300,300)))
print(thresh)
lastthresh=threshold(P,thresh,1,0)
thresh,lastthresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#lastthresh = pad(lastthresh,1)
plotc = Plotim(name + " overlayed lastthresh", overlay(fore.copy(), lastthresh * 255, alpha=lastthresh))
plotc.show()

# find biggest area
contours,hierarchy = cv2.findContours(lastthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("objects: ",len(contours))
index = 0
maxarea = 0
#objectarea = np.sum(lastthresh)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area>maxarea:
        index = i
        maxarea = area

print("area contour:",maxarea,"index: ",index)
cnt = contours[index]
print("optaining polygon test...")
polygontest = graphpolygontest((P,cnt)).show()

#DEFECTS
pallet = [[0,0,0],[255,255,255]]
pallet = np.array(pallet,np.uint8)
imdefects = pallet[lastthresh]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
distances = defects[:,0,3]
two_max = np.argpartition(distances, -2)[-2:] # get indeces of two maximum values

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(imdefects,start,end,[0,255,0],2)
    cv2.circle(imdefects,far,5,[0,0,255],-1)

#SEPARATING LINE
points = defects[:,0,2]
x1,y1 = tuple(cnt[points[two_max[0]]][0])
x2,y2 = tuple(cnt[points[two_max[1]]][0])
m = old_div((y2-y1),float(x2-x1))
b = int(y1-x1*m)
# find interception with xf and yf axis
if b>imdefects.shape[0]: # if start outside yf
    start = int(old_div((imdefects.shape[0]-b),m)),imdefects.shape[0] # (yf-b)/m, yf
else: # if start inside yf
    start = 0,b # 0,y
y = int(m*imdefects.shape[1]+b) # m*xf+b
if y<0: # if end outside yf
    end = int(old_div(-b,m)),0# x,0
else: # if end inside yf
    end = imdefects.shape[1],y # xf, y

cv2.line(imdefects,start,end,[0,0,100],2)

plotc = Plotim(name + " defects", imdefects)
plotc.show()

#ROI
ROI = np.zeros(P.shape,dtype=np.uint8)
cv2.drawContours(ROI,[cnt],0,1,-1)
plotc = Plotim(name + " ROI", ROI)
plotc.show()
M = cv2.moments(cnt) # find moments
#M2 = invmoments(ROI,Area=None,center=None)
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#x,y = centroid(ROI,maxarea)
#normalizedinvariantmoment(ROI,maxarea,0,0,x,y)
#n00 = bwmoment(ROI,0,0,cx,cy)
#print "(cx,cy)",(cx,cy)
#print "x,y",x,y
#cv2.circle(fore, (cx,cy), 10, (0, 0, 255), -1, 8)
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