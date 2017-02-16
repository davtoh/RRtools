from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
__author__ = 'Davtoh'
import cv2
import numpy as np
from .tesisfunctions import padVH,Plotim,graphpolygontest,polygontest
import time
import itertools as it
from multiprocessing.pool import ThreadPool
pool=ThreadPool(processes = cv2.getNumberOfCPUs())


def polygontestPool(src,cnt,pool=None):
    def f(data):
        i,j = data
        return i,j,cv2.pointPolygonTest(cnt,(j,i),True)
    #src = np.zeros(src.shape,np.float32)
    params = list(zip(*np.where(src==src)))
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)
    for i,j,n in ires:
        src.itemset((i,j),n)
    return src

r = 100
src = np.zeros((4*r,4*r),np.uint8)
# draw an polygon on image src
points = [ [1.5*r,1.34*r], [r,2*r], [1.5*r,2.866*r], [2.5*r,2.866*r],[3*r,2*r],[2.5*r,1.34*r] ]
points = np.array(points,np.int0)
#cv2.polylines(src,[points],True,255,3)
cv2.fillPoly(src,[points],255)
#Plotim.Plotim("object",src).show()
contours,hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0] # We take only one contour for testing
time1 = time.time()
test1 = polygontest(src.copy().astype(np.int32),cnt)
time2 = time.time()
test2 = cv2.distanceTransform(src,cv2.DIST_LABEL_PIXEL,5).astype(np.float)
test2[test2==0] = -1
#test2 = polygontestPool(src.copy().astype(np.int32),cnt,pool)
time3 = time.time()
print("Time normal: ",time2-time1,"Time pool: ",time3-time2)
test1 = graphpolygontest(test1,"poligon Test").data
test2 = graphpolygontest(test2,"distance Transform").data
grapth = padVH([[test1,test2]])[0]
plot = Plotim("poligon Test / distance Transform", grapth)
plot.show()