import cv2
import numpy as np
import pylab as plt
from RRtoolbox.lib.arrayops import points2mask
from Equations.Eq_HomogeniousTransform import HZrotate, applyTransformations,getApplyCenter
from RRtoolbox.lib.arrayops.basic import transformPoints
from RRtoolbox.lib.plotter import Plotim, plotPointsContour, fastplt
from RRtoolbox.lib.image import getcoors, drawcoorperspective,quadrants, Imcoors

random = np.random.random

h,w = 10,10
pts = np.array([[0, 0], [w, 0], [w, h], [0, h]],np.float32) # get list of image corners
#pts = np.array([[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]])
#pts = random_points([(-100, 100), (-100, 100)])
# perspective:  top_left, top_right, bottom_left, bottom_right
# corners and areas: top_left, top_right, bottom_right, bottom_left

#test_random([polyArea0,polyArea1,polyArea2], axes_range = ((0, 300),),points=[pts])

H = np.array([[random(),random(),random()], [random(),random(),random()], [random(),random(),random()]]) # impredictable transformation
transformations = [HZrotate(3.14*random())]
sM_rel_ = applyTransformations(transformations,False,getApplyCenter(w,h)) # apply relative transformations # symbolic transformation matrix
H = np.array(sM_rel_)[0:3,0:3].astype(np.float)#*H # array transformation matrix
#H = np.array([[1,1,1], [random(),1,1], [random(),random(),1]])
H = None

if H is not None:
    projections = transformPoints(pts, H) # get perspective of corners with transformation matrix
else:
    pts1 = getcoors(np.ones((h,w)),"get pixel coordinates", updatefunc=drawcoorperspective)
    pts2 = pts[[0,1,3,2]] # np.float32([[0,0],[w,0],[0,h],[w,h]]) # top_left,top_right,bottom_left,bottom_right
    if pts1:
        pts1 = np.float32(pts1)
    else:
        pts1 = pts2
    H = cv2.getPerspectiveTransform(pts1,pts2)
    projections = pts1[[0,1,3,2]]


def drawContours(pts, cor = (0,0,255), array = None):
    if array is None:
        pts = pts-pts.min(0) # shift
        xmax,ymax = pts.max(0)
        array = np.zeros((ymax,xmax,3),np.uint8)
    cv2.drawContours(array,[pts],0,cor,2)
    return array

rect = cv2.minAreaRect(projections)
box = cv2.cv.BoxPoints(rect)
#box = np.int0(box)

plotPointsContour(projections, deg=True)
plt.hold(True)
plotPointsContour(box, lcor="b", deg=True)
plt.show()
#fastplt(points2mask(box))