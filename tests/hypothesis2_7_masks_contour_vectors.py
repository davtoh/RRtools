# -*- coding: utf-8 -*-
__author__ = 'Davtoh'
from tesisfunctions import plotim,overlay,polygontest, polycenter, biggestCntData, graphpolygontest
from RRtoolbox.lib.arrayops import relativeQuadrants, relativeVectors,angle, contour2points, anorm, anorm2
import cv2
import numpy as np
import pylab as plt

#from invariantMoments import centroid,invmoments,normalizedinvariantmoment,bwmoment
from tesisfunctions import sigmoid,histogram,brightness,getthresh,threshold,pad

def plotPointToPoint(pts1, pts2, ax= None, cor="k", annotate =u'{i}({x:1.1f}, {y:1.1f})'):
    """
    Plots points and joining lines in axes.

    :param pts1: from points. [(x0,y0)...(xN,yN)]
    :param pts2: to points. [(x0,y0)...(xN,yN)]
    :param ax: axes handle to draw points.
    :param cor: color of joining lines.
    :return: ax.
    """
    # http://stackoverflow.com/a/12267492/5288758
    ax = ax or plt.gca() # get axes, creates and show figure if interactive is ON, disable with plt.ioff()
    for i,((x,y),(u,v)) in enumerate(zip(pts1, pts2)): # annotate each point
        ax.quiver(x, y, u - x, v - y, angles='xy', scale_units='xy',scale=1, width=0.004, color = cor)
        ax.annotate(annotate.format(i=i,x=x,y=y), xy=(x,y), textcoords='data')
    return ax

def reducePoints(cnt):
    epsilon = 0.1*cv2.arcLength(cnt,True)
    return cv2.approxPolyDP(cnt,epsilon,True)

#http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
fn1 = r'im1_2.jpg'
#fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300))
name = fn1.split('\\')[-1].split(".")[0]
P = brightness(fore)
thresh = getthresh(cv2.resize(P,(300,300)))
print thresh
lastthresh=threshold(P,thresh,1,0)
thresh,lastthresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#lastthresh = pad(lastthresh,1)
#plotc = plotim(name+" overlayed lastthresh",overlay(fore.copy(),lastthresh*255,alfa=lastthresh)).show()

# find biggest area
contours,hierarchy = cv2.findContours(lastthresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print "objects: ",len(contours)
index,maxarea = biggestCntData(contours)
print "area contour:",maxarea,"index: ",index
cnt = contours[index]
pts = contour2points(cnt)
test = polygontest(P.copy().astype(np.int32),cnt,mask = None)
#pg = graphpolygontest(test).show()
center,center_pts = polycenter(test)
magnitudes = anorm((np.array(pts-(center))))
mean = np.sum(magnitudes)/np.float(len(magnitudes))
deviation = np.abs(magnitudes-mean)
variance = anorm2(deviation)/np.float(len(magnitudes))
standard_deviation= np.sqrt(variance)
bads = pts[deviation>standard_deviation]
goods = pts[deviation<=standard_deviation]
ax = plotPointToPoint(np.array([(center)]*len(goods)),goods,annotate="")
#plotPointToPoint(np.array([(center)]*len(bads)), bads, cor="r", annotate="")
plt.show()