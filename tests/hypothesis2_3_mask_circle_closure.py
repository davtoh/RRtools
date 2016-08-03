__author__ = 'Davtoh'

from tesisfunctions import Plotim,overlay
import cv2
import numpy as np
from tesisfunctions import brightness, IMAGEPATH,graphpolygontest,thresh_biggestCnt,\
    CircleClosure,twoMaxTest,graphDeffects,extendedSeparatingLine


fn1 = r'im1_2.jpg'
#fn1 = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300))

P = brightness(fore)
thresh,lastthresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
Plotim(name + " overlayed lastthresh", overlay(fore.copy(), lastthresh * 255, alpha=lastthresh * 0.2)).show()

for i in xrange(2): # test multiple applications to results
    # SIMULATE polygon test
    dist_transform = cv2.distanceTransform(lastthresh,cv2.DIST_LABEL_PIXEL,5)
    dist_transform[lastthresh==0] = -1 # simulate outside points
    graph = graphpolygontest(dist_transform,name+" dist_transform")
    center = graph.center
    cx,cy = center
    centerVal = dist_transform[cy,cx]

    print "center: ", center, " Value: ", centerVal
    graph.show()
    overcircle = np.zeros_like(lastthresh,np.uint8)
    cv2.circle(overcircle,center,centerVal,1,-1)
    overcircle[lastthresh==0]=0
    Plotim(name + " overlayed circle", overcircle).show()

    #DEFECTS
    pallet = [[0,0,0],[255,255,255]]
    pallet = np.array(pallet,np.uint8)
    imdefects = pallet[overcircle]
    imdefects = overlay(fore.copy(), imdefects, alpha=brightness(imdefects))

    cnt = thresh_biggestCnt(overcircle)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    if twoMaxTest(defects,epsilon=0.5):
        graphDeffects(imdefects,cnt,defects)
        #SEPARATING LINE
        start,end = extendedSeparatingLine(imdefects.shape, cnt, defects)
        cv2.line(imdefects,start,end,[0,0,100],2)
        Plotim(name + " and circle defects", imdefects).show()

        cv2.line(lastthresh,start,end,0,2)
        cnt = thresh_biggestCnt(lastthresh)
    else:
        cnt = CircleClosure(lastthresh)

    ellipse = cv2.fitEllipse(cnt)
    mask = np.ones(P.shape,dtype=np.uint8)
    cv2.ellipse(mask,ellipse,0,-1)
    fore[mask>0]=0
    Plotim(name + " result", fore).show()
    #cv2.imwrite("mask_"+name+".png",fore)