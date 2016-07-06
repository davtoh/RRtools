__author__ = 'Davtoh'
import cv2
import numpy as np
from RRtoolbox.lib.directory import getData,mkPath
from RRtoolbox.lib.arrayops import angle2D, vectorsAngles, contour2points, points2contour, vectorsQuadrants
from tesisfunctions import plotim,overlay,sigmoid,histogram,brightness,getthresh,threshold,pad,\
    IMAGEPATH,SAVETO,thresh_biggestCnt,extendedSeparatingLine,graphDeffects
from glob import glob
name_script = getData(__file__)[-2]

#http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

## ARGUMENTS
fn1 = r'im1_2.jpg'
#fn1 = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
#fn1 = glob(IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/*.jpg")[19]
#fn1 = r"asift2Result_with_alfa1.png"
#fn1 = r"im_completer_Result2.png"
name_image = getData(fn1)[-2]#fn1.split('\\')[-1].split(".")[0]
saveTo = SAVETO+name_script+"/"
save = True
iterations = 3
if save: mkPath(saveTo)
show = False
shape = None #100,200

# INIT CONTROL VARIABLES
fore = cv2.imread(fn1)
if shape:
    fore = cv2.resize(fore,shape)

P = brightness(fore) # get gray image
#thresh = getthresh(cv2.resize(P,shape)) # obtain threshold value
#bImage=threshold(P, thresh, 1, 0) #binary image
thresh, bImage = cv2.threshold(P, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # threshold value, and binary image
ROI = bImage

if show or save:
    for i in xrange(iterations): # test iterations over the same algorithm
        print "ITERATION",i
        bImage = ROI
        fore1 = overlay(fore.copy(), bImage * 255, alpha=bImage)
        plot = plotim("{} ITER {} STEP 1 get binary image".format(name_image,i), fore1)
        if save: plot.save("{}{{win}}".format(saveTo))
        if show: plot.show()
        if i>= 17:
            pass
        # find biggest cnt
        cnt = thresh_biggestCnt(bImage)

        #DEFECTS
        pallet = [[0,0,0],[255,255,255]]
        pallet = np.array(pallet,np.uint8)
        imdefects = pallet[bImage]

        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        graphDeffects(imdefects,cnt,defects)

        #SEPARATING LINE
        if defects.size<=4:
            break
        start,end = extendedSeparatingLine(imdefects.shape, cnt, defects)
        cv2.line(imdefects,start,end,[0,0,100],2)
        plot = plotim("{} ITER {} STEP 2 get contour defects".format(name_image,i), imdefects)
        if save: plot.save("{}{{win}}".format(saveTo))
        if show: plot.show()

        #ROI
        bImage = np.zeros(P.shape, dtype=np.uint8)
        cv2.drawContours(bImage, [cnt], 0, 1, -1)
        cv2.line(bImage, start, end, 0, 2)
        # find biggest cnt
        cnt = thresh_biggestCnt(bImage)

        #ROI
        ROI = np.zeros(P.shape,dtype=np.uint8)
        cv2.drawContours(ROI,[cnt],0,1,-1)
        plot = plotim("{} ITER {} STEP 3 get ROI".format(name_image,i), ROI*255)
        if save: plot.save("{}{{win}}".format(saveTo))
        if show: plot.show()

        M = cv2.moments(cnt) # find moments
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print "(cx,cy)",(cx,cy)
        fore1 = fore.copy()
        cv2.circle(fore1, (cx,cy), 10, (0, 0, 255), -1, 8)
        cv2.drawContours(fore1,[cnt],0,(0, 0, 255),2)

        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(fore1,ellipse,(0,255,0),2)

        plot = plotim("{} ITER {} STEP 4 ROI description".format(name_image,i), fore1)
        if save: plot.save("{}{{win}}".format(saveTo))
        if show: plot.show()

        mask = np.ones(P.shape,dtype=np.uint8)
        cv2.ellipse(mask,ellipse,0,-1)
        fore1 = fore.copy()
        fore1[mask>0]=0
        plot = plotim("{} ITER {} STEP 5 masked".format(name_image,i),fore1)
        if save: plot.save("{}{{win}}".format(saveTo))
        if show: plot.show()
else:
    raise Exception("save or show flags not set in {} to process {}".format(name_script,name_image))