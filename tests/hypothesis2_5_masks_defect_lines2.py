__author__ = 'Davtoh'
import cv2
import numpy as np

from RRtoolbox.lib.arrayops import convexityRatio
from RRtoolbox.lib.directory import getData,mkPath
from recommended import getKernel
from tesisfunctions import Plotim,overlay, brightness, SAVETO, StdoutLOG,thresh_biggestCnt,\
    extendedSeparatingLine,graphDeffects,printParams

name_script = getData(__file__)[-2]

def separeByDefect(cnt,defects):
    """
    Separate contour at two defects with biggest distances. (contours must be index ordered)

    :param cnt: contour
    :param defects: convexity defects
    :return: contour a side, contour at the other side
    """
    positions = defects[:,0,2] # get positions
    distances = defects[:,0,3] # get its distances from hull
    two_max = np.argpartition(distances, -2)[-2:] # get indexes of two maximum distances in defects
    two_indexes = positions[two_max] # get indexes in cnt
    # assuming that cnts indexes are ordered then get slicing indexes
    imin,imax = np.min(two_indexes),np.max(two_indexes) # get slices
    #assert np.any(np.vstack((cnt[:imin],cnt[imin:imax],cnt[imax:])) == cnt)
    #return cnt[imin:imax],np.vstack((cnt[:imin],cnt[imax:])) # side A, side B # FIXME it separates the objects but with ranges open
    #cnt = np.vstack((cnt[0:1],cnt,cnt[-1:]))
    return cnt[imin:imax+1],np.vstack((cnt[:imin+1],cnt[imax:])) # side A, side B # this separates cnt with ranges closed

def mask_hull_lines(image, iterations):
    """

    :param image:
    :param iterations:
    :return:
    """
    biggest = thresh_biggestCnt(image)
    for _ in iterations:
        hull = cv2.convexHull(biggest,returnPoints = False)
        defects = cv2.convexityDefects(biggest,hull)
        biggest,b = separeByDefect(biggest,defects)
        if cv2.contourArea(b)>cv2.contourArea(biggest):
            biggest =  b # new biggest is b
    return biggest

def sortCnts(cnts, reverse=False):
    """
    order

    :param cnts: list of cnts
    :return: ordered cnts
    """
    cnts = list(cnts)
    cnts.sort(reverse=reverse,key=lambda x: cv2.contourArea(x))
    return cnts

def drawSeparated(a, b, shape, pallet=None, win ="separated"):
    """

    :param a: biggest
    :param b: smallest
    :param shape: shape of output
    :param pallet: numpy pallete
    :param win: window name
    :return:
    """
    ROIA = np.zeros(shape,dtype=np.uint8)
    cv2.drawContours(ROIA, [a], 0, 1, -1)
    ROIB = np.zeros(shape,dtype=np.uint8)
    cv2.drawContours(ROIB, [b], 0, 1, -1)

    if pallet is None:
        pallet = np.array([[0,0,0],[100,100,100],[255,255,255]],np.uint8)
    imgA = pallet[ROIA]
    imgB = pallet[ROIB*2]
    img = overlay(imgA, imgB, alpha=ROIB)
    plot = Plotim(win, img)
    plot.a = ROIA
    plot.b = ROIB
    return plot

def invert(gray):
    return (-1 * (gray.astype(np.int32) - gray.max())).astype(np.uint8)

def demo():
    #http://stackoverflow.com/questions/14725181/speed-up-iteration-over-numpy-arrays-opencv-cv2-image
    #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

    ## ARGUMENTS
    #fn1 = r'im1_2.jpg'
    fn1 = "inputImage3.png"
    #fn1 = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
    #fn1 = glob(IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/*.jpg")[19]
    #fn1 = r"asift2Result_with_alfa1.png"
    #fn1 = r"im_completer_Result2.png"
    fns = [fn1]
    #fns = glob(IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/*.jpg")

    # important variables
    show = True # show test
    save = False # to save tests
    saveTo_root = SAVETO+name_script+"/" # this should not be changed for standart-like outputs
    resultToRoot = True # save last result in parent directory. Useful for many tests
    invertROI = True # to invert the ROI if object is inverted
    filterROI = True # to filter the ROI if object is not smooth
    ratioCheck = 0.99 # check convexity ratio: 0.99 is used for regular objects, 0.97 for irregular objects
    # optional variables
    maxIter = 3 # max iterations if ratio fails
    forceIterations = True # can produce arrors by tries reach maxIter
    shape = None# 200,200 # shapes
    logical = 1 # what is considered to be logical True
    # FIXME look for a better pallet or color system
    # pallte = 0 -background,1- biggestObj,2-smallestObject,3- ROI_desc, 4-elliptical_desc, 5-points
    pallet= [[255,255,255],[0,0,0],[100,100,100],(255, 0, 0),(0, 255, 0),(0, 0, 255)] # pallet to color all
    alfapallet = np.array([1,0,0,1,1,1],np.uint8) # alfa used for plots
    alfa1 = 0.8 # alfa of lines
    alfa2 = 0.4 # alfa of ROI
    cline,cpoint = tuple(pallet[3]),tuple(pallet[5]) # color of line, color of point
    pallet = np.array(pallet,np.uint8) # pallet converted to numpy array to work as pallet
    if save: config = locals().copy() # save configuration for logging

    if show or save:
        if shape: # if the shape is known then use predefined variables too
            template = np.zeros(shape, dtype=np.uint8) # template to use in each binary image
            k = getKernel(shape[0]*shape[1])
            ks = k.shape[0]/3 # lines thickness

        for fn in fns:
            name_image = getData(fn)[-2]#fn1.split('\\')[-1].split(".")[0]
            saveTo = saveTo_root+name_image+"/"

            if save:
                mkPath(saveTo) # make directory
                StdoutLOG(saveTo + "log") # save file output
                # now everything printed will be logged
                printParams(config) # this is logged too
            # INIT CONTROL VARIABLES
            fore = cv2.imread(fn,1)
            if shape:
                fore = cv2.resize(fore,shape)
            else:
                template = np.zeros(fore.shape[:2], dtype=np.uint8) # template to use in each binary image
                k = getKernel(fore.shape[0]*fore.shape[1])
                ks = k.shape[0]/3 # lines thickness

            P = brightness(fore) # get gray image
            #thresh = getthresh(cv2.resize(P,shape)) # obtain threshold value
            #bImage=threshold(P, thresh, 1, 0) #binary image
            thresh, ROI = cv2.threshold(P, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # threshold value, and binary image

            if invertROI: ROI = invert(ROI) # this is used when the object is inverted
            if filterROI: ROI = cv2.morphologyEx(ROI,3,k,iterations = 2)
            print "performed morphological operation: closing with kernel of shape {}".format(k.shape)
            # find biggest cnt
            cnt = thresh_biggestCnt(ROI)

            for i in xrange(maxIter): # test iterations over the same algorithm
                print "ITERATION",i
                fore1 = overlay(fore.copy(), pallet[ROI], alpha=ROI * alfa2)
                plot = Plotim("ITER {} STEP 1 get binary image".format(i), fore1)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()
                 # get convex hull indexes
                hull = cv2.convexHull(cnt, returnPoints = False)
                defects = cv2.convexityDefects(cnt, hull) # get defects

                if defects is None: # there are no defects
                    print "defects is None"
                    if i == 0:
                        print "perhaps you should use ROI = invert(ROI) for this image: {}".format(name_image)
                    if not forceIterations:
                        print "breaking algorithm"
                        break
                    else:
                        print "forced to continue"

                if defects.size<=4: # cannot separate by defects
                    print "not enough defects to create lines"
                    if not forceIterations:
                        print "breaking algorithm"
                        break
                    else:
                        print "forced to continue"

                distances = defects[:,0,3] # get its distances from hull
                two_max = np.argpartition(distances, -2)[-2:] # get indexes of two maximum distances
                print "two max indexes values",distances[np.argpartition(distances, -2)[-2:]]

                ratio = convexityRatio(cnt, hull)
                print "convexity Ratio",ratio

                if ratio>ratioCheck: # there should not be a good defect
                    print "image should not have more defects"
                    if not forceIterations:
                        print "breaking algorithm"
                        break
                    else:
                        print "forced to continue"

                graphDeffects(fore1, cnt, defects,cline=cline,cpoint=cpoint, alfa=alfa1,thickness=ks)
                plot = Plotim("ITER {} STEP 2 get contour defects".format(i), fore1)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()

                a,b = sortCnts(separeByDefect(cnt, defects), reverse=True) # a must be the biggest
                p = drawSeparated(a,b,P.shape, pallet=pallet)
                #DEFECTS
                imdefects = p.img
                graphDeffects(imdefects, cnt, defects,cline=cline,cpoint=cpoint, alfa=alfa1,thickness=ks)
                start,end = extendedSeparatingLine(imdefects.shape, cnt, defects)
                cv2.line(imdefects,start,end,[0,0,100],ks)
                plot = Plotim("ITER {} STEP 3 separate objects by defects".format(i), imdefects)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()

                # fillet ROI
                ROI = template.copy()
                cv2.drawContours(ROI, [a], 0, logical, -1) # get current fillet ROI
                # contour of ROI
                ROI_contour = template.copy()
                cv2.drawContours(ROI_contour, [a], 0, logical, ks) # draw new object contour
                """
                # center of ROI
                ROI_center = template.copy()
                M = cv2.moments(a) # find moments
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #print "(cx,cy)",(cx,cy)
                cv2.circle(ROI_center, (cx,cy), ks, logical, -1, ks)# draw center"""
                # elliptical ROI
                ellipse = cv2.fitEllipse(a)
                ROI_elliptical = template.copy()
                cv2.ellipse(ROI_elliptical,ellipse,logical,-1) # get elliptical ROI
                # elliptical contour of ROI
                ROI_contour_elliptical = template.copy()
                cv2.ellipse(ROI_contour_elliptical,ellipse,logical,ks) # get elliptical ROI

                # plot a description
                img = overlay(fore.copy(), pallet[ROI_contour*3], alpha=ROI_contour) # draw ROI contour
                #img = overlay(img,pallet[ROI_center*5],alfa=ROI_center) # draw center
                img = overlay(img, pallet[ROI_contour_elliptical*4], alpha=ROI_contour_elliptical) # draw elliptical ROI
                #hf.graphDeffects(img, cnt, defects,cline=cline,cpoint=cpoint, alfa=0.3,thickness=ks) # draw defects
                plot = Plotim("ITER {} STEP 4 get biggest object ROI or its elliptical mask".format(i), img)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()

                img = overlay(fore.copy(), pallet[ROI], alpha=alfapallet[ROI])
                plot = Plotim("ITER {} STEP 5 ROI in image".format(i), img)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()

                img = overlay(fore.copy(), pallet[ROI_elliptical], alpha=alfapallet[ROI_elliptical])
                plot = Plotim("ITER {} STEP 6 elliptical ROI in image".format(i), img)
                if save: plot.save("{}{{win}}".format(saveTo))
                if show: plot.show()
                cnt = a

            fore1 = overlay(fore.copy(), pallet[ROI], alpha=ROI * alfa2)
            plot = Plotim("ITER {} STEP final binary image".format(i), fore1)
            if save: plot.save("{}{{win}}".format(saveTo))
            if save and resultToRoot: plot.save("{}{}{{win}}".format(saveTo_root,name_image))
            if show: plot.show()
    else:
        raise Exception("save or show flags not set in {}".format(name_script))

if __name__ == "__main__":
    demo()