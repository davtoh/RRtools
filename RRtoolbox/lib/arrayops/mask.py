# -*- coding: utf-8 -*-
"""
    This module contains all basic masking and pre-processing (as in segmenting phase) methods
"""
from __future__ import division
__author__ = 'Davtoh'

import cv2
import numpy as np
from basic import findminima, im2shapeFormat, getOtsuThresh
from filters import smooth

def brightness(img):
    """
    get brightness from an image
    :param img: BGR or gray image
    :return:
    """
    ### LESS BRIGHT http://alienryderflex.com/hsp.html
    #b,g,r = cv2.split(img.astype("float"))
    #return np.sqrt( .299*(b**2) + .587*(g**2) + .114*(r**2)).astype("uint8")
    ### Normal gray
    return im2shapeFormat(img,img.shape[:2])
    ### HSV
    #return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]

def background(gray, mask = None, iterations = 3):
    """
    get the background mask of a gray image. (this it the inverted of :func:`foreground`)

    :param gray: gray image
    :param mask: (None) input mask to process gray
    :param iterations: (3) number of iterations to detect background
            with otsu threshold.
    :return: output mask
    """
    #gray = works with any gray image
    if mask is None: mask = np.ones_like(gray)
    for i in xrange(iterations):
        hist, bins = np.histogram(gray[mask.astype(bool)].flatten(), 256, [0, 256])
        thresh = getOtsuThresh(hist)
        cv2.threshold(gray, thresh, 1, cv2.THRESH_BINARY_INV, dst=mask)
    return mask

def foreground(gray, mask = None, iterations = 3):
    """
    get the foreground mask of a gray image. (this it the inverted of :func:`background`)

    :param gray: gray image
    :param mask: (None) input mask to process gray
    :param iterations: (3) number of iterations to detect foreground
            with otsu threshold.
    :return: output mask
    """
    return 1-background(gray,mask,iterations)

def multiple_otsu(gray, mask = None, flag = cv2.THRESH_BINARY, iterations = 1):
    """
    get the mask of a gray image applying Otsu threshold.

    :param gray: gray image
    :param mask: (None) input mask to process gray
    :param iterations: (1) number of iterations to detect Otsu threshold.
    :return: thresh, mask
    """
    #get mask
    if mask is None and flag == cv2.THRESH_BINARY_INV:
        mask = np.ones_like(gray)
    if mask is None and flag == cv2.THRESH_BINARY:
        mask = np.zeros_like(gray)

    if iterations>0:
        for i in xrange(iterations):
            hist, bins = np.histogram(gray[mask.astype(bool)].flatten(), 256, [0, 256])
            thresh = getOtsuThresh(hist)
            cv2.threshold(gray, thresh, 1, flag, dst=mask)
        return thresh, mask
    else:
        raise Exception("iterations must be greater than 0 and got {}".format(iterations))

def hist_cdf(img,window_len=0,window='hanning'):
    """
    Get image histogram and the normalized cumulative distribution function.

    :param img: imaeg
    :param window_len:
    :param window:
    :return: histogram (int), normalized cdf (float)
    """
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    if window_len: hist = smooth(hist,window_len,window) # if window_len=0 => no filter
    cdf = hist.cumsum() # cumulative distribution function
    cdf_normalized = cdf*(hist.max())/cdf.max() #normalized cdf
    return hist,cdf_normalized

def thresh_hist(gray):
    """
    Get best possible thresh to threshold object from the gray image.

    :param gray: gray image.
    :return: thresh value.
    """
    hist,cdf = hist_cdf(gray,11)
    th1 = 130 #np.min(np.where(cdf.max()*0.2<=cdf))
    th2 = np.max(np.where(hist.max()==hist))
    th3 = np.min(np.where(np.mean(cdf)<=cdf))
    th4=findminima(hist,np.mean([th1,th2,th3]))
    return th4

def threshold_opening(src, thresh, maxval, type):
    """
    Eliminate small objects from threshold.

    :param src:
    :param thresh:
    :param maxval:
    :param type:
    :return:
    """
    kz = np.mean(src.shape)/50 # proportion to src
    kernel = np.ones((kz,kz),np.uint8) # kernel of ones
    retval,th = cv2.threshold(src, thresh, maxval, type) # apply threshold
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) # apply opening
    return th

def biggestCntData(contours):
    """
    Gets index and area of biggest contour.

    :param contours:
    :return: index, area
    """
    index,maxarea = 0,0
    for i in xrange(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>maxarea: index, maxarea = i, area
    return index,maxarea

def biggestCnt(contours):
    """
    Filters contours to get biggest contour.

    :param contours:
    :return: cnt
    """
    if contours:
        return contours[biggestCntData(contours)[0]]
    # return empty array if there is not anything to choose
    return np.array([])

def cnt_hist(gray):
    """
    Mask of a ellipse enclosing retina using histogram threshold.

    :param gray: gray image
    :param invert: invert mask
    :return: mask
    """
    thresh = thresh_hist(gray) # obtain optimum threshold
    rough_mask=threshold_opening(gray,thresh,1,0)
    contours,hierarchy = cv2.findContours(rough_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return biggestCnt(contours)

def mask_watershed(BGR, GRAY = None):
    """
    Get retinal mask with watershed method.

    :param BGR:
    :param GRAY:
    :return: mask
    """
    if GRAY is None: GRAY = brightness(BGR) # get image brightness
    thresh,sure_bg = cv2.threshold(GRAY,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # obtain over threshold
    thresh,sure_fg = cv2.threshold(GRAY,thresh+10,1,cv2.THRESH_BINARY)
    markers = np.ones_like(GRAY,np.int32) # make background markers
    markers[sure_bg==1]=0 # mark unknown markers
    markers[sure_fg==1]=2 # mark sure object markers
    cv2.watershed(BGR,markers) # get watershed on markers
    retval,mask = cv2.threshold(markers.astype("uint8"),1,1,cv2.THRESH_BINARY) # get binary image of contour
    return mask

def thresh_biggestCnt(thresh):
    """
    From threshold obtain biggest contour.

    :param thresh: binary image
    :return: cnt
    """
    #http://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html#gsc.tab=0
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return biggestCnt(contours)

def gethull(contours):
    """
    Get convex hull.

    :param contours: contours or mask array
    :return: cnt
    """
    if type(contours).__module__ == np.__name__:
        contours, _ = cv2.findContours(contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    allcontours = np.vstack(contours[i] for i in np.arange(len(contours)))
    return cv2.convexHull(allcontours)