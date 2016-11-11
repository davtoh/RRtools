# -*- coding: utf-8 -*-
__author__ = 'Davtoh'

import cv2
import numpy as np
from ..lib.arrayops import normsigmoid, normalize, Bandpass, Bandstop,\
    findminima, findmaxima, find_near, smooth, getOtsuThresh, convexityRatio, \
    filterFactory, brightness, background, thresh_biggestCnt, contours2mask

def _getBrightAlpha(backgray, foregray, window = None):
    """
    Get alpha transparency for merging foreground to
    background gray image according to brightness.
    (This is a test and not intended for practical use)

    :param backgray: background image.
    :param foregray: foreground image.
    :param window: window used to customizing alfa. It can be a binary or alpha mask,
            values go from 0 for transparency to any value where the maximum is visible
            i.e a window with all the same values does nothing. A binary mask can be used,
            where 0 is transparent and 1 is visible. If not window is given alfa is not
            altered and the intended alpha is returned.
    :return: alfa mask
    """
    # this method was obtained for a particular problem, change to an automated one
    backmask = normalize(normsigmoid(backgray,10,180) + normsigmoid(backgray,3.14,192) +
                         normsigmoid(backgray,-3.14,45))
    foremask = normalize(normsigmoid(foregray,-1,242)*normsigmoid(foregray,3.14,50))
    foremask = normalize(foremask * backmask)
    foremask[foremask>0.9] = 2.0
    ksize = (21,21)
    foremask = normalize(cv2.blur(foremask,ksize))
    if window is not None: foremask *= normalize(window) # ensures that window is normilized to 1
    return foremask

def get_beta_params_hist(P):
    """
    Automatically find parameters for bright alpha masks
    using a histogram analysis method.

    :param P: gray image
    :return: beta1 for minimum valley left of body, beta2 for brightest valley right of body
            where the body starts at the tallest peak in the histogram.
    """
    hist_P, bins = np.histogram(P.flatten(),256,[0,256])
    window = "hamming" # 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    window_len = 51
    hist_PS = smooth(hist_P,window_len,window=window, correct=True)
    otsu = getOtsuThresh(hist_P) # Otsu value
    minima = findminima(hist=hist_PS) # minima values
    maxima = findmaxima(hist=hist_PS) # maxima values
    if minima != [] and maxima !=[]:
        data_body = find_near(maxima, thresh=otsu,side="right")
        beta1 = find_near(minima,thresh=data_body,side="left")
        beta2 = find_near(minima,thresh=data_body,side="right")
    else: # case where histogram is uniform and does not have minima or maxima values
        beta1 = beta2 = otsu
    if beta2==beta1:
        beta2 +=1 # prevents overlapping
    return beta1,beta2

def get_bright_alpha(backgray, foregray, window = None):
    """
    Get alpha transparency for merging foreground to
    background gray image according to brightness.

    :param backgray: background image. (as float)
    :param foregray: foreground image. (as float)
    :param window: window used to customizing alfa. It can be a binary or alpha mask,
            values go from 0 for transparency to any value where the maximum is visible
            i.e a window with all the same values does nothing. A binary mask can be used,
            where 0 is transparent and 1 is visible. If not window is given alfa is not
            altered and the intended alpha is returned.
    :return: alfa mask
    """
    backmask = Bandpass(3, *get_beta_params_hist(backgray))(backgray) # beta1 = 50, beta2 = 190
    foremask = Bandpass(3, *get_beta_params_hist(foregray))(foregray) # beta1 = 50, beta2 = 220
    foremask = normalize(foremask * backmask)
    if window is not None: foremask *= normalize(window) # ensures that window is normilized to 1
    return foremask

def get_beta_params_Otsu(P):
    """
    Automatically find parameters for alpha masks using Otsu threshold value.

    :param P: gray image
    :return: beta1 for minimum histogram value, beta2 for Otsu value
    """
    # process histogram for uint8 gray image
    if P.any(): # if array is not empty
        hist, bins = np.histogram(P.flatten(), 256)

        # get Otsu thresh as beta2
        beta2 = bins[getOtsuThresh(hist)]
        return np.min(P), beta2 # beta1, beta2
    return 0.0,1.0

def get_layered_alpha(back, fore):
    """
    Get bright alpha mask (using Otsu method)

    :param back: BGR background image
    :param fore: BGR foreground image
    :return: alpha mask
    """
    # find retinal area and its alpha
    mask_back, alpha_back = retinal_mask(back,biggest=True,addalpha=True)
    mask_fore, alpha_fore = retinal_mask(fore,biggest=True,addalpha=True)

    # convert uint8 to float
    backgray = brightness(back).astype(float)
    foregray = brightness(fore).astype(float)

    # scale from 0-1 to 0-255
    backm = alpha_back*255
    forem = alpha_fore*255

    # get alpha masks fro background and foreground
    backmask = Bandstop(3, *get_beta_params_Otsu(backm[mask_back.astype(bool)]))(backgray)
    foremask = Bandpass(3, *get_beta_params_Otsu(forem[mask_fore.astype(bool)]))(foregray)

    # merge masks
    alphamask = normalize(foremask * backmask * (backm/255.))
    return alphamask

def retina_markers_thresh(P):
    """
    Retinal markers thresholds to find background,
    retinal area and optic disc with flares based
    in the histogram.

    :param P: gray image
    :return: min,b1,b2,max

    .. were::
        black background < min
        b1 > retina < b2
        flares > max
    """
    # calculate histogram
    #hist_P = histogram(P)[0]
    hist_P, bins = np.histogram(P.flatten(),256,[0,256])

    # filter histogram to smooth it.
    window = "hamming" # 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    window_len = 51
    hist_PS = smooth(hist_P,window_len,window=window, correct=True) # it smooths and expands the histogram
    #from scipy.signal import savgol_filter
    #hist_PS = savgol_filter(hist_P, window_len, polyorder = 3)

    otsu = getOtsuThresh(hist_P) # Otsu value
    minima = findminima(hist=hist_PS) # minima values
    maxima = findmaxima(hist=hist_PS) # maxima values

    # INITIAL DATA
    data_min_left = P.min() # initial minimum value
    data_min = find_near(maxima, thresh=data_min_left)
    #data_min_right = find_near(minima, thresh=data_min, side="right")
    data_max = P.max()
    data_max_left = find_near(minima, thresh=data_max, side="left")

    # SECONDARY DATA
    data_body = find_near(maxima, thresh=otsu,side="right")
    data_body_left = find_near(minima,thresh=data_body,side="left")
    #data_body_right = find_near(minima,thresh=data_body,side="right")
    return data_min,data_body_left,data_body,data_max_left


def find_optic_disc_watershed(img, P):
    """
    Find optic disk in image using a watershed method.

    :param img: BGR image
    :param P: gray image
    :return: optic_disc, Crs, markers, watershed
    """
    assert img is not None and P is not None
    #fore = cv2.cvtColor(P,cv2.COLOR_GRAY2BGR)
    # create watershed
    data_min,data_body_left,data_body,data_max_left = retina_markers_thresh(P)
    watershed = np.zeros_like(P).astype("int32")
    mk_back,mk_body,mk_flare = 1,2,3
    watershed[P <= data_min]=mk_back # background FIXMED use P.min() and aproaching to local maxima
    watershed[np.bitwise_and(P>data_body_left,P<data_body)]=mk_body # main body
    # find bright objects
    flares_thresh = (P >= data_max_left).astype(np.uint8)
    contours,hierarchy = cv2.findContours(flares_thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mks_flare = []
    for i,cnt in enumerate(contours):
        index = mk_flare+i
        mks_flare.append(index)
        watershed[contours2mask(cnt,shape=watershed.shape)]=index # Flares. this can be used approaching
                                        # to local maxima, but brightest areas are almost
                                        # always saturated so no need to use it
    markers = watershed.copy()
    # apply watershed to watershed
    cv2.watershed(img,watershed) # FIXME perhaps the function should be cv2.floodFill?

    # different classification algorithms could be used using watershed
    contours_flares = []
    for mk_flare in mks_flare:
        brightest = np.uint8(watershed==mk_flare)
        contours,hierarchy = cv2.findContours(brightest,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours_flares.extend(contours)
    Crs = [(i,j) for i,j in [(convexityRatio(cnt),cnt) for cnt in contours_flares] if i != 0] # convexity ratios
    Crs.sort(reverse=True,key=lambda x:x[0])
    candidate = Crs[-1]
    ellipse = cv2.fitEllipse(candidate[1])
    optic_disc = np.zeros_like(P)
    cv2.ellipse(optic_disc, ellipse, 1, -1) # get elliptical ROI
    return optic_disc, Crs, markers, watershed

def layeredfloods(img, gray = None, backmask = None, step = 1, connectivity = 4, weight = False):
    """
    Create an alpha mask from an image using a weighted layered flooding algorithm,

    :param img: BGR image
    :param gray: Gray image
    :param backmask: background mask
    :param step: step to increase upDiff in the floodFill algorithm. If weight is True
            step also increases the weight of the layers.
    :param connectivity: pixel connectivity of 4 or 8 to use in the floodFill algorithm
    :param weight: Increase progressively the weight of the layers using the step parameter.
    :return: alpha mask
    """
    if gray is None:
        if len(img.shape)>2:
            gray = brightness(img)
        else:
            gray = img

    # initialization of flags for flooding
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    h, w = gray.shape# get image shape

    # pad mask to add 1 layer of pixels
    # this is needed in the flood operation
    mask = np.zeros((h+2, w+2), np.uint8) #

    # create the background mask
    if backmask is None:
        backmask = background(gray)
    mask_background = np.zeros((h+2, w+2), np.uint8)
    mask_background[1:-1,1:-1] = backmask

    # get area of background mask
    area_background = np.sum(mask_background)

    # low and high limits for flooding
    lo,hi = 255,0

    # first seed point of the minimum color
    xs,ys = np.where(gray == np.min(gray))
    seed_pt = xs[0],ys[0]

    #flood adding overal layers starting from background
    area = 0
    hi = step
    all = np.zeros_like(mask,np.float32)
    while hi <= 255:
        mask[:]=0
        cv2.floodFill(img, mask, seed_pt, (255, 255, 255), (lo,) * 3, (hi,) * 3, flags)
        if weight and area >= area_background:
            all += mask.astype("float")*hi
        elif not weight and area >= area_background:
            all += mask.astype("float")
        else:
            area = np.sum(np.bitwise_and(mask,mask_background))
        hi += step
    all = all[1:-1,1:-1]
    all /= all.max()
    return all

def retinal_mask(img, biggest = False, addalpha = False):
    """
    Obtain the mask of the retinal area in an image.
    For a simpler and lightweight algorithm see :func:`retinal_mask_watershed`.

    :param img: BGR or gray image
    :param biggest: True to return only biggest object
    :param addalpha: True to add additional alpha mask parameter
    :return: if addalpha:
                binary mask, alpha mask
            else:
                binary mask
    """

    # pad image to add 1 layer of pixels
    # this is used to correctly flood all the background
    if len(img.shape)>2:
        h, w, c = img.shape
        imgP = np.zeros((h+2, w+2,3), np.uint8)
        imgP[1:-1,1:-1,:] = img
        P = brightness(imgP)
    else:
        h, w = img.shape
        imgP = np.zeros((h+2, w+2), np.uint8)
        imgP[1:-1,1:-1] = img
        P = imgP

    # get flooded alpha mask
    mask_alpha = 1-layeredfloods(imgP, gray=P)[1:-1,1:-1]
    hist, bins = np.histogram(mask_alpha.flatten(),256)
    thresh = bins[getOtsuThresh(hist)]
    mask_binary = (mask_alpha > thresh).astype(np.uint8)

    if biggest: # process to give only the biggest object
        a = thresh_biggestCnt(mask_binary)
        mask_binary = np.zeros_like(mask_binary)
        cv2.drawContours(mask_binary, [a], 0, 1, -1) # get current fillet ROI

    if addalpha:
        #if biggest: mask_alpha[mask_binary==0] = 0
        return mask_binary,mask_alpha
    return mask_binary

def retinal_mask_watershed(img, parameters = (10, 30, None), addMarkers = False):
    """
    Quick and simple watershed method to obtain the mask of the retinal area in an image.
    For a more robust algorithm see :func:`retinal_mask`.

    :param img: BGR or gray image
    :param parameters: tuple of parameters to pass to :func:`filterFactory`
    :param addMarkers: True to add additional Marker mask. It contains 0 for unknown
            areas, 1 for background and 2 for retinal area.
    :return: if addMarkers:
                binary mask, Markers mask
            else:
                binary mask
    """

    if parameters is not None:
        myfilter = filterFactory(*parameters) # alfa,beta1,beta2
        img=(myfilter(img.astype("float"))*255).astype("uint8")#*fore.astype("float")

    P = brightness(img) # get scaled image brightness
    thresh,sure_bg = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # obtain over threshold
    thresh,sure_fg = cv2.threshold(P,thresh+10,1,cv2.THRESH_BINARY)

    markers = np.ones_like(sure_fg).astype("int32") # make background markers
    markers[sure_bg==1]=0 # mark unknown markers
    markers[sure_fg==1]=2 # mark sure object markers

    cv2.watershed(img,markers) # get watershed on markers

    thresh,mask = cv2.threshold(markers.astype("uint8"),1,1,cv2.THRESH_BINARY) # get binary image of contour
    if addMarkers:
        return mask, thresh
    return mask