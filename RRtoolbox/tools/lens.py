# -*- coding: utf-8 -*-
__author__ = 'Davtoh'

from ..lib.arrayops import spoint2opointfunc, thresh_biggestCnt
from segmentation import retinal_mask
from ..lib.plotter import fastplt
import cv2
import numpy as np

def drawEllipse(array, cnt, color=0):
    """
    project ellipse over array.

    :param array: array to draw ellipse
    :param cnt: contours of segmentation to fit ellipse
    :param color: color of lens
    :return: array
    """
    ellipse = cv2.fitEllipse(cnt) # get ellipse
    # project ellipse over array
    cv2.ellipse(array, ellipse, color, -1)
    return array

def drawCircle(array, cnt, color=0):
    """
    project circle over array.

    :param array: array to draw circle
    :param cnt: contours of segmentation to fit circle
    :param color: color of lens
    :return: array
    """
    center, radius = cv2.minEnclosingCircle(cnt) # get circle
    # project circle over array
    cv2.circle(array, tuple(map(int, center)), int(radius), color, -1)
    return array

def fitLens(img, mask, color = 0, asEllipse = False, addmask = False):
    """
    Place lens-like object in image.

    :param img: image to place lens
    :param mask: mask to fit lens
    :param color: color of the lens
    :param asEllipse: True to fit lens as a ellipse, False to fit circle.
    :param addmask: return additional mask parameter
    :return: image with simulated lens
    """
    # scaling operation
    sz = img.shape[:2] # get original image size
    pshape = mask.shape
    if sz != pshape:
        # make rescaling function: scaled point -to- original point function
        scalepoints = spoint2opointfunc(pshape,sz)
    else:
        scalepoints = lambda x: x # return the same points
    # find biggest area and contour
    cnt = thresh_biggestCnt(mask)
    # rescale contour to original image contour
    cnt2 = np.int32(scalepoints(cnt))

    mask_lens = np.ones(sz,dtype=np.uint8) # create mask
    if asEllipse:
        # get ellipse for original image to simulate lens
        drawEllipse(mask_lens,cnt2)
    else:
        # get circle for original image to simulate lens
        drawCircle(mask_lens,cnt2)

    # simulate lens
    img[mask_lens>0] = color # use mask to project black color over original image

    if addmask:
        return img, mask_lens
    return img

def simulateLens(img, threshfunc = None, pshape = (300, 300), color = 0, asEllipse=True):
    """
    Place lens-like object in image.

    :param img: image to place lens.
    :param threshfunc: function to segment retinal area and get its mask.
    :param pshape: shape to resize processing image to increase performance.
    :param color: color of the lens.
    :param asEllipse: True to fit lens as a ellipse, False to fit circle.
    :return: image with simulated lens.
    """
    # scaling operation
    if pshape is not None:
        img_resized = cv2.resize(img, pshape) # resize to scaled image
    else:
        img_resized = img
    # select threshold function
    if threshfunc is None:
        threshfunc = retinal_mask
    # segment retinal area
    segmented = threshfunc(img_resized)
    # fit lens
    return fitLens(img, segmented, color, asEllipse=asEllipse)


if __name__ == "__main__":
    pass