# -*- coding: utf-8 -*-
"""
Contour analysis, polygon area calculations, and mask generation operations.
"""

import cv2
import numpy as np

from .canvas import isnumpy
from .geometry import standarizePoints


def findContours(*args, **kwargs):
    """
    Compatibility wrapper around cv2.findContour to support both openCV 2 and
    openCV 3.

    findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
    """
    try:
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    except ValueError:  # opencv >= 3.2
        image, contours, hierarchy = cv2.findContours(*args, **kwargs)
    return contours, hierarchy


def contoursArea(contours):
    """
    Accumulates areas from list of contours.

    :param contours: list of contours or binary array.
    :return: area.
    """
    if isnumpy(contours):
        contours, _ = findContours(
            contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0  # accumulate area
    for cnt in contours:
        area += cv2.contourArea(cnt.astype(np.int32))  # get each contour area
    return area


def points2mask(pts, shape=None, astype=bool):
    """
    Creates an array with the filled polygon formed by points.

    :param pts: points.
    :param shape: (None) shape of array. If None it creates an array fitted to points.
    :param astype: (bool) numpy type
    :return: array.

    Example::

        pts = random_points([(-100, 100), (-100, 100)])
        img = points2mask(pts)
        Plotim("filled",img).show()
    """
    if shape is None:
        pts = pts - pts.min(0)  # shift
        xmax, ymax = pts.max(0)
        array = np.zeros((ymax, xmax), astype)
    else:
        array = np.zeros(shape[:2], astype)
    # cv2.drawContours(array,[pts.astype(np.int32)],0,1,-1)
    cv2.fillConvexPoly(array, pts.astype(np.int32), 1, 0)
    return array


def contours2mask(contours, shape=None, astype=bool):
    """
    Creates an array with filled polygons formed by contours.

    :param contours: list of contour or points forming objects
    :param shape: (None) shape of array. If None it creates an array fitted to contours.
    :param astype: (bool) numpy type
    :return:
    """
    ncoors = []  # adequate contours
    # calculate variables
    minx, miny = np.inf, np.inf
    maxx, maxy = -np.inf, -np.inf
    for coors in contours:
        coors = standarizePoints(coors, aslist=False).astype(np.int32)
        if coors.size != 0:
            ncoors.append(coors)
            # calculate variables for shape
            if shape is None:
                pminx, pminy = coors.min(0)
                pmaxx, pmaxy = coors.max(0)
                if pminx < minx:
                    minx = pminx
                if pminy < miny:
                    miny = pminy
                if pmaxx > maxx:
                    maxx = pmaxx
                if pmaxy > maxy:
                    maxy = pmaxy
    if shape is None:
        minv = np.array([(minx, miny)])
        ncoors = [c - minv for c in ncoors]  # shift
        mask = np.zeros((maxy, maxx))
    else:
        mask = np.zeros(shape[:2])
    # for coors in contours:
    #    mask = np.logical_or(mask, points2mask(coors, shape))
    cv2.drawContours(mask, ncoors, -1, 1, -1)
    return mask.astype(astype)


def polygonArea_contour(pts):
    """
    Area of points using contours.

    :param pts: points.
    :return: area value.

    ..note:: if polygon is incomplete (last is not first point) it completes the array.
    """
    return contoursArea([pts])


def polygonArea_fill(pts):
    """
    Area of points using filled polygon and pixel count.

    :param pts: points.
    :return: area value.

    ..note:: if polygon is incomplete (last is not first point) it completes the array.
    """
    return points2mask(pts).sum()


def polygonArea_calcule(pts):
    """
    Area of points calculating polygon Area.

    :param pts: points.
    :return: area value.

    ..note::
        - If polygon is incomplete (last is not first point) it completes the array.
        - If the polygon crosses over itself the algorithm will fail.
        - Based on http://www.mathopenref.com/coordpolygonarea.html
    """
    pts = standarizePoints(pts)
    if (pts[0] == pts[-1]).all():  # test if last is the first (closed points)
        x, y = pts[:, 0], pts[:, 1]
    else:  # complete the points (last point must be the first)
        x = pts[list(range(len(pts))) + [0], 0]
        y = pts[list(range(len(pts))) + [0], 1]
    xmul = x[:] * np.roll(y, -1)  # shifted multiplication in axis x
    ymul = y[:] * np.roll(x, -1)  # shifted multiplication in axis y
    return np.abs(np.sum(xmul) - np.sum(ymul)) / 2.  # summation and absolute area


polygonArea = polygonArea_calcule


def convexityRatio(cnt, hull=None):
    """
    Ratio to test if contours are irregular

    :param cnt: contour
    :param hull:(None) convex hull
    :return: ratio
    """
    if hull is None:
        hull = cv2.convexHull(cnt)  # get convex hull
    if len(hull.shape) == 2:  # convert back from indexes
        from .convert import points2contour
        hull = points2contour(cnt[hull])
    ahull = cv2.contourArea(hull)
    acnt = cv2.contourArea(cnt)
    if ahull == 0:  # solves issue dividing by 0
        ahull = 0.0000001
    return acnt / ahull  # contour area / hull area

