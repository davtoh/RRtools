# -*- coding: utf-8 -*-
r"""
Coordinate representations, geometric analytics, and coordinate drawing utilities.
"""
from __future__ import division
import cv2
import numpy as np
from ..config import FLOAT, INT
from ..cache import cache
from ..arrayops.basic import polygonArea, angle, vectorsAngles, overlay
from ..plotter import limitaxis
from .conversions import black, red, green


class ImCoors(object):
    """
    Image's coordinates class.
    Example::

        a = ImCoors(np.array([(116, 161), (295, 96), (122, 336), (291, 286)]))
        print a.__dict__
        print "mean depend on min and max: ", a.mean
        print a.__dict__
        print "after mean max has been already been calculated: ", a.max
        a.data = np.array([(116, 161), (295, 96)])
        print a.__dict__
        print "mean and all its dependencies are processed again: ", a.mean
    """

    def __init__(self, pts, dtype=FLOAT, deg=False):
        """
        Initializes ImCoors.

        :param pts: list of points
        :param dtype: return data as dtype. Default is config.FLOAT
        """
        self._pts = pts  # supports bigger numbers
        self._dtype = dtype
        self._deg = deg

    @property
    def pts(self):
        return self._pts

    @pts.setter
    def pts(self, value):
        getattr(self, "__dict__").clear()
        self._pts = value

    @pts.deleter
    def pts(self):
        raise Exception("Cannot delete attribute")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        getattr(self, "__dict__").clear()
        self._dtype = value

    @dtype.deleter
    def dtype(self):
        raise Exception("Cannot delete attribute")

    # DATA METHODS

    def __len__(self):
        return len(self._pts)

    @cache
    def max(self):
        """
        Maximum in each axis.

        :return: x_max, y_max
        """
        return tuple(np.max(self._pts, 0))

    @cache
    def min(self):
        """
        Minimum in each axis.

        :return: x_min, y_min
        """
        return tuple(np.min(self._pts, 0))

    @cache
    def rectbox(self):
        """
        Rectangular box enclosing points (origin and end point or rectangle).

        :return: (x0,y0),(x,y)
        """
        return (self.min, self.max)

    @cache
    def boundingRect(self):
        """
        Rectangular box dimensions enclosing points.

        example::

            P = np.ones((400,400))
            a = ImCoors(np.array([(116, 161), (295, 96), (122, 336), (291, 286)]))
            x0,y0,w,h = a.boundingRect
            P[y0:y0+h,x0:x0+w] = 0

        :return: x0,y0,w,h
        """
        return cv2.boundingRect(self._pts)

    @cache
    def minAreaRect(self):
        return cv2.minAreaRect(self._pts)

    @cache
    def rotatedBox(self):
        """
        Rotated rectangular box enclosing points.

        :return: 4 points.
        """
        try:  # opencv 2
            return self._dtype(cv2.cv.BoxPoints(self.minAreaRect))
        except AttributeError:  # opencv 3
            return self._dtype(cv2.boxPoints(self.minAreaRect))

    @cache
    def boxCenter(self):
        """
        Mean in each axis.

        :return: x_mean, y_mean
        """
        xX, xY = self.max
        nX, nY = self.min
        return tuple(self._dtype((xX + nX, xY + nY)) / 2)

    @cache
    def mean(self):
        """
        Center or mean.
        :return: x,y
        """
        return tuple(np.mean(self._pts, 0, dtype=self._dtype))

    center = mean

    @cache
    def area(self):
        """
        Area of points.

        :return: area number
        """
        return polygonArea(self._pts)

    @cache
    def rectangularArea(self):
        """
        Area of rectangle enclosing points aligned with x,y axes.

        :return: area number
        """
        (x0, y0), (x, y) = self.rectbox
        return self.dtype(np.abs((x - x0) * (y - y0)))

    @cache
    def rotatedRectangularArea(self):
        """
        Area of Rotated rectangle enclosing points.

        :return: area number
        """
        return polygonArea(self.rotatedBox)

    @cache
    def rectangularity(self):
        """
        Ratio that represent a perfect square aligned with x,y axes.

        :return: ratio from 1 to 0, 1 representing a perfect rectangle.
        """
        return self.dtype(self.area / self.rectangularArea)

    @cache
    def rotatedRectangularity(self):
        """
        Ratio that represent a perfect rotated square fitting points.

        :return: ratio from 1 to 0, 1 representing a perfect rotated rectangle.
        """
        if self.area < 1:
            area = 0  # needed to prevent false values
        else:
            area = self.area
        return self.dtype(area / self.rotatedRectangularArea)

    @cache
    def regularity(self):
        """
        Ratio of forms with similar measurements and angles.
        e.g. squares and rectangles have rect angles so they are regular.
        For regularity object must give 1.

        :return:
        """
        pi = angle((1, 0), (0, 1), deg=self._deg)
        av = self.vertexesAngles
        return pi * (len(av)) / np.sum(av)  # pi*number_angles/sum_angles

    @cache
    def relativeVectors(self):
        """
        Form vectors from points.

        :return: array of vectors [V0, ... , (V[n] = x[n+1]-x[n],y[n+1]-y[n])].
        """
        pts = np.array(self._pts)
        # adds last vector from last and first point.
        pts = np.append(pts, [pts[0]], axis=0)
        return np.stack([np.diff(pts[:, 0]), np.diff(pts[:, 1])], 1)

    @cache
    def vertexesAngles(self):
        """
        Relative angle of vectors formed by vertexes.

        i.e. angle between vectors "v01" formed by points "p0-p1" and "v12"
        formed by points "p1-p2" where "p1" is seen as a vertex (where vectors cross).

        :return: angles.
        """
        vs = self.relativeVectors  # get all vectors from points.
        # add last vector to first position
        vs = np.roll(np.append(vs, [vs[-1]], axis=0), 2)
        return np.array([angle(vs[i - 1], vs[i], deg=self._deg)
                         for i in range(1, len(vs))], self._dtype)  # calculate angles

    @cache
    def pointsAngles(self):
        """
        Angle of vectors formed by points in Cartesian plane with respect to x axis.

        i.e. angle between vector "v01" (formed by points "p0-p1") and vector unity in axis x.

        :return: angles.
        """
        vs = self.relativeVectors  # get all vectors from points.
        return vectorsAngles(pts=vs, dtype=self._dtype, deg=self._deg)

    @cache
    def vectorsAngles(self):
        """
        Angle of vectors in Cartesian plane with respect to x axis.

        i.e. angle between vector "v0" (formed by point "p0" and the origin) and vector unity in axis x.
        :return: angles.
        """
        return np.array([angle((1, 0), i, deg=self._deg) for i in self._pts], self._dtype)


def drawcoorpoints(vis, points, col_out=black, col_in=red, radius=2):
    """
    Function to draw points.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points, INT)
    radius_in = radius - 1
    for x, y in points:
        cv2.circle(vis, (x, y), radius, col_out, -1)
        cv2.circle(vis, (x, y), radius_in, col_in, -1)
    return vis


def myline(img, pt1, pt2, color, thickness=None):
    """
    Function to draw points (experimental).

    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :return:
    """
    # y=m*x+b
    x1, y1 = np.array(pt1, dtype=FLOAT)
    x2, y2 = np.array(pt2, dtype=FLOAT)
    m = (y2 - y1) / (x2 - x1)
    xmin, xmax = np.sort([x1, x2])
    xvect = np.arange(xmin, xmax + 1).astype('int')
    yvect = np.array(xvect * m + int(y1 - x1 * m), dtype=int)
    for i in zip(yvect, xvect):
        img[i] = color


def drawcooraxes(vis, points, col_out=black, col_in=green, radius=2):
    """
    Function to draw axes instead of points.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points, INT)
    thickness = radius - 1
    h1, w1 = vis.shape[:2]  # obtaining image dimensions
    for i in points:
        h1pt1 = (0, i[1])
        h1pt2 = (w1, i[1])
        w2pt1 = (i[0], 0)
        w2pt2 = (i[0], h1)
        cv2.line(vis, h1pt1, h1pt2, col_in, thickness)
        cv2.line(vis, w2pt1, w2pt2, col_in, thickness)
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    return vis


def drawcoorpolyline(vis, points, col_out=black, col_in=red, radius=2):
    """
    Function to draw interaction with points to obtain polygonal.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    thickness = radius - 1
    if len(points) > 1:
        points = np.array(points, INT)
        cv2.polylines(vis, [points], False, col_in, thickness)
    else:
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    return vis


def drawcoorarea(vis, points, col_out=black, col_in=red, radius=2):
    """
    Function to draw interaction with points to obtain area.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    if len(points) > 2:
        mask = np.zeros(vis.shape[:2])
        cv2.drawContours(mask, [np.array(points, np.int32)], 0, 1, -1)
        vis = overlay(vis, np.array(
            [(0,) * len(col_in), col_in])[mask.astype(int)], alpha=mask * 0.5)
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    else:
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    return vis


def drawcoorpolyArrow(vis, points, col_out=black, col_in=red, radius=2):
    """
    Function to draw interaction with vectors to obtain polygonal.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points, INT)
    thickness = radius - 1
    if len(points) > 1:
        for i in range(len(points) - 1):
            pt1 = (points[i][0], points[i][1])
            pt2 = (points[i + 1][0], points[i + 1][1])
            cv2.arrowedLine(vis, pt1, pt2, col_in, thickness)
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    else:
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    return vis


def drawcoorperspective(vis, points, col_out=black, col_in=red, radius=2):
    """
    Function to draw interaction with points to obtain perspective.

    :param vis: image array.
    :param points: list of points.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    :param radius: radius of drawn points.
    :return:
    """
    points = np.array(points, INT)
    thickness = radius - 1
    if len(points) > 1 and len(points) < 5:
        for i in range(len(points) - 1):
            if i % 2:
                for j in range(i + 1, min(len(points), i + 3)):
                    if j % 2:
                        pt1 = (points[i][0], points[i][1])
                        pt2 = (points[j][0], points[j][1])
                        cv2.arrowedLine(vis, pt1, pt2, col_in, thickness)
            else:
                for j in range(i + 1, min(len(points), i + 3)):
                    pt1 = (points[i][0], points[i][1])
                    pt2 = (points[j][0], points[j][1])
                    cv2.arrowedLine(vis, pt1, pt2, col_in, thickness)
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    else:
        vis = drawcoorpoints(vis, points, col_out, col_in, radius)
    return vis


def limitaxispoints(c, maxc, minc=0):
    """
    Limit a point in axis.

    :param c: list of points.
    :param maxc: maximum value of point.
    :param minc: minimum value of point.
    :return: return limited points.
    """
    x = np.zeros(len(c), dtype=int)
    for i, j in enumerate(c):
        x[i] = limitaxis(j, maxc, minc)
    return tuple(x)
