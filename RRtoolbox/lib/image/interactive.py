# -*- coding: utf-8 -*-
r"""
Interactive coordinate selectors, window handlers, and geometric grouping functions.
"""
from __future__ import division
import cv2
import numpy as np
from ..arrayops.basic import anorm, standarizePoints
from ..plotter import Plotim
from .coordinates import (ImCoors, drawcoorpoints, drawcooraxes, drawcoorpolyline,
                          drawcoorperspective, limitaxispoints)
from .conversions import black, red, green


class GetCoors(Plotim):
    """
    Create window to select points from image.

    :param im: image to get points.
    :param win: window name.
    :param updatefunc: function to draw interaction with points.
            (e.g. limitaxispoints, drawcoorperspective, etc.).
    :param prox: proximity to identify point.
    :param radius: radius of drawn points.
    :param unique: If True no point can be repeated,
            else selected points can be repeated.
    :param col_out: outer color of point.
    :param col_in: inner color of point.
    """

    def __init__(self, im, win="get coordinates", updatefunc=drawcoorpoints,
                 unique=True, col_out=black, col_in=red):
        super(GetCoors, self).__init__(win, im)
        self.updatefunc = updatefunc
        self.userupdatefunc = updatefunc
        self.prox = 8  # proximity to keypoint
        self.radius = 3
        self.unique = unique
        self.col_out = col_out
        self.col_in = col_in
        self.interpolation = cv2.INTER_AREA
        self._coors = []
        self.rcoors = []  # rendered coordinates
        self.coorlen = 0
        self.showstats = False
        self.mapdata2 = [None, None, None]
        self.data2 = np.zeros((self.rH, self.rW, 1), dtype=np.uint8)
        self.drawcooraxes = drawcooraxes
        self.drawcoorperspective = drawcoorperspective
        self.drawcoorpolyline = drawcoorpolyline
        self.drawcoorpoints = drawcoorpoints
        self.controlText[0].extend([" No. coordinates: {self.coorlen}. "])
        self.cmdeval.update({"points": "self.updatefunc = self.drawcoorpoints",
                             "polyline": "self.updatefunc = self.drawcoorpolyline",
                             "perspective": "self.updatefunc = self.drawcoorperspective",
                             "axes": "self.updatefunc = self.drawcooraxes",
                             "user": "self.updatefunc = self.userupdatefunc",
                             "end": ["self.updatecoors()", "self.mousefunc()"]})
        self.cmdlist.extend(["unique", "showstats", "user", "points",
                             "polyline", "perspective", "axes"])

    @property
    def coors(self):
        return self._coors

    @coors.setter
    def coors(self, value):
        self._coors = value
        self.updatecoors()

    @coors.deleter
    def coors(self):
        self._coors = []
        self.updatecoors()

    def drawstats(self, points, col_out=black, col_in=green, radius=2):
        vis = self.rimg
        p = ImCoors(points)
        self.data2 = np.zeros((vis.shape[0], vis.shape[1], 1), dtype=np.uint8)
        drawcooraxes(vis, [p.boxCenter], col_out, col_in, radius)
        drawcooraxes(self.data2, [p.boxCenter], 1, 1, self.prox)
        drawcooraxes(vis, [p.mean], col_in, col_out, radius)
        drawcooraxes(self.data2, [p.mean], 2, 2, self.prox)
        p1 = ImCoors(self.coors)
        self.mapdata2 = [None, "center at " +
                         str(p1.boxCenter), "mean at " + str(p1.mean)]

    def updatecoors(self):
        self.coorlen = len(self.coors)
        self.updaterenderer()
        if self.coors != []:
            self.rcoors = self.coors[:]
            newc = self.coors[:]
            for j, i in enumerate(self.coors):
                newc[j] = self.real2render(i[0], i[1])
                self.rcoors[j] = limitaxispoints(newc[j], 10000, -10000)
            if self.showstats:
                self.drawstats(newc, radius=self.radius)
            self.rimg = self.updatefunc(
                self.rimg, newc, self.col_out, self.col_in, self.radius)
        else:
            self.data2[:] = 0
            self.coordinateText = [["xy({self.x},{self.y})"]]

    def mousefunc(self):
        controlled = self.builtincontrol()
        drawed = False
        isnear = False
        idx = None

        if self.coors != [] and self.rx is not None and self.ry is not None:
            vals = anorm(np.int32(self.rcoors) - (self.rx, self.ry))
            near_point = np.logical_and(vals < self.prox, vals == np.min(vals))
            if np.any(near_point):
                idx = np.where(near_point)[0]
                isnear = True
                val = self.coors[idx[0]]
                count = self.coors.count(val)
                self.coordinateText = [
                    ["point " + str(idx[0]) + " at " + str(val) + "x" + str(count)]]
            else:
                self.coordinateText = [["xy({self.x},{self.y})"]]

        if not controlled and bool(self.flags):
            if self.event == cv2.EVENT_RBUTTONDBLCLK:
                self.coors = []
                self.img = self.data.copy()
                drawed = True
            elif isnear and self.event == cv2.EVENT_RBUTTONDOWN and idx is not None:
                self.coors.pop(idx[0])
                drawed = True
            elif self.event == cv2.EVENT_LBUTTONDOWN:
                val = (self.x, self.y)
                if not self.coors.count(val) or not self.unique:
                    self.coors.append(val)
                drawed = True

        if controlled or drawed:
            self.updatecoors()

        if self.y is not None and self.x is not None:
            if self.showstats:
                data = self.mapdata2[self.data2[self.ry, self.rx]]
                if not isnear and data is not None:
                    self.coordinateText = [[data]]
            self.builtinplot(self.data[int(self.y), int(self.x)])


def getcoors(im, win="get coordinates", updatefunc=drawcoorpoints, coors=None,
             prox=8, radius=3, unique=True, col_out=black, col_in=red):
    self = GetCoors(im, win, updatefunc, unique=unique,
                    col_out=col_out, col_in=col_in)
    self.radius = radius
    self.prox = prox
    if coors is not None:
        self.coors = standarizePoints(coors, aslist=True)
    self.show(clean=False)
    coors = self.coors
    self.clean()
    return coors


def separe(values, sep, axis=0):
    """
    Separate values from separator or threshold.

    :param values: list of values
    :param sep: separator value
    :param axis: axis in each value
    :return: lists of greater values, list of lesser values
    """
    greater, lesser = [], []
    for i in values:
        if i[axis] > sep:
            greater.append(i)
        else:
            lesser.append(i)
    return greater, lesser


def getrectcoors(*data):
    """
    Get ordered points.

    :param data: list of points
    :return: [Top_left,Top_right,Bottom_left,Bottom_right]
    """
    if len(data) == 1:
        points = data[0]
    else:
        points = getcoors(*data)

    p = ImCoors(points)
    min_x, min_y = p.min
    max_x, max_y = p.max
    Top_left = (min_x, min_y)
    Top_right = (max_x, min_y)
    Bottom_left = (min_x, max_y)
    Bottom_right = (max_x, max_y)
    return [Top_left, Top_right, Bottom_left, Bottom_right]


def quadrants(points):
    """
    Separate points respect to center of gravity point.

    :param points: list of points
    :return: [[Top_left],[Top_right],[Bottom_left],[Bottom_right]]
    """
    p = ImCoors(points)
    mean_x, mean_y = p.mean
    Bottom, Top = separe(points, mean_y, axis=1)
    Top_right, Top_left = separe(Top, mean_x, axis=0)
    Bottom_right, Bottom_left = separe(Bottom, mean_x, axis=0)
    return [Top_left, Top_right, Bottom_left, Bottom_right]


def getgeometrycoors(*data):
    """
    Get filled object coordinates.
    """
    if len(data) == 1:
        points = data[0]
    else:
        points = getcoors(*data)
    return points


def random_color(channels=1, min=0, max=256):
    """
    Random color.

    :param channels: number of channels
    :param min: min color in any channel
    :param max: max color in any channel
    :return: random color
    """
    return [np.random.randint(min, max) for i in range(channels)]

