# -*- coding: utf-8 -*-
"""
Interactive Canny edge tuner with trackbars, filtering, and equalization.
"""

import cv2
from ..arrayops.filters import bilateralFilter
from .viewer import Plotim

__all__ = ["Edger"]


class Edger(Plotim):
    """
    Test visualization for edge detection tuning.

    self.edge -> the edges in processed image
    self.img -> the processed image
    self.sample -> the rendered processed image
    """

    def __init__(self, img, isSIZE=True, isEQUA=False, isCLAHE=False, isBFILTER=False):
        self.load(img, False)
        super(Edger, self).__init__("Edger " + self.pathdata[2], self.data2)
        self._th1 = 3000
        self._th2 = 6000
        self._maxth = 10000
        self._showgray = False
        self._isSIZE = isSIZE
        self._isEQUA = isEQUA
        self._isCLAHE = isCLAHE
        self._isBFILTER = isBFILTER
        self._size = (400, 400)
        self.edgecolor = (0, 255, 0)
        self.initname = "edge_"
        self.cmdlist.extend(["isSIZE", "isEQUA", "isBFILTER", "isCLAHE", "showgray"])
        self.d, self.sigmaColor, self.sigmaSpace = 10, 20, 20
        self.clipLimit, self.tileGridSize = 2.0, (8, 8)
        self.apertureSize, self.L2gradient = 7, True
        self.computeAll()

    def getParameters(self, params=("d", "sigmaColor", "sigmaSpace", "clipLimit", "tileGridSize",
                                    "isSIZE", "isEQUA", "isCLAHE", "isBFILTER", "th1", "th2", "size",
                                    "apertureSize", "L2gradient")):
        if isinstance(params, (str, bytes)):
            params = (params,)
        return {param: getattr(self, param) for param in params}

    def save(self, strname=None, ext=".png", name="img"):
        if not strname:
            temp = "E{}_{}_{}_{}_".format(self.th1, self.th2, self.apertureSize, self.L2gradient)
            name_parts = self.pathdata[:2] + [self.initname] + self.savename + [temp] + self.pathdata[2:]
            strname = "".join(name_parts)
        return super(Edger, self).save(strname, ext, name)

    def load(self, img, compute=True):
        if isinstance(img, (str, bytes)):
            from ..directory import getData
            self.pathdata = getData(img)
            self.data2 = cv2.imread(img)
        else:
            self.pathdata = ["", "", "img", ".png"]
            self.data2 = img
        if compute:
            self.computeAll()

    def _apply_filters(self, gray):
        """Apply EQUA, CLAHE, and Bilateral filtering to grayscale image."""
        if self.isEQUA:
            gray = cv2.equalizeHist(gray)
            self.savename.append("EQUA_")
        if self.isCLAHE:
            clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
            gray = clahe.apply(gray)
            self.savename.append("CLAHE{}_{}_{}_".format(self.clipLimit, self.tileGridSize[0], self.tileGridSize[1]))
        if self.isBFILTER:
            gray = bilateralFilter(gray, self.d, self.sigmaColor, self.sigmaSpace)
            self.savename.append("FILTER{}_{}_{}_".format(self.d, self.sigmaColor, self.sigmaSpace))
        return gray

    def computeAll(self):
        """Recompute preprocessing pipeline and edge extraction."""
        self.savename = [""]
        if self.isSIZE:
            img = cv2.resize(self.data2.copy(), self._size)
            self.savename.append("SIZE{}_{}_".format(self._size[0], self._size[1]))
        else:
            img = self.data2.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.bgr = img
        self.gray = self._apply_filters(gray)
        if self.showgray:
            self.sample = self.gray.copy()
            self.data = cv2.cvtColor(self.sample, cv2.COLOR_GRAY2BGR)
        else:
            self.sample = self.bgr.copy()
            self.data = self.sample
        self.init()
        try:
            self.computeEdge()
        except Exception:
            pass

    @property
    def size(self):
        return self.gray.shape

    @size.setter
    def size(self, value):
        if bool(self._size != value):
            self._size = value
            if self.isSIZE:
                self.computeAll()

    @size.deleter
    def size(self):
        del self._size

    @property
    def showgray(self):
        return self._showgray

    @showgray.setter
    def showgray(self, value):
        if bool(self._showgray != value):
            self._showgray = value
            if self._showgray:
                self.sample = self.gray.copy()
                self.data = cv2.cvtColor(self.sample, cv2.COLOR_GRAY2BGR)
            else:
                self.sample = self.bgr.copy()
                self.data = self.sample
            self.init()
            try:
                self.computeEdge()
            except Exception:
                pass

    @showgray.deleter
    def showgray(self):
        del self._showgray

    @property
    def isSIZE(self):
        return self._isSIZE

    @isSIZE.setter
    def isSIZE(self, value):
        if bool(self._isSIZE != value):
            self._isSIZE = value
            self.computeAll()

    @isSIZE.deleter
    def isSIZE(self):
        del self._isSIZE

    @property
    def isEQUA(self):
        return self._isEQUA

    @isEQUA.setter
    def isEQUA(self, value):
        if bool(self._isEQUA != value):
            self._isEQUA = value
            self.computeAll()

    @isEQUA.deleter
    def isEQUA(self):
        del self._isEQUA

    @property
    def isCLAHE(self):
        return self._isCLAHE

    @isCLAHE.setter
    def isCLAHE(self, value):
        if bool(self._isCLAHE != value):
            self._isCLAHE = value
            self.computeAll()

    @isCLAHE.deleter
    def isCLAHE(self):
        del self._isCLAHE

    @property
    def isBFILTER(self):
        return self._isBFILTER

    @isBFILTER.setter
    def isBFILTER(self, value):
        if bool(self._isBFILTER != value):
            self._isBFILTER = value
            self.computeAll()

    @isBFILTER.deleter
    def isBFILTER(self):
        del self._isBFILTER

    @property
    def th1(self):
        return self._th1

    @th1.setter
    def th1(self, value):
        if bool(self._th1 != value):
            self._th1 = value
            if self.isActiveWindow():
                cv2.createTrackbar('th1', self.win, value, self.maxth, self.onTrackbar1)
                self.computeEdge()

    @th1.deleter
    def th1(self):
        del self._th1

    @property
    def th2(self):
        return self._th2

    @th2.setter
    def th2(self, value):
        if bool(self._th2 != value):
            self._th2 = value
            if self.isActiveWindow():
                cv2.createTrackbar('th2', self.win, value, self.maxth, self.onTrackbar2)
                self.computeEdge()

    @th2.deleter
    def th2(self):
        del self._th2

    @property
    def maxth(self):
        return self._maxth

    @maxth.setter
    def maxth(self, value):
        if bool(self._maxth != value):
            self._maxth = value
            if self.isActiveWindow():
                self.th1 = min(self.th1, value)
                self.th2 = min(self.th2, value)
                cv2.createTrackbar('th1', self.win, self.th1, self._maxth, self.onTrackbar1)
                cv2.createTrackbar('th2', self.win, self.th2, self._maxth, self.onTrackbar2)
                self.computeEdge()

    @maxth.deleter
    def maxth(self):
        del self._maxth

    def computeEdge(self):
        """Compute Canny edges and overlay on display image."""
        edge = cv2.Canny(self.gray, self.th1, self.th2,
                         apertureSize=self.apertureSize, L2gradient=self.L2gradient)
        vis = self.data.copy()
        vis[edge != 0] = self.edgecolor
        self.edge = edge
        self.img = vis
        self.updaterenderer()
        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[int(self.y), int(self.x)])

    def onTrackbar1(self, *args):
        self.th1 = cv2.getTrackbarPos('th1', self.win)

    def onTrackbar2(self, *args):
        self.th2 = cv2.getTrackbarPos('th2', self.win)

    def isActiveWindow(self):
        try:
            cv2.getTrackbarPos('th1', self.win)
            return True
        except Exception:
            return False

    def windowfunc(self):
        cv2.namedWindow(self.win, self.wintype)
        cv2.createTrackbar('th1', self.win, self.th1, self.maxth, self.onTrackbar1)
        cv2.createTrackbar('th2', self.win, self.th2, self.maxth, self.onTrackbar2)
        cv2.resizeWindow(self.win, self.rW, self.rH)
        self.computeEdge()

