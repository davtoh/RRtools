# -*- coding: utf-8 -*-
"""
Interactive threshold and filter testing environment for image processing.
"""

import cv2
import numpy as np

from ..arrayops.basic import overlay, padVH, histogram
from ..arrayops.filters import sigmoid
from .viewer import Plotim

__all__ = ["Imtester"]


class Imtester(Plotim):
    """Used to test concepts such as thresholds and filters interactively."""

    def __init__(self, img, win="Imtester plot"):
        super(Imtester, self).__init__(win, img, (191, 191, 191))
        self.rW = 1080
        self.rH = 700
        self.showcoors = 0
        self.info = ""
        self.textbackground = (191, 191, 191, 200)
        self.errorbackground = (0, 0, 255, 150)
        self.data0 = img.copy()
        self.d = 9
        self.sigmaColor = self.sigmaSpace = 75
        self.ksize = (5, 5)
        self.filter = 5
        self.blocksz1 = self.blocksz2 = 11
        self.c1 = self.c2 = 30
        self.maximum = 255
        self.th1 = 255 // 2
        self.th2 = 0
        self.type1 = self.type2 = cv2.THRESH_BINARY
        self.binary = cv2.THRESH_BINARY
        self.binary_inv = cv2.THRESH_BINARY_INV
        self.tozero = cv2.THRESH_TOZERO
        self.tozero_inv = cv2.THRESH_TOZERO_INV
        self.trunc = cv2.THRESH_TRUNC
        self.showimg = self.showhist = self.portablehist = 0
        self.thtoggle1 = self.thtoggle2 = 1
        self.adaptoggle1 = self.adaptoggle2 = 0
        self.showchannel = 0
        self.alfa = self.beta = 10
        self.sigmoide = self.doubleth = 0
        self.operation = self.overlay = 1
        sz = self.data0.shape
        self.channels = ["channel gray"] if len(sz) == 2 else ["channel b", "channel g", "channel r", "channel c"]
        self.show()

    def windowfunc(self):
        cv2.namedWindow(self.win, self.wintype)
        cv2.resizeWindow(self.win, self.rW, self.rH)
        self.builtcmd()
        self.computefunc()

    def detectType(self, type, i="", info=""):
        mapping = {
            self.binary: ", type{}= binary".format(i),
            self.binary_inv: ", type{}= binary inv".format(i),
            self.tozero: ", type{}= tozero".format(i),
            self.tozero_inv: ", type{}= tozero inv".format(i),
            self.trunc: ", type{}= truncate".format(i),
        }
        return info + mapping.get(type, "")

    @staticmethod
    def applythresh(img, type, adaptativetoggle, threshtoggle, th, blocksz, c, i="", ti="", info="", title=""):
        if adaptativetoggle:
            method = cv2.ADAPTIVE_THRESH_MEAN_C if threshtoggle else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            prefix = ". Mean" if threshtoggle else ". Gauss"
            thresh = cv2.adaptiveThreshold(img, 1, method, type, blocksz, c)
            title += prefix + ti + ": "
            info += "blocksz{}= {}, c{}= {}".format(i, blocksz, i, c)
        else:
            flag = type if threshtoggle else (type + cv2.THRESH_OTSU)
            prefix = ". Normal" if threshtoggle else ". Otsu"
            thresh = cv2.threshold(img, th, 1, flag)[1]
            title += prefix + ti + ": "
            info += "th{}= {}".format(i, th)
        return thresh, info, title

    @staticmethod
    def formatinfo(info, words=9):
        infolist = info.split()
        newlist = [[""]]
        j = 1
        for i, word in enumerate(infolist):
            if i > words * j:
                j += 1
                newlist.append([""])
            newlist[-1][0] += word + " "
        return newlist

    def visualize(self):
        self.img = self.data.copy()
        self.sample = self.img
        temp = self.img.shape
        self.maxY, self.maxX = temp[0], temp[1]
        self.minY = self.minX = 0
        self.y = self.x = 0
        self.rx2, self.ry2 = self.maxX, self.maxY
        self.rx1 = self.ry1 = 0
        self.mousemoved = False
        self.event = self.flags = 0
        self.updaterenderer()

    def _prepare_plot_data(self, image, channel, th):
        if self.showhist and not self.portablehist:
            from ..image import fig2bgra
            return [fig2bgra(histogram(image, False))]
        data = []
        if self.showimg or th is None:
            data.append(image)
        if self.showchannel:
            data.append(channel)
        if th is not None:
            data.append(th)
        return data

    def updatevisualization(self, image, channel, th=None, items=None, thresh1=None, thresh2=None):
        if th is not None:
            th = th * self.maximum
            if self.overlay:
                image = overlay(image.copy(), th, alpha=th)
        data = self._prepare_plot_data(image, channel, th)
        if self.showhist and self.portablehist:
            from ..image import fig2bgra
            hst = fig2bgra(histogram(image, False))
            sz = int(hst.shape[1] / 2), int(hst.shape[0] / 2)
            resized_hst = [cv2.resize(hst, sz)]
            items = [resized_hst] if items is None else ([resized_hst] + items)
        self.controlText = items if items is not None else [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
        if thresh1 is not None and thresh2 is not None:
            self.data = padVH([data, [thresh1 * self.maximum, thresh2 * self.maximum]], 100, 100)[0]
        else:
            self.data = padVH([data], 100, 100)[0]
        self.visualize()

    def _get_threshold_cmdeval(self):
        return {
            "binary": ["if self.doubleth: self.type2=self.binary", "if not self.doubleth: self.type1=self.binary"],
            "binary inv": ["if self.doubleth: self.type2=self.binary_inv", "if not self.doubleth: self.type1=self.binary_inv"],
            "tozero": ["if self.doubleth: self.type2=self.tozero", "if not self.doubleth: self.type1=self.tozero"],
            "tozero inv": ["if self.doubleth: self.type2=self.tozero_inv", "if not self.doubleth: self.type1=self.tozero_inv"],
            "truncate": ["if self.doubleth: self.type2=self.trunc", "if not self.doubleth: self.type1=self.trunc"],
            "normal threshold": ["if self.doubleth: self.thtoggle2=1; self.adaptoggle2=0",
                                 "if not self.doubleth: self.thtoggle1=1; self.adaptoggle1=0"],
            "otsu threshold": ["if self.doubleth: self.thtoggle2=0; self.adaptoggle2=0",
                               "if not self.doubleth: self.thtoggle1=0; self.adaptoggle1=0"],
            "adaptive mean threshold": ["if self.doubleth: self.thtoggle2=1; self.adaptoggle2=1",
                                       "if not self.doubleth: self.thtoggle1=1; self.adaptoggle1=1"],
            "adaptive gaussian threshold": ["if self.doubleth: self.thtoggle2=0; self.adaptoggle2=1",
                                           "if not self.doubleth: self.thtoggle1=0; self.adaptoggle1=1"],
        }

    def builtcmd(self):
        cmdlist = ["limitrender", "limitaxes", "showcoors", "showcontrol", "showpixel", "staticcoors",
                   "showpixelvalue", "pixels", "showimg", "showhist", "adaptoggle1", "thtoggle1",
                   "thtoggle2", "portablehist", "cmdformatter", "doubleth", "operation", "overlay",
                   "sigmoide", "showchannel"]
        for i in cmdlist:
            if i not in self.cmdlist:
                self.cmdlist.append(i)
        cmdeval = {
            "pixels": ["showpixel", "showpixelvalue"],
            "+1th1": "self.th1=(1+self.th1)%256", "1th/2": "self.th1=(self.th1/2)%256", "-1th": "self.th1=(self.th1-1)%256",
            "+2th": "self.th2=(1+self.th2)%256", "2th/2": "self.th2=(self.th2/2)%256", "-2th": "self.th2=(self.th2-1)%256",
            "and": "self.operation=1", "or": "self.operation=0",
            "convolution": "self.filter=0", "gaussian filter": "self.filter=1", "blur": "self.filter=2",
            "median filter": "self.filter=3", "bilateral filter": "self.filter=4", "no filter": "self.filter=5",
            "end": ["self.computefunc()"],
        }
        cmdeval.update(self._get_threshold_cmdeval())
        self.cmdeval.update(cmdeval)
        sz = self.data0.shape
        if len(sz) == 2:
            self.channel = self.data0.copy()
            self.cmdeval[self.channels[0]] = "self.channel=self.data0.copy()"
        elif len(sz) == 3:
            self.channel = self.data0[:, :, 0].copy()
            for i in range(sz[2]):
                self.cmdeval[self.channels[i]] = "self.channel=self.data0[:,:," + str(i) + "].copy()"
        self.cmdlist.extend(list(self.cmdeval.keys()))

    def _apply_filter_flag(self, chimg):
        flags, ksize = self.filter, self.ksize
        if flags == 0 or isinstance(ksize, np.ndarray):
            k = (np.ones(ksize, np.float32) / self.d) if not isinstance(ksize, np.ndarray) else ksize
            return cv2.filter2D(chimg, -1, k), ". Convolution: ksize= " + str(tuple(k.shape))
        if flags == 1:
            return cv2.GaussianBlur(chimg, ksize, 0), ". Gaussian F: ksize= " + str(ksize)
        if flags == 2:
            return cv2.blur(chimg, ksize), ". Blur F: ksize= " + str(ksize)
        if flags == 3:
            return cv2.medianBlur(chimg, self.d), ". Median F: d= " + str(self.d)
        if flags == 4:
            res = cv2.bilateralFilter(chimg, self.d, self.sigmaColor, self.sigmaSpace)
            return res, ". Bilateral F: d= {}, sCol= {}, sSp= {}".format(self.d, self.sigmaColor, self.sigmaSpace)
        return chimg, ". No Filter"

    def computefunc(self, image=None):
        img = self.data0 if image is None else image
        sz = img.shape
        info = self.channels[0] if (len(sz) == 2 and np.array_equal(self.channel, img)) else "unknown channel"
        if len(sz) == 3:
            for i in range(sz[2]):
                if np.array_equal(self.channel, self.data0[:, :, i]):
                    info = self.channels[i]
                    break
        chimg = self.channel
        if self.sigmoide:
            info += ". Sigmoide: alfa= {}, beta= {}".format(self.alfa, self.beta)
            chimg = sigmoid(chimg.astype("float"), self.alfa, self.beta).astype("uint8")

        chimg, filter_info = self._apply_filter_flag(chimg)
        info += filter_info + ". Maximum: " + str(self.maximum)

        if self.doubleth:
            t1, i1, ti1 = self.applythresh(chimg, self.type1, self.adaptoggle1, self.thtoggle1, self.th1, self.blocksz1, self.c1, i="1", ti="1")
            t2, i2, ti2 = self.applythresh(chimg, self.type2, self.adaptoggle2, self.thtoggle2, self.th2, self.blocksz2, self.c2, i="2", ti="2")
            self.th = (t1 & t2) if self.operation else (t1 | t2)
            op = " AND" if self.operation else " OR"
            info += ti1 + self.detectType(self.type1, i="1", info=i1) + op + ti2[1:] + self.detectType(self.type2, i="2", info=i2)
            th1, th2 = t1, t2
        else:
            self.th, i1, ti1 = self.applythresh(chimg, self.type1, self.adaptoggle1, self.thtoggle1, self.th1, self.blocksz1, self.c1, i="1", ti="1")
            info += ti1 + self.detectType(self.type1, i="1", info=i1)
            th1 = th2 = None

        self.info = info
        items = [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
        items.extend(self.formatinfo(info, words=5))
        self.updatevisualization(img, chimg, self.th, items, th1, th2)

