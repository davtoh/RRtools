# -*- coding: utf-8 -*-
from __future__ import division
from past.builtins import basestring
from past.utils import old_div
import cv2
import numpy as np
from ..lib.arrayops import entroyTest
from ..lib.config import FLOAT
from ..lib.directory import getData
from ..lib.plotter import Plotim, limitaxis


try:  # opencv 2
    hist_map = dict((("correlation", (cv2.cv.CV_COMP_CORREL, True)),
                     ("chi-squared", (cv2.cv.CV_COMP_CHISQR, False)),
                     ("intersection", (cv2.cv.CV_COMP_INTERSECT, True)),
                     ("hellinger", (cv2.cv.CV_COMP_BHATTACHARYYA, False))))
except AttributeError:  # opencv 3
    hist_map = dict((("correlation", (cv2.HISTCMP_CORREL, True)),
                     ("chi-squared", (cv2.HISTCMP_CHISQR, False)),
                     ("intersection", (cv2.HISTCMP_INTERSECT, True)),
                     ("hellinger", (cv2.HISTCMP_BHATTACHARYYA, False))))


def entropy(imlist, loadfunc=None, invert=False):
    """
    Entropy function modified from:

    Yan Liu, Feihong Yu, An automatic image fusion algorithm for unregistered multiply multi-focus images,
    Optics Communications, Volume 341, 15 April 2015, Pages 101-113, ISSN 0030-4018,
    http://dx.doi.org/10.1016/j.optcom.2014.12.015.
    (http://www.sciencedirect.com/science/article/pii/S0030401814011559)

    :param imlist: list of path to images or arrays
    :return: sortedD,sortedImlist,D,fns

    where sortedD is the ranking of the Entropy test, D = [D0,...,DN] D0>DN
          sortedImlist is fns sorted to match sortedD,
          D is the list of the absolute difference between entropy and the root mean square, D = ||E-RMS||
    """
    # assert(len(fns)>=2) # no images to compare. There must be 2 or more
    E = np.zeros(len(imlist), FLOAT)  # pre-allocate array
    for num, im in enumerate(imlist):  # for each image get data

        if isinstance(im, basestring):
            if loadfunc is None:
                def loadfunc(im):
                    return cv2.imread(im, 0)
            gray_img = loadfunc(im)  # read gray image
        else:
            gray_img = im
        E[num] = entroyTest(gray_img)

    RMS = np.sqrt(old_div(np.sum(E**2), len(imlist)))  # get root mean square
    D = np.abs(E - RMS)  # absolute difference error
    sortedD = np.sort(D)  # sort errors # order from minor to greater
    if invert:
        sortedD = sortedD[::-1]  # order from greater to minor
    sortedImlist = [imlist[np.where(D == s)[0]]
                    for s in sortedD]  # get sorted images
    return sortedD, sortedImlist, D, imlist


def hist_comp(imlist, loadfunc=None, method="correlation"):
    """
    Histogram comparison

    :param imlist: list of path to images or arrays
    :return: comparison
    """
    # http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # assert(len(fns)>=2) # no images to compare. There must be 2 or more
    def helper(im, loadfunc=loadfunc):
        if isinstance(im, basestring):
            if loadfunc is None:
                def loadfunc(im):
                    return cv2.imread(im)
            im = loadfunc(im)  # read BGR image
        return im
    method, reverse = hist_map[method]
    comp, comparison = None, []
    for i, im in enumerate(imlist):  # for each image get data
        hist = cv2.calcHist([helper(im)], [0, 1, 2], None, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        if i == 0:
            comp = hist
        comparison.append((cv2.compareHist(comp, hist, method), im))

    comparison.sort(key=lambda x: x[0], reverse=reverse)  # sort comparisons
    return comparison


class EntropyPlot(Plotim):
    """
    Plot entropy test
    """

    def __init__(self, images, win="Entropy tests", func=None):
        if func is None:
            def loadfunc(im, size=(400, 400)):
                return cv2.resize(cv2.imread(im, 0), size)
            func = loadfunc
        self.sortedD, self.sortedImages, self.D, self.images = entropy(images)
        self.loadfunc = func
        self.selectlist(self.sortedImages)
        super(EntropyPlot, self).__init__(
            win, self.getImage(self.imlist[self.index]))
        self.controlText.append(self.getData(self.imlist[self.index]))
        self.init()

    def getImage(self, im):
        if isinstance(im, basestring):  # list is reference to image
            return self.loadfunc(im)
        elif type(im) in (float, int):  # is entropy value
            return self.getImage(self.sortedImages[self.sortedD.index(im)])
        else:  # list is image itself
            return im

    def getData(self, im):
        if isinstance(im, basestring):  # list is reference to image
            name = "".join(getData(im)[2:])
            number = str(1 + self.index)
            D = str(self.sortedD[self.sortedImages.index(im)])
        elif type(im) in (float, int):  # is entropy value
            return self.getData(self.sortedImages[self.sortedD.index(im)])
        else:  # list is image itself
            name = "Sorted image"
            number = str(1 + self.index)
            D = str(self.sortedD[self.index])
        return [name, " {}/{}".format(number, len(self.sortedD)), " Entropy = " + D]

    def selectlist(self, imlist):  # actual list
        self.imlist = imlist
        self.index = 0

    def nextim(self):
        self.index = limitaxis(self.index + 1, len(self.imlist) - 1)
        self.data = self.getImage(self.imlist[self.index])
        self.sample = self.data
        self.controlText[-1] = self.getData(self.imlist[self.index])
        self.init()

    def previousim(self):
        self.index = limitaxis(self.index - 1, len(self.imlist) - 1)
        self.data = self.getImage(self.imlist[self.index])
        self.sample = self.data
        self.controlText[-1] = self.getData(self.imlist[self.index])
        self.init()

    def keyfunc(self):
        replaced = False
        if self.pressedkey in (2555904, 65363):  # if right key (win,linux)
            self.nextim()
            replaced = True
        elif self.pressedkey in (2424832, 65361):  # if left key (win,linux)
            self.previousim()
            replaced = True
        if replaced or self.builtincmd():  # replaced left and right keys
            if self.y is not None and self.x is not None:
                self.builtinplot(self.sample[self.y, self.x])
            else:
                self.builtinplot()
