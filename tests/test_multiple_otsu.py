from __future__ import print_function
from builtins import range
import numpy as np
import cv2
from RRtoolbox.lib.arrayops import getOtsuThresh, overlay, brightness, filterFactory, background
from RRtoolbox.lib.image import loadFunc
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.directory import getData

def d(vals):
    hist, bins = np.histogram(vals,256,[0,256])
    s_values, bin_idx, s_counts = np.unique(P, return_inverse=True,
                                                return_counts=True)
    m = hist.mean()

def body_best_parts(P, mask = None):
    #P = should be with cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    # this ensures a better enclosing of the retinal area
    if mask is None: mask = 1-background(P)#np.ones_like(P) #
    hist, bins = np.histogram(P[mask.astype(bool)].flatten(),256,[0,256])
    thresh = getOtsuThresh(hist)
    cv2.threshold(P,thresh,1,cv2.THRESH_BINARY,dst=mask)


def brightest(P, mask = None, iterations = 3):
    #P = should be with cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), this
    # prevents false bright areas and includes others
    if mask is None: mask = np.ones_like(P)
    for i in range(iterations):
        hist, bins = np.histogram(P[mask.astype(bool)].flatten(),256,[0,256])
        thresh = getOtsuThresh(hist)
        cv2.threshold(P,thresh,1,cv2.THRESH_BINARY,dst=mask)
    return mask

root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/" # lighting/
fns = glob(root+"*")
#fn = root+"20150730_105818.jpg"
#fn = "im1_2.jpg"
#fn = root+"IMG_0404.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0421.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0401.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0419.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0180.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0243.jpg"
fn = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/20150730_092758(0).jpg"
#fns = [fn]


for fn in fns:
    print("processing {}".format(fn))
    img = loadFunc(1,(300,300))(fn)
    P = brightness(img) #img[:,:,2]#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #
    mask = 1-background(P)#np.ones_like(P) #
    iterations = 3
    for i in range(iterations):
        hist, bins = np.histogram(P[mask.astype(bool)].flatten(),256,[0,256])
        """
        # faster but not precise
        if i > 0:
            thresh = np.mean(img[mask.astype(bool)])
        else:
            thresh = getOtsuThresh(hist)"""
        if i == 0:
            pass
            #P = filterFactory(5, getOtsuThresh(hist))(P.astype(np.float32)).astype(np.uint8)*P
            #hist, bins = np.histogram(P[mask.astype(bool)].flatten(),256,[0,256])

        thresh = getOtsuThresh(hist)
        cv2.threshold(P,thresh,1,cv2.THRESH_BINARY,dst=mask)
        # np.mean(img[mask.astype(bool)],0),np.mean(img[mask.astype(bool)]),thresh
        if i == iterations-1:
            t = "{} mask with thresh = {}".format(getData(fn)[-2],thresh)
            fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),title=t,block=len(fns)>1)