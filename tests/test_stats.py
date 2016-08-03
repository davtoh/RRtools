import numpy as np
import cv2
from RRtoolbox.lib.arrayops import overlay, brightness, process_as_blocks, rescale, getOtsuThresh
from RRtoolbox.lib.image import loadFunc
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.image import GetCoors
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.directory import getData
from tesisfunctions import std_deviation, plainness

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

pallet = np.array([(0,0,0),(10, 10, 255)],np.uint8)
shape_block = (12,12)
useWindows = False
drawMask = False
test = 0

for fn in fns:
    print "processing {}".format(fn)
    img = loadFunc(1,(300,300))(fn) # load image
    h, w, c = img.shape # get image shape

    P = brightness(img) #img[:,:,2]#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #

    if drawMask:
        gcoors = GetCoors(P)
        coors = np.array(gcoors.show(clean=False).coors)
        gcoors.clean()
        mask = np.zeros_like(P)
        cv2.fillConvexPoly(mask,coors.astype(np.int32),1,0)
        ROI = P[mask]
    else:
        mask = None
        ROI = P

    pl = plainness(ROI)
    sd = std_deviation(ROI)
    print "plainness = {}, deviation = {}".format(pl,sd)


    if test==0:
        props = process_as_blocks(P,plainness,shape_block,mask,useWindows)
        testtext = "plainness"
        thr = pl # np.mean(props[mask==1])
    elif test == 1:
        props = process_as_blocks(P,std_deviation,shape_block,mask,useWindows)
        testtext = "deviation"
        thr = np.mean(props[mask==1]) # sd
    else:
        raise Exception("not thes implemented")

    #fastplt(img,block=len(fns)>1)
    #fastplt(rescale(props,255,0), title="{} blocks of shape = {}".format(testtext,shape_block),block=len(fns)>1)
    th = props >= thr
    fastplt(overlay(img.copy(),pallet[th.astype(np.uint8)],alpha=th*0.5),
            title="Threshold above overall {}".format(testtext),block=len(fns)>1)
    hist, bins = np.histogram(props[mask==1].flatten(),256)
    fastplt(rescale(props >= bins[getOtsuThresh(hist)],255,0),
            title="Otsu over {}".format(testtext),block=len(fns)>1)