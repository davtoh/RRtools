from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from past.utils import old_div
import numpy as np
import cv2
from RRtoolbox.lib.arrayops import overlay, brightness, background, getOtsuThresh, \
    process_as_blocks, rescale, getBilateralParameters
from RRtoolbox.lib.image import loadFunc
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.directory import getData
from .tesisfunctions import std_deviation, plainness



root = "/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/" # lighting/
#fns = glob(root+"*")

# most relevant images with problems
fns = ["_good.jpg"
        ,root+"20150730_105818.jpg"
        ,"im1_2.jpg"
        ,root+"IMG_0404.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0421.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0401.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0419.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0180.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0243.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/20150730_092758(0).jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/IMG_0426.jpg"
        ,"/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/analysis/20150730_092934.jpg"]

fixed_range = True
connectivity = 4
newVal = (255, 255, 255) # color of flooding
test = 3 # tests 0,1,2,3

flags = connectivity
step = 1 # step to increas flooding
shape_block = (4,4) # processing block shapes for tests 2,3
useWindows = True # mode to process blocks in tests 2,3

# initialization of flags for flooding
if fixed_range:
    flags |= cv2.FLOODFILL_FIXED_RANGE
flags |= cv2.FLOODFILL_MASK_ONLY

pallet = np.array([(0,0,0),newVal],np.uint8) # pallet to color masks

for fn in fns:
    print("processing {}".format(fn))
    img = loadFunc(1,(300,300))(fn) # load image
    #params = getBilateralParameters(img.shape) # calculate bilateral parameters (21,82,57)
    #img = cv2.bilateralFilter(img, *params)

    # pad image to add 1 layer of pixels
    # this is used to correctly flood all the background
    h, w, c = img.shape
    img2 = np.zeros((h+2, w+2,3), np.uint8)
    img2[1:-1,1:-1,:] = img
    img = img2

    h, w, c = img.shape# get image shape

    P = brightness(img) #img[:,:,2]#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #

    # pad mask to add 1 layer of pixels
    # this is needed in the flood operation
    mask = np.zeros((h+2, w+2), np.uint8) #

    # create the background maks
    mask_background = np.zeros((h+2, w+2), np.uint8)
    mask_background[1:-1,1:-1] = background(P)

    # get area of background mask
    area_background = np.sum(mask_background)

    # low and high limits for flooding
    lo,hi = 255,0

    # first seed point of the minimum color
    xs,ys = np.where(P==np.min(P))
    seed_pt = xs[0],ys[0]

    ### METHOD 1 flood individual ponds starting from background
    if test == 0:
        area = 0
        while area < area_background:
            mask[:]=0
            hi += step
            cv2.floodFill(img, mask, seed_pt, newVal, (lo,)*3, (hi,)*3, flags)
            area = np.sum(np.bitwise_and(mask,mask_background))
            print("{} and  {}".format(hi,area))
        mask = mask[1:-1,1:-1]

        # cv2.fillConvexPoly(array,pts.astype(np.int32),1,0)
        #contours, _ = cv2.findContours(contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #points2mask(pts)
        #levels = 255.0-hi
        fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),block=len(fns)>1)

    ### METHOD 2 flood adding overal layers starting from background
    if test in (1,2,3):
        area = 0
        hi = step
        all = np.zeros_like(mask,np.float32)
        while hi <= 255:
            mask[:]=0
            cv2.floodFill(img, mask, seed_pt, newVal, (lo,)*3, (hi,)*3, flags)
            if area >= area_background:
                all += mask.astype("float")
            else:
                area = np.sum(np.bitwise_and(mask,mask_background))
            """
            all += mask.astype("float")
            """
            hi += step
        all = all[1:-1,1:-1]
        all /= all.max()

        fastplt(1-all,block=len(fns)>1, title="alpha mask")
        hist, bins = np.histogram((all*255).astype(np.uint8).flatten(),256,[0,256])
        thresh = old_div(getOtsuThresh(hist),255.0)
        mask = (all < thresh).astype(np.uint8)

        hist, bins = np.histogram((all[mask==1]*255).astype(np.uint8).flatten(),256,[0,256])
        thresh = old_div(getOtsuThresh(hist),255.0)
        mask = (all < thresh).astype(np.uint8)

        t = "{} mask with thresh = {}".format(getData(fn)[-2],thresh)
        fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),block=len(fns)>1, title=t)

        all[mask==0] = all.max()
        all[:,:] = rescale(all)
        fastplt(1-all,block=len(fns)>1, title="processed alpha mask")


    if test == 2:
        props = process_as_blocks(P,plainness,shape_block,mask,useWindows)
        testtext = "plainness"
        thr = np.mean(props[mask==1])
    elif test == 3:
        props = process_as_blocks(P,std_deviation,shape_block,mask,useWindows)
        testtext = "deviation"
        thr = np.mean(props[mask==1]) # sd

    if test in (2,3):
        #fastplt(img,block=len(fns)>1)
        fastplt(rescale(props,255,0), title="{} blocks of shape = {}".format(testtext,shape_block),block=len(fns)>1)
        th = props >= thr
        fastplt(overlay(img.copy(),pallet[th.astype(np.uint8)],alpha=th*0.5),
                title="Threshold above overall {}".format(testtext),block=len(fns)>1)
        hist, bins = np.histogram(props[mask==1].flatten(),256)
        fastplt(rescale(props >= bins[getOtsuThresh(hist)],255,0),
                title="Otsu over {}".format(testtext),block=len(fns)>1)