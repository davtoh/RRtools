import numpy as np
import cv2
from RRtoolbox.lib.arrayops import overlay, brightness, background, getOtsuThresh, \
    process_as_blocks, rescale, getBilateralParameters
from RRtoolbox.lib.image import loadFunc
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.root import glob
from RRtoolbox.lib.directory import getData
from tesisfunctions import std_deviation, plainness, thresh_biggestCnt
from RRtoolbox.tools.segmentation import layeredfloods

def pad(IMG):

    if len(IMG.shape)>2:
        # pad image to add 1 layer of pixels
        # this is used to correctly flood all the background
        h, w, c = IMG.shape
        img = np.zeros((h+2, w+2,3), np.uint8)
        img[1:-1,1:-1,:] = IMG
    else:
        h, w = IMG.shape
        img = np.zeros((h+2, w+2), np.uint8)
        img[1:-1,1:-1] = IMG
    return img

def layeredfloods2(img, gray = None, backmask = None, step = 1, connectivity = 4, increase = False):

    if gray is None:
        if len(img.shape)>2:
            gray = brightness(img)
        else:
            gray = img

    # initialization of flags for flooding
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    h, w = gray.shape# get image shape

    # pad mask to add 1 layer of pixels
    # this is needed in the flood operation
    mask = np.zeros((h+2, w+2), np.uint8) #

    # create the background mask
    if backmask is None:
        backmask = background(gray)
    mask_background = np.ones((h+2, w+2), np.uint8)
    mask_background[1:-1,1:-1] = backmask

    # get area of background mask
    area_background = np.sum(mask_background)

    # low and high limits for flooding
    lo,hi = 255,0

    # first seed point of the minimum color
    xs,ys = np.where(gray == np.min(gray))
    seed_pt = xs[0],ys[0]

    #flood adding overal layers starting from background
    area = 0
    hi = step
    all = np.zeros_like(mask,np.float32)
    while area < area_background and hi <= 255:
        mask[:]=0
        cv2.floodFill(img, mask, seed_pt, (255, 255, 255), (lo,) * 3, (hi,) * 3, flags)
        if increase:
            all += mask.astype("float")*hi
        elif not increase:
            all += mask.astype("float")
        area = np.sum(np.bitwise_and(mask,mask_background))
        hi += step
    all = all[1:-1,1:-1]
    all /= all.max()
    return all

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

test = 1 # tests 0,1,2,3

shape_block = (4,4) # processing block shapes for tests 2,3
useWindows = True # mode to process blocks in tests 2,3

for fn in fns:
    print "processing {}".format(fn)
    img = pad(loadFunc(1,(300,300))(fn)) # load image with padding
    P = brightness(img)
    all = layeredfloods(img, gray=P)
    fastplt(all,block=len(fns)>1, title="alpha mask")
    hist, bins = np.histogram(all.flatten(),256)
    thresh = bins[getOtsuThresh(hist)]
    mask = (all < thresh).astype(np.uint8)

    #props = process_as_blocks(P,std_deviation,shape_block,mask,useWindows)
    #thresh = np.mean(all[props!=0])
    #mask = (all < thresh).astype(np.uint8)

    t = "{} mask with thresh = {}".format(getData(fn)[-2],thresh)
    fastplt(overlay(img.copy(),mask*255,alpha=mask*0.5),block=len(fns)>1, title=t)

    #TODO beging flooding the brightest part
    #all = layeredfloods(img, gray=((1-all)*255).astype(np.uint8))
    all = layeredfloods2(img, gray=((1-all)*255).astype(np.uint8), backmask=mask)
    #all[mask==0] = all.max()
    #all[:,:] = rescale(all)
    fastplt(all,block=len(fns)>1, title="processed alpha mask")