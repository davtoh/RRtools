"""
    Robust thresholds to segment retinal images
"""
from __future__ import print_function
from __future__ import absolute_import
# TODO, complete

import numpy as np
import cv2
import matplotlib.pyplot as plt
from RRtoolbox.lib.arrayops import findminima, getOtsuThresh, brightness, hist_cdf
from .tesisfunctions import threshold
from glob import glob

imlist= glob("im*.jpg")
imlist.extend(glob("good_*.jpg"))
imlist.append("_good.jpg")
dpi = 100 # use 100 for better resolution
grapths = True # True to plot images
proposed = False # True to use proposed method
ishull = False # True to apply convex hull
save = False # True to save figure
show = True # True to show figure
shape = None #(20,5*len(fns))
print("thresholdgin ",len(imlist),"images")

fig = plt.figure("threshs",figsize=shape) # window name
# make title
title = "Normal brightness - "
if proposed: title += "Proposed thresh"
else: title += "Otsu thresh"
if ishull: title += " with convex hull"

for i,fn in enumerate(imlist):

    ## get data
    img =cv2.imread(fn) # read image
    P = brightness(img) # get brightness
    hist,cdf = hist_cdf(P)
    if proposed:
        fhist,fcdf = hist_cdf(P,2) # get filtered histogram
        hist_shift = len(fhist)-256
        th1 = getOtsuThresh(fhist) - hist_shift #np.min(np.where(cdf.max()*0.5<=cdf)) # user criteria
        th2 = np.max(np.where(fhist.max()==fhist))-hist_shift # max value
        th3 = np.min(np.where(np.mean(fcdf)<=fcdf))-hist_shift # mean of cdf
        thresh=findminima(fhist,np.mean([th1,th2,th3]))
        #thresh = np.mean([th1,th2,th3,th4])
        #thresh2 = findminima(fhist,len(fhist)-10)-hist_shift
        #th = cv2.inRange(P, thresh, thresh2)
        th = threshold(P,thresh,255,0)
    else:
        #thresh = getOtsuThresh(hist)
        #th = threshold(P,thresh,255,0)
        thresh,th = cv2.threshold(P,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if ishull:
        contours,hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        allcontours = np.vstack(contours[i] for i in np.arange(len(contours)))
        hull = cv2.convexHull(allcontours)
        cv2.drawContours(th,[hull],-1,255,-1)

    ## plot image
    if grapths:
        plt.subplot(len(imlist),3,i*3+1),plt.imshow(cv2.resize(P,(300,300)),'gray')
        plt.title(fn)
        plt.xticks([]),plt.yticks([])

    ## plot thresh
    if grapths:
        plt.subplot(len(imlist),3,i*3+3),plt.imshow(cv2.resize(th,(300,300)),'gray')
        plt.title("thresh="+str(thresh))
        plt.xticks([]),plt.yticks([])

    ## plot data
    if grapths: plt.subplot(len(imlist),3,i*3+2)
    else: plt.subplot(len(imlist),1,i+1)
    if i==0: plt.title(title)
    plt.plot(hist, color = 'r') # plot histogram
    plt.plot(cdf, color = 'b') # plot cumulative distribution function
    # colors: b: blue, g: green, r: red, c: cyan, m: magenta, y: yellow, k: black, w: white
    x = np.arange(len(hist)) # get x axis
    if proposed:
        plt.plot(x[th1],hist[th1], "o",color="g") # plot user criteria
        plt.plot(x[th2],hist[th2], "o",color="c") # plot max value
        plt.plot(x[th3],cdf[th3], "o",color="y") # plot mean of cdf
        #plt.plot(x[thresh2], hist[thresh2], "ro",color="orangered") # plot selected threshold
    plt.plot(x[thresh], hist[thresh], "ro",color="r") # plot selected threshold
    plt.xlim([0,len(hist)]) # space the x axis
    #if len(fns)>1 and i != len(fns)-1: pass#plt.xticks([])
    #plt.legend(('histogram','cdf'), loc = 'upper left')
    #plt.legend(('histogram','cdf'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #saving
    if save and i == len(imlist)-1:
        fileName = title+"_th"+str(thresh)
        if len(imlist)==1:
            fileName +="_"+fn.split(".")[0]
        fileName += ".jpg"
        fig.savefig(fileName,dpi=dpi)

if show: plt.show()