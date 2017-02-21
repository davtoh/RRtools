from __future__ import division
from __future__ import absolute_import
from builtins import range
from past.utils import old_div

import numpy as np
import cv2
import matplotlib.pyplot as plt
from .tesisfunctions import hist_cdf,findminima,threshold
import glob

def brightness(img):
    ### LESS BRIGHT http://alienryderflex.com/hsp.html
    #b,g,r = cv2.split(img.astype("float"))
    #return np.sqrt( .299*(b**2) + .587*(g**2) + .114*(r**2)).astype("uint8")
    ### HSV
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]

def stem(x,y,color):
    markerline, stemlines, baseline = plt.stem(x,y,linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.setp(stemlines, linewidth=1, color = color)     # set stems
    plt.setp(markerline, 'markerfacecolor', color)    # make points

def otsuthresh(hist):
    #http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    # find normalized_histogram, and its cumulative distribution function
    hist_norm = old_div(hist.astype("float").ravel(),hist.max())
    Q = hist_norm.cumsum()

    bins = np.arange(len(hist_norm))

    fn_min = np.inf
    thresh = -1

    for i in range(1,len(hist_norm)):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[len(hist_norm)-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = old_div(np.sum(p1*b1),q1), old_div(np.sum(p2*b2),q2)
        v1,v2 = old_div(np.sum(((b1-m1)**2)*p1),q1),old_div(np.sum(((b2-m2)**2)*p2),q2)

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh

imlist= glob.glob("im*.jpg")
imlist.extend(glob.glob("good_*.jpg"))
dpi = 100
grapths = True # True to plot images
proposed = False # True to use proposed method
ishull = False # True to apply convex hull
save = False # True to save figure
show = True # True to show figure
shape = (20,5*len(imlist))


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
        th1 = otsuthresh(fhist)-hist_shift #np.min(np.where(cdf.max()*0.5<=cdf)) # user criteria
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