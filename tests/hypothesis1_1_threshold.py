"""
 Some test to make thresholds
"""
__author__ = 'Davtoh'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tesisfunctions import hist_cdf,findminima,threshold,brightness,getOtsuThresh
from glob import glob

root = ""
#fns= glob(root+"im*.jpg")
#fn = fns[0]
fn = root +"_good.jpg"
#fn = "inputImage3.png"
fns = [fn]
dpi = 1000 # quality of the plot
grapths = False # True to plot images
proposed = True # True to use proposed method
ishull = False # True to apply convex hull
save = False # True to save figure
show = True # True to show figure
shape = None# (20,5*len(fns))

fig = plt.figure("threshs",figsize=shape) # window name
# make title
title = "Normal brightness - "
if proposed: title += "Proposed thresh"
else: title += "Otsu thresh"
if ishull: title += " with convex hull"

for i,fn in enumerate(fns):

    ## get data
    img =cv2.imread(fn) # read image
    P = brightness(img) # get brightness
    hist,cdf = hist_cdf(P)
    if proposed:
        fhist,fcdf = hist_cdf(P,2) # get filtered histogram
        hist_shift = len(fhist)-256
        th1 = np.min(np.where(cdf.max()*0.5<=cdf)) # user criteria # getOtsuThresh(fhist)-hist_shift
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
        plt.subplot(len(fns), 3, i * 3 + 1), plt.imshow(cv2.resize(P, (300, 300)), 'gray')
        plt.title(fn)
        plt.xticks([]),plt.yticks([])

    ## plot thresh
    if grapths:
        plt.subplot(len(fns), 3, i * 3 + 3), plt.imshow(cv2.resize(th, (300, 300)), 'gray')
        plt.title("thresh="+str(thresh))
        plt.xticks([]),plt.yticks([])

    ## plot data
    if grapths: ax =  plt.subplot(len(fns), 3, i * 3 + 2)
    else: ax = plt.subplot(len(fns), 1, i + 1)
    if i==0: plt.title(title)
    ax.plot(hist, color = 'r') # plot histogram
    ax.plot(cdf, color = 'b') # plot cumulative distribution function
    # colors: b: blue, g: green, r: red, c: cyan, m: magenta, y: yellow, k: black, w: white
    x = np.arange(len(hist)) # get x axis
    if proposed:
        ax.plot(x[th1],hist[th1], "o",color="g") # plot user criteria
        ax.annotate(u' user criteria', xy=(x[th1],hist[th1]), textcoords='data',verticalalignment='top')
        ax.plot(x[th2],hist[th2], "o",color="c") # plot max value
        ax.annotate(u' max value', xy=(x[th2],hist[th2]), textcoords='data',verticalalignment='bottom')
        ax.plot(x[th3],cdf[th3], "o",color="y") # plot mean of cdf
        ax.annotate(u' cdf mean', xy=(x[th3],cdf[th3]), textcoords='data')
        #ax.plot(x[thresh2], hist[thresh2], "ro",color="orangered") # plot selected threshold
    ax.plot(x[thresh], hist[thresh], "ro",color="r") # plot selected threshold
    ax.annotate(u' threshold', xy=(x[thresh], hist[thresh]),
                textcoords='data',
                xytext=(100, hist.max()/2),
                horizontalalignment='left',
                verticalalignment='bottom',
                arrowprops=dict(facecolor='black',
                            shrink=0.05)
                )
    plt.xlim([0,len(hist)]) # space the x axis
    #if len(fns)>1 and i != len(fns)-1: pass#plt.xticks([])
    plt.legend(('histogram','cdf'), loc = 'upper left')
    #plt.legend(('histogram','cdf'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #saving
    if save and i == len(fns)-1:
        fileName = title+"_th"+str(thresh)
        if len(fns)==1:
            fileName +="_"+fn.split(".")[0]
        fileName += ".jpg"
        fig.savefig(fileName,dpi=dpi)

if show: plt.show()