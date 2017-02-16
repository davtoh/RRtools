from __future__ import absolute_import
__author__ = 'Davtoh'
from .tesisfunctions import histogram, brightness,Plotim, graphmath, graphHistogram, \
    overlay, findmaxima, findminima, smooth, graph_filter
from RRtoolbox.lib.arrayops import Bandpass
import cv2
import numpy as np
import pylab as plt
# I should  try by finding the minimum value or maximum value, find their indexes and then apply region growing from there
# to detect the main body, perhaps i should find the maximum value in the histogram and then find that color, and apply
# region growing from there. Using these three i should be able to distinct the background, main body and brightest areas

fn1 = r'im5_1.jpg'
name = fn1.split('\\')[-1].split(".")[0]

# read image
fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300)) # resize image
#Plotim(name,fore).show() # show image

# get intensity
P = brightness(fore)
#P = (255-P)

# create color pallet: unknown, background, body, brightest, boundaries
pallet = np.array([[255,0,0],[0,0,0],[0,0,255],[255,255,255],[255,0,255]],np.uint8)

lines,comments = [],[]
fig1 = plt.figure("Histogram")
# calculate histogram
#hist_P = histogram(P)[0]
hist_P, bins = np.histogram(P.flatten(),256,[0,256])
lines.extend(graphmath(hist_P,show=False)[1]) # plots original histogram

# filter histogram to smooth it.
hist_PS = smooth(hist_P,correct=True) # it smooths and expands the histogram
lines.extend(graphmath(hist_PS, show=False)[1]) # plots smooth histogram

x = bins[:-1] #np.indices(hist_P.shape)[0] # get indexes

# INITIAL DATA
data_min_left = P.min() # initial minimum value
data_min = findmaxima(hist=hist_PS, thresh=data_min_left)
data_min_right = findminima(hist=hist_PS, thresh=data_min, side="right")
data_max = P.max()
data_max_left = findminima(hist=hist_PS, thresh=data_max, side="left")

# filter to reduce histogram ends
# create Filter to reduce histogram ends
filterr = Bandpass(10, data_min, data_max) # FIXMED this should use P.min(), P.max() and local maximas, or a relation of porcentage
                              # like min_val % body_val % max_val
graph_filter(filterr,show=False)

plt.figure(fig1.number) # continue Histogram
hist_PF = filterr(x)*hist_P # this is applyed to find the maximum value at the main body.
lines.extend(graphmath(hist_PF,show=False)[1]) # plots damped histogram at the ends

# SECONDARY DATA
data_body = np.where(hist_PF==hist_PF.max())[0]
data_body_left = findminima(hist=hist_PS,thresh=data_body,side="left")
data_body_right = findminima(hist=hist_PS,thresh=data_body,side="right")
annotate = [(x[data_min_left], hist_P[data_min_left], "min left"),
            (x[data_min], hist_P[data_min], "min"),
            (x[data_min_right], hist_P[data_min_right], "min right"),
            (x[data_body_left], hist_P[data_body_left],"body left"),
            (x[data_body], hist_P[data_body],"body"),
            (x[data_body_right], hist_P[data_body_right],"body right"),
            (x[data_max_left], hist_P[data_max_left],"max left"),
            (x[data_max], hist_P[data_max],"max")]
arrowprops = dict(facecolor='black', shrink=0.05)

ax = plt.gca() # get current axes
for xp,yp,ann in annotate:
    ax.annotate(ann.title(), xy=(xp,yp), textcoords='data', xytext=(xp,yp), arrowprops=arrowprops)
    plt.plot(xp,yp, "o", label=ann)
#plt.legend(handles=lines,loc = 'upper left')
#graphHistogram(P)
plt.show()
# create markers
markers = np.zeros_like(P).astype("int32")
markers[P <= data_min]=1 # background FIXMED use P.min() and aproaching to local maxima
markers[np.bitwise_and(P>data_body_left,P<data_body)]=2 # main body
markers[P >= data_max_left]=3 # Flares. this can be used approaching to local maxima, but brightest areas are almost
                      # always saturated so no need to use it
plotc = Plotim(name + " markers", overlay(fore.copy(), pallet[markers], alpha=0.5))
plotc.sample = P
plotc.show() # show markers using pallet

# apply watershed to markers
cv2.watershed(fore,markers) # FIXME perhaps the function should be cv2.floodFill?
# convert processed markers to colors
water = pallet[markers]
plotc = Plotim(name + " watershed", overlay(fore.copy(), water, alpha=0.3)).show() # show colored watershed in image