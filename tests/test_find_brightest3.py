__author__ = 'Davtoh'
from tesisfunctions import histogram, brightness,Plotim, graphmath, graphHistogram, \
    overlay, findmaxima, findminima, smooth, graph_filter, getOtsuThresh, find_near
from RRtoolbox.lib.arrayops import Bandpass,convexityRatio
import cv2
import numpy as np
import pylab as plt
from scipy.signal import savgol_filter
# I should  try by finding the minimum value or maximum value, find their indexes and then apply region growing from there
# to detect the main body, perhaps i should find the maximum value in the histogram and then find that color, and apply
# region growing from there. Using these three i should be able to distinct the background, main body and brightest areas

fn1 = r'im5_1.jpg'
#fn1 = r'im1_1.jpg'
#fn1 = r'im3_1.jpg'
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
window = "hamming" # 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
window_len = 51
hist_PS = smooth(hist_P,window_len,window=window, correct=True) # it smooths and expands the histogram
#hist_PS = savgol_filter(hist_P, window_len, polyorder = 3)
lines.extend(graphmath(hist_PS, show=False)[1]) # plots smooth histogram

x = bins[:-1] #np.indices(hist_P.shape) #np.arange(len(hist_PS)) # get indexes

otsu = getOtsuThresh(hist_P) # Otsu value
minima = findminima(hist=hist_PS) # minima values
maxima = findmaxima(hist=hist_PS) # maxima values
fig2 = plt.figure("Min - Max")
plt.plot(x,hist_PS)
plt.plot(x[minima], hist_PS[minima], "o", label="min")
plt.plot(x[maxima], hist_PS[maxima], "o", label="max")
plt.plot(x[otsu], hist_PS[otsu], "o", label="Otsu")
plt.legend()

# INITIAL DATA
data_min_left = P.min() # initial minimum value
data_min = find_near(maxima, thresh=data_min_left)
data_min_right = find_near(minima, thresh=data_min, side="right")
data_max = P.max()
data_max_left = find_near(minima, thresh=data_max, side="left")
"""
# filter to reduce histogram ends
# create Filter to reduce histogram ends
filterr = Bandpass(10, data_min, data_max) # FIXMED this should use P.min(), P.max() and local maximas, or a relation of porcentage
                              # like min_val % body_val % max_val
graph_filter(filterr,show=False)


hist_PF = filterr(x)*hist_P # this is applyed to find the maximum value at the main body.
lines.extend(graphmath(hist_PF,show=False)[1]) # plots damped histogram at the ends"""

plt.figure(fig1.number) # continue Histogram

# SECONDARY DATA
data_body = find_near(maxima, thresh=otsu,side="right")
data_body_left = find_near(minima,thresh=data_body,side="left")
data_body_right = find_near(minima,thresh=data_body,side="right")
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
mk_back,mk_body,mk_flare = 1,2,3
markers[P <= data_min]=mk_back # background FIXMED use P.min() and aproaching to local maxima
markers[np.bitwise_and(P>data_body_left,P<data_body)]=mk_body # main body
markers[P >= data_max_left]=mk_flare # Flares. this can be used approaching
                                    # to local maxima, but brightest areas are almost
                                    # always saturated so no need to use it
plotc = Plotim(name + " markers", overlay(fore.copy(), pallet[markers], alpha=0.5))
plotc.sample = P
plotc.show() # show markers using pallet

# apply watershed to markers
cv2.watershed(fore,markers) # FIXME perhaps the function should be cv2.floodFill?
# convert processed markers to colors
water = pallet[markers]
plotc = Plotim(name + " watershed", overlay(fore.copy(), water, alpha=0.3)).show() # show colored watershed in image

#data = water==pallet[mk_flare]
#brightest = np.uint8(data[:,:,0]&data[:,:,1]&data[:,:,2])
brightest = np.uint8(markers==mk_flare)

#plotc = Plotim(name +" flares", brightest).show()

contours,hierarchy = cv2.findContours(brightest.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
Crs = [(convexityRatio(cnt),cnt) for cnt in contours] # convexity ratios
Crs.sort(reverse=True)
optic_disc = Crs.pop()
ellipse = cv2.fitEllipse(optic_disc[1])
optic_disc_ellipse = np.zeros_like(brightest)
cv2.ellipse(optic_disc_ellipse, ellipse, 1, -1) # get elliptical ROI

#plotc = Plotim(name +" optic disc mask", optic_disc_ellipse).show()
plotc = Plotim(name + " optic_dist", overlay(fore.copy(), optic_disc_ellipse * 255, alpha=0.3)).show()