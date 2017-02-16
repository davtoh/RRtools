from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from past.utils import old_div
__author__ = 'Davtoh'
from .tesisfunctions import getcoors, drawcoorperspective, Plotim
import cv2
import numpy as np
from .tesisfunctions import histogram,graphmath,filterFactory,normsigmoid

def histogram_filters(bgr,filters,title="bgr filters",win="histogram"):
    """
    Apply filters to histogram.

    :param bgr: image
    :param filters: list of filters for each color band
    :param title: title of plot
    :param win: window name
    :return:
    """
    y = histogram(bgr)
    # calculate filter
    levels = np.linspace(0, 256, 256)
    for f in filters:
        filtered = f(levels)
        # calculate maximum value
        maxval = 1
        for i in y:
            try:
                val = np.max(i[filtered==1])
            except:
                val = np.max(i)
            if val>maxval:
                maxval = val
        # append filter with maxvalue and graph
        y.append(filtered*maxval)
    return graphmath(y, ("b", "g", "r", "b", "g", "r"), win=win, title=title)

def enhancer(alfa,beta1,beta2=None):
    def filter(levels):
        #return np.log(levels)
        return old_div((normsigmoid(-50+levels,alfa,0+beta1)+normsigmoid(levels,alfa,255-beta2)),1.7)
    return filter

fn1 = r'im1_1.jpg'
img = cv2.resize(cv2.imread(fn1),(300,300))
"""
points =getcoors(img,"get pixel coordinates")
pixels = np.array([list(img[i[::-1]]) for i in points])
pmin = np.min(pixels,0)
pmax = np.max(pixels,0)
pmean = np.mean(pixels,0).astype("uint8")
"""
pmin = [59,50,182]
pmax = [170,166,255]
pmean= [127,106,238]

#print "pixels: ", pixels
print("min: ",pmin)
print("max: ",pmax)
print("mean: ",pmean)

alfa = 20
beta = 10
filters = []
for i in range(len(pmin)):
    #filters.append(filter(alfa,pmin[i]-beta,pmax[i]+beta))
    #filters.append(filter(alfa,pmean[i]))
    #filters.append(enhancer(pmean[i],pmin[i]-beta,pmax[i]+beta))
    filters.append(enhancer(20,20,10))

histogram_filters(img,filters)

fimg = img.copy()
for i in range(len(pmin)):
    channel = img[:,:,i].astype("float")
    fchannel=filters[i](channel)*255
    fimg[:,:,i] = fchannel.astype("uint8")

fy = histogram(fimg)
fig = graphmath(fy, ("b", "g", "r"), win="filtered histogram")

plot = Plotim("image - filtered image", np.hstack([img, fimg]))
plot.show()