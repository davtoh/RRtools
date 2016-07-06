# http://dsp.stackexchange.com/questions/16166/histogram-matching-of-two-images-using-cdf
# http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
# https://isis.astrogeology.usgs.gov/Application/presentation/Tabbed/histmatch/histmatch.html
# http://www.fmwconcepts.com/imagemagick/histmatch/index.php

import numpy as np
from RRtoolbox.lib.arrayops.basic import overlay
from RRtoolbox.lib.image import hist_match

def filtering(im, core =9):
    #im = cv2.GaussianBlur(im, (core, core), 0)
    im = cv2.medianBlur(im, core)
    #im = cv2.bilateralFilter(im, core,75,75)
    return im

from matplotlib import pyplot as plt
from scipy.misc import lena, ascent
import cv2

shape = (400,400)
#source = lena()
#source = cv2.cvtColor(cv2.resize(cv2.imread("im1_2.jpg"),shape),cv2.COLOR_BGR2RGB)
source = cv2.cvtColor(cv2.resize(cv2.imread("im1_1.jpg",0),shape),cv2.COLOR_GRAY2RGB)
#source = cv2.bilateralFilter(source, 9,75,75)
#template = ascent()
template = cv2.cvtColor(cv2.resize(cv2.imread("im1_1.jpg"),shape),cv2.COLOR_BGR2RGB)

matched = hist_match(source, template)
#matched = overlay(source.copy(),filtering(matched),0)
#matched = cv2.bilateralFilter(matched, 5,75,75)

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

x1, y1 = ecdf(source.ravel())
x2, y2 = ecdf(template.ravel())
x3, y3 = ecdf(matched.ravel())

fig = plt.figure()
gs = plt.GridSpec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[1, :])
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(source, cmap=plt.cm.gray)
ax1.set_title('Source')
ax2.imshow(template, cmap=plt.cm.gray)
ax2.set_title('Template')
ax3.imshow(matched, cmap=plt.cm.gray)
ax3.set_title('Matched')

ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
ax4.set_xlim(x1[0], x1[-1])
ax4.set_xlabel('Pixel value')
ax4.set_ylabel('Cumulative %')
ax4.legend(loc=5)

plt.show()