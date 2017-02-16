from __future__ import absolute_import
__author__ = 'Davtoh'

import cv2
import numpy as np
from .tesisfunctions import histogram,graphmath,Plotim,IMAGEPATH,filterFactory

fn1 = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_131444.jpg"
#fn1 = r'C:\Users\Davtoh\Dropbox\PYTHON\projects\Descriptors\im4_1.jpg'
alfa = 50
beta1 = 30 # if beta = 50 with noise, if beta = 200 without noise
beta2 = 270

alfa = 10
beta1 = 30 # if beta = 50 with noise, if beta = 200 without noise
beta2 = None

title = "alfa = "+str(alfa)
if beta2 is None:
    title+= ", beta = "+str(beta1)
else:
    assert(beta2>beta1) # beta2 must be greater than beta1
    title+= ", beta1 = "+str(beta1)+", beta2 = "+str(beta2)

# calculate bgr histograms
bgr = cv2.resize(cv2.imread(fn1),(300,300))
y = histogram(bgr)
# calculate gray histograms
gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
y_gray = histogram(gray)
y.extend(y_gray)
# calculate filter
levels = np.linspace(0, 256, 256)
myfilter = filterFactory(alfa, beta1, beta2)
filtered = myfilter(levels)
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
#y.append(filtered*maxval)
fig = graphmath(y, ("b", "g", "r", "k", "m"), win="histogram", title=title)

#apply filter
fbgr=myfilter(bgr.astype("float"))*bgr.astype("float")
fgray=myfilter(gray.astype("float"))*gray.astype("float")
fbgr = fbgr.astype("uint8")
fgray = fgray.astype("uint8")
fy = histogram(fbgr)
fy_gray = histogram(fgray)
fy.extend(fy_gray)
fig = graphmath(fy, ("b", "g", "r", "k"), win="filtered histogram")

plot = Plotim("fbgr", fbgr)
plot.show()
plot = Plotim("fgray", fgray)
plot.show()