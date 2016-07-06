__author__ = 'Davtoh'

import time

from tesisfunctions import plotim,overlay
import cv2
from tesisfunctions import normsigmoid,normalize

fn1 = r'im1_1.jpg'
fn2 = r"asift2fore.png"
back = cv2.imread(fn1)
fore = cv2.imread(fn2,-1)
foregray = cv2.cvtColor(fore.copy(),cv2.COLOR_BGRA2GRAY).astype("float")
backgray = cv2.cvtColor(back.copy(),cv2.COLOR_BGR2GRAY).astype("float")
fore = fore.astype("float")
t1 = time.time()
backmask = normalize(normsigmoid(backgray,10,180)+normsigmoid(backgray,3.14,192)+normsigmoid(backgray,-3.14,45))
a = normsigmoid(foregray,-3,242)*0.5
a[a<0.1]=0
b = normsigmoid(foregray,3.14,50)
b[b<0.1]=0
foremask = a*b*normsigmoid(foregray,20,112)
window = normalize(fore[:,:,3].copy())
print time.time()-t1
plot = plotim("foremask",foremask)
plot.show()
plot = plotim("backmask",backmask)
plot.show()
plot = plotim("window",window)
plot.show()
foremask = foremask * backmask
foremask[foremask>0.9] = 2.0
ksize = (21,21)
foremask = normalize(cv2.blur(foremask,ksize))
plot = plotim("foremask * backmask",foremask)
plot.show()
foremask*=window
plot = plotim("foremask*window",foremask)
plot.show()

fore[:,:,3]=foremask
result = overlay(back,fore)

#imtester(result)
plot = plotim("result",result)
plot.show()
