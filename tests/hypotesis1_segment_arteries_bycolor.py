"""
Trying to threshold only the arteries from a retinal image, silly idea
"""
__author__ = 'Davtoh'

import cv2
from tesisfunctions import Plotim

fn1 = 'im1_2.jpg'
fore = cv2.imread(fn1)

fb = fore[:,:,0]
fg = fore[:,:,1]
fr = fore[:,:,2]
fgray = cv2.cvtColor(fore,cv2.COLOR_BGR2GRAY)


for i in xrange(fore.shape[1]): # rows
    for j in xrange(fore.shape[0]): # columns
        b,g,r = fb.item(j, i),fg.item(j, i),fr.item(j, i)
        if (b-g)>0 and r>g and r>b and b>70 and g>70:
            cv2.circle(fore, (i,j), 1, (0, 0, 255), -1, 8)

plotc = Plotim("blood vessels", fore)
plotc.show()

"""
for i in xrange(bgr.shape[1]): # rows
    for j in xrange(bgr.shape[0]): # columns
        a,b,c = fb.item(j, i),fg.item(j, i),fr.item(j, i)
        detA = a*c - b*b
        traceA = a + c
        harmonic_mean = detA/traceA
        if harmonic_mean > thresh:
            cv2.circle(bgr, (i,j), 1, (0, 0, 255), -1, 8)


detA = fb*fr - np.power(fg,2)
traceA = fb + fr
harmonic_mean = detA/traceA
for j, i in np.argwhere(harmonic_mean > thresh):
    cv2.circle(bgr, (i,j), 1, (0, 0, 255), -1, 8)
"""