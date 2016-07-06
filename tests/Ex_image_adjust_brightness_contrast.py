"""
This is a sample code that works as in http://stackoverflow.com/a/10561359/5288758 to adjust sheets of papers' colors.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
gs = plt.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


gray = cv2.resize(cv2.imread("im1_2.jpg",0),(400,400))

ax1.imshow(gray, cmap=plt.cm.gray)
ax1.set_title('Source')

gray = (255-gray) # TODO: this is a tweak of the code

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res = (255-res) # TODO: this is a tweak of the code

ax2.imshow(res, cmap=plt.cm.gray)
ax2.set_title('adjusted')

plt.show()