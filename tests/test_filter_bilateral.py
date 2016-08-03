__author__ = 'Davtoh'

import cv2
import numpy as np
from tesisfunctions import histogram,graphmath,filterFactory,Plotim

fn1 = r'im1_2.jpg'
bgr = cv2.resize(cv2.imread(fn1),(300,300))

d = 9
sigmaColor = 35
sigmaSpace = 20

chimg = cv2.bilateralFilter(bgr,d,sigmaColor,sigmaSpace)

plot = Plotim("filter", np.hstack([bgr, chimg]))
plot.show()