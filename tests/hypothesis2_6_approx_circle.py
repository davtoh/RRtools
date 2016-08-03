__author__ = 'Davtoh'
from tesisfunctions import Plotim,overlay
import cv2
import numpy as np
import tesisfunctions as tf
from recommended import getKernel

fn1 = r'im1_2.jpg'
#fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(800,800))

P = tf.brightness(fore)
th,thresh = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

hull = tf.gethull(thresh)
ROI = np.zeros(thresh.shape,dtype=np.uint8)
cv2.drawContours(ROI,[hull],-1,1,-1)
Plotim("First ROI", ROI).show()

# for erotion
iterations = 1
#kernel = np.ones((10,10),np.uint8)
kernel = getKernel(ROI.size)
while not np.all(thresh[ROI==1]):
    ROI = cv2.erode(ROI,kernel,iterations = iterations)
    #ROI = cv2.dilate(ROI,kernel,iterations = iterations)

Plotim("Last ROI", ROI).show()
