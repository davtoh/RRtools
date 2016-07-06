__author__ = 'Davtoh'


import cv2
import numpy as np
from tesisfunctions import brightness,sigmoid,IMAGEPATH,plotim
import glob

rootpath = IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/"
imlist= glob.glob(rootpath+"IMG*.jpg")
fn1 = imlist[3]
#fn1 = r"C:\Users\Davtoh\Documents\2015_01\Tesis tests\retinal photos\ALCATEL ONE TOUCH IDOL X\left_DAVID\IMG_20150730_115534_1.jpg"
#fn1 = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\tesis\im4_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(400,400))

im2 = brightness(fore)
im = sigmoid(im2,50,133).astype(np.uint8)
plotim("levels",im).show()