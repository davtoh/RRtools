__author__ = 'Davtoh'
import cv2
from tesisfunctions import plotim,overlay,padVH
import numpy as np
from RRtoolbox.lib.plotter import imtester

# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# http://people.csail.mit.edu/sparis/bf_course/

if __name__=="__main__":
    #img= cv2.resize(cv2.imread(r"asift2fore.png"),(400,400))
    img = cv2.resize(cv2.imread(r'im1_2.jpg'),(400,400))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    test = imtester(img)
    print test.info