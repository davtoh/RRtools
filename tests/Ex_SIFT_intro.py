#Introduction to SIFT (Scale-Invariant Feature Transform)
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
# http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0
import cv2
import numpy as np

img = cv2.resize(cv2.imread('im2_1.jpg'),(800, 600))
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)

#kp, des = sift.detectAndCompute(gray,None)
kp,des = sift.compute(gray,kp)
