import cv2
import matplotlib as plt
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
img = cv2.resize(cv2.imread('im2_1.jpg'),(800, 600))
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.SURF(400)
# In actual cases, it is better to have a value 300-500
surf.hessianThreshold = 500
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)


img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()
