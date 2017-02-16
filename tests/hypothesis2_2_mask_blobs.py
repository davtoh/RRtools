from __future__ import absolute_import
__author__ = 'Davtoh'

from .tesisfunctions import Plotim,overlay
import cv2
import numpy as np
from . import tesisfunctions as tf

def blobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.thresholdStep = 1
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector(params)

    return detector


fn1 = r'im1_2.jpg'
fn1 = tf.IMAGEPATH+r"cellphone_retinal/ALCATEL ONE TOUCH IDOL X/left_DAVID/IMG_20150730_115534_1.jpg"
name = fn1.split('\\')[-1].split(".")[0]

fore = cv2.imread(fn1)
fore = cv2.resize(fore,(300,300))
padval = 100
fore = tf.pad(fore,(np.min(fore[:,:,0]),np.min(fore[:,:,1]),np.min(fore[:,:,2])),padval,True)


P = tf.brightness(fore)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
P2 = clahe.apply(P)
"""
thresh,P2 = cv2.threshold(P,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
P2 = cv2.distanceTransform(P2,cv2.DIST_LABEL_PIXEL,5)
P2 = tf.normalizeToRange(P2).astype(np.uint8)
"""
P2 = (255-P2)

detector = blobDetector()

# Detect blobs.
keypoints = detector.detect(P2)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(P2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
im_with_keypoints = tf.croppad(im_with_keypoints,padval)
Plotim("Keypoints", im_with_keypoints).show()

