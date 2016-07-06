__author__ = 'Davtoh'

import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    imgf,imgb - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
    #http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    This function applies a ratio test
    :param kp1: raw keypoint 1
    :param kp2: raw keypoint 2
    :param matches: raw matches
    :param ratio: filtering ratio
    :return: filtered keypoint 1, filtered keypoint 2, keypoint pairs
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )  # keypoint with Index of the descriptor in query descriptors
            mkp2.append( kp2[m.trainIdx] )  # keypoint with Index of the descriptor in train descriptors
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


fn1 = r'C:\Users\Davtoh\Dropbox\PYTHON\projects\Descriptors\Tests\im2_1.jpg'  # queryImage
fn2 = r'C:\Users\Davtoh\Dropbox\PYTHON\projects\Descriptors\Tests\im2_2.jpg'  # trainImage

img1 = cv2.resize(cv2.imread(fn1, 0), (800, 600))
img2 = cv2.resize(cv2.imread(fn2, 0), (800, 600))

sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
# Match descriptors.
kp_pairs = bf.match(des1,des2)
# Sort them in the order of their distance.
kp_pairs = sorted(kp_pairs, key = lambda x:x.distance)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = drawMatches(img1,kp1,img2,kp2,kp_pairs)

plt.imshow(img3),plt.show()