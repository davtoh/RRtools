#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.

  Example: $python asift.py --feature=sift-flann im1_1.jpg im1_2.jpg

EDIT: This code has been obtained from OpenCV examples and modified by Davtoh 15/04/2015.
      e-mail: davsamirtor@hotmail.com ; davsamirtor@gmail.com
      Documentation has been expanded for better understanding of ASIFT implementation
      explained at http://www.ipol.im/pub/algo/my_affine_sift/
'''

import itertools as it
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from common import Timer
from find_obj import init_feature, filter_matches, explore_match


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    #bulding parameters of tilt and rotation variations
    params = [(1.0, 0.0)]
    #phi rotations for t tilts of the image
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p #tilt, phi (rotation)
        #computing the affine transform
        timg, tmask, Ai = affine_skew(t, phi, img)
        #Find keypoints and descriptors with the detector
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs
    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)
    for i, (k, d) in enumerate(ires):
        print 'affine sampling: %d / %d\r' % (i+1, len(params)),
        keypoints.extend(k)
        descrs.extend(d)
    print
    return keypoints, np.array(descrs)

if __name__ == '__main__':
    print __doc__

    #getting commands from command pront
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk-flann') #default is 'sift-flann'
    try: fn1, fn2 = args
    except:
        fn1 = r'im5_1.jpg'
        fn2 = r'im5_3.jpg'
    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    detector, matcher = init_feature(feature_name)
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)

    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    with Timer('detecting features...'):
        kp1, desc1 = affine_detect(detector, img1, pool=pool)
        kp2, desc2 = affine_detect(detector, img2, pool=pool)
        print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))

    def match_and_draw(win):
        with Timer('matching'):
            # BFMatcher.knnMatch() returns k best matches where k is specified by the user
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
            # If k=2, it will draw two match-lines for each keypoint.
            # So we have to pass a mask if we want to selectively draw it.
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches) #ratio test of 0.75
        if len(p1) >= 4:
            #status = mask and H = transformation or homography matrix
            # X1 = H * X2
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        vis = explore_match(win, img1, img2, kp_pairs, None, H)

    match_and_draw('affine find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html