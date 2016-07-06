#!/usr/bin/env python

'''
Feature-based image matching sample.

USAGE
  find_obj.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

import cv2
import numpy as np

from tests.deprecated.common import anorm

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    """
    This function takes parameters from a command to initialize a detector and matcher
    :param name: "<sift|surf|orb>[-flann]" (str) Ex: "sift-flann"
    :return: detector, matcher
    """
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()  # Scale-invariant feature transform
        norm = cv2.NORM_L2  # distance measurement to be used
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)  # Hessian Threshold to 800
        norm = cv2.NORM_L2  # distance measurement to be used
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)  # binary string based descriptors
        norm = cv2.NORM_HAMMING  # Hamming distance
    else:
        return None, None
    if 'flann' in chunks:  # FLANN based Matcher
        if norm == cv2.NORM_L2:  # for SIFT ans SURF
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:  # for ORB
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:  # brute force matcher
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


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

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    """
    This function draws a set of keypoint pairs obtained on a match method of a descriptor
    on two images imgf and imgb
    :param win: window's name (str)
    :param img1: image1 (numpy array)
    :param img2: image2 (numpy array)
    :param kp_pairs: zip(keypoint1, keypoint2)
    :param status: obtained from cv2.findHomography
    :param H: obtained from cv2.findHomography (default=None)
    :return: vis (image with matching result) (default=None)
    """
    # colors to use
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)

    h1, w1 = img1.shape[:2]  # obtaining image1 dimensions
    h2, w2 = img2.shape[:2]  # obtaining image2 dimensions
    # imgf and imgb will be visualized horizontally (left-right)
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)  # making visualization image
    vis[:h1, :w1] = img1  # imgf at the left of vis
    vis[:h2, w1:w1+w2] = img2  # imgf at the right of vis
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # changing color attribute to background image

    if H is not None:  # enclosing object
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, white)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)  # making sure every pair of keypoints is graphed
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  # pair of coordinates for imgf
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) # pair of coordinates for imgb

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:  # drawing circles (good keypoints)
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)  # for left keypoint (imgf)
            cv2.circle(vis, (x2, y2), 2, col, -1)  # for right keypoint (imgf)
        else:  # drawing x marks (wrong keypoints)
            col = red
            r = 2
            thickness = 3
            # for left keypoint (imgf)
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            # for right keypoint (imgf)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()  # saving state of the visualization for onmouse event
    # drawing lines for non-onmouse event
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.namedWindow(win,cv2.WINDOW_NORMAL) # Can be resized
    cv2.imshow(win, vis)  # show static image as visualization for non-onmouse event
    def onmouse(event, x, y, flags, param):
        cur_vis = vis  # actual visualization
        if flags & cv2.EVENT_FLAG_LBUTTON:  # if onmouse
            cur_vis = vis0.copy()
            r = 8  # proximity to keypoint
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]  # get indexes near pointer
            kp1s, kp2s = [], []
            for i in idxs:  # for all keypints near pointer
                 (x1, y1), (x2, y2) = p1[i], p2[i]  # my keypoint
                 col = (red, green)[status[i]]  # choosing False=red,True=green
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)  # drawing line
                 # keypoints to show on event
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            # drawing keypoints near pointer for imgf and imgb
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)  # show visualization
    cv2.setMouseCallback(win, onmouse)
    return vis

if __name__ == '__main__':
    print __doc__

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try: fn1, fn2 = args
    except:
        fn1 = '../c/box.png'
        fn2 = '../c/box_in_scene.png'

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature_name)
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)


    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))

    def match_and_draw(win):
        print 'matching...'
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        vis = explore_match(win, img1, img2, kp_pairs, status, H)

    match_and_draw('find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()
