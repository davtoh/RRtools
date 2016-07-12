# -*- coding: utf-8 -*-
# ----------------------------    IMPORTS    ---------------------------- #
from __future__ import division
# multiprocessing
import itertools as it
from multiprocessing.pool import ThreadPool as Pool
# three-party
import cv2
import numpy as np
# custom
from config import MANAGER, FLAG_DEBUG
from cache import memoize
from arrayops import SimKeyPoint, normsigmoid

# ----------------------------    GLOBALS    ---------------------------- #
cpc = cv2.getNumberOfCPUs()
if FLAG_DEBUG: print "configured to use {} cpus".format(cpc)
pool = Pool(processes = cpc) # DO NOT USE IT when module is imported and this runs with it. It creates a deadlock"
feature_name = 'sift-flann'

# ----------------------------SPECIALIZED FUNCTIONS---------------------------- #

class Feature(object):
    """
    Class to manage detection and computation of features

    :param pool: multiprocessing pool (dummy, it uses multithreading)
    :param useASIFT: if True adds Affine perspectives to the detector.
    :param debug: if True prints to the stdout debug messages.
    """
    def __init__(self,pool=pool,useASIFT = True, debug = True):
        self.pool=pool
        self.detector = None
        self.matcher = None
        self.useASIFT = useASIFT
        self.debug = debug

    def detectAndCompute(self, img, mask=None):
        """
        detect keypoints and descriptors

        :param img: image to find keypoints and its descriptors
        :param mask: mask to detect keypoints (it uses default, mask[:] = 255)
        :return: keypoints,descriptors
        """
        # bulding parameters of tilt and rotation variations
        if self.useASIFT:
            params = [(1.0, 0.0)] # first tilt and rotation
            # phi rotations for t tilts of the image
            for t in 2**(0.5*np.arange(1,6)):
                for phi in np.arange(0, 180, 72.0 / t):
                    params.append((t, phi))

            def helper(param):
                t, phi = param #tilt, phi (rotation)
                # computing the affine transform
                timg, tmask, Ai = affine_skew(t, phi, img, mask) # get tilted image, mask and transformation
                # Find keypoints and descriptors with the detector
                keypoints, descrs = self.detector.detectAndCompute(timg, tmask) # use detector
                for kp in keypoints:
                    x, y = kp.pt # get actual keypoints
                    kp.pt = tuple( np.dot(Ai, (x, y, 1)) ) # transform keypoints to original img
                if descrs is None: descrs = [] # faster than: descrs or []
                return keypoints, descrs

            if pool is None:
                ires = it.imap(helper, params) # process asynchronously
            else:
                ires = pool.imap(helper, params)  # process asynchronously in pool
            keypoints, descrs = [], []
            for i, (k, d) in enumerate(ires):
                keypoints.extend(k)
                descrs.extend(d)
                if self.debug: print 'affine sampling: %d / %d\r' % (i+1, len(params)),
        else:
            keypoints, descrs = self.detector.detectAndCompute(img, mask) # use detector

        keypoints = [getattr(SimKeyPoint(obj),"__dict__") for obj in keypoints] # convert to dictionaries
        #return keyPoint2tuple(keypoints), np.array(descrs)
        return keypoints, np.array(descrs)

    def config(self, name, separator = "-"):
        """
        This function takes parameters from a command to initialize a detector and matcher.

        :param name: "[a-]<sift|surf|orb>[-flann]" (str) Ex: "a-sift-flann"
        :param features: it is a dictionary containing the mapping from name to the
                initialized detector, matcher pair. If None it is created.
                This feature is to reduce time by reusing created features.
        :return: detector, matcher
        """
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        FLANN_INDEX_LSH    = 6

        chunks = name.split(separator)
        index = 0
        if chunks[index] == "a":
            self.useASIFT = True
            index+=1
        if chunks[index] == 'sift':
            detector = cv2.SIFT()  # Scale-invariant feature transform
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[index] == 'surf':
            detector = cv2.SURF(500)  # Hessian Threshold to 800, 500 # http://stackoverflow.com/a/18891668/5288758
            # http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[index] == 'orb':
            detector = cv2.ORB(400)  # binary string based descriptors
            norm = cv2.NORM_HAMMING  # Hamming distance
        else:
            raise Exception("name {} with detector {} not valid".format(name,chunks[index]))
        index +=1
        if len(chunks)-1 >= index and chunks[index] == 'flann':
            # FLANN based Matcher, Fast Approximate Nearest Neighbor Search Library
            if norm == cv2.NORM_L2:  # for SIFT ans SURF
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:  # for ORB
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        else:  # brute force matcher
            matcher = cv2.BFMatcher(norm) # difference in norm http://stackoverflow.com/a/32849908/5288758
        self.detector, self.matcher = detector, matcher
        return detector, matcher

def init_feature(name, features = None):
    """
    This function takes parameters from a command to initialize a detector and matcher.

    :param name: "<sift|surf|orb>[-flann]" (str) Ex: "sift-flann"
    :param features: it is a dictionary containing the mapping from name to the
            initialized detector, matcher pair. If None it is created.
            This feature is to reduce time by reusing created features.
    :return: detector, matcher
    """
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6
    if features is None: features = {} # reset features
    if name not in features: # if called with a different name
        chunks = name.split('-')
        if chunks[0] == 'sift':
            detector = cv2.SIFT()  # Scale-invariant feature transform
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[0] == 'surf':
            detector = cv2.SURF(500)  # Hessian Threshold to 800, 500 # http://stackoverflow.com/a/18891668/5288758
            # http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
            norm = cv2.NORM_L2  # distance measurement to be used
        elif chunks[0] == 'orb':
            detector = cv2.ORB(400)  # binary string based descriptors
            norm = cv2.NORM_HAMMING  # Hamming distance
        else:
            return None, None
        if 'flann' in chunks:  # FLANN based Matcher, Fast Approximate Nearest Neighbor Search Library
            if norm == cv2.NORM_L2:  # for SIFT ans SURF
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:  # for ORB
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        else:  # brute force matcher
            matcher = cv2.BFMatcher(norm) # difference in norm http://stackoverflow.com/a/32849908/5288758
        features[name] = detector, matcher # cache detector and matcher
    return features[name] # get buffered: detector, matcher

def affine_skew(tilt, phi, img, mask=None):
    """
    Increase robustness to descriptors by calculating other invariant perspectives to image.

    :param tilt: tilting of image
    :param phi: rotation of image (in degrees)
    :param img: image to find Affine transforms
    :param mask: mask to detect keypoints (it uses default, mask[:] = 255)
    :return: skew_img, skew_mask, Ai (invert Affine Transform)

    Ai - is an affine transform matrix from skew_img to img

    """
    h, w = img.shape[:2] # get 2D shape
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]]) # init Transformation matrix
    if phi != 0.0: # simulate rotation
        phi = np.deg2rad(phi) # convert degrees to radian
        s, c = np.sin(phi), np.cos(phi) # get sine, cosine components
        A = np.float32([[c,-s], [ s, c]]) # build partial Transformation matrix
        corners = [[0, 0], [w, 0], [w, h], [0, h]] # use corners
        tcorners = np.int32( np.dot(corners, A.T) ) # transform corners
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2)) # get translations
        A = np.hstack([A, [[-x], [-y]]]) # finish Transformation matrix build
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1) # get sigma
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01) # blur image with gaussian blur
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST) # resize
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2] # get new 2D shape
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST) # also get mask transformation
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai

@memoize(MANAGER["TEMPPATH"],ignore=["pool"])
def ASIFT(feature_name, img, mask=None, pool=pool):
    """
    asift(feature_name, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.

    :param feature_name: feature name to create detector.
    :param img: image to find keypoints and its descriptors
    :param mask: mask to detect keypoints (it uses default, mask[:] = 255)
    :param pool: multiprocessing pool (dummy, it uses multithreading)
    :return: keypoints,descriptors
    """
    # bulding parameters of tilt and rotation variations
    detector = init_feature(feature_name)[0] # it must get detector object of cv2 here to prevent conflict with memoizers
    params = [(1.0, 0.0)] # first tilt and rotation
    # phi rotations for t tilts of the image
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def helper(param):
        t, phi = param #tilt, phi (rotation)
        # computing the affine transform
        timg, tmask, Ai = affine_skew(t, phi, img, mask) # get tilted image, mask and transformation
        # Find keypoints and descriptors with the detector
        keypoints, descrs = detector.detectAndCompute(timg, tmask) # use detector
        for kp in keypoints:
            x, y = kp.pt # get actual keypoints
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) ) # transform keypoints to original img
        if descrs is None: descrs = [] # faster than: descrs or []
        return keypoints, descrs
    if pool is None:
        ires = it.imap(helper, params) # process asynchronously
    else:
        ires = pool.imap(helper, params)  # process asynchronously in pool
    keypoints, descrs = [], []
    for i, (k, d) in enumerate(ires):
        keypoints.extend(k)
        descrs.extend(d)
        if FLAG_DEBUG: print 'affine sampling: %d / %d\r' % (i+1, len(params)),
    keypoints = [getattr(SimKeyPoint(obj),"__dict__") for obj in keypoints] # convert to dictionaries
    #return keyPoint2tuple(keypoints), np.array(descrs)
    return keypoints, np.array(descrs)


def ASIFT_iter(imgs, feature_name=feature_name):
    """
    Affine-SIFT for N images.

    :param imgs: images to apply asift
    :param feature_name: eg. SIFT SURF ORB
    :return: [(kp1,desc1),...,(kpN,descN)]
    """
    #print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))
    for img in imgs: yield ASIFT(feature_name, img, pool=pool)

def ASIFT_multiple(imgs, feature_name=feature_name):
    """
    Affine-SIFT for N images.

    :param imgs: images to apply asift
    :param feature_name: eg. SIFT SURF ORB
    :return: [(kp1,desc1),...,(kpN,descN)]
    """
    #print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))
    return [ASIFT(feature_name, img, pool=pool) for img in imgs]

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    This function applies a ratio test.

    :param kp1: raw keypoints 1
    :param kp2: raw keypoints 2
    :param matches: raw matches
    :param ratio: filtering ratio of distance
    :return: filtered keypoint 1, filtered keypoint 2, keypoint pairs
    """
    mkp1, mkp2 = [], [] # initialize matched keypoints
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio: # by Hamming distance
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )  # keypoint with Index of the descriptor in query descriptors
            mkp2.append( kp2[m.trainIdx] )  # keypoint with Index of the descriptor in train descriptors
    p1 = np.float32([kp["pt"] for kp in mkp1])
    p2 = np.float32([kp["pt"] for kp in mkp2])
    return p1, p2, zip(mkp1, mkp2) # p1, p2, kp_pairs

@memoize(MANAGER["TEMPPATH"])
def MATCH(feature_name,kp1,desc1,kp2,desc2):
    """
    Use matcher and asift output to obtain Transformation matrix (TM).

    :param feature_name: feature name to create detector. It is the same used in the detector
            which is used in init_feature function but the detector itself is ignored.
            e.g. if 'detector' uses BFMatcher, if 'detector-flann' uses FlannBasedMatcher.
    :param kp1: keypoints of source image
    :param desc1: descriptors of kp1
    :param kp2: keypoints of destine image
    :param desc2: descriptors of kp2
    :return: TM
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    """
    matcher = init_feature(feature_name)[1] # it must get matcher object of cv2 here to prevent conflict with memoizers
    # BFMatcher.knnMatch() returns k best matches where k is specified by the user
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    # If k=2, it will draw two match-lines for each keypoint.
    # So we have to pass a status if we want to selectively draw it.
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches) #ratio test of 0.75
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0) # status specifies the inlier and outlier points
        if FLAG_DEBUG: print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        # do not draw outliers (there will be a lot of them)
        #kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag] # uncomment to give only good kp_pairs
    else:
        H, status = None, None
        if FLAG_DEBUG: print '%d matches found, not enough for homography estimation' % len(p1)
    return H, status, kp_pairs


def MATCH_multiple(pairlist, feature_name=feature_name):
    """
    :param pairlist: list of keypoint and descriptors pair e.g. [(kp1,desc1),...,(kpN,descN)]
    :param feature_name: feature name to create detector
    :return: [(H1, mask1, kp_pairs1),....(HN, maskN, kp_pairsN)]
    """
    kp1,desc1 = pairlist[0]
    return [MATCH(feature_name,kp1,desc1,kpN,descN) for kpN,descN in pairlist[1:]]


def inlineRatio(inlines,lines, thresh = 30):
    """
    Probability that a match was correct.

    :param inlines: number of matched lines
    :param lines: number lines
    :param thresh: threshold for lines (i.e. very low probability <= thresh < good probability)
    :return:
    """
    return (inlines/lines)*normsigmoid(lines,30,thresh) # less than 30 are below 0.5