from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from past.utils import old_div
__author__ = 'Davtoh'

from .asift import affine_detect, init_feature, filter_matches, explore_match
from .common import Timer
from tests.tesisfunctions import Plotim,overlay
import sys

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

def scaled2realfunc(real_shape,scaled_shape):
    """

    :param real_shape:
    :param scaled_shape:
    :return:
    # Example:
    forefunc = scaled2realfunc(fore.shape,imgf.shape)
    backfunc = scaled2realfunc(back.shape,imgb.shape)
    p1fore = np.array([forefunc(i) for i in p1])
    p2back = np.array([backfunc(i) for i in p2])
    """
    rH = float(real_shape[0])
    rW = float(real_shape[1])
    sH = float(scaled_shape[0])
    sW = float(scaled_shape[1])
    op = np.array([old_div(rW,sW),old_div(rH,sH)],dtype=np.float32)
    def scaled2real(p):
        #rx = sx*rW/sW
        #ry = sy*rH/sH
        return p*op
    return scaled2real

def hs2hr(H,shape_rf,shape_sf,shape_rb,shape_sb):
    H2 = H.copy()
    Hrf,Wrf = float(shape_rf[0]),float(shape_rf[1])
    Hsf,Wsf = float(shape_sf[0]),float(shape_sf[1])
    Hrb,Wrb = float(shape_rb[0]),float(shape_rb[1])
    Hsb,Wsb = float(shape_sb[0]),float(shape_sb[1])
    H2[0] = H2[0]*Wrb/Wsb
    H2[1] = H2[1]*Hrb/Hsb
    H2[:,0] = H2[:,0]*Wsf/Wrf
    H2[:,1] = H2[:,1]*Hsf/Hrf
    return H2

def getmask(img,thresh=4,maximum=1):
    ret,thresh1 = cv2.threshold(img,thresh,maximum,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,thresh,maximum,cv2.THRESH_BINARY_INV)


if __name__ == '__main__':

    #getting commands from command pront
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift-flann') #default is 'sift-flann'
    #feature_name = 'sift-flann'
    try: fn1, fn2 = args
    except:
        pass
    fn1 = r'im1_2.jpg'
    fn2 = r'im1_1.jpg'
    #fn1 = r'im5_1.jpg'
    #fn2 = r'im5_3.jpg'

    fore = cv2.imread(fn1)
    back = cv2.imread(fn2)
    rzyf = 900
    rzxf = 900
    rzyb = 900
    rzxb = 900
    imgf = cv2.resize(cv2.imread(fn1, 0), (rzxf, rzyf))
    imgb = cv2.resize(cv2.imread(fn2, 0), (rzxb, rzyb))

    detector, matcher = init_feature(feature_name)
    if detector != None:
        print('using', feature_name)
    else:
        print('unknown feature:', feature_name)
        sys.exit(1)

    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    with Timer('detecting features...'):
        kp1, desc1 = affine_detect(detector, imgf, pool=pool)
        kp2, desc2 = affine_detect(detector, imgb, pool=pool)
        print('imgf - %d features, imgb - %d features' % (len(kp1), len(kp2)))

    win = 'affine find_obj'
    with Timer('matching'):
        # BFMatcher.knnMatch() returns k best matches where k is specified by the user
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        # If k=2, it will draw two match-lines for each keypoint.
        # So we have to pass a mask if we want to selectively draw it.
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches) #ratio test of 0.75
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        H2 = hs2hr(H,fore.shape,imgf.shape,back.shape,imgb.shape)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    vis = explore_match(win, imgf, imgb, kp_pairs, None, H)
    cv2.waitKey()
    cv2.destroyAllWindows()

    if H is not None:
        def normsigmoide(x,alfa,beta):
            return old_div(1,(np.exp(old_div((beta*1.0-x),alfa))+1))
        fore2 = cv2.cvtColor(fore.copy(),cv2.COLOR_BGR2BGRA)
        dst = cv2.warpPerspective(fore2.copy(),H2,(back.shape[1],back.shape[0]))
        cv2.imwrite("asift2fore.png",dst)
        foregray = cv2.cvtColor(dst.copy(),cv2.COLOR_BGRA2GRAY)
        dst[:,:,3]= normsigmoide(foregray,1,60)
        cv2.imwrite("asift2alfa.png",dst[:,:,3])
        result = overlay(back.copy(),dst)
        #plot = Plotim("result",result)
        # cv2.compare(src1, src2, cmpop[, dst])
        # http://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
        #plot.show()
        cv2.imwrite("asift2Result.png",result)