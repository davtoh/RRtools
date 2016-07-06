__author__ = 'Davtoh'

import sys
import time
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from tesisfunctions import normsigmoid,normalize,sh2oh,plotim,overlay
from tests.deprecated.asift import affine_detect, init_feature, filter_matches, explore_match,Timer


def getalfa_(foregray,backgray,window):
    backmask = normalize(normsigmoid(backgray,20,180)+normsigmoid(backgray,3.14,192)+normsigmoid(backgray,-3.14,45))
    a = normsigmoid(foregray,-3,242)*0.5
    a[a<0.1]=0
    b = normsigmoid(foregray,3.14,50)
    b[b<0.1]=0
    foremask = a*b*normsigmoid(foregray,20,112)
    foremask = normalize(foremask*backmask)
    foremask *= normalize(window)
    return foremask

def getalfa(foregray,backgray,window):
    backmask = normalize(normsigmoid(backgray,10,180)+normsigmoid(backgray,3.14,192)+normsigmoid(backgray,-3.14,45))
    foremask = normalize(normsigmoid(foregray,-1,242)*normsigmoid(foregray,3.14,50))
    foremask = normalize(foremask * backmask)
    foremask[foremask>0.9] = 2.0
    ksize = (21,21)
    foremask = normalize(cv2.blur(foremask,ksize))
    foremask *= normalize(window)
    return foremask

if __name__ == '__main__':
    t1 = time.time()
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
    #fn1 = r'im5_2.jpg'
    #fn2 = r'im5_3.jpg'
    #fn1 = r"im3_1.jpg"
    #fn2 = r"im3_2.jpg"
    #fn1 = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\Image\t2.jpg"
    #fn2 = r"C:\Users\Davtoh\Dropbox\PYTHON\projects\Image\p1.jpg"

    original_fore = cv2.imread(fn1)
    original_back = cv2.imread(fn2)
    rzyf,rzxf = 900,900
    rzyb,rzxb = 900,900
    scaled_fore = cv2.resize(cv2.imread(fn1, 0), (rzxf, rzyf))
    scaled_back = cv2.resize(cv2.imread(fn2, 0), (rzxb, rzyb))

    detector, matcher = init_feature(feature_name)
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)

    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    with Timer('detecting features...','detecting features...'):
        kp1, desc1 = affine_detect(detector, scaled_fore, pool=pool)
        kp2, desc2 = affine_detect(detector, scaled_back, pool=pool)
        print 'imgf - %d features, imgb - %d features' % (len(kp1), len(kp2))

    win = 'affine find_obj'
    with Timer('matching'):
        # BFMatcher.knnMatch() returns k best matches where k is specified by the user
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        # If k=2, it will draw two match-lines for each keypoint.
        # So we have to pass a mask if we want to selectively draw it.
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches) #ratio test of 0.75
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print '%d matches found, not enough for homography estimation' % len(p1)

    t2 = time.time()
    vis = explore_match(win, scaled_fore, scaled_back, kp_pairs, None, H)
    cv2.waitKey()
    cv2.destroyAllWindows()
    t2 = time.time()-t2

    if H is not None:
        # get perspective
        H2 = sh2oh(H,original_fore.shape,scaled_fore.shape,original_back.shape,scaled_back.shape)
        bgra_fore = cv2.cvtColor(original_fore,cv2.COLOR_BGR2BGRA)
        fore_in_back = cv2.warpPerspective(bgra_fore,H2,(original_back.shape[1],original_back.shape[0]))
        # convert formats to float
        foregray = cv2.cvtColor(fore_in_back,cv2.COLOR_BGRA2GRAY).astype(float)
        fore_in_back = fore_in_back.astype(float)
        # find alfa and do overlay
        window = fore_in_back[:,:,3].copy()
        for i in xrange(1): # testing damage by repetition
            backgray = cv2.cvtColor(original_back.astype("uint8"),cv2.COLOR_BGR2GRAY).astype(float)
            fore_in_back[:,:,3]= getalfa(foregray,backgray,window)
            original_back = overlay(original_back,fore_in_back)
        # plot and save result
        original_back = original_back.astype("uint8")
        cv2.imwrite("im_completer_Result.png",original_back)
        print "process finished... ",time.time()-t1-t2
        plot = plotim("result",original_back)
        plot.show()
        # cv2.compare(src1, src2, cmpop[, dst])
        # http://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
        # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
