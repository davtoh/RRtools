# encoding: utf-8
# module cv2.ximgproc
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc

# imports
from __future__ import absolute_import
# encoding: utf-8
# module cv2.ximgproc
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc

# imports
# <module 'cv2.ximgproc.segmentation'>
from . import segmentation as segmentation

# Variables with simple values

AM_FILTER = 4

ARO_0_45 = 0

ARO_315_0 = 3
ARO_315_135 = 6
ARO_315_45 = 4

ARO_45_135 = 5
ARO_45_90 = 1

ARO_90_135 = 2

ARO_CTR_HOR = 7
ARO_CTR_VER = 8

DTF_IC = 1
DTF_NC = 0
DTF_RF = 2

FHT_ADD = 2
FHT_AVE = 3
FHT_MAX = 1
FHT_MIN = 0

GUIDED_FILTER = 3

HDO_DESKEW = 1
HDO_RAW = 0

SLIC = 100
SLICO = 101

THINNING_GUOHALL = 1
THINNING_ZHANGSUEN = 0

WMF_COS = 3
WMF_EXP = 0
WMF_IV1 = 1
WMF_IV2 = 2
WMF_JAC = 4
WMF_OFF = 5

__loader__ = None

__spec__ = None

# functions


def AdaptiveManifoldFilter_create():  # real signature unknown; restored from __doc__
    """ AdaptiveManifoldFilter_create() -> retval """
    pass


# real signature unknown; restored from __doc__
def amFilter(joint, src, sigma_s, sigma_r, dst=None, adjust_outliers=None):
    """ amFilter(joint, src, sigma_s, sigma_r[, dst[, adjust_outliers]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def bilateralTextureFilter(src, dst=None, fr=None, numIter=None, sigmaAlpha=None, sigmaAvg=None):
    """ bilateralTextureFilter(src[, dst[, fr[, numIter[, sigmaAlpha[, sigmaAvg]]]]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def covarianceEstimation(src, windowRows, windowCols, dst=None):
    """ covarianceEstimation(src, windowRows, windowCols[, dst]) -> dst """
    pass


# real signature unknown; restored from __doc__
def createAMFilter(sigma_s, sigma_r, adjust_outliers=None):
    """ createAMFilter(sigma_s, sigma_r[, adjust_outliers]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createDisparityWLSFilter(matcher_left):
    """ createDisparityWLSFilter(matcher_left) -> retval """
    pass


# real signature unknown; restored from __doc__
def createDisparityWLSFilterGeneric(use_confidence):
    """ createDisparityWLSFilterGeneric(use_confidence) -> retval """
    pass


# real signature unknown; restored from __doc__
def createDTFilter(guide, sigmaSpatial, sigmaColor, mode=None, numIters=None):
    """ createDTFilter(guide, sigmaSpatial, sigmaColor[, mode[, numIters]]) -> retval """
    pass


def createEdgeAwareInterpolator():  # real signature unknown; restored from __doc__
    """ createEdgeAwareInterpolator() -> retval """
    pass


# real signature unknown; restored from __doc__
def createFastGlobalSmootherFilter(guide, lambda_, sigma_color, lambda_attenuation=None, num_iter=None):
    """ createFastGlobalSmootherFilter(guide, lambda, sigma_color[, lambda_attenuation[, num_iter]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createFastLineDetector(_length_threshold=None, _distance_threshold=None, _canny_th1=None, _canny_th2=None, _canny_aperture_size=None, _do_merge=None):
    """ createFastLineDetector([, _length_threshold[, _distance_threshold[, _canny_th1[, _canny_th2[, _canny_aperture_size[, _do_merge]]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createGuidedFilter(guide, radius, eps):
    """ createGuidedFilter(guide, radius, eps) -> retval """
    pass


def createRFFeatureGetter():  # real signature unknown; restored from __doc__
    """ createRFFeatureGetter() -> retval """
    pass


# real signature unknown; restored from __doc__
def createRightMatcher(matcher_left):
    """ createRightMatcher(matcher_left) -> retval """
    pass


# real signature unknown; restored from __doc__
def createStructuredEdgeDetection(model, howToGetFeatures=None):
    """ createStructuredEdgeDetection(model[, howToGetFeatures]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createSuperpixelLSC(image, region_size=None, ratio=None):
    """ createSuperpixelLSC(image[, region_size[, ratio]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels, prior=None, histogram_bins=None, double_step=None):
    """ createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels[, prior[, histogram_bins[, double_step]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createSuperpixelSLIC(image, algorithm=None, region_size=None, ruler=None):
    """ createSuperpixelSLIC(image[, algorithm[, region_size[, ruler]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def dtFilter(guide, src, sigmaSpatial, sigmaColor, dst=None, mode=None, numIters=None):
    """ dtFilter(guide, src, sigmaSpatial, sigmaColor[, dst[, mode[, numIters]]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def fastGlobalSmootherFilter(guide, src, lambda_, sigma_color, dst=None, lambda_attenuation=None, num_iter=None):
    """ fastGlobalSmootherFilter(guide, src, lambda, sigma_color[, dst[, lambda_attenuation[, num_iter]]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def guidedFilter(guide, src, radius, eps, dst=None, dDepth=None):
    """ guidedFilter(guide, src, radius, eps[, dst[, dDepth]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def jointBilateralFilter(joint, src, d, sigmaColor, sigmaSpace, dst=None, borderType=None):
    """ jointBilateralFilter(joint, src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def l0Smooth(src, dst=None, lambda_=None, kappa=None):
    """ l0Smooth(src[, dst[, lambda[, kappa]]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def niBlackThreshold(_src, maxValue, type, blockSize, delta, _dst=None):
    """ niBlackThreshold(_src, maxValue, type, blockSize, delta[, _dst]) -> _dst """
    pass


# real signature unknown; restored from __doc__
def rollingGuidanceFilter(src, dst=None, d=None, sigmaColor=None, sigmaSpace=None, numOfIter=None, borderType=None):
    """ rollingGuidanceFilter(src[, dst[, d[, sigmaColor[, sigmaSpace[, numOfIter[, borderType]]]]]]) -> dst """
    pass


# real signature unknown; restored from __doc__
def thinning(src, dst=None, thinningType=None):
    """ thinning(src[, dst[, thinningType]]) -> dst """
    pass

# no classes
