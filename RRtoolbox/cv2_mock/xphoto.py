# encoding: utf-8
# module cv2.xphoto
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

BM3D_STEP1 = 1
BM3D_STEP2 = 2
BM3D_STEPALL = 0

HAAR = 0

INPAINT_SHIFTMAP = 0

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def applyChannelGains(src, gainB, gainG, gainR, dst=None):
    """ applyChannelGains(src, gainB, gainG, gainR[, dst]) -> dst """
    pass


def bm3dDenoising(src, dstStep1, dstStep2=None, h=None, templateWindowSize=None, searchWindowSize=None, blockMatchingStep1=None, blockMatchingStep2=None, groupSize=None, slidingStep=None, beta=None, normType=None, step=None, transformType=None):  # real signature unknown; restored from __doc__
    """ bm3dDenoising(src, dstStep1[, dstStep2[, h[, templateWindowSize[, searchWindowSize[, blockMatchingStep1[, blockMatchingStep2[, groupSize[, slidingStep[, beta[, normType[, step[, transformType]]]]]]]]]]]]) -> dstStep1, dstStep2  or  bm3dDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize[, blockMatchingStep1[, blockMatchingStep2[, groupSize[, slidingStep[, beta[, normType[, step[, transformType]]]]]]]]]]]]) -> dst """
    pass


def createGrayworldWB():  # real signature unknown; restored from __doc__
    """ createGrayworldWB() -> retval """
    pass


# real signature unknown; restored from __doc__
def createLearningBasedWB(path_to_model=None):
    """ createLearningBasedWB([, path_to_model]) -> retval """
    pass


def createSimpleWB():  # real signature unknown; restored from __doc__
    """ createSimpleWB() -> retval """
    pass


# real signature unknown; restored from __doc__
def dctDenoising(src, dst, sigma, psize=None):
    """ dctDenoising(src, dst, sigma[, psize]) -> None """
    pass


def inpaint(src, mask, dst, algorithmType):  # real signature unknown; restored from __doc__
    """ inpaint(src, mask, dst, algorithmType) -> None """
    pass

# no classes
