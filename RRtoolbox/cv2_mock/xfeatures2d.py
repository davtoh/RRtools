# encoding: utf-8
# module cv2.xfeatures2d
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

DAISY_NRM_FULL = 102
DAISY_NRM_NONE = 100
DAISY_NRM_PARTIAL = 101
DAISY_NRM_SIFT = 103

FREAK_NB_ORIENPAIRS = 45
FREAK_NB_PAIRS = 512
FREAK_NB_SCALES = 64

PCTSIGNATURES_GAUSSIAN = 1

PCTSignatures_GAUSSIAN = 1
PCTSignatures_HEURISTIC = 2

PCTSIGNATURES_HEURISTIC = 2

PCTSignatures_L0_25 = 0

PCTSIGNATURES_L0_25 = 0
PCTSIGNATURES_L0_5 = 1

PCTSignatures_L0_5 = 1

PCTSIGNATURES_L1 = 2

PCTSignatures_L1 = 2

PCTSIGNATURES_L2 = 3

PCTSignatures_L2 = 3
PCTSignatures_L2SQUARED = 4

PCTSIGNATURES_L2SQUARED = 4
PCTSIGNATURES_L5 = 5

PCTSignatures_L5 = 5

PCTSignatures_L_INFINITY = 6

PCTSIGNATURES_L_INFINITY = 6

PCTSIGNATURES_MINUS = 0

PCTSignatures_MINUS = 0
PCTSignatures_NORMAL = 2

PCTSIGNATURES_NORMAL = 2
PCTSIGNATURES_REGULAR = 1

PCTSignatures_REGULAR = 1
PCTSignatures_UNIFORM = 0

PCTSIGNATURES_UNIFORM = 0

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def BoostDesc_create(desc=None, use_scale_orientation=None, scale_factor=None):
    """ BoostDesc_create([, desc[, use_scale_orientation[, scale_factor]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def BriefDescriptorExtractor_create(bytes=None, use_orientation=None):
    """ BriefDescriptorExtractor_create([, bytes[, use_orientation]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def DAISY_create(radius=None, q_radius=None, q_theta=None, q_hist=None, norm=None, H=None, interpolation=None, use_orientation=None):
    """ DAISY_create([, radius[, q_radius[, q_theta[, q_hist[, norm[, H[, interpolation[, use_orientation]]]]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def FREAK_create(orientationNormalized=None, scaleNormalized=None, patternScale=None, nOctaves=None, selectedPairs=None):
    """ FREAK_create([, orientationNormalized[, scaleNormalized[, patternScale[, nOctaves[, selectedPairs]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def LATCH_create(bytes=None, rotationInvariance=None, half_ssd_size=None):
    """ LATCH_create([, bytes[, rotationInvariance[, half_ssd_size]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def LUCID_create(lucid_kernel=None, blur_kernel=None):
    """ LUCID_create([, lucid_kernel[, blur_kernel]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def PCTSignaturesSQFD_create(distanceFunction=None, similarityFunction=None, similarityParameter=None):
    """ PCTSignaturesSQFD_create([, distanceFunction[, similarityFunction[, similarityParameter]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def PCTSignatures_create(initSampleCount=None, initSeedCount=None, pointDistribution=None):
    """ PCTSignatures_create([, initSampleCount[, initSeedCount[, pointDistribution]]]) -> retval  or  PCTSignatures_create(initSamplingPoints, initSeedCount) -> retval  or  PCTSignatures_create(initSamplingPoints, initClusterSeedIndexes) -> retval """
    pass


# real signature unknown; restored from __doc__
def PCTSignatures_drawSignature(source, signature, result=None, radiusToShorterSideRatio=None, borderThickness=None):
    """ PCTSignatures_drawSignature(source, signature[, result[, radiusToShorterSideRatio[, borderThickness]]]) -> result """
    pass


# real signature unknown; restored from __doc__
def PCTSignatures_generateInitPoints(initPoints, count, pointDistribution):
    """ PCTSignatures_generateInitPoints(initPoints, count, pointDistribution) -> None """
    pass


# real signature unknown; restored from __doc__
def SIFT_create(nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None):
    """ SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def StarDetector_create(maxSize=None, responseThreshold=None, lineThresholdProjected=None, lineThresholdBinarized=None, suppressNonmaxSize=None):
    """ StarDetector_create([, maxSize[, responseThreshold[, lineThresholdProjected[, lineThresholdBinarized[, suppressNonmaxSize]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def SURF_create(hessianThreshold=None, nOctaves=None, nOctaveLayers=None, extended=None, upright=None):
    """ SURF_create([, hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def VGG_create(desc=None, isigma=None, img_normalize=None, use_scale_orientation=None, scale_factor=None, dsc_normalize=None):
    """ VGG_create([, desc[, isigma[, img_normalize[, use_scale_orientation[, scale_factor[, dsc_normalize]]]]]]) -> retval """
    pass

# no classes
