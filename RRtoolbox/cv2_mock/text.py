# encoding: utf-8
# module cv2.text
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

ERFILTER_NM_IHSGrad = 1
ERFILTER_NM_IHSGRAD = 1
ERFILTER_NM_RGBLGRAD = 0
ERFILTER_NM_RGBLGrad = 0

ERGROUPING_ORIENTATION_ANY = 1
ERGROUPING_ORIENTATION_HORIZ = 0

OCR_DECODER_VITERBI = 0

OCR_LEVEL_TEXTLINE = 1
OCR_LEVEL_WORD = 0

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def computeNMChannels(_src, _channels=None, _mode=None):
    """ computeNMChannels(_src[, _channels[, _mode]]) -> _channels """
    pass


# real signature unknown; restored from __doc__
def createERFilterNM1(cb, thresholdDelta=None, minArea=None, maxArea=None, minProbability=None, nonMaxSuppression=None, minProbabilityDiff=None):
    """ createERFilterNM1(cb[, thresholdDelta[, minArea[, maxArea[, minProbability[, nonMaxSuppression[, minProbabilityDiff]]]]]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createERFilterNM2(cb, minProbability=None):
    """ createERFilterNM2(cb[, minProbability]) -> retval """
    pass


# real signature unknown; restored from __doc__
def createOCRHMMTransitionsTable(vocabulary, lexicon):
    """ createOCRHMMTransitionsTable(vocabulary, lexicon) -> retval """
    pass


# real signature unknown; restored from __doc__
def detectRegions(image, er_filter1, er_filter2):
    """ detectRegions(image, er_filter1, er_filter2) -> regions """
    pass


# real signature unknown; restored from __doc__
def erGrouping(image, channel, regions, method=None, filename=None, minProbablity=None):
    """ erGrouping(image, channel, regions[, method[, filename[, minProbablity]]]) -> groups_rects """
    pass


def loadClassifierNM1(filename):  # real signature unknown; restored from __doc__
    """ loadClassifierNM1(filename) -> retval """
    pass


def loadClassifierNM2(filename):  # real signature unknown; restored from __doc__
    """ loadClassifierNM2(filename) -> retval """
    pass


# real signature unknown; restored from __doc__
def loadOCRBeamSearchClassifierCNN(filename):
    """ loadOCRBeamSearchClassifierCNN(filename) -> retval """
    pass


# real signature unknown; restored from __doc__
def loadOCRHMMClassifierCNN(filename):
    """ loadOCRHMMClassifierCNN(filename) -> retval """
    pass


# real signature unknown; restored from __doc__
def loadOCRHMMClassifierNM(filename):
    """ loadOCRHMMClassifierNM(filename) -> retval """
    pass


# real signature unknown; restored from __doc__
def OCRBeamSearchDecoder_create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode=None, beam_size=None):
    """ OCRBeamSearchDecoder_create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table[, mode[, beam_size]]) -> retval """
    pass


# real signature unknown; restored from __doc__
def OCRHMMDecoder_create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode=None):
    """ OCRHMMDecoder_create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table[, mode]) -> retval """
    pass


# real signature unknown; restored from __doc__
def OCRTesseract_create(datapath=None, language=None, char_whitelist=None, oem=None, psmode=None):
    """ OCRTesseract_create([, datapath[, language[, char_whitelist[, oem[, psmode]]]]]) -> retval """
    pass

# no classes
