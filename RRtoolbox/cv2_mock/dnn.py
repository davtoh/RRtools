# encoding: utf-8
# module cv2.dnn
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

BLOB_ALLOC_BOTH = 3

Blob_ALLOC_BOTH = 3

BLOB_ALLOC_MAT = 1

Blob_ALLOC_MAT = 1

BLOB_ALLOC_UMAT = 2

Blob_ALLOC_UMAT = 2

BLOB_HEAD_AT_MAT = 1

Blob_HEAD_AT_MAT = 1
Blob_HEAD_AT_UMAT = 2

BLOB_HEAD_AT_UMAT = 2

BLOB_SYNCED = 3

Blob_SYNCED = 3
Blob_UNINITIALIZED = 0

BLOB_UNINITIALIZED = 0

EltwiseLayer_MAX = 2
EltwiseLayer_PROD = 0
EltwiseLayer_SUM = 1

ELTWISE_LAYER_MAX = 2
ELTWISE_LAYER_PROD = 0
ELTWISE_LAYER_SUM = 1

LRNLayer_CHANNEL_NRM = 0

LRNLAYER_CHANNEL_NRM = 0

LRNLAYER_SPATIAL_NRM = 1

LRNLayer_SPATIAL_NRM = 1

PoolingLayer_AVE = 1
PoolingLayer_MAX = 0
PoolingLayer_STOCHASTIC = 2

POOLING_LAYER_AVE = 1
POOLING_LAYER_MAX = 0
POOLING_LAYER_STOCHASTIC = 2

__loader__ = None

__spec__ = None

# functions

def AbsLayer_create(): # real signature unknown; restored from __doc__
    """ AbsLayer_create() -> retval """
    pass

def BNLLLayer_create(): # real signature unknown; restored from __doc__
    """ BNLLLayer_create() -> retval """
    pass

def ConcatLayer_create(axis=None): # real signature unknown; restored from __doc__
    """ ConcatLayer_create([, axis]) -> retval """
    pass

def ConvolutionLayer_create(kernel=None, stride=None, pad=None, dilation=None): # real signature unknown; restored from __doc__
    """ ConvolutionLayer_create([, kernel[, stride[, pad[, dilation]]]]) -> retval """
    pass

def createCaffeImporter(prototxt, caffeModel=None): # real signature unknown; restored from __doc__
    """ createCaffeImporter(prototxt[, caffeModel]) -> retval """
    pass

def createTorchImporter(filename, isBinary=None): # real signature unknown; restored from __doc__
    """ createTorchImporter(filename[, isBinary]) -> retval """
    pass

def DeconvolutionLayer_create(kernel=None, stride=None, pad=None, dilation=None): # real signature unknown; restored from __doc__
    """ DeconvolutionLayer_create([, kernel[, stride[, pad[, dilation]]]]) -> retval """
    pass

def initModule(): # real signature unknown; restored from __doc__
    """ initModule() -> None """
    pass

def InnerProductLayer_create(axis=None): # real signature unknown; restored from __doc__
    """ InnerProductLayer_create([, axis]) -> retval """
    pass

def LRNLayer_create(type=None, size=None, alpha=None, beta=None, bias=None, normBySize=None): # real signature unknown; restored from __doc__
    """ LRNLayer_create([, type[, size[, alpha[, beta[, bias[, normBySize]]]]]]) -> retval """
    pass

def LSTMLayer_create(): # real signature unknown; restored from __doc__
    """ LSTMLayer_create() -> retval """
    pass

def MVNLayer_create(normVariance=None, acrossChannels=None, eps=None): # real signature unknown; restored from __doc__
    """ MVNLayer_create([, normVariance[, acrossChannels[, eps]]]) -> retval """
    pass

def Net(): # real signature unknown; restored from __doc__
    """ Net() -> <dnn_Net object> """
    pass

def PoolingLayer_create(type=None, kernel=None, stride=None, pad=None, padMode=None): # real signature unknown; restored from __doc__
    """ PoolingLayer_create([, type[, kernel[, stride[, pad[, padMode]]]]]) -> retval """
    pass

def PoolingLayer_createGlobal(type=None): # real signature unknown; restored from __doc__
    """ PoolingLayer_createGlobal([, type]) -> retval """
    pass

def PowerLayer_create(power=None, scale=None, shift=None): # real signature unknown; restored from __doc__
    """ PowerLayer_create([, power[, scale[, shift]]]) -> retval """
    pass

def readNetFromCaffe(prototxt, caffeModel=None): # real signature unknown; restored from __doc__
    """ readNetFromCaffe(prototxt[, caffeModel]) -> retval """
    pass

def readTorchBlob(filename, isBinary=None): # real signature unknown; restored from __doc__
    """ readTorchBlob(filename[, isBinary]) -> retval """
    pass

def ReLULayer_create(negativeSlope=None): # real signature unknown; restored from __doc__
    """ ReLULayer_create([, negativeSlope]) -> retval """
    pass

def ReshapeLayer_create(newShape, applyingRange=None, enableReordering=None): # real signature unknown; restored from __doc__
    """ ReshapeLayer_create(newShape[, applyingRange[, enableReordering]]) -> retval """
    pass

def RNNLayer_create(): # real signature unknown; restored from __doc__
    """ RNNLayer_create() -> retval """
    pass

def SigmoidLayer_create(): # real signature unknown; restored from __doc__
    """ SigmoidLayer_create() -> retval """
    pass

def SliceLayer_create(axis): # real signature unknown; restored from __doc__
    """ SliceLayer_create(axis) -> retval  or  SliceLayer_create(axis, sliceIndices) -> retval """
    pass

def SoftmaxLayer_create(axis=None): # real signature unknown; restored from __doc__
    """ SoftmaxLayer_create([, axis]) -> retval """
    pass

def SplitLayer_create(outputsCount=None): # real signature unknown; restored from __doc__
    """ SplitLayer_create([, outputsCount]) -> retval """
    pass

def TanHLayer_create(): # real signature unknown; restored from __doc__
    """ TanHLayer_create() -> retval """
    pass

# no classes
