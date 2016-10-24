# -*- coding: utf-8 -*-
"""
    This module contains custom 1D adn 2D-array filters and pre-processing (as in filtering phase) methods
"""
from __future__ import division

import cv2
import numpy as np
try:
    from sympy.functions import exp
except ImportError: # if not sympy installed
    #from math import exp
    exp = np.exp

#from RRtoolbox.lib.cache import memoize
#from RRtoolbox.lib.config import MANAGER

__author__ = 'Davtoh'

#@memoize(MANAGER["TEMPPATH"]) # convert cv2.bilateralfilter to memoized bilateral filter
def bilateralFilter(im,d,sigmaColor,sigmaSpace):
    """
    Apply bilateral Filter.

    :param im:
    :param d:
    :param sigmaColor:
    :param sigmaSpace:
    :return: filtered image
    """
    return cv2.bilateralFilter(im,d,sigmaColor,sigmaSpace)

def normsigmoid(x, alpha, beta):
    """
    Apply normalized sigmoid filter.

    :param x: data to apply filter
    :param alpha: if alpha > 0: pass high filter, if alpha < 0: pass low filter, alpha must be != 0
    :param beta: shift from origin
    :return: filtered values normalized to range [-1 if x<0, 1 if x>=0]
    """
    try:
        return 1/(np.exp((beta-x) / alpha) + 1)
    except:
        return 1/(exp((beta-x) / alpha) + 1)

class FilterBase(object):
    """
    base filter to create custom filters
    """
    def __init__(self, alpha=None, beta1=None, beta2 = None):
        """

        :param alpha:
        :param beta1:
        :param beta2:
        :return:
        """
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        if alpha is not None: self.alpha = alpha
        if beta1 is not None: self.beta1 = beta1
        if beta2 is not None: self.beta2 = beta2

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self,value):
        self._test_alpha(value)
        self._alpha = value
    @alpha.deleter
    def alpha(self):
        del self._alpha

    @property
    def beta1(self):
        return self._beta1
    @beta1.setter
    def beta1(self,value):
        if self._beta2 is not None:
            self._test_beta1(value)
        self._beta1 = value
    @beta1.deleter
    def beta1(self):
        del self._beta1

    @property
    def beta2(self):
        return self._beta2
    @beta2.setter
    def beta2(self,value):
        if self._beta1 is not None:
            self._test_beta2(value)
        self._beta2 = value
    @beta2.deleter
    def beta2(self):
        del self._beta2

    def _test_alpha(self, value):
        assert value>0
    def _test_beta1(self, value):
        assert self.beta2>value
    def _test_beta2(self, value):
        assert value>self._beta1

    def __call__(self, levels):
        raise NotImplementedError

class Lowpass(FilterBase):
    """
    Lowpass filter (recommended to use float types)
    """
    def __init__(self, alpha, beta1):
        super(Lowpass, self).__init__(alpha=alpha, beta1=beta1)

    def _test_beta1(self, value):
        pass
    def _test_beta2(self, value):
        raise Exception("Lowpass filter does not implement beta2")

    def __call__(self, levels):
        return normsigmoid(levels, -self._alpha, self._beta1)

class Highpass(FilterBase):
    """
    Highpass filter (recommended to use float types)
    """
    def __init__(self, alpha, beta1):
        super(Highpass, self).__init__(alpha=alpha, beta1=beta1)

    def _test_beta1(self, value):
        pass
    def _test_beta2(self, value):
        raise Exception("Highpass filter does not implement beta2")

    def __call__(self, levels):
        return normsigmoid(levels, self.alpha, self._beta1)

class Bandstop(FilterBase):
    """
    Bandstop filter (recommended to use float types)
    """
    def __init__(self, alpha, beta1, beta2):
        super(Bandstop, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)
    def __call__(self, levels):
        return normsigmoid(levels, -self._alpha, self._beta1) - normsigmoid(levels, -self._alpha, self._beta2) + 1.0

class Bandpass(FilterBase):
    """
    Bandpass filter (recommended to use float types)
    """
    def __init__(self, alpha, beta1, beta2):
        super(Bandpass, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)
    def __call__(self, levels):
        return normsigmoid(levels,self._alpha,self._beta1)-normsigmoid(levels,self._alpha,self._beta2)

class InvertedBandstop(Bandstop):
    """
    inverted Bandstop filter (recommended to use float types)
    """
    def __call__(self, levels):
        return normsigmoid(levels,-self._alpha,self._beta2)-normsigmoid(levels,-self._alpha,self._beta1)-1.0

class InvertedBandpass(Bandpass):
    """
    inverted Bandpass filter (recommended to use float types)
    """
    def __call__(self, levels):
        return normsigmoid(levels,self._alpha,self._beta2)-normsigmoid(levels,self._alpha,self._beta1)

def filterFactory(alpha, beta1, beta2=None):
    """
    Make filter.

    :param alpha: steepness of filter
    :param beta1: first shift from origin
    :param beta2: second shift from origin::

        alpha must be != 0
        if beta2 = None:
            if alpha > 0: high-pass filter, if alpha < 0: low-pass filter
        else:
            if beta2 > beta1:
                if alpha > 0: band-pass filter, if alpha < 0: band-stop filter
            else:
                if alpha > 0: inverted-band-pass filter, if alpha < 0: inverted-band-stop filter
    :return: filter funtion with intup levels

    Example::

        alpha,beta1,beta2 = 10,20,100
        myfilter = filter(alpha,beta1,beta2)
        print myfilter,type(myfilter)
        print myfilter.alpha,myfilter.beta1,myfilter.beta2
    """
    #http://en.wikipedia.org/wiki/Filter_%28signal_processing%29
    if beta2 is None:
        if alpha < 0: # low pass
            func = Lowpass(alpha=-alpha, beta1=beta1)
        else: # high pass
            func = Highpass(alpha=alpha, beta1=beta1)
    else:
        if beta2>beta1:
            if alpha < 0: # band stop
                func = Bandstop(alpha=-alpha, beta1=beta1, beta2=beta2)
            else: # band pass
                func = Bandpass(alpha=alpha, beta1=beta1, beta2=beta2)
        else: # inverted
            beta1,beta2 = beta2,beta1
            if alpha < 0: # inverted band stop
                func = InvertedBandstop(alpha=-alpha, beta1=beta1, beta2=beta2)
            else: # inverted band pass
                func = InvertedBandpass(alpha=alpha, beta1=beta1, beta2=beta2)
    return func

def sigmoid(x,alpha,beta,max=255,min=0):
    """
    Apply sigmoid filter.

    :param x: data to apply filter
    :param alpha: if alpha > 0: pass high filter, if alpha < 0: pass low filter, alpha must be != 0
    :param beta: shift from origin
    :param max: maximum output value
    :param min: minimum output value
    :return: filtered values ranging as [min,max]

    .. note:: Based from http://www.itk.org/Doxygen/html/classitk_1_1SigmoidImageFilter.html
    """
    if min: return (max-min)*normsigmoid(x,alpha,beta)+min
    else: return max*normsigmoid(x,alpha,beta)  # speeds up operation

def smooth(x,window_len=11,window='hanning', correct = False):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    Example::

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

    .. seealso:: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve, scipy.signal.lfilter

    .. note:: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    #TODO: the window parameter could be the window itself if an array instead of a string

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    if correct:
        displaced = len(y) - len(x) # how much it was expanded, increased or displaced
        y = y[displaced // 2:len(y) - (displaced - displaced // 2)] # adjusted to original size
    return y

class BilateraParameter(Bandstop):
    """
    bilateral parameter
    """
    def __init__(self, scale, shift = 33, name=None, alpha=100, beta1=-400, beta2=200):
        super(BilateraParameter, self).__init__(alpha=alpha, beta1=beta1, beta2=beta2)
        self.scale = scale
        self.shift = shift
        self.name = name
    def __call__(self, levels):
        return np.int32(super(BilateraParameter, self).__call__(
            levels) * self.scale + self.shift)

class BilateralParameters(object):
    """
    create instance to calculate bilateral
    parameters from image shape.

    d -> inf then:
        * computation is slower
        * filtering is better to eliminate noise
        * images look more cartoon-like

    :param d: distance
    :param sigmaColor: sigma in color
    :param sigmaSpace: sigma in space

    """
    d = BilateraParameter(scale = 31, shift=15, alpha=150, beta1 = 60, beta2=800, name ="d")
    sigmaColor = BilateraParameter(scale = 50, name ="sigmaColor")
    sigmaSpace = BilateraParameter(scale = 25, name ="sigmaSpace")

    def __init__(self,d=None,sigmaColor=None,sigmaSpace=None):
        """
        replace bilateral limit parameters. It can be a
        instance from BilateraParameter or a value.
        """
        if d is not None:
            self.d = d
        if sigmaColor is not None:
            self.sigmaColor = sigmaColor
        if sigmaSpace is not None:
            self.sigmaSpace = sigmaSpace

    @property
    def filters(self):
        """
        list of filters
        """
        def create_func(val):
            def custom_val(x):
                return np.ones_like(x)*val
            custom_val.name = "Custom value {}".format(val)
            return custom_val
        fs = []
        for i,f in enumerate((self.d, self.sigmaColor, self.sigmaSpace)):
            if callable(f):
                fs.append(f)
            elif i == 0:
                fs.append(create_func(self.d))
            elif i == 1:
                fs.append(create_func(self.sigmaColor))
            elif i == 2:
                fs.append(create_func(self.sigmaSpace))
        return fs

    def __call__(self, shape):
        """
        calculate bilateral parameters from image shape.

        :param shape: image shape
        :return: d,sigmaColor,sigmaSpace
        """
        return [i(np.min(shape[:2])) for i in self.filters]

def getBilateralParameters(shape=None,mode=None):
    """
    Calculate from shape bilateral parameters.

    :param shape: image shape. if None it returns the instace to use with shapes.
    :param mode: "mild", "heavy" or "normal" to process noise
    :return: instance or parameters
    """
    #15,82,57 # 21,75,75 # faster and for low noise
    if mode == "mild" or mode is None:
        mode = 9
    elif mode == "heavy":
        mode = None
    elif mode == "normal":
        mode = 27
    elif isinstance(mode,(int,float)):
        pass
    else:
        raise Exception("Bilateral Parameter mode '{}' not recognised".format(mode))
    bp = BilateralParameters(mode)
    if shape is None:
        return bp
    return bp(shape[:2])

