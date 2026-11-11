# -*- coding: utf-8 -*-
"""
Signal processing, vector calculations, histogram analytics, and noise operations.
"""

import numpy as np

try:
    from skimage.util import view_as_windows, view_as_blocks
except ImportError:
    from numpy.lib.stride_tricks import as_strided as _ast

    def view_as_blocks(arr_in, block_shape=(3, 3)):
        """
        Provide a 2D block_shape view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of arr_in.

        :param arr_in:
        :param block_shape:
        :return:
        """
        # http://stackoverflow.com/a/5078155/5288758
        shape = (int(arr_in.shape[0] / block_shape[0]),
                 int(arr_in.shape[1] / block_shape[1])) + block_shape
        strides = (int(block_shape[0] * arr_in.strides[0]),
                   int(block_shape[1] * arr_in.strides[1])) + arr_in.strides
        return _ast(arr_in, shape=shape, strides=strides)

    def view_as_windows(arr_in, window_shape, step=1):
        """
        Provide a 2D block_shape rolling view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of arr_in.

        :param arr_in:
        :param window_shape:
        :param step:
        :return:
        """
        # taken from /skimage/util/shape.py
        arr_shape = np.array(arr_in.shape)
        arr_in = np.ascontiguousarray(arr_in)

        new_shape = (tuple((arr_shape - window_shape) // step + 1) +
                     tuple(window_shape))

        arr_strides = np.array(arr_in.strides)
        new_strides = np.concatenate((arr_strides * step, arr_strides))

        arr_out = _ast(arr_in, shape=new_shape, strides=new_strides)

        return arr_out


def isnumpy(arr):
    """
    Test whether an object is a numpy array.

    :param arr:
    :return: True if numpy array, else false.
    """
    return type(arr).__module__ == np.__name__


def anorm2(a):
    """
    Summation of squares (helper function for :func:`anorm`)

    :param a:
    :return:
    """
    return (np.square(a)).sum(-1)


def anorm(a):
    """
    norm in array.

    :param a:
    :return:
    """
    return np.sqrt(anorm2(a))


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def angle(v1, v2, deg=False):
    """
    Angle between two N dimmensional vectors.

    :param v1: vector 1.
    :param v2: vector 2.
    :param deg: if True angle is in Degrees, else radians.
    :return: angle in radians.

    .. note:: obtained from http://stackoverflow.com/a/13849249/5288758
            and tested in http://onlinemschool.com/math/assistance/vector/angl/
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    a = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if deg:
        return np.rad2deg(a)
    return a


def angle2D(v1, v2, deg=False, absolute=None):
    """
    Angle between two 2 dimensional vectors.

    :param v1: vector 1.
    :param v2: vector 2.
    :param deg: if True angle is in Degrees, else radians.
    :param absolute: if None returns the angle (0 yo 180(pi)) between v1 and v2.
                    if True returns the absolute angle (0 to 360(2pi)) from v1 as axis to v2.
                    if False returns the angle (0 to 180 or 0 to -180) from v1 as axis to v2,
                    where v2 angle relative to v1 is positive or negative if counter-clock or clock wise.
    :return: angle in radians.

    .. note:: implemented according to http://math.stackexchange.com/a/747992
            and tested in http://onlinemschool.com/math/assistance/vector/angl/
    """
    cross = np.cross(v1, v2)
    if cross.size != 1:
        raise Exception("angle2D receives only cartesian coordinates")
    mag = 1
    if absolute is not None:
        if cross == 0:
            for i in np.array(v1) + v2:
                if i < 0:
                    mag = -1
                    break
        elif cross < 0:
            mag = -1
    if absolute and mag < 0:
        return 360.0 - angle(v1=v1, v2=v2, deg=deg)
    return mag * angle(v1=v1, v2=v2, deg=deg)


def relativeVectors(pts, all=True):
    """
    Form vectors from points.

    :param pts: array of points [p0, ... ,(x,y)].
    :param all: (True) if True adds last vector from last and first point.
    :return: array of vectors [V0, ... , (V[n] = x[n+1]-x[n],y[n+1]-y[n])].
    """
    pts = np.array(pts)
    if all:
        pts = np.append(pts, [pts[0]], axis=0)
    return np.stack([np.diff(pts[:, 0]), np.diff(pts[:, 1])], 1)


def vertexesAngles(pts, dtype=None, deg=False):
    """
    Relative angle of vectors formed by vertexes (where vectors cross).

    :param pts: points seen as vertexes.
    :param dtype: return array of type supported by numpy.
    :param deg: if True angle is in Degrees, else radians.
    :return: angles.
    """
    vs = relativeVectors(pts, all=True)
    vs = np.roll(np.append(vs, [vs[-1]], axis=0), 2)
    return np.array([angle(vs[i - 1], vs[i], deg) for i in range(1, len(vs))], dtype)


def vectorsAngles(pts, ptaxis=(1, 0), origin=(0, 0), dtype=None, deg=False, absolute=None):
    """
    Angle of formed vectors in Cartesian plane with respect to formed axis vector.

    :param pts: points to form vectors from origin
    :param ptaxis: point to form axis from origin
    :param origin: origin
    :param dtype: return array of type supported by numpy.
    :param deg: if True angle is in Degrees, else radians.
    :param absolute: angle mode
    :return:
    """
    if np.sum(origin) == 0:
        return np.array([angle2D(v1=ptaxis, v2=i, deg=deg, absolute=absolute) for i in np.float32(pts)], dtype)
    ptaxis = np.float32(ptaxis) - origin
    return np.array([angle2D(v1=ptaxis, v2=i - origin, deg=deg, absolute=absolute) for i in np.float32(pts)], dtype)


def entroyTest(arr):
    """
    Entropy test of intensity arrays. (Helper function for :func:`entropy`)

    :param arr: array MxN of dim 2.
    :return: entropy.
    """
    N = arr.shape[0] * arr.shape[1]
    E = 0.0
    for i in range(256):
        Ni = np.count_nonzero(arr == i)
        if Ni:
            pi = Ni / N
            E += -pi * np.log(pi)
    return E


def normalize(arr):
    """Normalize array to ensure range [0,1]"""
    arr = np.float32(arr - np.min(arr))
    return arr / np.max(arr)


def normalize2(arr):
    """
    Normalize with factor of absolute maximum value.

    .. notes:: it does not ensures range [0,1], array must be shifted to 0 to achieve this.
    """
    return arr / np.max([np.max(arr), -1.0 * np.min(arr)])


def rescale(arr, max=1, min=0):
    """
    Rescales array values to range [min,max].

    :param arr: array.
    :param max: maximum value in range.
    :param min: minimum value in range.
    :return: rescaled array.
    """
    return (max - min) * normalize(arr) + min


def normalizeCustom(arr, by=np.max, axis=None):
    """
    Normalize array with custom operations.

    :param arr: array.
    :param by: np.max, np.sum or any function to obtain factor.
    :param axis: if None normalizes in all axes else in selected axis.
    :return: normalized array.
    """
    factor = by(arr, axis=axis)
    if axis is None:
        return np.float32(arr) / factor
    return np.float32(arr) / np.expand_dims(factor, axis=axis)


def instability_bf(funcs, step=10, maximum=300, guess=0, tolerance=0.01):
    """
    Find the instability of function approaching value by brute force.

    :param funcs: list of functions
    :param step: (10) step to close guess to maximum
    :param maximum: (300) maximum value
    :param guess: (0) initial guess
    :param tolerance: (0.01) tolerance
    :return: (state, updated guess)
    """
    if guess < maximum:
        s = 1
    else:
        s = -1
    step = s * abs(step)
    while s * (maximum - guess) > 0:
        val = np.linspace(start=guess, stop=maximum)
        val = np.sum(np.abs(np.diff(np.array([f(val) for f in funcs]))))
        if val <= tolerance:
            return True, guess
        guess += step
    return False, guess


def get_x_space(funcs, step=10, xleft=-300, xright=300):
    """
    get X axis space by brute force.

    :param funcs: list of functions
    :param step: (10) step to close guess to maximum
    :param xleft: maximum left limit
    :param xright: maximum right limit
    :return: linspace
    """
    assert xleft < xright
    l1 = instability_bf(funcs, step=step, maximum=xleft, guess=0)[1]
    l2 = instability_bf(funcs, step=step, maximum=xright, guess=0)[1]
    return np.linspace(l1, l2, int(l2 - l1))


def histogram(img):
    """
    Get image histogram.

    :param img: gray or image with any bands
    :return: histogram of every band
    """
    sz = img.shape
    if len(sz) == 2:
        histr = [np.histogram(img.flatten(), 256, [0, 256])[0]]
    else:
        histr = [np.histogram(img[:, :, i].flatten(), 256, [0, 256])[0] for i in range(sz[2])]
    return histr


def find_near(m, thresh=None, side=None):
    """
    helper function for findminima and findmaxima
    :param m: minima or maxima points
    :param thresh: guess or seed point
    :param side: left or right
    :return: value
    """
    if thresh is None:
        return m
    if side == "left":
        if thresh > m[0]:
            m = m[m <= thresh]
        else:
            return m[0]
    if side == "right":
        if thresh < m[-1]:
            m = m[m >= thresh]
        else:
            return m[-1]
    distances = np.sqrt((m - thresh) ** 2)
    idx = np.where(np.min(distances) == distances)[0]
    return m[idx][0]


def findminima(hist, thresh=None, side=None):
    """
    Get nearest valley value to a thresh point from a histogram.

    :param hist: histogram
    :param thresh: initial seed
    :param side: find valley from left or right of thresh
    :return:
    """
    # http://stackoverflow.com/a/9667121/5288758
    mn = (np.diff(np.sign(np.diff(hist))) > 0).nonzero()[0] + 1
    return find_near(mn, thresh, side)


def findmaxima(hist, thresh=None, side=None):
    """
    Get nearest peak value to a thresh point from a histogram.

    :param hist: histogram
    :param thresh: initial seed
    :param side: find valley from left or right of thresh
    :return:
    """
    # http://stackoverflow.com/a/9667121/5288758
    mn = (np.diff(np.sign(np.diff(hist))) < 0).nonzero()[0] + 1
    return find_near(mn, thresh, side)


def getOtsuThresh(hist):
    """
    From histogram calculate Otsu threshold value.

    :param hist: histogram
    :return: otsu threshold value
    """
    # http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    hist_norm = hist.astype(np.float32).ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(len(hist_norm))
    fn_min = np.inf
    thresh = -1
    for i in range(1, len(hist_norm)):
        p1, p2 = np.hsplit(hist_norm, [i])
        q1, q2 = Q[i], Q[len(hist_norm) - 1] - Q[i]
        b1, b2 = np.hsplit(bins, [i])
        if q1 == 0 or q2 == 0:
            continue
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = (np.sum(((b1 - m1)**2) * p1) / q1, np.sum(((b2 - m2)**2) * p2) / q2)
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def noisy(arr, mode):
    """
    Add noise to arrays

    :param arr: Input ndarray data.
    :param mode: noise method ('gauss', 'poisson', 's&p', 'speckle').
    :return: noisy arr

    .. notes:: code was based from http://stackoverflow.com/a/30609854/5288758
    """
    if mode == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, arr.shape)
        gauss = gauss.reshape(*arr.shape)
        return arr + gauss
    elif mode == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        num_salt = np.ceil(amount * arr.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
        arr[coords] = 1

        num_pepper = np.ceil(amount * arr.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
        arr[coords] = 0
        return arr
    elif mode == "poisson":
        vals = len(np.unique(arr))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(arr * vals) / float(vals)
    elif mode == "speckle":
        gauss = np.random.randn(*arr.shape)
        gauss = gauss.reshape(*arr.shape)
        return arr + arr * gauss
    else:
        raise Exception(f"mode {mode} not recognized")


def process_as_blocks(arr, func, block_shape=(3, 3), mask=None, asWindows=False):
    """
    process with function over an array using blocks (using re-striding).

    :param arr: array to process
    :param func: function to feed blocks
    :param block_shape: (3,3) shape of blocks
    :param mask: (None) mask to process arr
    :param asWindows: (False) if True all blocks overlap
    :return: processed array.
    """
    result = np.zeros_like(arr, dtype=np.float32)
    if asWindows:
        blocks_in = view_as_windows(arr, block_shape)
        if mask is not None:
            blocks_mask = view_as_windows(mask, block_shape)
        sr, sc = blocks_in.shape[:2]
        for row in range(sr):
            for col in range(sc):
                b_in = blocks_in[row, col]
                if mask is not None:
                    b_mask = blocks_mask[row, col]
                    indexes = b_mask == 1
                    if np.any(indexes):
                        nrow, ncol = (row + block_shape[0] // 2, col + block_shape[1] // 2)
                        result[nrow, ncol] = func(b_in[indexes])
                else:
                    nrow, ncol = (row + block_shape[0] // 2, col + block_shape[1] // 2)
                    result[nrow, ncol] = func(b_in)
    else:
        blocks_in = view_as_blocks(arr, block_shape)
        blocks_result = view_as_blocks(result, block_shape)
        if mask is not None:
            blocks_mask = view_as_blocks(mask, block_shape)
        sr, sc = blocks_in.shape[:2]
        for row in range(sr):
            for col in range(sc):
                b_in = blocks_in[row, col]
                b_result = blocks_result[row, col]
                if mask is not None:
                    b_mask = blocks_mask[row, col]
                    indexes = b_mask == 1
                    if np.any(indexes):
                        b_result[indexes] = func(b_in[indexes])
                else:
                    b_result[:, :] = func(b_in)
    return result

