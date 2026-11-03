# -*- coding: utf-8 -*-
r"""
Format conversions, color definitions, and array conversions for images.
"""
from __future__ import division
import numpy as np
import cv2
import matplotlib.axes
import matplotlib.figure

supported_formats = ("bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png",
                     "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif")


def transposeIm(im):
    if len(im.shape) == 2:
        return im.transpose(1, 0)
    else:
        return im.transpose(1, 0, 2)


# colors to use
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)
orange = (51, 103, 236)
black = (0, 0, 0)
blue = (255, 0, 0)

# dictionary of colors to use
colors = {
    "blue": blue,
    "green": green,
    "red": red,
    "white": white,
    "orange": orange,
    "black": black}

# look for these as <numpy array>.dtype.names
bgra_dtype = np.dtype({'b': (np.uint8, 0), 'g': (
    np.uint8, 1), 'r': (np.uint8, 2), 'a': (np.uint8, 3)})


def plt2bgr(image):
    """
    Convert a Matplotlib plot/figure to a BGR image.
    """
    if isinstance(image, matplotlib.axes.SubplotBase):
        image = fig2bgr(image.figure)
    elif isinstance(image, matplotlib.figure.Figure):
        image = fig2bgr(image)
    return image


def plt2bgra(image):
    """
    Convert a Matplotlib plot/figure to a BGRA image.
    """
    if isinstance(image, matplotlib.axes.SubplotBase):
        image = fig2bgra(image.figure)
    elif isinstance(image, matplotlib.figure.Figure):
        image = fig2bgra(image)
    return image


def fig2bgr(fig):
    """
    Convert a Matplotlib figure to a RGB image.

    :param fig: a matplotlib figure
    :return: RGB image.
    """
    fig.canvas.draw()
    buf = np.fromstring(fig.canvas.tostring_rgb(),
                        dtype=np.uint8, sep='')  # get bgr
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def np2str(arr):
    if hasattr(arr, "tobytes"):
        return arr.tobytes()
    return arr.tostring()


def str2np(string, shape):
    if hasattr(np, "frombuffer"):
        buf = np.frombuffer(string, dtype=np.uint8)
    else:
        buf = np.fromstring(string, dtype=np.uint8, sep='')  # get bgr
    return buf.reshape(shape)


def fig2bgra(fig):
    """
    Convert a Matplotlib figure to a RGBA image.

    :param fig: a matplotlib figure
    :return: RGBA image.
    """
    # http://www.icare.univ-lille1.fr/drupal/node/1141
    # http://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)  # get bgra
    buf = buf.reshape(fig.canvas.get_width_height()[
                      ::-1] + (4,))  # reshape to h,w,c
    return np.roll(buf, 3, axis=2)  # correct channels


def qi2np(qimage, dtype='array'):
    """
    Convert QImage to numpy.ndarray.  The dtype defaults to uint8
    for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
    for 32bit color images.  You can pass a different dtype to use, or
    'array' to get a 3D uint8 array for color images.

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    from pyqtgraph import QtGui
    if qimage.isNull():
        raise IOError("Image is Null")
    result_shape = (qimage.height(), qimage.width())
    temp_shape = (qimage.height(), qimage.bytesPerLine() * 8 / qimage.depth())
    if qimage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                           QtGui.QImage.Format_ARGB32,
                           QtGui.QImage.Format_RGB32):
        if dtype == 'rec':
            dtype = bgra_dtype
        elif dtype == 'array':
            dtype = np.uint8
            result_shape += (4, )
            temp_shape += (4, )
    elif qimage.format() == QtGui.QImage.Format_Indexed8:
        dtype = np.uint8
    else:
        raise ValueError("qi2np only supports 32bit and 8bit images")
    # FIXME: raise error if alignment does not match
    buf = qimage.bits().asstring(qimage.numBytes())
    result = np.frombuffer(buf, dtype).reshape(result_shape)
    if result_shape != temp_shape:
        result = result[:, :result_shape[1]]
    if qimage.format() == QtGui.QImage.Format_RGB32 and dtype == np.uint8:
        result = result[..., :3]
    return result


def np2qi(array):
    """
    Convert numpy array to Qt Image.

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py

    :param array:
    :return:
    """
    if np.ndim(array) == 2:
        return gray2qi(array)
    elif np.ndim(array) == 3:
        return rgb2qi(array)
    raise ValueError("can only convert 2D or 3D arrays")


def gray2qi(gray):
    """
    Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!).

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    from pyqtgraph import QtGui
    if len(gray.shape) != 2:
        raise ValueError("gray2QImage can only convert 2D arrays")

    gray = np.require(gray, np.uint8, 'C')

    h, w = gray.shape

    result = QtGui.QImage(gray.data, w, h, QtGui.QImage.Format_Indexed8)
    result.ndarray = gray  # let object live to avoid garbage collection
    """
    for i in xrange(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())"""
    return result


def rgb2qi(rgb):
    """
    Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!).

    source from: https://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    """
    from pyqtgraph import QtGui
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain "
                         "exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape
    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[:, :, 2] = rgb[:, :, 2]
    bgra[:, :, 1] = rgb[:, :, 1]
    bgra[:, :, 0] = rgb[:, :, 0]
    # dstack, dsplit, stack
    if rgb.shape[2] == 3:
        bgra[..., 3].fill(255)
        fmt = QtGui.QImage.Format_RGB32
    else:
        bgra[..., 3] = rgb[..., 3]
        fmt = QtGui.QImage.Format_ARGB32
    result = QtGui.QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra  # let object live to avoid garbage collection
    return result


def bgra2bgr(im, bgrcolor=colors["white"]):
    """
    Convert BGR to BGRA image.

    :param im: image
    :param bgrcolor: BGR color representing transparency. (information is lost when
                    converting BGRA to BGR) e.g. [200,200,200].
    :return:
    """
    # back[chanel] = bgr[chanel]*(bgr[3]/255.0) + back[chanel]*(1-bgr[3]/255.0)
    temp = im.shape
    im2 = np.zeros((temp[0], temp[1], 3), np.uint8)
    im2[:, :, :] = bgrcolor
    for c in range(0, 3):  # looping over channels
        im2[:, :, c] = (im[:, :, c] * (im[:, :, 3] / 255.0) + im2[:, :, c] *
                        (1.0 - im[:, :, 3] / 255.0))
    return im2


def hist_match(source, template, alpha=None):
    """
    Adjust the pixel values of an image to match those of a template image.

    :param source: image to transform colors to template
    :param template: template image ()
    :param alpha:
    :return: transformed source
    """
    # theory https://en.wikipedia.org/wiki/Histogram_matching
    # explanation http://dsp.stackexchange.com/a/16175 and
    # http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html
    # based on implementation http://stackoverflow.com/a/33047048/5288758
    # see http://www.mathworks.com/help/images/ref/imhistmatch.html
    if len(source.shape) > 2:
        matched = np.zeros_like(source)
        for i in range(3):
            matched[:, :, i] = hist_match(source[:, :, i], template[:, :, i])
        return matched
    else:
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)  # reconstruct image
