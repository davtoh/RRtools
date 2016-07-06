"""
runs CLAHE equalization example based from tests/Ex_histogram_equalization.py
"""

from preamble import *
from tests.tesisfunctions import hist_cdf, equalization

script_name = os.path.basename(__file__).split(".")[0]

def applyCHAHE(img,clipLimit = 2.0, tileGridSize = (8,8), useHSV = False):
    """
    apply Contrast Limited Adaptive Histogram Equalization to BGR images

    :param img: BGR image
    :param clipLimit: level limit
    :param tileGridSize: tile size
    :param useHSV: True to process BGR image as SVH image.
    :return: equalized image.
    """
    clahe = cv2.createCLAHE(clipLimit,tileGridSize)
    if useHSV:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for i in xrange(img.shape[2]):
        img[:,:,i] = clahe.apply(img[:,:,i])
    if useHSV:
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def graph(pytex =None, name = script_name, fn = None, clipLimit = 4., tileGridSize = (5,5), useHSV = False, figsize=(20,10)):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param fn: (None) file name of figure to load
    :param figsize: figure shape (width,height) in inches
    :return: locals()
    """
    # http://funcvis.org/blog/?p=54
    gd = graph_data(pytex)
    if fn is None:
        fn = "im1_1.jpg"
    gd.load(fn)

    figure(figsize=figsize)#
    img = gd.BGR
    subplot(121),imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    title(gd.wrap_title("Input"))
    xticks([]), yticks([])

    subplot(122),imshow(cv2.cvtColor(applyCHAHE(img,clipLimit=clipLimit,
                                                tileGridSize=tileGridSize,
                                                useHSV=useHSV),cv2.COLOR_BGR2RGB))
    title(gd.wrap_title("CLAHE"))
    xticks([]), yticks([])

    gd.output(name)
    return locals()


if __name__ == "__main__":
    graph(pytex.context, useHSV = False)