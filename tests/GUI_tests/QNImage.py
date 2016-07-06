import numpy as np
from pyqtgraph import QtGui
QImage = QtGui.QImage

# look for these as <numpy array>.dtype.names
bgra_dtype = np.dtype({'b': (np.uint8, 0), 'g': (np.uint8, 1), 'r': (np.uint8, 2), 'a': (np.uint8, 3)})

def qi2np(qimage, dtype ='array'):
    """
    Convert QImage to numpy.ndarray.  The dtype defaults to uint8
    for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
    for 32bit color images.  You can pass a different dtype to use, or
    'array' to get a 3D uint8 array for color images.
    """
    if qimage.isNull():
        raise IOError("Image is Null")
    result_shape = (qimage.height(), qimage.width())
    temp_shape = (qimage.height(), qimage.bytesPerLine() * 8 / qimage.depth())
    if qimage.format() in (QImage.Format_ARGB32_Premultiplied,
                            QImage.Format_ARGB32,
                            QImage.Format_RGB32):
        if dtype == 'rec':
            dtype = bgra_dtype
        elif dtype == 'array':
            dtype = np.uint8
            result_shape += (4, )
            temp_shape += (4, )
    elif qimage.format() == QImage.Format_Indexed8:
        dtype = np.uint8
    else:
        raise ValueError("qi2np only supports 32bit and 8bit images")
    # FIXME: raise error if alignment does not match
    buf = qimage.bits().asstring(qimage.numBytes())
    result = np.frombuffer(buf, dtype).reshape(result_shape)
    if result_shape != temp_shape:
        result = result[:,:result_shape[1]]
    if qimage.format() == QImage.Format_RGB32 and dtype == np.uint8:
        result = result[...,:3]
    return result

def np2qi(array):
    if np.ndim(array) == 2:
        return gray2qi(array)
    elif np.ndim(array) == 3:
        return rgb2qi(array)
    raise ValueError("can only convert 2D or 3D arrays")

def gray2qi(gray):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(gray.shape) != 2:
        raise ValueError("gray2QImage can only convert 2D arrays")

    gray = np.require(gray, np.uint8, 'C')

    h, w = gray.shape

    result = QImage(gray.data, w, h, QImage.Format_Indexed8)
    result.ndarray = gray # let object live to avoid garbage collection
    """
    for i in xrange(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())"""
    return result

def rgb2qi(rgb):
    """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape
    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[:,:,2] = rgb[:,:,2]
    bgra[:,:,1] = rgb[:,:,1]
    bgra[:,:,0] = rgb[:,:,0]
    # dstack, dsplit, stack
    if rgb.shape[2] == 3:
        bgra[...,3].fill(255)
        fmt = QtGui.QImage.Format_RGB32
    else:
        bgra[...,3] = rgb[...,3]
        fmt = QtGui.QImage.Format_ARGB32
    result = QtGui.QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra  # let object live to avoid garbage collection
    return result

if __name__ == '__main__':

    import sys
    import time
    app = QtGui.QApplication(sys.argv)

    def makeIm():
        spectroWidth=1000
        spectroHeight=1000
        a= np.random.random(spectroHeight * spectroWidth) * 255
        a=np.reshape(a, (spectroHeight, spectroWidth))
        a=np.require(a, np.uint8, 'C')
        COLORTABLE=[]
        for i in range(256): COLORTABLE.append(QtGui.qRgb(i/4,i,i/2))
        a=np.roll(a, -5)
        QI = QtGui.QImage(a.data, spectroWidth, spectroHeight, QtGui.QImage.Format_Indexed8)
        QI.setColorTable(COLORTABLE)
        return QI

    path = r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtoolbox/tests/im1_3.png"

    """
    import pylab

    i = QtGui.QImage()
    i.load(path)
    #i = makeIm()
    v2 = qi2np(i, "array")
    pylab.imshow(v2)
    pylab.show()

	# v is a recarray; make it MPL-compatible for showing:
    v = qi2np(i, "rec")
    rgb = np.empty(v.shape + (3,), dtype = np.uint8)
    rgb[...,0] = v["r"]
    rgb[...,1] = v["g"]
    rgb[...,2] = v["b"]
    pylab.imshow(rgb)
    pylab.show()
    """

    import cv2
    def loadcv(pth,mode=-1,shape=None):
        im = cv2.imread(pth,mode)
        if shape:
            im = cv2.resize(im,shape)
        return im

    class thiswin(QtGui.QMainWindow):
        def __init__(self,image):
            super(thiswin, self).__init__()
            self.printer = QtGui.QPrinter()
            self.scaleFactor = 0.0

            self.imageLabel = QtGui.QLabel()
            self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
            self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,QtGui.QSizePolicy.Ignored)
            self.imageLabel.setScaledContents(True)

            self.scrollArea = QtGui.QScrollArea()
            self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
            self.scrollArea.setWidget(self.imageLabel)
            self.scrollArea.setWidgetResizable(True)
            self.setCentralWidget(self.scrollArea)

            self.setWindowTitle("Image Viewer")
            self.resize(500, 400)
            self.image = image
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
            self.imageLabel.adjustSize()

    # TODO examine when the image is updated
    import threading
    import pyqtgraph as pg


    def func():
        time.sleep(2)
        img[100:1000,100:1000] = 0
        #shit.setImage(qim.ndarray)
        shit.updateImage()
        QtGui.qApp.processEvents() # update
        print "image changed, try minimizing and then maximazing"

    t = threading.Thread(target=func)
    win_main = QtGui.QMainWindow()
    img = loadcv(path)
    qim = np2qi(img)
    uimain = thiswin(qim)
    uimain.show()
    axes = None
    if img.ndim == 3:
        if img.shape[2] <= 4:
            axes = {'t': None, 'x': 2, 'y': 1, 'c': 0}
        else:
            axes = {'t': 2, 'x': 1, 'y': 0, 'c': None}
    if axes:
        print "sending with axes"
        shit = pg.image(img)
    else:
        shit = pg.image(img)
    t.start()
    sys.exit(app.exec_())
