# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
from __future__ import print_function
# SEE:http://www.pyqtgraph.org/documentation/how_to_use.html
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')

## Create random 3D data set with noisy signals
import cv2
img = cv2.imread(r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtoolbox/tests/im1_3.png")


img = np.ascontiguousarray(img.transpose(1,0,2))
imv.setImage(img)
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # TODO examine when the image is updated
    import threading
    import time
    def func():
        #time.sleep(2)
        img[1100:1200,100:200] = 0
        print("image changed, try resizing")
        imv.setImage(img)
    t = threading.Thread(target=func)
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        t.start()
        QtGui.QApplication.instance().exec_() # execute GUI
    t.join()
