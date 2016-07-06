# https://groups.google.com/forum/#!topic/pyqtgraph/EBzl12n8YYs
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

STEPS = np.array([0.0, 0.2, 0.6, 1.0])
CLRS =           ['k', 'r', 'y', 'w']
clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))

data = np.random.normal(size=(100, 200, 200))

## Create window with ImageView widget
imv = pg.image(data)
imv.ui.histogram.gradient.setColorMap(clrmp)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
   import sys
   if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
       QtGui.QApplication.instance().exec_()