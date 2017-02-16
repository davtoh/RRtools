from __future__ import print_function
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from RRtoolbox import loadcv,MANAGER

# http://stackoverflow.com/q/25956163/5288758
# Load image from disk and reorient it for viewing
#from matplotlib import image
fname = MANAGER.TESTPATH + r"im1_3.png"    # This can be any photo image file
#photo = image.imread(fname)
photo=loadcv(fname,1)
# photo[photo.shape[0]-1,photo.shape[1]-1,photo.shape[2]-1],np.empty(photo.shape) # that is how numpy works
# but numpy has an inverted coordinate system (heigth,width) or (y,x)
# so we have to transpose the array to convert coordinate system to (x,y)
if len(photo.shape) == 2:
    img = photo.transpose(1,0)
else:
    img = np.ascontiguousarray(photo.transpose(1,0,2)) # http://stackoverflow.com/a/27601130/5288758
# select for red color and extract as monochrome image
#img = img[0,:,:]  # WHAT IF I WANT TO DISPLAY THE ORIGINAL RGB IMAGE?

# Create app
app = QtGui.QApplication([])
imitem = pg.ImageItem()
imitem.setImage(img.astype(np.float32))
imitem.setLevels(np.array([[0,1],[0,1],[0,1]],np.float64),False)
## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(1200,800)
imv = pg.ImageView(imageItem=imitem)
win.setCentralWidget(imv)
win.show()
win.setWindowTitle(fname)
## Display the data
#imv.setImage(img)#, levels=np.array([[0,1],[0,1],[0,1]],np.float64),autoLevels=False) # it uses BGR! so no problem with opencv, you can set useRGBA to invert though.
# TODO integrate with main
def click(event):
    event.accept()
    pos = event.pos()
    print((int(pos.x()),int(pos.y())))

g = imv.getImageItem()
g.mouseClickEvent = click

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()