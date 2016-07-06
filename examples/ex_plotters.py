from RRtoolFC.lib.app import execApp, app
import cv2
def loadcv(pth,mode=-1,shape=None):
    im = cv2.imread(pth,mode)
    if shape:
        im = cv2.resize(im,shape)
    return im
## Set the raw data as the input value to the flowchart
data=loadcv(r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/im1_1.jpg",mode=0,shape=(300,300))
from RRtoolbox.lib.plotter import plotim,fastplt
#fastplt(data)
p = plotim("plotim",data)
p.show(block=False)
print "after plots"