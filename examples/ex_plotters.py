from __future__ import print_function
import sys, os
sys.path.insert(0,os.path.abspath("../")) # add relative path
from RRtoolFC.lib.app import execApp, app
import cv2
def loadcv(pth,mode=-1,shape=None):
    im = cv2.imread(pth,mode)
    if shape:
        im = cv2.resize(im,shape)
    return im
win = True
## Set the raw data as the input value to the flowchart
data=loadcv(r"../tests/im1_1.jpg",mode=0,shape=(300,300))
from RRtoolbox.lib.plotter import Plotim,fastplt
#fastplt(data,block=True) # FIXME block=False does not work under windows
p = Plotim("Plotim",data)
p.show(block=True)
print("after plots")