from RRtoolbox.lib.image import loadFunc, getData, imLoader
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.config import MANAGER
from RRtoolbox.lib.cache import mapper
# help(imLoader) # test documentation

import sys
import cv2
path = "/home/davtoh/Desktop/untitled.png"#MANAGER.TESTPATH + r"im1_3.png"

class ima:
    # TEST mapper! it was demostrated that first saved data is returned regardlest that data was changed
    # (that's because only path to map file is given and not the other arguments)
    im = None
    flag = None
    dsize = None
    fx = None
    fy = None
    interpolation = None
    def __init__(self,im,flag=0,dsize=0,fx=0,fy=0,interpolation=0):
        self.im = im
        self.flag = flag
        self.dsize = dsize
        self.fx = fx
        self.fy = fy
        self.interpolation = interpolation

if False: # example test
    flag = 1
    dsize= None
    dst=None
    fx=0#None
    fy=1#None
    interpolation=0#None
    img = cv2.imread(path, flag)
    im = cv2.resize(img, dsize, dst, fx, fy, interpolation)

if False: # print interpolation flag values
    print cv2.INTER_NEAREST
    print cv2.INTER_LINEAR
    print cv2.INTER_CUBIC
    print cv2.INTER_AREA
    print cv2.INTER_LANCZOS4

if False: # mmapped test of loading the same mapped file
    f = loadFunc(interpolation=cv2.INTER_NEAREST)
    # mmap is like a pointer to a file where data can be changed and all their
    #f = loadFunc(flag=-2, fx=3, fy=1,mpath=MANAGER.TEMPPATH+"",mmode="r+")
    #f = loadFunc(flag=-2, dsize=(300,100),mpath=MANAGER.TEMPPATH+"",mmode="r+") pointers are updated.
    fastplt(f(path))

    if False:
        images = []
        while True:
            images.append(f(path))
            #fastplt(images[-1])
            if raw_input("continue? (y/n)") in ("n","not"):
                break

if True: # load numpy array test
    f = loadFunc()
    fastplt(f("testim.npy"))