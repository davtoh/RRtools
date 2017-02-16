from __future__ import print_function
from builtins import input
from builtins import object
from RRtoolbox.lib.image import loadFunc, ImLoader, convertAs
from RRtoolbox.lib.plotter import fastplt
from RRtoolbox.lib.config import MANAGER
from RRtoolbox.lib.cache import mapper
from RRtoolbox.lib.root import glob
# help(imLoader) # test documentation

import sys,os
import cv2
path = MANAGER["TESTPATH"] + r"im1_1.jpg"

class ima(object):
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
    dsize= (2448,None)
    dst=None
    fx=1#None
    fy=1#None
    interpolation=0#None
    loader = loadFunc(flag=flag,dsize=dsize,dst=dst,fx=fx,fy=fy,interpolation=interpolation)
    img = loader(path)
    print(img.shape)

if False: # print interpolation flag values
    print(cv2.INTER_NEAREST)
    print(cv2.INTER_LINEAR)
    print(cv2.INTER_CUBIC)
    print(cv2.INTER_AREA)
    print(cv2.INTER_LANCZOS4)

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
            if input("continue? (y/n)") in ("n","not"):
                break

if False: # load numpy array test
    f = loadFunc()
    fastplt(f("testim.npy"))

if True:
    # applies to a set of folders
    stats = convertAs(glob("/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/", check=os.path.isdir),
                      base="/mnt/4E443F99443F82AF/results/",loader=loadFunc(1,dsize=(500,None)),
                      overwrite=True, folder= True, ext=True)
    """
    # applies to a folder
    stats = convertAs("/mnt/4E443F99443F82AF/MEGAsync/TESIS/DATA_RAW/classified/set1",
                      base="/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/thesis/images",
                      overwrite=False, folder= True, ext=True,simulate=True)
    """
    successes = [(k,nk) for k,v,nk in stats if not v]
    if successes:
        print("These succeeded:")
        for s,ns in successes:
            print(s, "as", ns)

    fails = [k for k,v,nk in stats if v]
    if fails:
        print("These failed:")
        for failed in fails:
            print(failed)