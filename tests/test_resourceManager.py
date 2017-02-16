from __future__ import division
from __future__ import print_function

from builtins import input
from builtins import object
import weakref
import numpy as np
import inspect
import gc
from random import random
from time import time, sleep
from collections import MutableMapping, OrderedDict
import sys
import cv2
from RRtoolbox.lib.cache import ResourceManager
# https://mindtrove.info/python-weak-references/
# http://stackoverflow.com/a/3387975/5288758
# references = weakref.WeakValueDictionary()

def getParameters(methods):
    """

    :param methods:
    :return:
    """
    """

        data = [] #
        for key,val in self.iteritems():
            calls = int(10*random())#val._calls
            fails = int(10*random())#val._fails
            means = 10*random()#val._means
            if val._ref is not None:
                val = val._ref()
                if val is not None:
                    data.append((key,sys.getsizeof(val),calls,fails,means))

        if method: # to sort (bit 1): order
            op = "{0:b}".format(method)[::-1]
            isMeans = len(op)>4 and op[4]=='1' # (bit4): fails[3]
            isFails = len(op)>3 and op[3]=='1' # (bit4): fails[3]
            isCalls = len(op)>2 and op[2]=='1' # (bit3): calls[2]
            isSizes = len(op)>1 and op[1]=='1' # (bit2): size[1]
            isReverse = op[0]=='1' # (bit 1): revert order
            if isSizes: data.sort(key=lambda x: x[1],reverse=isReverse) # sort by size always
            if isCalls: data.sort(key=lambda x: x[2],reverse=isReverse) # sort by calls
            if isFails: data.sort(key=lambda x: x[3],reverse=isReverse) # sort by fails
            if isMeans: data.sort(key=lambda x: x[4],reverse=isReverse) # sort by mean time
    """
    if methods:
        op = "{0:b}".format(methods)[::-1] # get each method
        fields = [True for i,val in enumerate(op) if val == "1"] # get fields to use
        return fields


def loadcv(pth,mode=-1,shape=None):
    """
    Simple function to load using opencv.

    :param pth: path to image.
    :param mode: (default -1)  0 to read as gray, 1 to read as BGR, -1 to read as BGRA.
    :param shape: shape to resize image.
    :return: loaded image
    """
    def myfunc():
        im = cv2.imread(pth,mode)
        if shape:
            im = cv2.resize(im,shape)
        return im
    return myfunc

def mymethod(shape=None,mode=-1):
    return loadcv("/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/equalization.png",mode,shape)

class MyObj(object):
    def __init__(self, size):
        self.size = size
    def __sizeof__(self):
        return self.size

def factoryfunc(time,memory):
    def myfunc():
        print("OBJECT IS BEING CREATED")
        sleep(random()*time)
        return MyObj(memory)
    return myfunc

entrada = input("Select units: ")
ret = ResourceManager(unit=entrada)
entrada = input("Max memory ({}): ".format(ret.unit))
ret.maxMemory = int(entrada)

#maxX,maxY = 1000,1000
# ret[name] = mymethod((int(random()*maxY),int(random()*maxX)))
# 1 mega = 1048576 bytes = 2 ** 20 bytes; 1 giga = 1000 megas
# mission load until there is 1 giga


keepOutSide = []
while True:
    name = input("name of object: ")
    if not name:
        break
    print("To create object select:")
    t = int(input("max time processing: "))
    s = ret.units2bytes(float(input("memory size ({}): ".format(ret.unit))))
    func = factoryfunc(t,s)
    val = eval(input("method 1,2,3,other: "))
    if val==1: # creation method is given but it once called it is kept alive
        ret[name] = func
        keepOutSide.append(ret[name])
    elif val==2: # object was just given alive
        ret.register(name,func,func())
        ret[name]
    elif val==3: # object was and always will be alive
        keepOutSide.append(func())
        ret.register(name,func,keepOutSide[-1])
        ret[name]
    else: # just method is given
        ret[name] = func

#ret.update(little=mymethod((500,500)),medium=mymethod((500,700)),big=mymethod((500,900)))# same as ret["obj"] = mymethod
ret._optimizeMemory()

