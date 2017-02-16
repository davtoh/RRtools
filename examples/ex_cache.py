from __future__ import print_function
from builtins import object
from RRtoolbox.lib.cache import MemoizedDict,mapper, memoize
from RRtoolbox.lib.session import saveSession,updateSession,readSession
from RRtoolbox.lib.root import TimeCode

tomemoize_processed = []

def tomemoize(val):
    """
    This is the doc of tomemoize. This function is intended to test Memoizers
    from cache module. internally it appends input variable "val" to the
    list tomemoize_processed, confirm that it was "processed".
    :param val: any value
    :return: val
    """
    tomemoize_processed.append(val) # this confirms if tomemize processed val
    return val

tomemoize_old = tomemoize
tomemoize = memoize("m")(tomemoize)

def test2(saveTo = "/mnt/4E443F99443F82AF/restoration_data2/"):
    with TimeCode("Loading descriptors"):
        descriptors = MemoizedDict(saveTo + "descriptors")
    with TimeCode("Loading shapes"):
        shapes = MemoizedDict(saveTo + "shapes")
    with TimeCode("Loading data"):
        data = MemoizedDict(saveTo + "data")
    return descriptors,shapes,data

#descriptors,shapes,data = test2()

mydict = MemoizedDict("mydict")
class textOp(object):
    pass

if "TextOp" not in mydict:
    print("inserting TextOp")
    mydict["TextOp"] = textOp()

print(mydict)
#memoize.ignore = [tomemoize] # or it can be tomemoize_old
v = "something"
print(v, tomemoize(v), tomemoize_processed)


