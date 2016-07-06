from RRtoolbox.lib.cache import memoizedDict,mapper, memoize
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
        descriptors = memoizedDict(saveTo+"descriptors")
    with TimeCode("Loading shapes"):
        shapes = memoizedDict(saveTo+"shapes")
    with TimeCode("Loading data"):
        data = memoizedDict(saveTo+"data")
    return descriptors,shapes,data

#descriptors,shapes,data = test2()

mydict = memoizedDict("mydict")
class textOp:
    pass

if "textOp" not in mydict:
    print "inserting textOp"
    mydict["textOp"] = textOp()

print mydict
#memoize.ignore = [tomemoize] # or it can be tomemoize_old
v = "something"
print v, tomemoize(v), tomemoize_processed


