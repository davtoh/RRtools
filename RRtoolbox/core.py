from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object
__author__ = 'Davtoh'

import time
from .lib.inspector import Logger
from .lib.cache import memoize
from .lib import config as cf
import inspect,types

pkl_path = __file__.split(".")[0]+".pkl"

mylogger = Logger()
mylogger.throwAtError = False
@mylogger.tracer
def f(a, b=1, *pos, **named):
   a = b
   return a

mylogger.report()

def tools(instance,modules):
    for key in list(modules.keys()):
        moduleTool = getattr(modules[key],"tool","tool")
        classmethods = dict(inspect.getmembers(moduleTool, predicate=inspect.ismethod))
        if "__init__" in list(classmethods.keys()): del classmethods["__init__"]
        for method in classmethods:
            fn = types.MethodType(classmethods[method], instance, instance.__class__) # convert to bound method
            setattr(instance, fn.__name__, fn) # set fn method with name fn.func_name in instance

def tools2(instance,modules):
    for key in list(modules.keys()):
        instance.__dict__[key] = getattr(modules[key],"tool","tool")

class rrbox(object):
    def __init__(self,*args):
        #configuration object
        self.tools = cf.ConfigTool().getTools(cf.MANAGER['TOOLPATH'])
        tools2(self,self.tools)
    @memoize
    def asift(self):
        pass

if __name__ == '__main__':
    import RRtoolbox  # use if no relative import
    a = rrbox()
    f(1, 2, 3)
    f(a=2, x=4)
    mylogger.report()
    print("tools:",list(a.tools.keys()))
    print(dir(a.tools['restoration']))
    print(getattr(a.tools['restoration'],"tool","tool"))
    #memoize = a.tools['restoration'].root.memoize
    #MEMORY = a.tools['restoration'].root.MEMORY
    #a.tools['restoration'].ASIFT_multiple.flush()

    @memoize(pkl_path)
    def memtest(x):
        k=1
        time.sleep(10)
        for i in range(x):
            k+=123
        return k

    def test(x):
        print("processing....")
        return memtest(x)

    #print test(123)
    a.tools['restoration'].asif_demo()
    #mm = a.tools['restoration'].ASIFT_multiple
    #help(mm)
    #help(mm.clear)
    #memtest.clear()