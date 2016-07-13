# -*- coding: utf-8 -*-
"""
    This module holds core-like methods for library modules but not for the hole package
"""
# system
import sys
import os
import inspect
import types
from time import time, sleep
import numpy as np
from contextlib import contextmanager

# ----------------------------GLOBAL VARIABLES---------------------------- #
__author__ = 'Davtoh'
# ----------------------------BASIC FUNCTIONS---------------------------- #


class stdoutSIM:
    """
    simple logger to simulate stdout output
    """
    def __init__(self, disable = False):
        self.stdout = sys.stdout
        self.disable = disable
    def write(self, text, **kwargs):
        if not self.disable:
            self.stdout.write(text,**kwargs)
    def printline(self, text, **kwargs):
        if not self.disable:
            self.stdout.printline(text, **kwargs)
    def printlines(self, lines, **kwargs):
        if not self.disable:
            self.stdout.printlines(lines, **kwargs)
    def flush(self):
        if not self.disable:
            self.stdout.flush()
    def close(self):
        pass

stdout = stdoutSIM()

class stdoutLOG:
    """
    simple logger to save stdout output
    so anything printed in the console is logged to a file.

    :param path: path to logging file
    :param mode: mode for opening the file.
    :param chain: if True closes previous logs and continues with new log
    """
    def __init__(self, path, mode = "w+", chain = False):
        self._path = path
        self._mode = mode
        self._chain = chain
        self._log = open(path,mode)
        self._closed = False

        # close previous stdoutLOG
        if chain and isinstance(sys.stdout,stdoutLOG):
                sys.stdout.close()

        if not chain and isinstance(sys.stdout,stdoutSIM):
            # continue using previous logs
            self.logs = [sys.stdout,self._log]
        else:
            # use new log and stdout
            self.logs = [stdout,self._log]

        sys.stdout = self # register log

    def write(self, text, **kwargs):
        for log in self.logs:
            log.write(text, **kwargs)

    def printline(self, text, **kwargs):
        for log in self.logs:
            log.printline(text, **kwargs)

    def printlines(self, lines, **kwargs):
        for log in self.logs:
            log.printlines(lines, **kwargs)

    def flush(self):
        for log in self.logs:
            log.flush()

    def close(self):
        # close according to chain option
        if self._chain:
            # close all the logs in the chain
            for log in self.logs:
                log.close()
        elif not self._closed:
            # only closes this log but keep alive
            # previous logs
            self.logs.remove(self._log)
            self._log.close()
            self._closed = True

def decorateInstanceMethods(self, decorator,excludeMth=("__init__"),includeMth=None):
    """
    Decorate methods in an instance. It should be used in the __init__ method of a class.

    :param self: class instance.
    :param decorator: decorator function to apply to self.
    :param excludeMth: list of methods to exclude.
    :param includeMth: list of methods to include if not in exclude.
            if excludeMth is None then decorateInstanceMethods checks for includeMth list.
            if includeMth and excludeMth is None then all methods of self are decorated.
    :return: self

    .. note:: It must be used at instance initialization (i.e. inside __init__ method)
    """
    classmethods = dict(inspect.getmembers(self, predicate=inspect.ismethod))
    #if "__init__" in classmethods.keys(): del classmethods["__init__"] # init method should not be decorated
    for key in classmethods:
        if excludeMth and key in excludeMth:
            continue
        if includeMth is None or key in includeMth:
            setattr(self, key, decorator(classmethods[key]))
    return self


# ----------------------------DECORATORS---------------------------- #

def addto(instance,funcname=None):
    """
    Decorator: Add function as method to instance.

    :param instance: class instance.
    :param funcname: name to register in instance.
    :return:
    """
    def decorator(fn):
        fn = types.MethodType(fn, instance, instance.__class__) # convert to bound method
        if funcname:
            setattr(instance, funcname, fn)
        else:
            setattr(instance, fn.func_name, fn) # set fn method with name fn.func_name in instance
        return fn
    return decorator

# ----------------------------CLASS OBJECTS---------------------------- #

# ----------------------------DECORATED FUNCTIONS---------------------------- #

# ----------------------------SPECIALISED FUNCTIONS---------------------------- #

class FactorConvert(object):
    """
    Keep track of factor and converts to any available factor.
    """
    _factors = (("exa", "E", 1000000000000000000),
                ("peta","P",1000000000000000),
                ("tera","T",1000000000000),
                ("giga","G",1000000000),
                ("mega","M",1000000),
                ("kilo","k",1000),
                ("hecto","h",100),
                ("deca","da",10),
                (None,"",1),
                ("deci","d",0.1),
                ("centi","c",0.01),
                ("milli","m",0.001),
                ("micro","u",0.000001),
                ("nano","n",0.000000001),
                ("pico","p",0.000000000001),
                ("femto","f",0.000000000000001),
                ("atto","a",0.000000000000000001))

    def __init__(self, factor = None, factorIndex = 1):
        """

        :param factor: anything to look in factors (i.e. factor list with Factor structures).
        :param factorIndex: index to return from Factor structure when factor is asked.

        .. notes: A factor structure is of the form ("Name","abbreviation",value)
        """
        self._factor = None
        self._factorsCache = None
        self.factorIndex = factorIndex
        self.factors = self._factors
        self.factor = factor

    @property
    def factors(self):
        return self._factors
    @factors.setter
    def factors(self, value):
        self._factorsCache = zip(*value) # itertools.izip is faster but this operation is one time
        self._factors = value
    @factors.deleter
    def factors(self):
        raise Exception("Property cannot be deleted")

    @property
    def factor(self):
        return self._factor[self.factorIndex]
    @factor.setter
    def factor(self, value):
        self._factor = self.getFactor(value) # transform units
    @factor.deleter
    def factor(self):
        raise Exception("Property cannot be deleted")

    def convert(self, factor, to = None):
        """
        Convert from actual factor to another factor.

        :param factor: number
        :param to: factor to convert
        :return: converted value, units
        """
        to = self.getFactor(to)
        return factor * self._factor[2] / float(to[2]), to[self.factorIndex] # converted value, factor

    def convert2sample(self, factor, to = None):
        """
        Convert to resemble sample.

        :param factor: number
        :param to: sample factor.
        :return: converted value, units
        """
        if to is None or to==0:
            unit = None
        else: # calculate as fast as possible
            try:
                unit = factor * self._factor[2] / to
            except:
                to = self.getFactor(to)[2]
                unit = factor * self._factor[2] / to
        return self.convert(factor, unit) # to units.

    def exactFactorIndex(self, key):
        """
        Find the index of a factor that contains a key.

        :param key: anything to look in factors (i.e. factor name, factor value, abbreviation).
        :return: factor structure, else None.
        """
        for i in self._factorsCache:# try to find it
            if key in i: return i.index(key)

    def nearFactorIndex(self, factor):
        """
        Find the index of nearest factor value.

        :param factor: factor value.
        :return: factor structure near factor value.
        """
        return (np.abs(np.array(self._factorsCache[2]) - factor)).argmin()

    def getFactor(self, key):
        """
        Tries to find factor value in factors.

        :param key: anything to look in factors (i.e. factor name, factor value, abbreviation).
                    If key is a factor value it will look for the nearest factor value.
        :return: factor structure, else raises error.
        """
        if key is None or isinstance(key, basestring):
            index = self.exactFactorIndex(key)
            if index is None: raise Exception("Factor not found for {}".format(key))
        else:
            index = self.nearFactorIndex(key)
        return self._factors[index]

    @staticmethod
    def split(value):
        """
        Get number fraction.

        :param value: number
        :return: integer, fraction
        """
        left = int(value)
        right = value-left
        return left,right

    @staticmethod
    def parts(value, precision = 4):
        """
        Get number parts.

        :param value: number
        :param precision: decimal precision
        :return: ([... ,Hundreds, Tens, Ones],[Tenths, ...])
        """
        p = "{{:0.{}f}}".format(precision).format(value).split(".")
        if len(p)>1:
            return [int(i) for i in p[0]],[int(i) for i in p[1]]
        else:
            return [int(i) for i in p[0]],[]

@contextmanager
def TimeCode(msg, unit = None, precision = None,
             abv=None, endmsg = "{time}\n", enableMsg= True,
             printfunc= sys.stdout.write):
    """
    Context to profile code by printing a prelude and prologue with time.

    :param msg: prelude or description message
    :param unit: unit supported by FactorConvert class
    :param precision: number of digits after a float point
    :param abv: if True prints "s", if False "seconds" for time
    :param endmsg: prologue message
    :param enableMsg: (True) A flag specifying if context
            should be printed or not.
    :param printfunc: function to print messages. By default it
            is sys.stdout.write
    :return:
    """
    if enableMsg:
        printfunc(msg)
        start = time() # begin chronometer
    try:
        yield # code to execute
    finally:
        if enableMsg:
            t = time()-start # end chronometer
            if precision is None:
                text = "{:f} {}{}"
            else:
                text = "{{:0.{}f}} {{}}{{}}".format(precision)
            # abbreviation
            if abv is True:
                u = "s"
                i = 1
            else:
                u = " seconds"
                i = 0
            if unit: #
                if isinstance(unit,basestring):
                    printfunc(endmsg.format(
                        time = text.format(*(FactorConvert(
                            factorIndex = i).convert(t, unit) + (u,)))))
                else:
                    printfunc(endmsg.format(
                        time = text.format(*(FactorConvert(
                            factorIndex = i).convert2sample(t, unit) + (u,)))))
            else:
                printfunc(endmsg.format(
                    time=text.format(t,"",u)))

@contextmanager
def Controlstdout(disable = True):
    """
    Context manager to control output to stdout

    :param disable: if True suppress output.
    """
    buffer = sys.stdout
    fakelog = stdoutSIM(disable)
    sys.stdout = fakelog
    try:
        yield # code to execute
    finally:
        sys.stdout = buffer

def glob(path, contents="*", check = os.path.isfile):
    """
    Return a list of paths matching a pathname pattern with valid files.

    :param path: path to process ing glob filter
    :param contents: If path is a folder then looks for contents using
    :param check: function to filter contents. it must receive the path
            and return True to let it pass and False to suppress it.
    :return: return list of files
    """
    from glob import glob
    fns = glob(path)
    # special case: Folder
    if len(fns) == 1 and not os.path.isfile(fns[0]):
        fns = glob(os.path.join(fns[0], contents))
    return [p for p in fns if check(p)]

def lookinglob(pattern, path, ext=None, returnAll = False, raiseErr = False):
    """

    :param pattern: look pattern in path
    :param path: path to look pattern
    :param ext: extension
    :param raiseErr: If true raise Exception if patter not found in path
    :return: fn or None
    """
    tests = [pattern, "{b}*", "*{b}", "*{b}*", "{p}{b}", "{p}*{b}", "{p}{b}*", "{p}*{b}*"]
    if ext is not None and not pattern.endswith(ext):
        if not ext.startswith("."):
            ext = "."+ext
        tests = [i + ext for i in tests]

    ress = []
    for test in tests:
        res = glob(test.format(p=path, b=pattern))
        if not res:
            continue
        if not returnAll and len(res)==1:
            return res[0]
        ress.append(res)

    if ress:
        if returnAll:
            unique = set()
            for res in ress:
                for i in res:
                    unique.add(i)
            return unique
        ress.sort(key=lambda x:len(x))
        if len(ress[0])==1:
            return ress[0][0]
        elif raiseErr:
            raise Exception("More than one file with pattern {}".format(path))
        return ress[0] # return possibilities

    if raiseErr:
        raise Exception("{} not in {}".format(pattern, path))


if __name__ == "__main__":
    def myfc():
        sleep(1)
        return "======"*10000

    with TimeCode("init",100,abv=True):
        myfc()
    pass
    fac = FactorConvert()
    print fac.convert(10,100)
    print fac.parts(1001010.01010101)
    print "{:f} {}".format(*FactorConvert("m").convert2sample(36797.59, "m"))
    print FactorConvert("m").convert(36797.59)
    print fac.convert(1000)