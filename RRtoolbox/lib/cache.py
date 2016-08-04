# -*- coding: utf-8 -*-
"""
    .. moduleauthor:: David Toro <davsamirtor@gmail.com>
    :platform: Unix, Windows
    :synopsis: Serialize and Memoize.

    Contains memoizing, caching, serializing and memory-mapping methods so as to let the package
    save its state (persistence) and to let a method "remember" what it processed in a session (with cache) or
    between sessions (memoization and serializization) of the same input contend once processed. It also wraps mmapping
    functions to let objects "live" in the disk (slower but almost unlimited) rather than in memory (faster but limited).

    *@cache* is used as replacement of *@property* to compute a class method once.
    It is computed only one time after which an attribute of the same name is generated in its place.

    *@cachedProperty* is used as replacement of *@property* to compute
    a class method depending on changes in its watched variables.

    *@memoize* used as a general memoizer decorator for functions
    where metadata is generated to disk for persistence.

    Made by Davtoh, powered by joblib.
    Dependent project: https://github.com/joblib/joblib
"""
from RRtoolbox.lib.root import NotCallable, NotCreatable, VariableNotSettable, VariableNotDeletable

__license__ = """

    joblib is BSD-licenced (3 clause):

    This software is OSI Certified Open Source Software. OSI Certified is a
    certification mark of the Open Source Initiative.

    Copyright (c) 2009-2011, joblib developpers All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    3. Neither the name of Gael Varoquaux. nor the names of other joblib contributors may be used
    to endorse or promote products derived from this software without specific prior written permission.

    This software is provided by the copyright holders and contributors "as is" and any
    express or implied warranties, including, but not limited to, the implied warranties
    of merchantability and fitness for a particular purpose are disclaimed.

    In no event shall the copyright owner or contributors be liable for any direct, indirect,
    incidental, special, exemplary, or consequential damages (including, but not limited to,
    procurement of substitute goods or services; loss of use, data, or profits; or business
    interruption) however caused and on any theory of liability, whether in contract, strict
    liability, or tort (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such damage.

"""
__author__ = 'Davtoh'
## READ: https://wiki.python.org/moin/PythonDecoratorLibrary

import joblib
from functools import wraps
from weakref import ref
from collections import MutableMapping
from time import time
from numpy.lib import load as numpyLoad, save as numpySave
import os
#print "using joblib version", joblib.__version__

class NotMemorizedFunc(joblib.memory.NotMemorizedFunc):
    pass

class MemorizedFunc(joblib.memory.MemorizedFunc):
    pass

class DynamicMemoizedFunc(object):
    def __init__(self, func, cachedir = None, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None, banned = False):
        self._func = func # the only one that should not be able to change
        self._mmap_mode = mmap_mode
        self._ignore = ignore
        self._verbose = verbose
        self._cachedir = cachedir
        self._compress = compress
        self._timestamp = timestamp
        self._enabled = banned
        self._use = None
        self._build()
    def _build(self):
        if self._cachedir is None or not self._enabled:
            self._use = NotMemorizedFunc(self._func)
        else:
            self._use = MemorizedFunc(func=self._func,cachedir= self._cachedir, ignore=self._ignore,
                                      mmap_mode=self._mmap_mode,compress=self._compress,
                                      verbose=self._verbose, timestamp=self._timestamp)
            self.__doc__ = self._use.__doc__
    @property
    def func(self):
        return self._func
    @func.setter
    def func(self,value):
        if value != self._func:
            if isinstance(value, (MemorizedFunc,NotMemorizedFunc,DynamicMemoizedFunc)):
                value = value.func
            self._func = value
            self._build()
    @func.deleter
    def func(self):
        raise Exception("property cannot be deleted")

    @property
    def mmap_mode(self):
        return self._mmap_mode
    @mmap_mode.setter
    def mmap_mode(self,value):
        if value != self._mmap_mode:
            self._mmap_mode = value
            self._build()
    @mmap_mode.deleter
    def mmap_mode(self):
        raise Exception("property cannot be deleted")

    @property
    def ignore(self):
        return self._ignore
    @ignore.setter
    def ignore(self,value):
        if value != self._ignore:
            self._ignore = value
            self._build()
    @ignore.deleter
    def ignore(self):
        raise Exception("property cannot be deleted")

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self,value):
        if value != self._verbose:
            self._verbose = value
            self._build()
    @verbose.deleter
    def verbose(self):
        raise Exception("property cannot be deleted")

    @property
    def cachedir(self):
        return self._cachedir
    @cachedir.setter
    def cachedir(self,value):
        if self._cachedir != value:
            self._cachedir = value
            self._build()
    @cachedir.deleter
    def cachedir(self):
        raise Exception("property cannot be deleted")

    @property
    def compress(self):
        return self._compress
    @compress.setter
    def compress(self,value):
        if self._compress != value:
            self._compress = value
            self._build()
    @compress.deleter
    def compress(self):
        raise Exception("property cannot be deleted")

    @property
    def enabled(self):
        return self._enabled
    @enabled.setter
    def enabled(self, value):
        if self._enabled != value:
            self._enabled = value
            self._build()
    @enabled.deleter
    def enabled(self):
        raise Exception("property cannot be deleted")

    def __call__(self, *args, **kwargs):
        return self._use(*args, **kwargs)

    def call_and_shelve(self, *args, **kwargs):
        return self._use.call_and_shelve(*args, **kwargs)

    def __reduce__(self):
        return self._use.__reduce__()

    def __repr__(self):
        return self._use.__repr__()
    def clear(self, warn=True):
        return self._use.clear(warn=warn)

class Memory(joblib.Memory):
    """
    A wrapper to joblib Memory to have better control.
    """
    def __init__(self, cachedir, mmap_mode=None, compress=False, verbose=1):
        super(Memory,self).__init__(None, mmap_mode, compress, verbose)
        if cachedir is None:
            self.cachedir = None
        else:
            self.cachedir = cachedir
            joblib.memory.mkdirp(self.cachedir)

    '''
    def cache(self, func=None, ignore=None, verbose=None,
                        mmap_mode=False):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return partial(self.cache, ignore=ignore,
                                     verbose=verbose, mmap_mode=mmap_mode)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, (MemorizedFunc,NotMemorizedFunc,DynamicMemoizedFunc)):
            func = func.func
        return DynamicMemoizedFunc(func, cachedir=self.cachedir,
                                   mmap_mode=mmap_mode,
                                   ignore=ignore,
                                   compress=self.compress,
                                   verbose=verbose,
                                   timestamp=self.timestamp)'''
    __call__ = joblib.Memory.cache

class Memoizer(object):
    memoizers = {}
    def __init__(self, ignore=(), ignoreAll=False):
        self._ignore = None
        self.ignoreAll = ignoreAll
        self.ignore = ignore
        self.memoized = {} # FOR CLEANING UP handle # flag = True is safe to remove, flag = False is unsafe to remove
        Memoizer.memoizers[id(self)] = ref(self)

    @property
    def ignore(self):
        return self._ignore
    @ignore.setter
    def ignore(self,value):
        temp = set()
        for f in value:
            if not callable(f):
                raise Exception("{} must be callable".format(f))
            temp.add(id(f))
        self._ignore = temp
    @ignore.deleter
    def ignore(self):
        self._ignore = set()

    def makememory(self,cachedir = None, mmap_mode=None, compress=False, verbose=0):
        """
        Make memory for :func:`memoize` decorator.

        :param cachedir: path to save metadata, if left None function is not cached.
        :param mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional.
                    The memmapping mode used when loading from cache
                    numpy arrays. See numpy.load for the meaning of the
                    arguments.
        :param compress: (boolean or integer)
                    Whether to zip the stored data on disk. If an integer is
                    given, it should be between 1 and 9, and sets the amount
                    of compression. Note that compressed arrays cannot be
                    read by memmapping.
        :param verbose: (int, optional)
                    Verbosity flag, controls the debug messages that are issued
                    as functions are evaluated.
        :return:
        """
        """
        if not cachedir:
            cachedir = tempfile.mkdtemp() # tempfile.mkdtemp(dir=cachedir)"""
        MEMORY = Memory(cachedir, mmap_mode, compress, verbose)
        return MEMORY

    def memoize(self, memory=None, ignore=None, verbose=0, mmap_mode=False):
        """
        Decorated functions are faster by trading memory for time, only hashable values can be memoized.

        :param memory: (Memory or path to folder) if left None function is not cached.
        :param ignore: (list of strings) A list of arguments name to ignore in the hashing.
        :param verbose: (integer) Verbosity flag, controls the debug messages that are issued as functions are evaluated.
        :param mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional. The memmapping mode used when loading from cache
                    numpy arrays. See numpy.load for the meaning of the arguments.
        :return: decorator
        """
        def decorator(fn):
            if isinstance(memory,joblib.Memory): # use provided memory
                memoizedfn = memory.cache(fn,ignore,verbose, mmap_mode)
            else:
                #memoizedfn = DynamicMemoizedFunc(func=fn,cachedir=str(memory),
                #                ignore=ignore,verbose=verbose, mmap_mode=mmap_mode)
                memoizedfn = self.makememory(cachedir=str(memory)).cache(func=fn,
                                ignore=ignore,verbose=verbose, mmap_mode=mmap_mode)
            @wraps(fn)
            def wrapper(*args,**kwargs):
                # TODO solve this, how to test quickly to ignore memoization
                #perhaps it is better to use dynamicMemoizedFuciton  to prevent comparitions when ignoring a memoization
                if not self.ignoreAll and id(fn) not in self._ignore and id(wrapper) not in self._ignore:
                    return memoizedfn(*args,**kwargs)
                else:
                    return fn(*args,**kwargs)
            self.memoized[id(fn)] = ref(memoizedfn) # safe to remove
            return wrapper
        return decorator
    __call__ = memoize

memoize = Memoizer() # make memoizer manager

class Cache(object):
    """
    Descriptor (non-data) for building an attribute on-demand at first use.
    @cache decorator is used for class methods without inputs (only self reference to the object)
    and it caches on first compute. ex::

        class x(object):
            @cache
            def method_x(self):
                return self.data

    .. note:: Cached data can be deleted in the decorated object to recalculate its value.
    """
    def __init__(self, func):
        """
        Initialize cache with a property function.
        """
        self.func = func # function handle
    # if method simulating getattr
    def __get__(self, instance, owner):  # if trying to get attribute
        # Build the attribute.
        cached = self.func(instance) #evaluate function over instance
        # Cache the value;
        # Creates variable (name) of value (cached) in (instance).
        # instance.name = cached
        setattr(instance, self.func.__name__, cached)
        return cached

def cachedProperty(watch=[],handle=[]):
    """
    A memoize decorator of @property decorator specifying what to trigger caching.

    :param watch: (list of strings) A list of arguments name to watch in the hashing.
    :param handle: (list of handles or empty list) Provided list is appended with the Memo
                handle were data is stored for the method and where a clear() function is provided.
    :return:
    """
    #http://code.activestate.com/recipes/576563-cached-property/
    class Memo:
        """ Memo function for cache """
        def __init__(self):
            self._cache ={}
            self._input_cache = {}
        def clear(self):
            self._cache ={}
            self._input_cache = {}

    this = Memo()
    handle.append(this)

    def noargs(f): # if not watch use this funciton
        @wraps(f)
        def get(self):
            try:
                return this._cache[f]
            except AttributeError:
                this._cache = {}
            except KeyError:
                pass
            ret = this._cache[f] = f(self)
            return ret
        return property(get)

    def withargs(f): # if watch use this function
        @wraps(f)
        def get(self):
            input_values = dict((key,getattr(self,key)) for key in watch )
            try:
                x = this._cache[f]
                if input_values == this._input_cache[f]:
                    return x
            except AttributeError:
                this._cache ={}
                this._input_cache = {}
            except KeyError:
                pass
            x = this._cache[f] = f(self)
            this._input_cache[f] = input_values
            return x
        return property(get)

    if type(watch) is not list: # if not arguments
        return noargs(watch)
    elif watch==[]:
        return noargs
    return withargs


class ObjectGetter(object):
    """
    Creates or get instance object depending if it is alive.
    """
    def __init__(self, callfunc = None, obj=None, callback=None,  **annotations):
        """
        :param callfunc: function to create object
        :param obj: (optional) alive object already obtained from callfunc
        :param callback: function called on object destruction
        :param annotations: annotations to self (this object can be used to save info or statistics)

        Example::

            class constructor: pass
            myobj = constructor() # created hard reference
            getobj = objectGetter(myobj, callfunc=constructor) # created getter
            assert myobj is getobj() # it uses the same reference as myobj
            del myobj # myobj reference lost
            a = getobj() # created (+1) other object from constructor
            b = getobj() # it uses the same hard reference a

            myobj = constructor()
            getobj = objectGetter(None, callfunc=constructor)
            assert myobj is not getobj() # created (+1) other object from constructor
            a = getobj() # this created (+1) again because there was not reference to object
            b = getobj() # it uses the same hard reference a

        """
        super(ObjectGetter, self).__init__()
        self._ref = None
        self._callfunc = None
        self._callback = None
        self.update(obj=obj, callback=callback, callfunc = callfunc, **annotations)

    def update(self, **kwargs):
        callfunc = kwargs.pop("callfunc",None)
        if callfunc:
            if not callable(callfunc):
                raise NotCallable("callfunc {} is not callable".format(callfunc))
            self._callfunc = callfunc
        obj = kwargs.pop("obj", None)
        callback = kwargs.pop("callback", None)
        if obj is not None or callback is not None:
            if callback is not None: self._callback = callback
            self._ref = ref(obj, self._callback)
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def getObj(self, throw = False):
        ob = self.raw()
        if ob is None: ob = self.create(throw)
        return ob
    __call__ = getObj

    def create(self, throw = False):
        """
        Creates an object and keep reference.

        :param throw: if there is not creation function throws error.
        :return: created object.

        .. warning:: previous object reference is lost even if it was alive.

        .. note:: Recommended only to use when object from current reference is dead.
        """
        if self.isCreatable(): # to create
            ob = self._callfunc()
            self._ref = ref(ob, self._callback)
            return ob
        elif throw:
            raise NotCreatable("No callfunc to create object")

    def raw(self):
        """
        get object from reference.
        :return: None if object is dead, object itself if is alive.
        """
        if self._ref: # if in references
            ob = self._ref()
            return ob

    def isAlive(self):
        """
        test if object of reference is alive
        """
        return self.raw() is not None

    def isCreatable(self):
        """
        test if can create object
        """
        return self._callfunc

    def isGettable(self):
        """
        test if object can be gotten either by reference or creation.
        """
        return self.isCreatable() or self.isAlive()

class Retriever(MutableMapping):
    """
    keep track of references and create objects on demand if needed.
    """
    def __init__(self):
        self.references = {}
        self._lastKey = None

    def register(self, key, method = None, instance = None):
        """
        Register object to retrieve.

        :param key: hashable key to retrieve
        :param method: callable method to get object
        :param instance: object instance already created from method
        :return:

        Example::

            def mymethod():
                class constructor: pass
                return constructor()

            ret = retriever()
            ret["obj"] = mymethod # register creating method in "obj"
            im = ret["obj"] # get object (created obj +1, with reference)
            assert im is ret["obj"] # check that it gets the same object
            # it remembers that "obj" is last registered or fetched object too
            assert ret() is ret()
            # lets register with better control (created obj2 +1, no reference)
            ret.register("obj2",mymethod(),mymethod)
            # proves that obj2 is not the same as obj (created obj2 +1, no reference)
            assert ret() is not ret["obj"]
            print list(ret.iteritems()) # get items
        """
        self.references[key] = ObjectGetter(obj=instance, callfunc=method)
        self._lastKey = key # register last key

    def __call__(self):
        return self[self._lastKey]

    def __getitem__(self, key):
        data = self.references[key]()
        self._lastKey = key
        return data

    def __setitem__(self, key, method):
        self.register(key, method=method)

    def __delitem__(self, key):
        del self.references[key]

    def __iter__(self):
        return iter(self.references)

    def __len__(self):
        return len(self.references)

class ResourceManager(Retriever):
    """
    keep track of references, create objects on demand, manage their memory and optimize for better performance.

    :param maxMemory: (None) max memory in specified unit to keep in check optimization (it does
                    not mean that memory never surpasses maxMemory).
    :param margin: (0.8) margin from maxMemory to trigger optimization.
                    It is in percentage of maxMemory ranging from 0 (0%) to maximum 1 (100%).
                    So optimal memory is inside range: maxMemory*margin < Memory < maxMemory
    :param unit: (MB) maxMemory unit, it can be GB (Gigabytes), MB (Megabytes), B (bytes)
    :param all: if True used memory is from all alive references,
                    if False used memory is only from keptAlive references.
    """
    def __init__(self, maxMemory = None, margin = 0.8, unit = "MB", all = True):
        super(ResourceManager, self).__init__()
        #self.references is a dictionary containing all the references
        #self._lastkey is effectively the last key to use when Manager is called
        self._keptMemory = 0 # used memory in bytes of references
        self._refMemory = 0 # used memory in bytes of keptAlive
        self._maxMemory = None # maximum memory in bytes
        self._unit = None # unit of memory
        self._conv = 1 # convert to any unit: equivalence with bytes from unit
        self._margin = None # private data of margin
        self._limit = None # private data representing _maxMemory*_margin in bytes
        self._all = all
        self.blacklist = set() # key of objects likely to destroy
        self.whitelist = set() # key of objects likely to keep in memory and only delete in extreme cases
        self.keptAlive = {} # objects currently being kept alive
        self.verbosity = True # if true print debugging messages
        self.invert = False # invert any order made by user
        self.methods = {} # mapping methods for user defined fields
        self.method = None # ("size","_call","_fail","_mean")
        self.unit = unit # set units
        self.margin = margin # margin of percentage (0 to 1) of memory
        if maxMemory is not None: self.maxMemory = maxMemory # set maximum memory

    def __getitem__(self, key):
        """
        gets object from key and collect statistical data
        """
        t1 = time()
        getter = self.references[key]
        getter._iddleT = (t1 - getter._iddleT) # iddle time
        wasAlive = getter.isAlive()
        wasAtFail = getter._fails>0
        try:
            obj = getter(throw = True)
            if wasAtFail: self.resetGetter(getter)
        except NotCreatable:
            obj = None
            getter._fails += 1 # keep accumulating fails
        getter._calls += 1 # actual calls
        # time it and increase _calls
        if getter._calls: # get successive times
            getter._processT = (time()-t1 + getter._processT)/2 # process time
        else: # if first call then get first profile time
            getter._processT = time()-t1

        if key not in self.keptAlive:
            toWhiteList =  getter._fails==0 and key not in self.whitelist and getter._processT > 3
            if not wasAlive: # it was not alive but now it was created
                self.optimizeObject(key,getter,toWhiteList=toWhiteList)

        self._lastKey = key # update key once finished
        return obj

    def __setitem__(self, key, value):
        self.register(key, value)

    def __delitem__(self, key):
        self._free(key) # tries to _free if kept alive
        del self.references[key] # delete entry
        if key in self.blacklist: # clear form black list
            self.blacklist.remove(key)
        if key in self.whitelist: # clear form white list
            self.whitelist.remove(key)

    def getSizeOf(self, item):
        return item.__sizeof__()#getsizeof(item,0)

    def optimizeObject(self, key, getter, toWhiteList = False):
        if getter.isAlive():
            obj = getter.raw()
            if obj is None:
                raise Exception("given a dead reference")
            flag = self.keepAlive(key,obj)
            if flag and toWhiteList:
                self.whitelist.add(key)
            return flag

    def keepAlive(self, key, obj):
        if key in self.keptAlive:
            #self._free(key)
            raise Exception("Already ketp alive")
        s = self.getSizeOf(obj)# needed memory to allocate
        flag = self._optimizeMemory(needed=s)
        if flag is not None and flag<=0: # manage memory to allocate new object
            self.keptAlive[key] = obj
            self._keptMemory += s # update kept memory
            return True

    def _free(self, key):
        """
        _free memory from keptAlive
        :param key:
        :return: Liberated memory
        """
        s = self.getSizeOf(self.keptAlive[key])
        del self.keptAlive[key]
        self._keptMemory -= s # update kept memory
        return s

    def bytes2units(self,value):
        """
        converts value from bytes to user units
        """
        return value/self._conv

    def units2bytes(self,value):
        """
        converts value from user units two bytes
        """
        return value*self._conv

    @property
    def usedMemory(self):
        """
        :return: used memory in user units
        """
        return self.bytes2units(self._keptMemory + self._refMemory)

    @usedMemory.setter
    def usedMemory(self, value):
        raise Exception("variable not settable")

    @property
    def maxMemory(self):
        if self._maxMemory is not None:
            return self.bytes2units(self._maxMemory)

    @maxMemory.setter
    def maxMemory(self, value):
        if value is None:
            print "WARNING: maximum memory configured to be unlimited"
            self._limit = None
            self._maxMemory = None
        else:
            print "WARNING: maximum memory is {} {}".format(value,self.unit)
            self._maxMemory = self.units2bytes(value) # pass in bytes
            self._limit = self._maxMemory * self.margin # re calculate limit
        self._optimizeMemory()

    @property
    def margin(self):
        """
        :return: margin used for triggering memory optimization from maxMemory.
        """
        return self._margin

    @margin.setter
    def margin(self, value):
        if value<0 or value>1: raise Exception("Margin must be between 0 and 1")
        if value is None:
            self._limit = self._maxMemory # set limit to maxMemory
        else:
            if self._maxMemory is None:
                self._limit = None
            else:
                self._limit = self._maxMemory * value

        self._margin = value
        self._optimizeMemory()

    @property
    def all(self):
        """
        :return: all flag, if True: used memory is from all alive references,
                        if False: used memory is only from keptAlive references.
        """
        return self._all

    @all.setter
    def all(self, flag):
        if self._all != flag:
            self._all = flag
            # TODO  recaculate usedMemory every time all changes

    @property
    def unit(self):
        """
        :return: user defined units
        """
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit.lower() in ("b","bytes","byte"):
            self._conv = 1
        elif unit.lower() in ("m","mb","megabytes","megas","mega"):
            self._conv = 2**20
        elif unit.lower() in ("g","gb","gigas","gigabytes","giga","gigabyte"):
            self._conv =  1000*2**20
        else:
            raise Exception("unit '{}' not supported".format(unit))
        self._unit = unit

    @staticmethod
    def resetGetter(getter):
        """
        Helper function to reset getter parameters.

        :param getter: any instance of objectGetter
        """
        getter._fails = 0 # init fail count
        getter._calls = 0 # init call count
        getter._processT = 0 # init mean of retrieving time
        getter._iddleT = time()

    def register(self, key, method = None, instance = None):
        """
        Register object to retrieve.

        :param key: hashable key to retrieve
        :param method: callable method to get object
        :param instance: object instance already created from method

        .. note:: This method is used in __setitem__ as self.register(key, value). Overwrite this
                    method to change key assignation behaviour.

        Example::

            def mymethod():
                class constructor: pass
                return constructor()

            ret = retriever()
            ret["obj"] = mymethod # register creating method in "obj"
            im = ret["obj"] # get object (created obj +1, with reference)
            assert im is ret["obj"] # check that it gets the same object
            # it remembers that "obj" is last registered or fetched object too
            assert ret() is ret()
            # lets register with better control (created obj2 +1, no reference)
            ret.register("obj2",mymethod(),mymethod)
            # proves that obj2 is not the same as obj (created obj2 +1, no reference)
            assert ret() is not ret["obj"]
            print list(ret.iteritems()) # get items
        """
        if key in self.references:
            getter = self.references[key]
            getter.update(obj = instance, callfunc=method)
        else:
            getter = ObjectGetter(obj = instance, callfunc=method)
            self.resetGetter(getter)
            self.references[key] = getter
        if instance: self.optimizeObject(key,getter,toWhiteList= not getter.isCreatable())
        self._lastKey = key # register last key

    def _checkMemory(self, needed = 0):
        """
        check if memory needs to be optimized.

        :param asValue: True to return positive (how much memory needed to _free)
                        or negative (how much memory until limit reached) value.
                        False to return Value (how much memory needed to _free) or None.
        :return Memory value or None if no capacity to allocate.
        """
        limit = self._limit

        if limit is not None:
            if needed> self._maxMemory:
                return # indicates not capacity
            if needed> limit: limit = self._maxMemory
            if self._all:
                val = self._keptMemory + self._refMemory + needed
            else:
                val = self._keptMemory + needed
            return val - limit
        else:
            return 0 # return 0 bytes to free

    def _getOrderedData(self, method = None, all = None):
        """
        Construct list of alive objects.

        :param method: method to use from self.methods, if None use self.method.
        :param all: if True: all alive objects in the references
                    (of course if they are kept alive by self.keptAlive then they are alive in references).
                    else False: only objects kept alive by self.keptAlive.
        :param self.method: specifies the method to calculate and
                            sort according to the third column in list
        :return: list with items (key,size,calculated)
        """
        method = method or self.method
        if all is None: all = self._all # let user data pass or choose default
        usemethod = method is not None and method != "size" and self.methods[method]

        data = [] # frame: key, size, calls, fails
        if all:
            for key,getter in self.references.iteritems():
                if getter.isAlive():
                    val = getter.raw()
                    size = self.getSizeOf(val)
                    if usemethod:
                        data.append((key,size,usemethod(val)))
                    else:
                        data.append((key,size))
        else:
            for key,val in self.keptAlive.iteritems():
                size = self.getSizeOf(val)
                if size:
                    if usemethod:
                        data.append((key,size,usemethod(val)))
                    else:
                        data.append((key,size))

        if method == "size": # sort just by size
            data.sort(key=lambda x:x[1],reverse=self.invert)
        elif usemethod: # sort by user defined val
            data.sort(key=lambda x:x[2],reverse=self.invert)
        return data

    def _optimizeMemory(self, needed = 0):
        """
        :param ret: dictionary or retriever
        :param _limit: _limit of memory
        :param margin:
        :param unit:
        :return:
        """
        # TODO liberate needed memory
        # ideal: total = media*len(sizes) < limit*margin
        # if limit< total > limit*percent then eliminate
        # methods: None, by ascendant, by descendant, by creations
        tofree = self._checkMemory(needed=needed)
        if tofree is None: return # nothing to do, not capacity to allocate

        if tofree<=0:
            return tofree # successful

        c = self.bytes2units
        unit = self.unit
        if self.verbosity:
            print "{1} {0} used of {2} {0}.".format(unit,self.usedMemory,self.maxMemory)
            print "{1} {0} needs to be freed to allocate {2} {0}.".format(unit,c(tofree),c(needed))

        # FIRST STAGE
        blacklist = list(self.blacklist)
        # first eliminate in black list
        freed = 0
        while len(blacklist) and freed>=tofree:
            key = blacklist.pop()
            if key in self.keptAlive:
                size = self._free(key)
                freed += size
                if self.verbosity: print "Eliminated '{}' of size {} {}".format(key,c(size),unit)
                self.blacklist.pop(key) # liminate in real black list

        tofree -= freed
        if tofree<=0:
            if self.verbosity: print "{0} {1} where freed".format(c(freed),unit)
            return tofree # successful

        # SECOND STAGE: if it did not work keep freeing
        data = self._getOrderedData(all=False) # get only in keptAlive
        if data:
            keys,sizes = zip(*data)[:2] # just needed keys and sizes
            total = sum(sizes) # total used memory in bytes
            if abs(total - self._keptMemory) > 10: # difference of 10 bytes
                self._keptMemory = total
                tofree = self._checkMemory(needed=needed)
                print("WARNING: data was not well collected")
            if tofree>0:
                #ratios = np.array(sizes)/total
                freed = 0
                for key,size in zip(keys,sizes): # free only normal ones
                    if freed>=tofree:
                        break
                    if key not in self.whitelist and key in self.keptAlive:
                        size2 = self._free(key)
                        freed += size2
                        if size2 != size:
                            print "DEBUG: key {0} had size {1} but was freed {2}".format(key,size,size2)
                        if self.verbosity: print "Eliminated '{}' of size {} {}".format(key,c(size),unit)
                tofree-= freed
                if tofree>0:
                    if self.verbosity: print "WARNING: {} {} not adequately freed".format(c(tofree),unit)
                if self.verbosity: print "{0} {2} where freed, remaining {1} {2}".format(c(freed),c(total-freed),unit)
            else:
                if self.verbosity: print "{} {} is an optimal memory. Not optimized.".format(c(total),unit)
        else:
            if self.verbosity: print "{} {} is considered low memory. Not optimized.".format(self.usedMemory, unit)
        return tofree # this means: > 0 bytes not able to free; < 0 bytes over freed; == 0 successful

def mapper(path, obj = None, mode =None, onlynumpy = False):
    """
    Save and load or map live objects to disk to free RAM memory.

    :param path: path to save mapped file.
    :param obj: the object to map, if None it tries to load obj from path if exist
    :param mode: {None, 'r+', 'r', 'w+', 'c'}.
    :param onlynumpy: if True, it saves a numpy mapper from obj.
    :return: mmap image, names of mmap files
    """
    names = None
    if onlynumpy:
        if not path.endswith(".npy"):
            path += ".npy" # correct path
        if obj is not None:
            numpySave(path,obj) # save numpy array
            names = [path] # simulates answer as onlynumpy = False
        return numpyLoad(path,mode),names
    else:
        if obj is not None:
            names = joblib.dump(obj, path) # dump object to file for mapping
        return joblib.load(path, mmap_mode=mode), names

class MemoizedDict(MutableMapping):
    """
    memoized dictionary with keys and values persisted to files.

    :param path: path to save memo file
    :param mode: loading mode from memo file {None, 'r+', 'r', 'w+', 'c'}

    .. notes:: If saveAtKey is True it will attempt to memoize each time a keyword is added
                and throw an error if not successful. But if saveAtKey is False this process
                will be carried out when the MemoizedDict instance is being destroyed in
                a proper deletion, that is, if the program ends unexpectedly all data will be
                lost or if data cannot be saved it will be lost without warning.

    .. warning:: Some data structures cannot be memoize, so this structure is not save yet.
                Use at your own risk.
    """
    def __init__(self, path, mode = None):
        #from directory import checkFile, checkDir, mkPath, rmFile
        from directory import mkPath
        import cPickle
        # TODO: change serializer to use json (it seems it is more reliable and compatible)
        # TODO: It is slow load key per key, consider making a way to load al de dictionary keys quickly
        # TODO: when clear is called, delete all memoized folder instead each key. it could be dangerous but faster
        self._map_serializer = cPickle # serializer for keys
        self._path = mkPath(path) # path to memoized keys and values
        self._map_file = os.path.join(self._path, "metadata") # file containing the keys
        #self._map = self._load_map() or {} # keeps the map to persistent files
        self._map_old = None
        self._mode = mode # mode to read the values
        self._loader = joblib.load # loader for values
        self._saver = joblib.dump # saver for values
        self._hasher = hash # function to hash the keys
        self._secure = False # this is an option to use dangerous and secure routines

    @property
    def _map(self):
        self._map_old = self._load_map()
        if self._map_old is None:
            self._map_old = {}
        return self._map_old
    @_map.setter
    def _map(self,value):
        raise VariableNotSettable("_map cannot be set")
    @_map.deleter
    def _map(self):
        raise VariableNotDeletable("_map cannot be deleted")

    def _getHash(self, key):
        """
        Hasher function to use in persistent files.

        :param key: key to hash
        """
        return '{}'.format(self._hasher(key))

    def _load(self, key):
        """
        Loads key from persistent file using map.

        :param key: hashable key.
        :return: value stored in key.
        """
        hashed,files = self._map[key]
        filename = os.path.join(self._path, hashed)
        if os.path.isfile(filename):
            try:
                return self._loader(filename, self._mode)
            except IOError:
                raise KeyError
        else:
            raise KeyError

    def _save(self, key, value):
        """
        saves key to persistent files.

        :param key: hashable key.
        :param value: serializable object.
        :return: None
        """
        try:
            hashed,files = self._map[key]
            for file in files: # remove old keys
                try:
                    os.remove(file) # FIXME enclose in try/except, who knows files could be deleted
                except OSError:
                    pass # file could have been deleted
            del self._map_old[key]
        except KeyError:
            hashed = self._getHash(key)

        filename = os.path.join(self._path, hashed)
        try:
            # TODO: Consider implementing a set to keep track of hashes so that a hash is not repeated
            # TODO: though add overloads, consider persisting the key along the value too for recovery purposes.
            self._map_old[key] = (hashed,self._saver(value, filename))
            self._save_map()
        except OSError:
            print " Race condition in the creation of the directory "

    def _save_map(self):
        """
        persist dictionary map to file (metadata).
        """
        if self._map_old is not None:
            with open(self._map_file, 'wb') as f: # FIXME: consumes too much time
                return self._map_serializer.dump(self._map_old, f) # save dictionary

    def _load_map(self):
        """
        loads metadata of dictionary map from persisted file.
        """
        try:
            with open(self._map_file, 'rb') as f:
                return self._map_serializer.load(f) # get session
        except IOError as e:
            return
        except EOFError as e:
            raise e

    def clear(self): # overloads clear in the abc class
        """
        Remove all items from D.
        """
        # for security it is better to wait until all keys are
        # safely deleted and not deleting everything at once
        for key in self._map.keys():
            try:
                del self[key]
            except KeyError:
                pass
        """
        # This is really dangerous, an user could change the _path
        # variable and delete something else. Or if there exists
        # a symbolic link it could bring problems.
        # http://stackoverflow.com/a/185941/5288758
        folder = self._path
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        self._map.clear()
        """

    def __setitem__(self, key, value):
        self._save(key,value)

    def __getitem__(self, key):
        return self._load(key)

    def __delitem__(self, key):
        hashed,files = self._map[key]
        for file in files: # remove old keys
            try:
                os.remove(file) # FIXME enclose in try/except, who knows files could be deleted
            except OSError:
                pass # file could have been deleted
        del self._map_old[key]
        self._save_map()

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        try:
            #self[key] # It has to load values taking too long
            hashed,files = self._map[key] # test that key is in map
            for file in files: # test that key really exists in disk
                if not os.path.exists(file):
                    raise KeyError
        except KeyError:
            return False
        else:
            return True