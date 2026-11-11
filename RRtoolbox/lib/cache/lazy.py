# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Lazy evaluation, object getters, and reference retrievers.

Provides proxy objects and mappings to defer expensive object creation
until accessed, while managing weak references to alive instances.
"""

from collections.abc import MutableMapping
from weakref import ref

try:
    from ..root import NotCallable, NotCreatable
except (ImportError, ValueError):
    from RRtoolbox.lib.root import NotCallable, NotCreatable

__all__ = ["ObjectGetter", "Retriever", "LazyDict"]


class ObjectGetter(object):
    """
    Creates or get instance object depending if it is alive.
    """

    def __init__(self, callfunc=None, obj=None, callback=None, **annotations):
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
        self.update(obj=obj, callback=callback,
                    callfunc=callfunc, **annotations)

    def update(self, **kwargs):
        callfunc = kwargs.pop("callfunc", None)
        if callfunc:
            if not callable(callfunc):
                raise NotCallable(
                    "callfunc {} is not callable".format(callfunc))
            self._callfunc = callfunc
        obj = kwargs.pop("obj", None)
        callback = kwargs.pop("callback", None)
        if obj is not None or callback is not None:
            if callback is not None:
                self._callback = callback
            self._ref = ref(obj, self._callback)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def getObj(self, throw=False):
        ob = self.raw()
        if ob is None:
            ob = self.create(throw)
        return ob
    __call__ = getObj

    def create(self, throw=False):
        """
        Creates an object and keep reference.

        :param throw: if there is not creation function throws error.
        :return: created object.

        .. warning:: previous object reference is lost even if it was alive.

        .. note:: Recommended only to use when object from current reference is dead.
        """
        if self.isCreatable():  # to create
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
        if self._ref:  # if in references
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

    def register(self, key, method=None, instance=None):
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
        self._lastKey = key  # register last key

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


class LazyDict(MutableMapping):
    """
    Create objects on demand if needed. call the instance with keys to prevent it
    from using lazy evaluations (e.g. instead of self[key] use self(key) to prevent
    recursion). Containing operations are safe to prevent recursion (e.g. if key in self
    instead of self[key]). In addition use self.isLazy flag to enable or disable lazy
    operations to prevent possible recursions when getter is called.
    """

    def __init__(self, getter, dictionary=None):
        if not callable(getter):
            raise NotCallable(
                "getter must be a callable function to process keys")
        if dictionary is None:
            dictionary = {}
        self.getter = getter
        self.dictionary = dictionary
        self._lastKey = None
        self.isLazy = True
        self.cached = True

    def __call__(self, key=None):
        if key is None:
            return self[self._lastKey]
        else:
            return self.dictionary[key]

    def __getitem__(self, key):
        try:
            if self.cached:
                # try to use cached data
                data = self.dictionary[key]
            else:  # always compute data
                data = self.getter(key)
                self.dictionary[key] = data
        except Exception as e:
            if self.isLazy or not self.cached:
                # for lazy or to recompute
                data = self.getter(key)
                self.dictionary[key] = data
            else:
                raise e
        self._lastKey = key
        return data

    def __setitem__(self, key, val):
        self.dictionary[key] = val

    def __delitem__(self, key):
        del self.dictionary[key]

    def __iter__(self):
        return iter(self.dictionary)

    def __len__(self):
        return len(self.dictionary)

    def __contains__(self, key):
        try:
            # self[key] # It can create a recursion
            self.dictionary[key]
        except KeyError:
            return False
        else:
            return True

