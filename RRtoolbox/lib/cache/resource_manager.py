# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Dynamic resource management and memory optimization.

Keeps track of object references, creates objects on demand, and dynamically
manages in-memory allocations according to memory budgets and policies.
"""

from time import time
from .lazy import Retriever, ObjectGetter

try:
    from ..root import NotCreatable
except (ImportError, ValueError):
    from RRtoolbox.lib.root import NotCreatable

__all__ = ["ResourceManager", "parse_memory_unit"]


def parse_memory_unit(unit_str):
    """
    Parse memory unit string and return conversion factor to bytes.

    :param unit_str: 'B', 'MB', 'GB', etc.
    :return: multiplier in bytes
    """
    u = unit_str.lower()
    if u in ("b", "bytes", "byte"):
        return 1
    elif u in ("m", "mb", "megabytes", "megas", "mega"):
        return 2**20
    elif u in ("g", "gb", "gigas", "gigabytes", "giga", "gigabyte"):
        return 1000 * 2**20
    else:
        raise ValueError("unit '{}' not supported".format(unit_str))


def _evict_blacklist_items(blacklist, kept_alive, free_func, tofree, verbosity, unit, b2u):
    """Stage 1: Evict items in the blacklist."""
    freed = 0
    bl_list = list(blacklist)
    while len(bl_list) and freed < tofree:
        key = bl_list.pop()
        if key in kept_alive:
            size = free_func(key)
            freed += size
            if verbosity:
                print("Eliminated '{}' of size {} {}".format(key, b2u(size), unit))
            blacklist.discard(key)
    return freed


def _evict_unwhitelisted_items(ordered_data, whitelist, kept_alive, free_func, tofree, verbosity, unit, b2u):
    """Stage 2: Evict non-whitelisted items according to ordering."""
    freed = 0
    keys = [item[0] for item in ordered_data]
    sizes = [item[1] for item in ordered_data]
    for key, size in zip(keys, sizes):
        if freed >= tofree:
            break
        if key not in whitelist and key in kept_alive:
            size2 = free_func(key)
            freed += size2
            if size2 != size and verbosity:
                print("DEBUG: key {0} had size {1} but was freed {2}".format(key, size, size2))
            if verbosity:
                print("Eliminated '{}' of size {} {}".format(key, b2u(size), unit))
    return freed


class ResourceManager(Retriever):
    """
    keep track of references, create objects on demand, manage their memory and optimize for better performance.

    :param maxMemory: (None) max memory in specified unit to keep in check optimization.
    :param margin: (0.8) margin from maxMemory to trigger optimization (0 to 1).
    :param unit: (MB) maxMemory unit, it can be GB, MB, B.
    :param all: if True used memory is from all alive references,
                if False used memory is only from keptAlive references.
    """

    def __init__(self, maxMemory=None, margin=0.8, unit="MB", all=True):
        super(ResourceManager, self).__init__()
        self._keptMemory = 0
        self._refMemory = 0
        self._maxMemory = None
        self._unit = None
        self._conv = 1
        self._margin = None
        self._limit = None
        self._all = all
        self.blacklist = set()
        self.whitelist = set()
        self.keptAlive = {}
        self.verbosity = True
        self.invert = False
        self.methods = {}
        self.method = None
        self.unit = unit
        self.margin = margin
        if maxMemory is not None:
            self.maxMemory = maxMemory

    def __getitem__(self, key):
        t1 = time()
        getter = self.references[key]
        getter._iddleT = t1 - getter._iddleT
        wasAlive = getter.isAlive()
        wasAtFail = getter._fails > 0
        try:
            obj = getter(throw=True)
            if wasAtFail:
                self.resetGetter(getter)
        except NotCreatable:
            obj = None
            getter._fails += 1
        getter._calls += 1
        if getter._calls > 1:
            getter._processT = (time() - t1 + getter._processT) / 2.0
        else:
            getter._processT = time() - t1

        if key not in self.keptAlive:
            toWhiteList = getter._fails == 0 and key not in self.whitelist and getter._processT > 3
            if not wasAlive:
                self.optimizeObject(key, getter, toWhiteList=toWhiteList)

        self._lastKey = key
        return obj

    def __setitem__(self, key, value):
        self.register(key, value)

    def __delitem__(self, key):
        self._free(key)
        del self.references[key]
        self.blacklist.discard(key)
        self.whitelist.discard(key)

    def getSizeOf(self, item):
        return item.__sizeof__()

    def optimizeObject(self, key, getter, toWhiteList=False):
        if getter.isAlive():
            obj = getter.raw()
            if obj is None:
                raise Exception("given a dead reference")
            flag = self.keepAlive(key, obj)
            if flag and toWhiteList:
                self.whitelist.add(key)
            return flag

    def keepAlive(self, key, obj):
        if key in self.keptAlive:
            raise Exception("Already kept alive")
        s = self.getSizeOf(obj)
        flag = self._optimizeMemory(needed=s)
        if flag is not None and flag <= 0:
            self.keptAlive[key] = obj
            self._keptMemory += s
            return True

    def _free(self, key):
        if key not in self.keptAlive:
            return 0
        s = self.getSizeOf(self.keptAlive[key])
        del self.keptAlive[key]
        self._keptMemory -= s
        return s

    def bytes2units(self, value):
        return value / float(self._conv)

    def units2bytes(self, value):
        return value * self._conv

    @property
    def usedMemory(self):
        return self.bytes2units(self._keptMemory + self._refMemory)

    @property
    def maxMemory(self):
        if self._maxMemory is not None:
            return self.bytes2units(self._maxMemory)

    @maxMemory.setter
    def maxMemory(self, value):
        if value is None:
            print("WARNING: maximum memory configured to be unlimited")
            self._limit = None
            self._maxMemory = None
        else:
            print("WARNING: maximum memory is {} {}".format(value, self.unit))
            self._maxMemory = self.units2bytes(value)
            self._limit = self._maxMemory * self.margin
        self._optimizeMemory()

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, value):
        if value is not None and (value < 0 or value > 1):
            raise Exception("Margin must be between 0 and 1")
        if value is None:
            self._limit = self._maxMemory
        else:
            self._limit = self._maxMemory * value if self._maxMemory is not None else None
        self._margin = value
        self._optimizeMemory()

    @property
    def all(self):
        return self._all

    @all.setter
    def all(self, flag):
        self._all = flag

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        self._conv = parse_memory_unit(unit)
        self._unit = unit

    @staticmethod
    def resetGetter(getter):
        getter._fails = 0
        getter._calls = 0
        getter._processT = 0
        getter._iddleT = time()

    def register(self, key, method=None, instance=None):
        if key in self.references:
            getter = self.references[key]
            getter.update(obj=instance, callfunc=method)
        else:
            getter = ObjectGetter(obj=instance, callfunc=method)
            self.resetGetter(getter)
            self.references[key] = getter
        if instance:
            self.optimizeObject(key, getter, toWhiteList=not getter.isCreatable())
        self._lastKey = key

    def _checkMemory(self, needed=0):
        limit = self._limit
        if limit is not None:
            if needed > self._maxMemory:
                return None
            if needed > limit:
                limit = self._maxMemory
            val = (self._keptMemory + self._refMemory + needed) if self._all else (self._keptMemory + needed)
            return val - limit
        return 0

    def _getOrderedData(self, method=None, all=None):
        method = method or self.method
        if all is None:
            all = self._all
        usemethod = method is not None and method != "size" and self.methods.get(method)
        data = []
        source = self.references.items() if all else self.keptAlive.items()
        for key, item in source:
            val = item.raw() if (all and hasattr(item, "raw")) else item
            if val is not None:
                size = self.getSizeOf(val)
                entry = (key, size, usemethod(val)) if usemethod else (key, size)
                data.append(entry)

        sort_idx = 2 if usemethod else (1 if method == "size" else None)
        if sort_idx is not None:
            data.sort(key=lambda x: x[sort_idx], reverse=self.invert)
        return data

    def _optimizeMemory(self, needed=0):
        tofree = self._checkMemory(needed=needed)
        if tofree is None or tofree <= 0:
            return tofree

        b2u, unit = self.bytes2units, self.unit
        if self.verbosity:
            print("{1} {0} used of {2} {0}.".format(unit, self.usedMemory, self.maxMemory))
            print("{1} {0} needs to be freed to allocate {2} {0}.".format(unit, b2u(tofree), b2u(needed)))

        # STAGE 1: Evict blacklisted items
        freed = _evict_blacklist_items(self.blacklist, self.keptAlive, self._free, tofree, self.verbosity, unit, b2u)
        tofree -= freed
        if tofree <= 0:
            if self.verbosity:
                print("{0} {1} were freed".format(b2u(freed), unit))
            return tofree

        # STAGE 2: Evict unwhitelisted items
        data = self._getOrderedData(all=False)
        if data:
            freed_s2 = _evict_unwhitelisted_items(
                data, self.whitelist, self.keptAlive, self._free, tofree, self.verbosity, unit, b2u)
            tofree -= freed_s2
            if self.verbosity:
                total_freed = freed + freed_s2
                print("{0} {1} were freed".format(b2u(total_freed), unit))
        return tofree

