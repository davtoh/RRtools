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
from fnmatch import fnmatch,fnmatchcase
from itertools import groupby
from collections import OrderedDict
from string import Formatter
formater = Formatter() # string formatter str.format
# ----------------------------GLOBAL VARIABLES---------------------------- #
__author__ = 'Davtoh'
# ----------------------------BASIC FUNCTIONS---------------------------- #


class StdoutSIM:
    """
    simple logger to simulate stdout output
    """
    def __init__(self, disable = False, stdout = None):
        if stdout is None:
            stdout = sys.stdout
        self.stdout = stdout
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

stdout = StdoutSIM()

class StdoutLOG:
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

        # close previous StdoutLOG
        if chain and isinstance(sys.stdout, StdoutLOG):
            sys.stdout.close()

        if not chain and isinstance(sys.stdout, StdoutSIM):
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

    def flush(self,**kwargs):
        for log in self.logs:
            log.flush(**kwargs)

    def close(self,**kwargs):
        # close according to chain option
        if self._chain:
            # close all the logs in the chain
            for log in self.logs:
                log.close(**kwargs)
        elif not self._closed:
            # only closes this log but keep alive
            # previous logs
            self.logs.remove(self._log)
            self._log.close()
            self._closed = True

class StdoutMULTI:
    """
    Enclose several file-like objects.

    :param filelist = list of file-like objects
    """
    def __init__(self, filelist):
        self.filelist = filelist

    def write(self, text, **kwargs):
        for log in self.filelist:
            log.write(text, **kwargs)

    def printline(self, text, **kwargs):
        for log in self.filelist:
            log.printline(text, **kwargs)

    def printlines(self, lines, **kwargs):
        for log in self.filelist:
            log.printlines(lines, **kwargs)

    def flush(self,**kwargs):
        for log in self.filelist:
            log.flush(**kwargs)

    def close(self,**kwargs):
        for log in self.filelist:
            log.close(**kwargs)

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

def formatOnly(format_string, **kwargs):
    """
    Format string only with provided keys

    :param format_string: string to format
    :param kwargs: format keys
    :return: formatted string
    """
    keys = [i[1] for i in formater.parse(format_string) if i[1] is not None]
    for key in keys:
        if key not in kwargs:
            kwargs[key] = "{{{}}}".format(key)
    return format_string.format(**kwargs)


def formatConsume(format_string, kwargs, formatter=None, handle = None):
    """
    Format with dictionary and consume keys.

    :param format_string: string to format
    :param kwargs: dictionary containing the keys and values to format string.
            The keys must be supported by the string formatter
    :param formatter: (None) formatter function to format string
    :return: formatted string
    """
    keys_in_str = [i[1] for i in formater.parse(format_string) if i[1] is not None]
    if formatter is None:
        formatted_string = format_string.format(**kwargs)
    else:
        formatted_string = formatter(format_string,**kwargs)
    for k in keys_in_str:
        if handle is not None:
            handle(kwargs,k)
        else:
            try:
                del kwargs[k]
            except:
                pass
    return formatted_string


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

    def __init__(self, factor = None, abbreviate = True):
        """

        :param factor: anything to look in factors (i.e. factor list with Factor structures).
        :param abbreviate: index to return from Factor structure when factor is asked.

        .. notes: A factor structure is of the form ("Name","abbreviation",value)
        """
        self._factor = None
        self._factorsCache = None
        self.abbreviate = abbreviate
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
        return self._factor[self.abbreviate]
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
        return factor * self._factor[2] / float(to[2]), to[self.abbreviate] # converted value, factor

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

class Magnitude(object):
    def __init__(self, value =0, factor = None, unit = None, precision = None, abbreviate = False):
        self.value = value
        self.precision = precision
        self.factor = factor
        self.unit = unit
        self.abbreviate = abbreviate
    def format_value(self, value):
        if self.precision is None:
            text = "{:f} {}{}"
        else:
            text = "{{:0.{}f}} {{}}{{}}".format(self.precision)
        if self.factor: #
            if isinstance(self.factor,basestring):
                return text.format(*(FactorConvert(
                        abbreviate= self.abbreviate).convert(value, self.factor) +
                                     (self.unit,)))
            else:
                return text.format(*(FactorConvert(
                        abbreviate= self.abbreviate).convert2sample(value, self.factor) +
                                     (self.unit,)))
        else:
            return text.format(value, "", self.unit)
    def __str__(self):
        return self.format_value(self.value)


class TimeCode(object):
    """
    Context to profile code by printing a prelude and prologue with time.

    :param msg: prelude or description message
    :param factor: factor supported by FactorConvert class
    :param precision: number of digits after a float point
    :param abv: if True prints "s", if False "seconds" for time
    :param endmsg: prologue message
    :param enableMsg: (True) A flag specifying if context
            should be printed or not.
    :param printfunc: function to print messages. By default it
            is sys.stdout.write
    """
    def __init__(self, msg = None, factor = None, precision = None,
                 abv=None, endmsg = "{time}\n", enableMsg= True,
                 printfunc= None, profiler = None, profile_point = None):
        self.msg = msg
        self.factor = factor
        self.precision = precision
        self.abv = abv
        self.endmsg = endmsg
        self.enableMsg = enableMsg
        self.printfunc = printfunc
        self.time_start = None
        self.time_end = None
        self.profiler = profiler
        self.profile_point = profile_point

    @property
    def time(self):
        if self.time_start is None:
            return 0
        else:
            return time()-self.time_start

    @property
    def time_end(self):
        if self._time_end is None:
            return self.time
        return self._time_end
    @time_end.setter
    def time_end(self,value):
        self._time_end = value
    @time_end.deleter
    def time_end(self):
        del self._time_end

    def __enter__(self):
        if self.printfunc is None:
            self.printfunc = sys.stdout.write
        if self.enableMsg and self.msg is not None:
            self.printfunc(self.msg)
        self.time_start = time() # begin chronometer
        self.time_end = None
        if self.profiler is not None:
            if self.profile_point is None:
                msg = self.msg
            else:
                msg = self.profile_point
            if isinstance(msg,basestring):
                self.profile_point = self.profiler.open_point(msg = msg)
            else:
                self.profile_point = self.profiler.open_point(*msg)
        elif isinstance(self.profile_point, Profiler):
            self.profile_point.time_start = self.time_start
        return self

    def __exit__(self, type, value, traceback):
        if isinstance(self.profile_point, Profiler):
            self.profile_point.close()
        t = self.time # end chronometer
        self.time_end = t
        if self.enableMsg and self.endmsg is not None:
            # abbreviation
            if self.abv is True:
                u,i = "s",True
            else:
                u,i = " seconds",False
            self.printfunc(self.endmsg.format(
                time=Magnitude(value = t, precision=self.precision,
                unit=u, factor=self.factor, abbreviate=i)))

class Profiler(object):
    """
    profiler for code points

    :param msg: custom comment for profiling point
    :param tag: classification tag
    :parameter space: (" ")
    :parameter format_line: ("{space}{tag}{msg}{time}")
    :parameter format_structure: ("\n{space}[{tag}{msg}{time}{child}]{side}")
    :parameter points: profile instaces which are divided in "side" or "children" points
            according if they are side by side or are inside of the profiler.
    """
    def __init__(self, msg = None, tag= None):
        self.time_start = time()
        self.msg = msg
        self.tag = tag
        self.space = " "
        self.format_line = "{space}{tag}{msg}{time}"
        self.format_structure = "\n{space}[{tag}{msg}{time}{child}]{side}"
        self.format_tag = "({}) "
        self.format_time = " -> {:0.4f} secs"
        self.format_msg = "{!r}"
        self.time_end = None
        self.points = []

    @property
    def time(self):
        """
        :return: overall time of profiling
        """
        try:
            return self.time_end -self.time_start
        except:
            points_diff = [i.time for i in self.points if i.time]
            if points_diff:
                return np.sum(points_diff)
            return None

    def _add_point(self, point):
        """
        keep track of points and organize self.points (not intended for the user)

        :param point: profile instance
        """
        if self.points and not point in self.points:
            child = self.points[-1]
            if child.time_end is None:
                child._add_point(point)
            else:
                self.points.append(point)
        elif not self.points:
            self.points.append(point)

    def _close_point(self, point):
        """
        close a points and adds to the list of points (not intended for the user)

        :param point: profile instance
        """
        point.close()
        self._add_point(point)

    def open_point(self, msg = None, tag= None):
        """
        Open a profiling point to track time.

        :param msg: custom comment for profiling point
        :param tag: classification tag
        :return:
        """
        point = Profiler(msg, tag)
        self._add_point(point)
        return point

    def close(self):
        """
        close profiler and all their points
        """
        for p in self.points:
            p.close()
        if self.time_end is None:
            self.time_end = time()

    def formatter(self, level, tag, msg, time):
        """
        format profiling point arguments.

        :param level:
        :param tag: classification tag
        :param msg: custom comment of profiling point
        :param time: time of profiling
        :return: formatted (spacing, tag, msg, time)
        """
        if tag:
            if isinstance(self.format_tag,basestring):
                tag = self.format_tag.format(tag)
            else:
                tag = self.format_tag(tag)
        else:
            tag = ""
        if time:
            if isinstance(self.format_time,basestring):
                time = self.format_time.format(time)
            else:
                time = self.format_time(time)
        else:
            time = ""
        if msg:
            if isinstance(self.format_msg,basestring):
                msg = self.format_msg.format(msg)
            else:
                msg = self.format_msg(msg)
        else:
            msg = ""
        return self.space * level, tag, msg, time

    def _collapse_structure(self, children, collapse = None):
        """
        Collapse list of profiles (not intended for the user)

        :param children: list of profiles
        :param collapse: list for collapsing repeated tags or messages.
        :return: filtered list
        """

        # collapse them
        if collapse is not None and children:
            new_children = [] # init new list of collapsed children
            indexes = dict() # keep cache of indexed children
            for child in children: # loop over children
                cmp = tuple(child[:2]) # comparison and key index
                if new_children \
                        and (collapse is True or cmp[0] in collapse or cmp[1] in collapse) \
                        and cmp in indexes:
                    index = indexes[cmp] # return cached index
                    new_children[index][2] += child[2] # add time
                    new_children[index][3].extend(child[3]) # add children of child
                else: # if not indexed
                    indexes[cmp] = len(new_children) # cache actual index
                    new_children.append(child) # append a new child
            return new_children # replace for new children list
        return children

    def restructure(self, structure, collapse):
        """
        reprocess an already created structure.

        :param structure: structure.
        :param collapse: list for collapsing repeated tags or messages.
        :return: reprocessed structure
        """
        tag, msg, time, children = structure
        if children:
            children = self._collapse_structure(children,collapse)
            for i,child in enumerate(children):
                children[i] = self.restructure(child, collapse)
            structure[3] = children
        return structure

    def structure(self, collapse = None):
        """
        profiling structure.

        :param collapse: list for collapsing repeated tags or messages.
        :return: structure with format [tag,msg,time,children]
        """
        # collect structure of children
        children = [point.structure(collapse) for point in self.points]
        children = self._collapse_structure(children, collapse)
        return [self.tag, self.msg, self.time, children] # permit item assignment

    def _helper_lines_unformatted(self, struct, level=0):
        """
        helper to generate lines (not intended for the user)

        :param struct: profiling structure
        :param level: structure level
        :return: generator with outputs (level, tag, msg, time)
        """
        tag,msg,time,children = struct
        yield (level, tag, msg, time)
        for child in children:
            for i in self._helper_lines_unformatted(child, level + 1):
                yield i

    def lines_unformatted(self, collapse =None):
        """
        generate structure lines

        :param collapse: list for collapsing repeated tags or messages.
        :return: generator with outputs (level, tag, msg, time)
        """
        return self._helper_lines_unformatted(self.structure(collapse))

    def lines_formatted(self, collapse = None):
        """
        generate string lines

        :param collapse: list for collapsing repeated tags or messages.
        :return: list of lines
        """
        lns = []
        for level,tag,msg,time in self.lines_unformatted(collapse):
            space,tag,msg,time = self.formatter(level,tag,msg,time)
            lns.append(self.format_line.format(space=space, tag=tag, msg=msg, time=time))
        return lns

    def string_lines(self):
        """
        string with plain structure of profiling
        """
        return "\n".join(self.lines_formatted())

    def _helper_string_structured(self, structs, level=0):
        """
        helper to generate string with structure of profiling

        :param structs: list of structures
        :param level: structure level
        :return: string
        """
        if structs:
            mystr = self.format_structure
            tag,msg,time,children = structs[0]
            space,tag,msg,time = self.formatter(level,tag,msg,time)
            mystr = mystr.format(space=space, tag=tag, msg=msg, time=time,
                                 child=self._helper_string_structured(children, level + 1),
                                 side=self._helper_string_structured(structs[1:], level))
            return mystr
        else:
            return ""

    def string_structured(self, collapse = None, structure = None):
        """
        string with plain structure of profiling

        :param collapse: list for collapsing repeated tags or messages.
        :param structure: (None) uses and already created structure. If None
                it creates the structure.
        :return: string
        """
        if structure is None:
            return self._helper_string_structured([self.structure(collapse)])
        else:
            return self._helper_string_structured([self.restructure(structure, collapse)])

class Controlstdout(object):
    """
    Context manager to control output to stdout

    :param disable: if True suppress output.
    :param buffer: (None) if True creates a buffer to collect all data
            printed to the stdout which can be retrieved with self.buffered.
            A file can be given but if it is write-only it cannot retrieve
            data to self.buffered so "w+" is recommended to be used with self.buffered.

    .. warning:: If a references to sys.stdout is kept before the Controlstdout
            instance then output can be printed trough it and cannot be
            controlled by the Controlstdout context.
    """

    def __init__(self, disable = True, buffer = None):
        self.disable = disable
        self.buffer = buffer
        self.buffered = ""
        self.stdout_old = None
        self.stdout_new = None

    def __enter__(self):
        self.stdout_old = sys.stdout
        self.stdout_new = StdoutSIM(self.disable)
        if self.buffer:
            if self.buffer is True: # in case it is not defined
                import StringIO
                self.buffer = StringIO.StringIO()
            self.stdout_new = StdoutMULTI([self.stdout_new, self.buffer])
        sys.stdout = self.stdout_new
        return self

    def __exit__(self, type, value, traceback):
        if self.buffer:
            try:
                self.buffer.seek(0)
                self.buffered = self.buffer.read()
                self.buffer.close()
            except:
                pass
        sys.stdout = self.stdout_old

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
    return filter(check,fns) # [p for p in fns if check(p)]

def ensureList(obj):
    """ ensures that object is list """
    if isinstance(obj,list):
        return obj # returns original lis
    elif hasattr(obj, '__iter__'): # for python 2.x check if obj is iterablet
        return list(obj) # converts to list
    else:
        return [obj] # object is returned inside list

class globFilter(object):
    '''glob filter for patterns'''
    def __init__(self, include=None, exclude=None, case= False):
        """
        :param include: permitted patterns
        :param exclude: excluded patterns. it takes priority over includes
        :param case: True or False for case sensitive patterns
        """
        if include is None: include = []
        self.include = ensureList(include)
        if exclude is None: exclude = []
        self.exclude = ensureList(exclude)
        if case:
            self.cmpfunc = fnmatchcase
        else:
            self.cmpfunc = fnmatch

    def __call__(self, cmp=None):
        """
         Evaluate filter.

        :param cmp: iterator or string
        :return: True or false if cmp pass filter test
        """
        if hasattr(cmp,"__iter__"):
            return [self(i) for i in cmp]
        else:
            cmpfunc = self.cmpfunc # prevents from accessing self
            for pattern in self.exclude: # if exclude is [] don't do the test
                if cmpfunc(cmp, pattern):
                    return False
            if self.include: # test includes
                for pattern in self.include:
                    if cmpfunc(cmp, pattern):
                        return True
            else: # if include is False or None return True anyway i.e. equivalent to ['*']
                return True
            return False


def lookinglob(pattern, path=None, ext=None, forward=None,
               filelist=None, aslist = False, raiseErr = False):
    """
    Look for patterns in Path. It looks as {if path}{if pattern}{if forward}{if ext}.

    :param pattern: string to look for pattern.
    :param path: (None) path to look for pattern
    :param ext: (None) extension of pattern in path
    :param forward: (None) look changes after pattern and before ext parameter.
    :param filelist: (None) simulates the files in path and look patterns in this list.
    :param aslist: (False) if False it returns the first match case string
            else the list of matching cases.
    :param raiseErr: If true raises Exception if patter is not found in path or there
            are more than one match
    :return: matched case if returnAll is False else the list of matched cases
            or if no match is found None
    """
    if path is None and ext is None and forward is None:
        # filter pure pattern
        tests = [pattern]
    else:
        if path is None:
            path = ""
        # create a list of test with pattern
        tests = [pattern, "*{pattern}", "{path}{pattern}", "{path}*{pattern}"]
        if forward:
            tests.extend(
                ["{pattern}*","*{pattern}*", "{path}{pattern}*", "{path}*{pattern}*"]
            )
        if ext is not None and not pattern.endswith(ext):
            if not ext.startswith("."):
                ext = "."+ext
            tests = [i + ext for i in tests]

    tests = set(tests) # eliminate repeated keys
    if filelist is None:
        # look in a real path
        ress = []
        for test in tests:
            # get list of matches
            res = glob(test.format(path=path, pattern=pattern))
            if not res: # if not match continue
                continue
            # return first match
            if not aslist and len(res)==1:
                return res[0]
            # keep acumulatting matches
            ress.extend(res)
    else:
        # look in a simulated path with files in filelist
        include = [test.format(path=path, pattern=pattern) for test in tests]
        ress = filter(globFilter(case=True,include=include),filelist)

    if len(ress)==1 and aslist:
        return ress # return list anyways
    elif len(ress)==1:
        return ress[0] # return only matched
    elif ress and raiseErr:
        raise Exception("More than one file with pattern '{}'".format(pattern))
    elif raiseErr:
        #if filelist is not None:
        #    path = filelist
        if path is None:
            raise Exception("pattern '{}' not found".format(pattern))
        raise Exception("pattern '{}' not in {}".format(pattern, path))
    elif aslist:
        return list(set(ress)) # return unique matches
    return None


if __name__ == "__main__":

    if False:
        def myfc():
            sleep(1)
            return "======"*10000
        with TimeCode("init",100,abv=True):
            myfc()

    if False:
        fac = FactorConvert()
        print fac.convert(10,100)
        print fac.parts(1001010.01010101)
        print "{:f} {}".format(*FactorConvert("m").convert2sample(36797.59, "m"))
        print FactorConvert("m").convert(36797.59)
        print fac.convert(1000)

    if True:
        pf = Profiler("Init")
        p1 = pf.open_point("for loop")
        for i in xrange(5):
            with TimeCode("loop {}".format(i),profiler=pf):
                pass
            with TimeCode("loop {}".format(i),profiler=pf):
                pass
        p2 = pf.open_point("in level of p1")
        p1.close()
        p3 = pf.open_point("other process")
        #print pf.string_structured()
        print pf.string_structured(True,pf.structure())

# ---------------------------- EXCEPTIONS ---------------------------- #

class TimeOutException(Exception): pass

class TransferExeption(Exception): pass

class VariableNotSettable(Exception): pass

class VariableNotDeletable(Exception): pass

class NotCallable(Exception):
    """
    Defines objectGetter error: given object is not callable.
    """
    pass

class NotCreatable(Exception):
    """
    Defines objectGetter error: objectGetter cannot create new object.
    """
    pass

class NoParserFound(Exception): pass


class NameSpace(object):
    """
    used to store variables
    """