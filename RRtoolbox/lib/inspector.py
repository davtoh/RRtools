# -*- coding: utf-8 -*-
"""
This module is an all purpose intended for debugging, tracking, auto-documenting and self-introspecting the
package

Made by Davtoh. Powered partially by pycallgraph.
Dependent project: https://github.com/gak/pycallgraph/#python-call-graph
"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import object
import sys,os
import tempfile
import traceback
import functools
import time
import re
import inspect
from pycallgraph.output import GraphvizOutput
import pycallgraph #PyCallGraph
from pycallgraph.tracer import AsyncronousTracer, SyncronousTracer
from importlib import import_module

__license__ = pycallgraph.__license__ + " see https://github.com/gak/pycallgraph/blob/develop/LICENSE"

from . import directory as dr




def funcData(func):
    argspec = inspect.getargspec(func) # get function data
    args = argspec.args
    defaults = argspec.defaults
    if defaults: # get args with their values
        defaults = {args[i-len(defaults)]:val for i,val in enumerate(defaults)}
    name = func.__name__
    doc = func.__doc__
    sourcefile = inspect.getsourcefile(func) # get path of source, unlike getfile that even gets compiled
    lines, line = inspect.getsourcelines(func) # get source
    imp_from = inspect.getmodule(func).__name__
    toreturn = dict(name = name,args=args, defaults = defaults, doc = doc,
                    keywords=argspec.keywords, varargs = argspec.varargs,
                    lines=lines,line=line,sourcefile=sourcefile,imp_from =imp_from)
    return toreturn

def reloadFunc(func):
    data = funcData(func)
    return load(data["imp_from"],data["name"])

def load(mod_name,obj_name):
    """
    Convert a string version of a class name to the object.

    For example, get_class('sympy.core.Basic') will return
    class Basic located in module sympy.core
    """
    #obj = getattr(__import__(mod_name, {}, {}, ['*']), obj_name)
    return getattr(import_module(mod_name),obj_name)

def tracer(instance, broadcast = True, report = True):
    """
    Tracer for decorated functions.

    :param instance: Logger instance
    :param broadcast:
    :param report:
    :return:
    """
    def decorator(func):
        #argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        funcname = func.__name__
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            instance.func = func
            instance.funcname = funcname
            instance.stack = inspect.stack() # traceback.extract_stack()
            instance.inputs = inspect.getcallargs(func,*args,**kwargs)
            instance.trace = inspect.trace()
            timeit = time.time()
            instance.time = time.localtime(timeit)
            e = None
            try:
                result = func(*args,**kwargs)
                if broadcast: instance.broadcast()
            except:
                result = None
                exc_type, exc_obj, exc_tb = e = sys.exc_info() # http://stackoverflow.com/a/1278740/5288758
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            instance.exectime = time.time()-timeit
            instance.outputs = result
            instance.error = e
            if report: instance.report()
            instance.throwError()
            instance.renew()
            return result
        return wrapper
    return decorator

class Logger(object):
    """
    Logger for decorated functions. Holds important information of an instanced object and
    can be used with @trace decorator for traceback purposes.

    :param func: object reference.
    :param funcname: object name.
    :param inputs: inputs pass to the object.
    :param outputs: outputs given by the object execution.
    :param time: initial time of execution.
    :param exectime: time of execution in seconds.
    :param writer: writer function where messages are passed.
    :param eventHandle: event function where object is
            passed when Logger.broadcast() is called.
    :param msg_report: message format to use in reports.
    :param msg_no_executed: massage format to pass to writer when object
            has not been executed and Logger.report() is called.
    :param msg_executed: massage format to use when object is
            executed and Logger.broadcast() is called.
    """
    eventHandle = None
    file = sys.stdout
    _msg_report = "\nName: {self.funcname}\n" \
           "Type: {self.Type_}\n" \
           "Time of execution: {self.Time_}\n" \
           "Execution time: {self.exectime} secs\n" \
           "Inputs: {self.inputs}\n" \
           "Outputs: {self.outputs}\n"
    _msg_no_executed = "No instance {self.funcname} has been executed"
    _msg_executed = "{self.funcname} has been executed..."
    def __init__(self, **kwargs):
        """
        """
        self.renew() # initialize info variables.
        self.__dict__.update(kwargs)

    def broadcast(self):
        """
        pass a notification message on object execution to the writer
        """
        self.writer(self.broadcast, self._msg_executed.format(self=self))
        if self.eventHandle: self.eventHandle(self)

    def report(self):
        """
        pass a report of the last executed object to the writer
        """
        if self.func is not None:
            self.writer(self.report, self._msg_report.format(self=self))
        else:
            self.writer(self.report, self._msg_no_executed.format(self=self))

    def throwError(self):
        """
        throw caught error
        :return:
        """
        if self.error:
            self.writer(self.throwError,self.error)

    def writer(self, sender, *arg):
        FILE = self.file
        HEADER = (sender.__self__, sender.__func__, self.func)
        if sender.__func__.__name__== "throwError":
            if self.throwAtError:
                raise arg[0]
            else:
                print(HEADER,">>> ", file=FILE)
                exc_type, exc_value, exc_traceback = arg[0]
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=FILE)
        else:
            print(HEADER,">>> ",*arg, file=FILE)
    @property
    def tracer(self):
        return tracer(self)
    @property
    def Type_(self):
        """
        returns type name (str)
        """
        return type(self.func).__name__
    @property
    def Time_(self):
        """
        returns formated time (str)
        """
        return time.asctime(self.time)
    def renew(self):
        """
        renew Instance
        """
        self.func = None
        self.funcname = ""
        self.inputs = {}
        self.outputs = ()
        self.stack = []
        self.trace = []
        self.error = None
        self.time = 0
        self.exectime = 0

### PYCALLGRAPH modification

# TODO: enhance decorator API to facilitate saving and more ergonomic feel
# TODO: let the user add the path to graphviz binaries from API

class Syncronous(SyncronousTracer):

    def start(self):
        self.old_trace = sys.gettrace()
        sys.settrace(self.tracer)

    def stop(self):
        sys.settrace(self.old_trace)

class Asyncronous(Syncronous):

    def start(self):
        self.processor.start()
        Syncronous.start(self)

    def tracer(self, frame, event, arg):
        self.processor.queue(frame, event, arg, self.memory())
        return self.tracer

    def done(self):
        self.processor.done()
        self.processor.join()

class GraphTraceOutput(GraphvizOutput):
    def __init__(self, source = None, saveflag = True, label = "", **kwargs):
        self.source = source
        self.saveflag = saveflag
        self.label = label
        GraphvizOutput.__init__(self, **kwargs)
        self.graph_attributes['graph']['label'] = self.label

    def done(self):
        source = self.generate()
        self.debug(source)
        self.source = source
        self.save()

    def saveSource(self, file, source = None):

        if not source:
            source = self.source

        cmd = '{} -T{} -o{} {}'.format(
            self.tool, self.output_type, self.output_file, file
        )

        with open(file, 'w') as f:
            f.write("// run as: {}\n\n".format(cmd)+source)

        self.cmd = cmd
        return cmd

    def save(self,file=None, source = None):
        if file: # save custom file
            istemp = False
            cmd = self.saveSource(file, source)
        else: # if no file a temporal file is made
            istemp = True
            if not source:
                source = self.source
            fd, file = tempfile.mkstemp()

            cmd = '{} -T{} -o{} {}'.format(
                self.tool, self.output_type, self.output_file, file
            )

            with os.fdopen(fd, 'w') as f:
                f.write("// run as: {}\n\n".format(cmd)+source)

            self.cmd = cmd

        self.verbose('Executing: {}'.format(cmd))
        try:
            ret = os.system(cmd)
            if ret:
                raise Exception(
                    'The command "%(cmd)s" failed with error '
                    'code %(ret)i.' % locals())
        finally:
            if istemp:
                os.remove(file) # remove file if it is temporal

        self.verbose('Generated {} with {} nodes.'.format(
            self.output_file, len(self.processor.func_count),
        ))

class GraphTrace(pycallgraph.PyCallGraph):

    def __enter__(self): # modified function
        # review for trace problems at \Python27\Lib\site-packages\pycallgraph\tracer.py
        self.start()
        return self

    @property
    def source(self): # modified function
        return [output.source for output in self.output]

    def saveSource(self,file): # added function
        name = dr.getData(file)
        outputs = self.output
        if len(outputs)>1: # save enumerated outputs
            for i,output in enumerate(self.output):
                tempname = name[:-1]+["_",str(i+1)]+name[-1:]
                output.saveSource("".join(tempname))
        else: # save single output
            outputs[0].saveSource("".join(name))

    def get_tracer_class(self): # modified function
        if self.config.threaded:
            return Asyncronous
        else:
            return Syncronous