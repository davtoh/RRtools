# http://stackoverflow.com/questions/15247075/how-can-i-dynamically-create-derived-classes-from-a-base-class
from pyqtgraph.flowchart import Node
from pyqtgraph.flowchart.library.common import CtrlNode
from RRtoolbox.lib.inspector import funcData

# TODO: see info under mylibs to come up with a way to convert functions to flowchar nodes.

class NodeGenerator:
    """
    Generate Nodes.

    :param nodeName: name of the node class. if None, it generates name on the fly from wrapped function
    :param terminals: generic inputs and outputs. if None, it generates
            terminals on the fly from wrapped function
    :param uiTemplate: template to use in UI controls. if None, it uses Node class
    :param nodeClass: class to use to generate the node. if None, it uses a convenient
            Node class on the fly from wrapped function
    """
    def __init__(self,nodeName=None,terminals=None,uiTemplate=None,nodeClass=None,classTemplate="{}Node",selfAs=None,addfuncs=None):
        """ define a customized NodeGenerator
        :return:
        """
        # these are totally needed for the node creation
        self.nodeName = nodeName # variable to use in the Node class to know it is a Node and show its name
        # generateUi in flowchar.library.common.py
        # currently it supports: 'intSpin', 'doubleSpin', 'spin', 'checkLoaded', 'combo', 'color', 'tip'
        self.nodeClass = nodeClass # either way all should be derived from Node class
        self.classTemplate = classTemplate
        # these are use for the node creation but not that important
        self.uiTemplate = uiTemplate
        self.terminals = terminals # see Terminal under flowchart/Terminal.py
        self.selfAs = selfAs
        self.addfuncs = addfuncs

    def config(self,func):
        # kwargs are temporal parameters tu use instead of defaults
        # func is a function that wants to be converted on the fly
        if self.uiTemplate:
            nodeClass = self.nodeClass or CtrlNode
            if not issubclass(nodeClass,CtrlNode):
                raise TypeError("nodeClass is not subclass of CtrlNode")
        else:
            nodeClass = self.nodeClass or Node
            if not issubclass(nodeClass,Node):
                raise TypeError("nodeClass is not subclass of Node")

        data = funcData(func)
        keywords, varargs = data["keywords"],data["varargs"]
        if False and (keywords or varargs):
            where = "and".join([i for i in (keywords,varargs) if i])
            raise Exception("generic function has {}. It must have explicit arguments".format(where))
        if False and varargs:
            raise Exception("NodeGenerator does not support positional arguments like '{}', try keywords arguments".format(varargs))
        args = data["args"]
        useDisplay = "display" in args
        nodeName = self.nodeName or data["name"]
        classname = self.classTemplate.format(nodeName)
        if not classname: raise Exception("classTemplate did not generate a classname")
        doc = data["doc"]
        templates = []
        if self.uiTemplate:
            for tmpl in self.uiTemplate:
                replace = tmpl[0]
                for i, arg in enumerate(args[:]):
                    if replace ==arg:
                        templates.append(replace) # register in process_handles
                        del args[i] # it won't apear in terminals
                    elif keywords: templates.append(replace) # register if support for more variables
        if self.terminals:
            terminals = self.terminals # replace by user terminals
        else:
            terminals = {arg:{"io":"in"} for arg in args} # only inputs registered
        classData = funcData(nodeClass.__init__)
        # know if nodeClass supports these parameters
        classArgs = classData["args"]
        if classData["keywords"]:
            useAllowAddInput = useAllowAddOutput = useAllowRemove = True
        else:
            useAllowAddInput = "allowAddInput" in classArgs
            useAllowAddOutput = "allowAddOutput" in classArgs
            useAllowRemove = "allowRemove" in classArgs
        useTerminals = "terminals" in classArgs
        # handle function should be
        # def hf(self,**kwargs):
        #   pass # process something here

        # initialize handles
        _init_handles = [] # it always must be
        # now begin to register
        # know if processing function supports these parameters
        allowAddInput = bool(data["keywords"])
        allowAddOutput = False
        allowRemove = allowAddInput or allowAddOutput
        if useAllowAddInput:
            def handle_allowAddInput(self,kwargs):
                kwargs["allowAddInput"] = allowAddInput
            _init_handles.append(handle_allowAddInput)
        if useAllowAddOutput:
            def handle_allowAddOutput(self,kwargs):
                kwargs["allowAddOutput"] = allowAddOutput
            _init_handles.append(handle_allowAddOutput)
        if useAllowRemove:
            def handle_allowRemove(self,kwargs):
                kwargs["allowRemove"] = allowRemove
            _init_handles.append(handle_allowRemove)
        if useTerminals:
            def handle_terminals(self,kwargs):
                kwargs["terminals"] = terminals
            _init_handles.append(handle_terminals)

        _process_handles = []
        ##
        if not useDisplay:
            def handle_display(self,kwargs):
                del kwargs["display"]
            _process_handles.append(handle_display)
        if self.selfAs:
            tempself = self.selfAs
            def handle_addself(self,kwargs):
                kwargs[tempself] = self
            _process_handles.append(handle_addself)


        d = {}
        for tmpl in templates:
            exec "def handle_{0}(self,kwargs): kwargs[{0}] = self.ctrls[{0}]".format(tmpl) in d
            _process_handles.append(d["handle_{}".format(tmpl)])

        def init(self,name,**kwargs):
            for h in self._init_handles:
                h(self,kwargs)
            nodeClass.__init__(self,name,**kwargs)
        def process(self, **kwargs):
            for h in self._process_handles:
                h(self,kwargs)
            return func(**kwargs)
        conf = dict(__init__=init,process=process,__doc__=doc,nodeName=nodeName,
                    _init_handles=_init_handles,_process_handles=_process_handles)
        if self.uiTemplate: conf["uiTemplate"] = self.uiTemplate
        if self.addfuncs: conf.update(self.addfuncs)
        # returns  parameters to use with type(what, bases, dict)
        return classname, (nodeClass,), conf

    def wrap(self, func):
        # inspect.getsourcelines(my_function)
        return type(*self.config(func))

    __call__ = wrap

if __name__ == "__main__":
    #new_class = type("NewClassName", (BaseClass), {"new_method": lambda self: ...})
    @NodeGenerator()
    def my_function1(param1, param2):
        "some comment here"
        print "processing something"
        output1,output2 = 10,100
        return output1,output2 # it must be clear

    @NodeGenerator()
    def my_function2(param1, param2, defparam1 = 10):
        print "processing something"
        output1,output2 = 10,100 # this shoud works
        return output1,output2 # it must be clear

    @NodeGenerator()
    def my_function3(param1, param2, defparam1 = 10, defparam2 = 20, *args, **kwargs):
        print "processing something"
        output1,output2 = 10,100
        return output1,output2 # it must be clear

    n1 = my_function1()
    n2 = my_function2
    n3 = my_function3

    print n1,n2,n3