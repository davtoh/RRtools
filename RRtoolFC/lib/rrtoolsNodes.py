from pyqtgraph.flowchart.library import NodeLibrary, LIBRARY
from pyqtgraph.flowchart import Flowchart, FlowchartCtrlWidget, FlowchartWidget, Node
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.flowchart.library.common import CtrlNode
import numpy as np
# LOAD funtions to convert to NODE
#from RRtoolbox.RRtools.restoration import init_feature, explore_match, ASIFT, MATCH, invertH, superpose, bilateralFilter
#from RRtoolbox.lib.image import imloader, loadcv
#from RRtoolbox.lib import config as cf

def isNodeClass(cls):
    try:
        if not issubclass(cls, Node):
            return False
    except:
        return False
    return hasattr(cls, 'nodeName')

def isNode(node):
    return isinstance(node,Node)

def isFlowCharClass(cls):
    try:
        if not issubclass(cls, Flowchart):
            return False
    except:
        return False
    return hasattr(cls, 'nodeName')

def isFlowChar(fc):
    return isinstance(fc,Flowchart)

class RRtoolLibrary(NodeLibrary):

    def reload(self):
        """
        Reload Node classes in this library.
        """
        raise NotImplementedError()


class RRtoolFlowchart(Flowchart):

    def widget(self):
        if self._widget is None:
            self._widget = RRtoolCtrlWidget(self)
            self.scene = self._widget.scene()
            self.viewBox = self._widget.viewBox()
            #self._scene = QtGui.QGraphicsScene()
            #self._widget.setScene(self._scene)
            #self.scene.addItem(self.chartGraphicsItem())

            #ci = self.chartGraphicsItem()
            #self.viewBox.addItem(ci)
            #self.viewBox.autoRange()
        return self._widget


class RRtoolCtrlWidget(FlowchartCtrlWidget):
    pass

class RRtoolFCWidget(FlowchartWidget):
    pass

class RRtoolConsole(ConsoleWidget):
    pass

class RRtoolCtrNode(CtrlNode):
    pass

def loadFromModules(modules, LIBRARY = RRtoolLibrary()):
    for mod in modules:
        nodes = [getattr(mod, name) for name in dir(mod) if isNodeClass(getattr(mod, name))]
        for node in nodes:
            LIBRARY.addNodeType(node, [(mod.__name__.split('.')[-1],)])