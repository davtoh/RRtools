# -*- coding: utf-8 -*-
"""
This example demonstrates writing a custom Node subclass for use with flowcharts customized
for RRtoolFC
"""
import pylab
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Node
from pyqtgraph.flowchart.library.common import CtrlNode
from RRtoolFC.lib.rrtoolsNodes import RRtoolFlowchart
from RRtoolbox.lib.image import loadcv

## Create an empty flowchart with a single input and output
fc = RRtoolFlowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}
})

## At this point, we need some custom Node classes since those provided in the library
## are not sufficient. Each node will define a set of input/output terminals, a
## processing function, and optionally a control widget (to be displayed in the
## flowchart control panel)

class Plot(Node):
    """Node that displays image data in Matplotlib"""
    nodeName = 'plot'

    def __init__(self, name):
        self.ax = None
        ## Initialize node with only a single input terminal
        Node.__init__(self, name, terminals={'data': {'io':'in'}},allowAddInput=False,
                      allowAddOutput=False,allowRemove=False)

    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        if display:
            if self.ax is None:
                self.fig = pylab.figure(self._name)
                self.ax = self.fig.gca()
            if data is not None:
                self.ax.imshow(data)
                self.fig.show()


class UnsharpMaskNode(CtrlNode):
    """Return the input data passed through pg.gaussianFilter."""
    nodeName = "UnsharpMask"
    uiTemplate = [
        ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'range': [0.0, None]}),
        ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5,
                              'minStep': 0.01, 'range': [0.0, None]}),
    ]
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'dataIn': dict(io='in'),    # each terminal needs at least a name and
            'dataOut': dict(io='out'),  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..

        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, dataIn, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        sigma = self.ctrls['sigma'].value()
        strength = self.ctrls['strength'].value()
        output = dataIn - (strength * pg.gaussianFilter(dataIn, (sigma,sigma)))
        return {'dataOut': output}


## To make our custom node classes available in the flowchart context menu,
## we can either register them with the default node library or make a
## new library.


## Method 1: Register to global default library:
#fclib.registerNodeType(ImageViewNode, [('Display',)])
#fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

## Method 2: If we want to make our custom node available only to this flowchart,
## then instead of registering the node type globally, we can create a new
## NodeLibrary:
library = fclib.LIBRARY.copy() # start with the default node set
library.addNodeType(Plot, [('Display',)])
# Add the unsharp mask node to two locations in the menu to demonstrate
# that we can create arbitrary menu structures
library.addNodeType(UnsharpMaskNode, [('Image',),('Submenu_test','submenu2','submenu3')])
fc.setLibrary(library)


## Now we will programmatically add nodes to define the function of the flowchart.
## Normally, the user will do this manually or by loading a pre-generated
## flowchart file.

fNode = fc.createNode('UnsharpMask', pos=(0, 0))
v1Node = fc.createNode('plot', pos=(0, -150))
v2Node = fc.createNode('plot', pos=(150, -150))
fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
fc.connectTerminals(fc['dataIn'], v1Node['data'])
fc.connectTerminals(fNode['dataOut'], v2Node['data'])
fc.connectTerminals(fNode['dataOut'], fc['dataOut'])
## Start Qt event loop unless running in interactive mode or using pyside.

if __name__ == "__main__": # test to convert to node
    import RRtoolFC.lib.RRtoolnamespace as ss
    ## Set the raw data as the input value to the flowchart
    fc.setInput(dataIn=loadcv(r"../../tests/im1_1.jpg",flags=0,shape=(300,300)))
    #fc.setInput(nameOfInputTerminal=newValue) # set default values
    output = fc.output() # get any time
    a = ss.plot(output["dataOut"])
    a.show() # activate mainApp
    #fc.process(dataIn=)