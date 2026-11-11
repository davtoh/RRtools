# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from ..Qt import QtCore
from .. import FileDialog
from .. import configfile as configfile
from ..debug import printExc
from .library import LIBRARY
from .Node import Node
from .execution import FlowchartExecutionMixin
from .ui_widgets import FlowchartCtrlWidget

__all__ = ['Flowchart', 'FlowchartNode', 'strDict']


def strDict(d):
    return dict([(str(k), v) for k, v in list(d.items())])


class Flowchart(FlowchartExecutionMixin, Node):
    sigFileLoaded = QtCore.Signal(object)
    sigFileSaved = QtCore.Signal(object)

    sigChartLoaded = QtCore.Signal()
    sigStateChanged = QtCore.Signal()  # called when output is expected to have changed
    sigChartChanged = QtCore.Signal(object, object, object)  # called when nodes are added, removed, or renamed. (self, action, node)

    def __init__(self, terminals=None, name=None, filePath=None, library=None):
        self.library = library or LIBRARY
        if name is None:
            name = "Flowchart"
        if terminals is None:
            terminals = {}
        self.filePath = filePath
        Node.__init__(self, name, allowAddInput=True, allowAddOutput=True)  ## create node without terminals; we'll add these later

        self.inputWasSet = False  ## flag allows detection of changes in the absence of input change.
        self._nodes = {}
        self.nextZVal = 10
        self._widget = None
        self._scene = None
        self.processing = False  ## flag that prevents recursive node updates

        self.widget()

        self.inputNode = Node('Input', allowRemove=False, allowAddOutput=True)
        self.outputNode = Node('Output', allowRemove=False, allowAddInput=True)
        self.addNode(self.inputNode, 'Input', [-150, 0])
        self.addNode(self.outputNode, 'Output', [300, 0])

        self.outputNode.sigOutputChanged.connect(self.outputChanged)
        self.outputNode.sigTerminalRenamed.connect(self.internalTerminalRenamed)
        self.inputNode.sigTerminalRenamed.connect(self.internalTerminalRenamed)
        self.outputNode.sigTerminalRemoved.connect(self.internalTerminalRemoved)
        self.inputNode.sigTerminalRemoved.connect(self.internalTerminalRemoved)
        self.outputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        self.inputNode.sigTerminalAdded.connect(self.internalTerminalAdded)

        self.viewBox.autoRange(padding=0.04)

        for name, opts in list(terminals.items()):
            self.addTerminal(name, **opts)

    def setLibrary(self, lib):
        self.library = lib
        self.widget().chartWidget.buildMenu()

    def setInput(self, **args):
        """Set the input values of the flowchart. This will automatically propagate
        the new values throughout the flowchart, (possibly) causing the output to change.
        """
        self.inputWasSet = True
        self.inputNode.setOutput(**args)

    def outputChanged(self):
        ## called when output of internal node has changed
        vals = self.outputNode.inputValues()
        self.widget().outputChanged(vals)
        self.setOutput(**vals)

    def output(self):
        """Return a dict of the values on the Flowchart's output terminals."""
        return self.outputNode.inputValues()

    def nodes(self):
        return self._nodes

    def addTerminal(self, name, **opts):
        term = Node.addTerminal(self, name, **opts)
        name = term.name()
        if opts['io'] == 'in':  ## inputs to the flowchart become outputs on the input node
            opts['io'] = 'out'
            opts['multi'] = False
            self.inputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
            try:
                term2 = self.inputNode.addTerminal(name, **opts)
            finally:
                self.inputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        else:
            opts['io'] = 'in'
            self.outputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
            try:
                term2 = self.outputNode.addTerminal(name, **opts)
            finally:
                self.outputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        return term

    def removeTerminal(self, name):
        term = self[name]
        inTerm = self.internalTerminal(term)
        Node.removeTerminal(self, name)
        inTerm.node().removeTerminal(inTerm.name())

    def internalTerminalRenamed(self, term, oldName):
        self[oldName].rename(term.name())

    def internalTerminalAdded(self, node, term):
        if term._io == 'in':
            io = 'out'
        else:
            io = 'in'
        Node.addTerminal(self, term.name(), io=io, renamable=term.isRenamable(), removable=term.isRemovable(), multiable=term.isMultiable())

    def internalTerminalRemoved(self, node, term):
        try:
            Node.removeTerminal(self, term.name())
        except KeyError:
            pass

    def terminalRenamed(self, term, oldName):
        newName = term.name()
        Node.terminalRenamed(self, self[oldName], oldName)
        for n in [self.inputNode, self.outputNode]:
            if oldName in n.terminals:
                n[oldName].rename(newName)

    def createNode(self, nodeType, name=None, pos=None):
        if name is None:
            n = 0
            while True:
                name = "%s.%d" % (nodeType, n)
                if name not in self._nodes:
                    break
                n += 1

        node = self.library.getNodeType(nodeType)(name)
        self.addNode(node, name, pos)
        return node

    def addNode(self, node, name, pos=None):
        if pos is None:
            pos = [0, 0]
        if type(pos) in [QtCore.QPoint, QtCore.QPointF]:
            pos = [pos.x(), pos.y()]
        item = node.graphicsItem()
        item.setZValue(self.nextZVal * 2)
        self.nextZVal += 1
        self.viewBox.addItem(item)
        item.moveBy(*pos)
        self._nodes[name] = node
        self.widget().addNode(node)
        node.sigClosed.connect(self.nodeClosed)
        node.sigRenamed.connect(self.nodeRenamed)
        node.sigOutputChanged.connect(self.nodeOutputChanged)
        self.sigChartChanged.emit(self, 'add', node)

    def removeNode(self, node):
        node.close()

    def nodeClosed(self, node):
        del self._nodes[node.name()]
        self.widget().removeNode(node)
        for signal in ['sigClosed', 'sigRenamed', 'sigOutputChanged']:
            try:
                getattr(node, signal).disconnect(self.nodeClosed)
            except (TypeError, RuntimeError):
                pass
        self.sigChartChanged.emit(self, 'remove', node)

    def nodeRenamed(self, node, oldName):
        del self._nodes[oldName]
        self._nodes[node.name()] = node
        self.widget().nodeRenamed(node, oldName)
        self.sigChartChanged.emit(self, 'rename', node)

    def arrangeNodes(self):
        pass

    def chartGraphicsItem(self):
        """Return the graphicsItem which displays the internals of this flowchart.
        (graphicsItem() still returns the external-view item)"""
        return self.viewBox

    def widget(self):
        if self._widget is None:
            self._widget = FlowchartCtrlWidget(self)
            self.scene = self._widget.scene()
            self.viewBox = self._widget.viewBox()
        return self._widget

    def saveState(self):
        state = Node.saveState(self)
        state['nodes'] = []
        state['connects'] = []

        for name, node in list(self._nodes.items()):
            cls = type(node)
            if hasattr(cls, 'nodeName'):
                clsName = cls.nodeName
                pos = node.graphicsItem().pos()
                ns = {'class': clsName, 'name': name, 'pos': (pos.x(), pos.y()), 'state': node.saveState()}
                state['nodes'].append(ns)

        conn = self.listConnections()
        for a, b in conn:
            state['connects'].append((a.node().name(), a.name(), b.node().name(), b.name()))

        state['inputNode'] = self.inputNode.saveState()
        state['outputNode'] = self.outputNode.saveState()

        return state

    def restoreState(self, state, clear=False):
        self.blockSignals(True)
        try:
            if clear:
                self.clear()
            Node.restoreState(self, state)
            nodes = state['nodes']
            nodes.sort(key=lambda a: a['pos'][0])
            for n in nodes:
                if n['name'] in self._nodes:
                    self._nodes[n['name']].restoreState(n['state'])
                    continue
                try:
                    node = self.createNode(n['class'], name=n['name'])
                    node.restoreState(n['state'])
                except:
                    printExc("Error creating node %s: (continuing anyway)" % n['name'])

            self.inputNode.restoreState(state.get('inputNode', {}))
            self.outputNode.restoreState(state.get('outputNode', {}))

            self._restoreConnections(state.get('connects', []))
        finally:
            self.blockSignals(False)

        self.sigChartLoaded.emit()
        self.outputChanged()
        self.sigStateChanged.emit()

    def _restoreConnections(self, connects):
        for n1, t1, n2, t2 in connects:
            try:
                self.connectTerminals(self._nodes[n1][t1], self._nodes[n2][t2])
            except:
                print(self._nodes[n1].terminals)
                print(self._nodes[n2].terminals)
                printExc("Error connecting terminals %s.%s - %s.%s:" % (n1, t1, n2, t2))

    def loadFile(self, fileName=None, startDir=None):
        if fileName is None:
            if startDir is None:
                startDir = self.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, "Load Flowchart..", startDir, "Flowchart (*.fc)")
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.loadFile)
            return
        fileName = str(fileName)
        state = configfile.readConfigFile(fileName)
        self.restoreState(state, clear=True)
        self.viewBox.autoRange()
        self.sigFileLoaded.emit(fileName)

    def saveFile(self, fileName=None, startDir=None, suggestedFileName='flowchart.fc'):
        if fileName is None:
            if startDir is None:
                startDir = self.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, "Save Flowchart..", startDir, "Flowchart (*.fc)")
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.saveFile)
            return
        fileName = str(fileName)
        configfile.writeConfigFile(self.saveState(), fileName)
        self.sigFileSaved.emit(fileName)

    def clear(self):
        for n in list(self._nodes.values()):
            if n is self.inputNode or n is self.outputNode:
                continue
            n.close()  ## calls self.nodeClosed(n) by signal
        self.widget().clear()

    def clearTerminals(self):
        Node.clearTerminals(self)
        self.inputNode.clearTerminals()
        self.outputNode.clearTerminals()


class FlowchartNode(Node):
    pass

