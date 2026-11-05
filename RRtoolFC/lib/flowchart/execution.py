# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from .. import functions as fn

__all__ = ['FlowchartExecutionMixin']


class FlowchartExecutionMixin(object):
    """
    Computational DAG execution, topological ordering, terminal data routing,
    and change propagation mixin for Flowchart.
    """

    def internalTerminal(self, term):
        """If the terminal belongs to the external Node, return the corresponding internal terminal"""
        if term.node() is self:
            if term.isInput():
                return self.inputNode[term.name()]
            else:
                return self.outputNode[term.name()]
        else:
            return term

    def connectTerminals(self, term1, term2):
        """Connect two terminals together within this flowchart."""
        term1 = self.internalTerminal(term1)
        term2 = self.internalTerminal(term2)
        term1.connectTo(term2)

    def listConnections(self):
        conn = set()
        for n in list(self._nodes.values()):
            terms = n.outputs()
            for n, t in list(terms.items()):
                for c in t.connections():
                    conn.add((t, c))
        return conn

    def process(self, **args):
        """
        Process data through the flowchart, returning the output.
        
        Keyword arguments must be the names of input terminals. 
        The return value is a dict with one key per output terminal.
        """
        data = {}  ## Stores terminal:value pairs
        order = self.processOrder()

        ## Record inputs given to process()
        for n, t in list(self.inputNode.outputs().items()):
            if n in args:
                data[t] = args[n]

        ret = {}

        ## process all in order
        for c, arg in order:
            if c == 'p':     ## Process a single node
                node = arg
                if node is self.inputNode:
                    continue  ## input node has already been processed.

                node_ret = self._processNode(node, data)
                if node is self.outputNode:
                    ret = node_ret
            elif c == 'd':   ## delete a terminal result
                if arg in data:
                    del data[arg]

        return ret

    def _processNode(self, node, data):
        """Process a single node and collect its output terminal values."""
        outs = list(node.outputs().values())
        ins = list(node.inputs().values())

        ## construct input value dictionary
        node_args = {}
        for inp in ins:
            inputs = inp.inputTerminals()
            if len(inputs) == 0:
                continue
            if inp.isMultiValue():  ## multi-input terminals require a dict of all inputs
                node_args[inp.name()] = dict([(i, data[i]) for i in inputs if i in data])
            else:                   ## single-inputs terminals only need the single input value available
                node_args[inp.name()] = data[inputs[0]]

        if node is self.outputNode:
            return node_args
        else:
            try:
                if node.isBypassed():
                    result = node.processBypassed(node_args)
                else:
                    result = node.process(display=False, **node_args)
            except:
                print("Error processing node %s. Args are: %s" % (str(node), str(node_args)))
                raise
            for out in outs:
                try:
                    data[out] = result[out.name()]
                except KeyError:
                    pass
            return None

    def processOrder(self):
        """Return the order of operations required to process this chart.
        The order returned should look like [('p', node1), ('p', node2), ('d', terminal1), ...] 
        where each tuple specifies either (p)rocess this node or (d)elete the result from this terminal
        """
        ## first collect list of nodes/terminals and their dependencies
        deps = {}
        tdeps = {}   ## {terminal: [nodes that depend on terminal]}
        for name, node in list(self._nodes.items()):
            deps[node] = node.dependentNodes()
            for t in list(node.outputs().values()):
                tdeps[t] = t.dependentNodes()

        order = fn.toposort(deps)
        ops = [('p', n) for n in order]

        ## determine when it is safe to delete terminal values
        dels = self._calculateDeletions(order, tdeps)
        for i, t in dels:
            ops.insert(i, ('d', t))
        return ops

    def _calculateDeletions(self, order, tdeps):
        dels = []
        for t, nodes in list(tdeps.items()):
            lastInd = 0
            lastNode = None
            for n in nodes:  ## determine which node is the last to be processed according to order
                if n is self:
                    lastInd = None
                    break
                else:
                    try:
                        ind = order.index(n)
                    except ValueError:
                        continue
                if lastNode is None or ind > lastInd:
                    lastNode = n
                    lastInd = ind
            if lastInd is not None:
                dels.append((lastInd + 1, t))
        dels.sort(key=lambda a: a[0], reverse=True)
        return dels

    def nodeOutputChanged(self, startNode):
        """Triggered when a node's output values have changed. (NOT called during process())
        Propagates new data forward through network."""
        if self.processing:
            return
        self.processing = True
        try:
            self._propagateOutputChanged(startNode)
        finally:
            self.processing = False
            if self.inputWasSet:
                self.inputWasSet = False
            else:
                self.sigStateChanged.emit()

    def _propagateOutputChanged(self, startNode):
        deps = {}
        for name, node in list(self._nodes.items()):
            deps[node] = []
            for t in list(node.outputs().values()):
                deps[node].extend(t.dependentNodes())

        ## determine order of updates 
        order = fn.toposort(deps, nodes=[startNode])
        order.reverse()

        ## keep track of terminals that have been updated
        terms = set(startNode.outputs().values())

        for node in order[1:]:
            for term in list(node.inputs().values()):
                deps = list(term.connections().keys())
                update = False
                for d in deps:
                    if d in terms:
                        update = True
                        term.inputChanged(d, process=False)
                if update:
                    node.update()
                    terms |= set(node.outputs().values())

