# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

try:
    from past.utils import old_div
except ImportError:
    def old_div(a, b):
        return a / b

from ..Qt import QtCore
from ..graphicsItems.GraphicsObject import GraphicsObject

__all__ = ['FlowchartGraphicsItem']


class FlowchartGraphicsItem(GraphicsObject):

    def __init__(self, chart):
        GraphicsObject.__init__(self)
        self.chart = chart  ## chart is an instance of Flowchart()
        self.updateTerminals()

    def updateTerminals(self):
        self.terminals = {}
        bounds = self.boundingRect()
        inp = self.chart.inputs()
        dy = old_div(bounds.height(), (len(inp) + 1))
        y = dy
        for n, t in list(inp.items()):
            item = t.graphicsItem()
            self.terminals[n] = item
            item.setParentItem(self)
            item.setAnchor(bounds.width(), y)
            y += dy
        out = self.chart.outputs()
        dy = old_div(bounds.height(), (len(out) + 1))
        y = dy
        for n, t in list(out.items()):
            item = t.graphicsItem()
            self.terminals[n] = item
            item.setParentItem(self)
            item.setAnchor(0, y)
            y += dy

    def boundingRect(self):
        return QtCore.QRectF()

    def paint(self, p, *args):
        pass

