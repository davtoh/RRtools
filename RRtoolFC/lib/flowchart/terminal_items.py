# -*- coding: utf-8 -*-
from __future__ import division
try:
    from past.utils import old_div
except ImportError:
    def old_div(a, b):
        return a / b
from ..Qt import QtCore, QtGui
from ..graphicsItems.GraphicsObject import GraphicsObject
from .. import functions as fn
from ..Point import Point

__all__ = ['TerminalGraphicsItem', 'ConnectionItem']


#class TerminalGraphicsItem(QtGui.QGraphicsItem):
class TerminalGraphicsItem(GraphicsObject):
    
    def __init__(self, term, parent=None):
        self.term = term
        #QtGui.QGraphicsItem.__init__(self, parent)
        GraphicsObject.__init__(self, parent)
        self.brush = fn.mkBrush(0,0,0)
        self.box = QtGui.QGraphicsRectItem(0, 0, 10, 10, self)
        self.label = QtGui.QGraphicsTextItem(self.term.name(), self)
        self.label.scale(0.7, 0.7)
        #self.setAcceptHoverEvents(True)
        self.newConnection = None
        self.setFiltersChildEvents(True)  ## to pick up mouse events on the rectitem
        if self.term.isRenamable():
            self.label.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
            self.label.focusOutEvent = self.labelFocusOut
            self.label.keyPressEvent = self.labelKeyPress
        self.setZValue(1)
        self.menu = None
            

    def labelFocusOut(self, ev):
        QtGui.QGraphicsTextItem.focusOutEvent(self.label, ev)
        self.labelChanged()
        
    def labelKeyPress(self, ev):
        if ev.key() == QtCore.Qt.Key_Enter or ev.key() == QtCore.Qt.Key_Return:
            self.labelChanged()
        else:
            QtGui.QGraphicsTextItem.keyPressEvent(self.label, ev)
        
    def labelChanged(self):
        newName = str(self.label.toPlainText())
        if newName != self.term.name():
            self.term.rename(newName)

    def termRenamed(self, name):
        self.label.setPlainText(name)

    def setBrush(self, brush):
        self.brush = brush
        self.box.setBrush(brush)

    def disconnect(self, target):
        self.term.disconnectFrom(target.term)

    def boundingRect(self):
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        return br | lr
        
    def paint(self, p, *args):
        pass
        
    def setAnchor(self, x, y):
        pos = QtCore.QPointF(x, y)
        self.anchorPos = pos
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        
        
        if self.term.isInput():
            self.box.setPos(pos.x(), pos.y()-old_div(br.height(),2.))
            self.label.setPos(pos.x() + br.width(), pos.y() - old_div(lr.height(),2.))
        else:
            self.box.setPos(pos.x()-br.width(), pos.y()-old_div(br.height(),2.))
            self.label.setPos(pos.x()-br.width()-lr.width(), pos.y()-old_div(lr.height(),2.))
        self.updateConnections()
        
    def updateConnections(self):
        for t, c in list(self.term.connections().items()):
            c.updateLine()
            
    def mousePressEvent(self, ev):
        #ev.accept()
        ev.ignore() ## necessary to allow click/drag events to process correctly

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.label.setFocus(QtCore.Qt.MouseFocusReason)
        elif ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)
            
    def raiseContextMenu(self, ev):
        ## only raise menu if this terminal is removable
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        
    def getMenu(self):
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self.menu.setTitle("Terminal")
            remAct = QtGui.QAction("Remove terminal", self.menu)
            remAct.triggered.connect(self.removeSelf)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
            if not self.term.isRemovable():
                remAct.setEnabled(False)
            multiAct = QtGui.QAction("Multi-value", self.menu)
            multiAct.setCheckable(True)
            multiAct.setChecked(self.term.isMultiValue())
            multiAct.setEnabled(self.term.isMultiable())
            
            multiAct.triggered.connect(self.toggleMulti)
            self.menu.addAction(multiAct)
            self.menu.multiAct = multiAct
            if self.term.isMultiable():
                multiAct.setEnabled = False
        return self.menu

    def toggleMulti(self):
        multi = self.menu.multiAct.isChecked()
        self.term.setMultiValue(multi)
    
    def removeSelf(self):
        self.term.node().removeTerminal(self.term)
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        ev.accept()
        if ev.isStart():
            if self.newConnection is None:
                self.newConnection = ConnectionItem(self)
                #self.scene().addItem(self.newConnection)
                self.getViewBox().addItem(self.newConnection)
                #self.newConnection.setParentItem(self.parent().parent())

            self.newConnection.setTarget(self.mapToView(ev.pos()))
        elif ev.isFinish():
            if self.newConnection is not None:
                items = self.scene().items(ev.scenePos())
                gotTarget = False
                for i in items:
                    if isinstance(i, TerminalGraphicsItem):
                        self.newConnection.setTarget(i)
                        try:
                            self.term.connectTo(i.term, self.newConnection)
                            gotTarget = True
                        except:
                            self.scene().removeItem(self.newConnection)
                            self.newConnection = None
                            raise
                        break
                
                if not gotTarget:
                    #print "remove unused connection"
                    #self.scene().removeItem(self.newConnection)
                    self.newConnection.close()
                self.newConnection = None
        else:
            if self.newConnection is not None:
                self.newConnection.setTarget(self.mapToView(ev.pos()))
        
    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptDrags(QtCore.Qt.LeftButton):
            ev.acceptClicks(QtCore.Qt.LeftButton) ## we don't use the click, but we also don't want anyone else to use it.
            ev.acceptClicks(QtCore.Qt.RightButton)
            self.box.setBrush(fn.mkBrush('w'))
        else:
            self.box.setBrush(self.brush)
        self.update()
        
    #def hoverEnterEvent(self, ev):
        #self.hover = True
        
    #def hoverLeaveEvent(self, ev):
        #self.hover = False
        
    def connectPoint(self):
        ## return the connect position of this terminal in view coords
        return self.mapToView(self.mapFromItem(self.box, self.box.boundingRect().center()))

    def nodeMoved(self):
        for t, item in list(self.term.connections().items()):
            item.updateLine()


#class ConnectionItem(QtGui.QGraphicsItem):
class ConnectionItem(GraphicsObject):
    
    def __init__(self, source, target=None):
        #QtGui.QGraphicsItem.__init__(self)
        GraphicsObject.__init__(self)
        self.setFlags(
            self.ItemIsSelectable | 
            self.ItemIsFocusable
        )
        self.source = source
        self.target = target
        self.length = 0
        self.hovered = False
        self.path = None
        self.shapePath = None
        self.style = {
            'shape': 'line',
            'color': (100, 100, 250),
            'width': 1.0,
            'hoverColor': (150, 150, 250),
            'hoverWidth': 1.0,
            'selectedColor': (200, 200, 0),
            'selectedWidth': 3.0,
            }
        #self.line = QtGui.QGraphicsLineItem(self)
        self.source.getViewBox().addItem(self)
        self.updateLine()
        self.setZValue(0)
        
    def close(self):
        if self.scene() is not None:
            #self.scene().removeItem(self.line)
            self.scene().removeItem(self)
        
    def setTarget(self, target):
        self.target = target
        self.updateLine()
    
    def setStyle(self, **kwds):
        self.style.update(kwds)
        if 'shape' in kwds:
            self.updateLine()
        else:
            self.update()
    
    def updateLine(self):
        start = Point(self.source.connectPoint())
        if isinstance(self.target, TerminalGraphicsItem):
            stop = Point(self.target.connectPoint())
        elif isinstance(self.target, QtCore.QPointF):
            stop = Point(self.target)
        else:
            return
        self.prepareGeometryChange()
        
        self.path = self.generatePath(start, stop)
        self.shapePath = None
        self.update()
        
    def generatePath(self, start, stop):
        path = QtGui.QPainterPath()
        path.moveTo(start)
        if self.style['shape'] == 'line':
            path.lineTo(stop)
        elif self.style['shape'] == 'cubic':
            path.cubicTo(Point(stop.x(), start.y()), Point(start.x(), stop.y()), Point(stop.x(), stop.y()))
        else:
            raise Exception('Invalid shape "%s"; options are "line" or "cubic"' % self.style['shape'])
        return path

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace:
            #if isinstance(self.target, TerminalGraphicsItem):
            self.source.disconnect(self.target)
            ev.accept()
        else:
            ev.ignore()
    
    def mousePressEvent(self, ev):
        ev.ignore()
        
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            sel = self.isSelected()
            self.setSelected(True)
            if not sel and self.isSelected():
                self.update()
                
    def hoverEvent(self, ev):
        if (not ev.isExit()) and ev.acceptClicks(QtCore.Qt.LeftButton):
            self.hovered = True
        else:
            self.hovered = False
        self.update()
            
            
    def boundingRect(self):
        return self.shape().boundingRect()
        ##return self.line.boundingRect()
        #px = self.pixelWidth()
        #return QtCore.QRectF(-5*px, 0, 10*px, self.length)
    def viewRangeChanged(self):
        self.shapePath = None
        self.prepareGeometryChange()
        
    def shape(self):
        if self.shapePath is None:
            if self.path is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            px = self.pixelWidth()
            stroker.setWidth(px*8)
            self.shapePath = stroker.createStroke(self.path)
        return self.shapePath
        
    def paint(self, p, *args):
        if self.isSelected():
            p.setPen(fn.mkPen(self.style['selectedColor'], width=self.style['selectedWidth']))
        else:
            if self.hovered:
                p.setPen(fn.mkPen(self.style['hoverColor'], width=self.style['hoverWidth']))
            else:
                p.setPen(fn.mkPen(self.style['color'], width=self.style['width']))
                
        #p.drawLine(0, 0, 0, self.length)
        
        p.drawPath(self.path)

