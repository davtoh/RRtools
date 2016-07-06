# http://stackoverflow.com/a/12465861/5288758

import sys
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import random

from PyQt4.QtCore import Qt
import math


class Overlay(QtGui.QLabel):

    def __init__(self, parent = None, draggable = False):
        QtGui.QWidget.__init__(self, parent)
        self.setMouseTracking(True)
        self.draggable = draggable

    def mousePressEvent(self, event):
        if self.draggable:
            self.__mousePressPos = None
            self.__mouseMovePos = None
            if event.button() == Qt.LeftButton:
                self.__mousePressPos = event.globalPos()
                self.__mouseMovePos = event.globalPos()
        super(QtGui.QLabel, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draggable and (event.buttons() == Qt.LeftButton):
            # adjust offset from clicked point to origin of widget
            currPos = self.mapToGlobal(self.pos())
            globalPos = event.globalPos()
            diff = globalPos - self.__mouseMovePos
            newPos = self.mapFromGlobal(currPos + diff)
            self.move(newPos)

            self.__mouseMovePos = globalPos
        super(QtGui.QLabel, self).mouseMoveEvent(event)
        QtGui.QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.draggable and self.__mousePressPos is not None:
            moved = event.globalPos() - self.__mousePressPos
            if moved.manhattanLength() > 3:
                event.ignore()
                return
        super(QtGui.QLabel, self).mouseReleaseEvent(event)

class OverlayOld(QtGui.QLabel):

    def __init__(self, parent = None):

        QtGui.QWidget.__init__(self, parent)
        m = 80
        self.setGeometry(m,m,m,m)
        #self.resize(parent.width(),parent.height())
        self.setAlignment(Qt.AlignBottom)
        self.setMouseTracking(True)
    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 102)))
        painter.setPen(QtGui.QPen(Qt.NoPen))

        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5)*32, 127, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                20, 20)
        painter.end()

    def mouseMoveEvent(self, QMouseEvent):
        #hoveredWidget = QtGui.QApplication.widgetAt(QMouseEvent.globalPos())
        QtGui.QWidget.mouseMoveEvent(self, QMouseEvent)
        #self.parent().mouseMoveEvent(QMouseEvent) # for some reason coordinates do not match with the parent

    def showEvent(self, event):

        self.timer = self.startTimer(50)
        self.counter = 0

    def timerEvent(self, event):

        self.counter += 1
        self.update()
        if self.counter == 60:
            self.killTimer(self.timer)
            self.hide()


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, child):
        FigureCanvas.__init__(self,child)

    def mouseMoveEvent(self, event):
        #FigureCanvas.mouseMoveEvent(self,event)
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        self.motion_notify_event(x, y, guiEvent=event)
        #FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)
        #super(MatplotlibCanvas, self).motion_notify_event(x, y, guiEvent=event)

    def timerEvent(self, event):
        FigureCanvas.timerEvent(self,event)
        self.counter += 1
        self.update()
        if self.counter == 60:
            self.killTimer(self.timer)
            #self.hide()

    def paintEvent(self, event):
        FigureCanvas.paintEvent(self,event)
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 0)))
        if self.counter<60:
            painter.setPen(QtGui.QPen(Qt.NoPen))
            for i in range(6):
                if (self.counter / 5) % 6 == i:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5)*32, 127, 127)))
                else:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
                painter.drawEllipse(
                    self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                    self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                    20, 20)
        painter.end()

class tools(QtGui.QToolBar):
    # TODO
    def __init__(self):
        QtGui.QToolBar.__init__(self)
        pass

class MatplotlibWidget(QtGui.QWidget):
    def __init__(self, parent=None, figure = None):
        super(MatplotlibWidget, self).__init__(parent)
        # TODO: add threading to canvas, it blocks the GUI
        # a figure instance to plot on
        self.figure = plt.figure()
        self.figure.suptitle("TITLE")
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = MatplotlibCanvas(self.figure)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.canvas.setPalette(palette)
        # Just some button connected to `plot` method
        self.button_plot = QtGui.QPushButton('Plot')
        self.button_plot.clicked.connect(self.plot)

        self.button_wait = QtGui.QPushButton("Wait")
        self.button_wait2 = QtGui.QPushButton("Wait2")

        self.overlay = Overlay(self.canvas)
        self.overlay.hide()
        self.button_wait2.clicked.connect(self.overlay.show)
        self.button_wait.clicked.connect(self.showEvent)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button_plot)
        layout.addWidget(self.button_wait)
        layout.addWidget(self.button_wait2)
        self.setLayout(layout)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        #data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        ax.hold(False)

        # plot data
        #ax.plot(data, '*-')
        fname = r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtoolbox/tests/im1_3.png"    # This can be any photo image file
        photo=np.array(mpimg.imread(fname))
        ax.imshow(photo)
        #ax.title = "im1_3.png"
        #ax.xticks([]),ax.yticks([])

        # refresh canvas
        self.canvas.draw()

    def showEvent(self, event):
        self.canvas.timer = self.canvas.startTimer(50)
        self.canvas.counter = 0



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = MatplotlibWidget()
    main.show()

    sys.exit(app.exec_())