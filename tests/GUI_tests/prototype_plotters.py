# http://stackoverflow.com/a/12465861/5288758

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyQt4.QtCore import Qt
from PyQt4 import QtGui

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

class tools(NavigationToolbar,QtGui.QToolBar):
    # TODO
    pass

class MatplotlibWidget(QtGui.QWidget):
    def __init__(self, parent=None, figure = None):
        super(MatplotlibWidget, self).__init__(parent)
        # TODO: add threading to canvas, it blocks the GUI
        # a figure instance to plot on
        self.image = figure
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
        self.button_plot.clicked.connect(self.imshow)

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
        #self.imshow(self.image)

    def imshow(self, image=None):
        ''' plot some random stuff '''
        # random data
        #data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        ax.hold(False)

        # plot data
        #ax.plot(data, '*-')
        ax.imshow(self.image)
        #ax.title = "im1_3.png"
        #ax.xticks([]),ax.yticks([])

        # refresh canvas
        self.canvas.draw()

wins = [0]

def fastplt(image, cmap = None, title = "visualazor", win = None, block = False):
    """
        fast pyplot
    :param image: image to show
    :param cmap: "gray" or None
    :param title: title of subplot
    :param win: title of window
    :param block: if True it wait for window close, else it detaches
    :param daemon: if True window closes if main thread ends, else windows must be closed to main thread to end
    :return: plt
    """
    def myplot():
        f = plt.figure()
        # Normally this will always be "Figure 1" since it's the first
        # figure created by this process. So do something about it.
        if win: f.canvas.set_window_title(win)
        else:f.canvas.set_window_title("Figure {}".format(wins[0]))
        plt.imshow(image,cmap)
        if title: plt.title(title)
        plt.xticks([]), plt.yticks([])
        #plt.colorbar()
        plt.show()
        return plt
    wins[0]+=1
    if block:
        return myplot()
    else:
        return plt

if __name__ == '__main__':
    from RRtoolFC.lib.app import execApp
    import numpy as np
    fname = r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/im1_1.jpg"    # This can be any photo image file
    photo=np.array(mpimg.imread(fname))
    figure = plt.figure("shitty one")
    figure.suptitle("TITLE")
    ax = figure.add_subplot(111)
    # discards the old graph
    ax.imshow(photo)
    ax.hold(False)

    main = MatplotlibWidget(figure= photo)
    main.show()
    execApp()