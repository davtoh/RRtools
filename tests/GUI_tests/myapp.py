import sys
from PyQt4 import QtGui

class Example(QtGui.QMainWindow):

    def __init__(self, image = None):
        super(Example, self).__init__()

        self.printer = QtGui.QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setWidgetResizable(True)
        self.setCentralWidget(self.scrollArea)

        self.setWindowTitle("Image Viewer")
        self.resize(500, 400)
        if image:
            self.image = image
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
            self.imageLabel.adjustSize()
        self.initUI()


    def initUI(self):

        exitAction = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)
        self.show()


def main():
    #TODO integrate with main
    import cv2
    from RRtoolbox import np2qi
    def loadcv(pth,mode=-1,shape=None):
        im = cv2.imread(pth,mode)
        if shape:
            im = cv2.resize(im,shape)
        return np2qi(im)
    fname = r"/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtoolbox/tests/im1_3.png"
    img = loadcv(fname)

    app = QtGui.QApplication(sys.argv)
    ex = Example(img)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()