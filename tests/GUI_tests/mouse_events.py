import sys
from PyQt4 import QtGui, QtCore
# http://stackoverflow.com/a/25340881/5288758
# http://stackoverflow.com/a/7410780/5288758

class QCustomLabel (QtGui.QLabel):
    def __init__ (self, parent = None):
        super(QCustomLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setTextLabelPosition(0, 0)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def mouseMoveEvent (self, eventQMouseEvent):
        self.setTextLabelPosition(eventQMouseEvent.x(), eventQMouseEvent.y())
        QtGui.QWidget.mouseMoveEvent(self, eventQMouseEvent)

    def mousePressEvent (self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            QtGui.QMessageBox.information(self, 'Position', '( %d : %d )' % (self.x, self.y))
        QtGui.QWidget.mousePressEvent(self, eventQMouseEvent)

    def setTextLabelPosition (self, x, y):
        self.x, self.y = x, y
        self.setText('Please click on screen ( %d : %d )' % (self.x, self.y))

class QCustomWidget (QtGui.QWidget):
    def __init__ (self, parent = None):
        super(QCustomWidget, self).__init__(parent)
        self.setWindowOpacity(0.7)
        # Init QLabel
        self.positionQLabel = QCustomLabel(self)
        # Init QLayout
        layoutQHBoxLayout = QtGui.QHBoxLayout()
        layoutQHBoxLayout.addWidget(self.positionQLabel)
        layoutQHBoxLayout.setMargin(0)
        layoutQHBoxLayout.setSpacing(0)
        self.setLayout(layoutQHBoxLayout)
        self.showFullScreen()


myQApplication = QtGui.QApplication(sys.argv)
myQTestWidget = QCustomWidget()
myQTestWidget.show()
myQApplication.exec_()