import sys
from pyqtgraph.Qt import QtCore, QtGui

def openApp(): # it must be the first line in the program before implementing any plot
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    return app

app = openApp()

def execApp(app=app):
    sys.exit(app.exec_())