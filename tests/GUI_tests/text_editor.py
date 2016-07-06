# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Development\Python\PyQt tests\text editor\edytor.ui'
#
# Created: Mon Apr 29 16:12:17 2013
#      by: PyQt4 UI code generator 4.10.1
#
# WARNING! All changes made in this file will be lost!
 #https://gist.github.com/FichteFoll/5477908
from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_edytor(object):
    def setupUi(self, edytor):
        edytor.setObjectName(_fromUtf8("edytor"))
        edytor.resize(451, 461)
        self.centralwidget = QtGui.QWidget(edytor)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.button_open = QtGui.QPushButton(self.centralwidget)
        self.button_open.setGeometry(QtCore.QRect(20, 10, 131, 31))
        self.button_open.setObjectName(_fromUtf8("button_open"))
        self.button_close = QtGui.QPushButton(self.centralwidget)
        self.button_close.setGeometry(QtCore.QRect(300, 10, 131, 31))
        self.button_close.setObjectName(_fromUtf8("button_close"))
        self.editor_window = QtGui.QTextEdit(self.centralwidget)
        self.editor_window.setGeometry(QtCore.QRect(20, 50, 411, 391))
        self.editor_window.setObjectName(_fromUtf8("editor_window"))
        self.button_save = QtGui.QPushButton(self.centralwidget)
        self.button_save.setEnabled(False)
        self.button_save.setGeometry(QtCore.QRect(160, 10, 131, 31))
        self.button_save.setObjectName(_fromUtf8("button_save"))
        edytor.setCentralWidget(self.centralwidget)

        self.retranslateUi(edytor)
        QtCore.QMetaObject.connectSlotsByName(edytor)

    def retranslateUi(self, edytor):
        edytor.setWindowTitle(_translate("edytor", "Edytor", None))
        self.button_open.setText(_translate("edytor", "Open File", None))
        self.button_close.setText(_translate("edytor", "Close", None))
        self.button_save.setText(_translate("edytor", "Save", None))



"""A simple text editor from a tutorial.
Originally obtained from http://www.rkblog.rk.edu.pl/w/p/simple-text-editor-pyqt4/
but modified for newer PyQt4 and Python 3.
"""
import sys
from os.path import isfile

from PyQt4.QtGui import QMessageBox, QApplication, QMainWindow, QFileDialog
from PyQt4.QtCore import pyqtSlot, QFileSystemWatcher

class PyQtRunner(QApplication):
    apps = []

    def __init__(self, args=None, apps=None, exec_=False):
        if args is None:
            args = sys.argv
        super(PyQtRunner,self).__init__(args)

        if apps is not None:
            self.apps.extend(App() for App in apps)
        if exec_:
            self.exec_()

    def add(self, App):
        self.apps.append(App())
        return self

    def exec_(self):
        for a in self.apps:
            a.show()

        sys.exit(super(PyQtRunner,self).exec_())


class Edytor(QMainWindow, Ui_edytor):
    fpath = ""

    def __init__(self):
        super(Edytor,self).__init__()
        self.setupUi(self)
        # Create watcher
        self.watcher = QFileSystemWatcher(self)

        # Register slots
        self.button_close.clicked.connect(self.close)
        self.button_open.clicked.connect(self.file_open)
        self.button_save.clicked.connect(self.file_save)
        self.editor_window.textChanged.connect(self.text_changed)
        self.watcher.fileChanged.connect(self.file_changed)

    @pyqtSlot()
    def file_open(self, fpath=""):
        if self.ask_discard():
            return

        fpath = fpath or QFileDialog.getOpenFileName(self, "Open file...")
        if isfile(fpath):
            # Switch watcher files
            if self.fpath:
                self.watcher.removePath(self.fpath)
            self.watcher.addPath(fpath)

            with open(fpath) as f:
                text = f.read()
            self.editor_window.setText(text)
            # Disable afterwards since `setText` triggers "textChanged" signal
            self.button_save.setEnabled(False)

            # Finally save the path
            self.fpath = fpath

    @pyqtSlot()
    def file_save(self):
        if isfile(self.fpath):
            # Do not trigger fileChanged when saving ourselves
            self.watcher.removePath(self.fpath)

            text = self.editor_window.toPlainText()
            with open(self.fpath, 'w') as f:
                f.write(text)

            self.button_save.setEnabled(False)
            self.watcher.addPath(self.fpath)

    @pyqtSlot()
    def text_changed(self):
        if self.fpath:
            self.button_save.setEnabled(True)

    @pyqtSlot(str)
    def file_changed(self, path):
        res = QMessageBox.question(
            self, "%s - File has been changed" % self.objectName(),
            "The opened document has been modified by another program.\n"
            + "Do you want to reload the file?",
            QMessageBox.Yes | QMessageBox.No
            | (QMessageBox.Save if self.button_save.isEnabled() else 0),
            QMessageBox.Yes
        )
        if res == QMessageBox.Yes:
            self.file_open(self.fpath)
        elif res == QMessageBox.Save:
            self.file_save()

    def ask_discard(self):
        if not self.button_save.isEnabled():
            return

        res = QMessageBox.question(
            self, "%s - Unsaved changes" % self.objectName(),
            "The document has been modified\n"
            + "Do you want to save your changes?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        print(res)
        if res == QMessageBox.Save:
            self.file_save()

        return res == QMessageBox.Cancel

    def closeEvent(self, event):
        # For some reason this is called twice when clicking the "Close" button SOMETIMES
        if self.ask_discard():
            event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    # PyQtRunner(None, (Edytor,), True)
    PyQtRunner().add(Edytor).exec_()