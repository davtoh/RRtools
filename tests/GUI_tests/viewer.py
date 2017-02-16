#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg20541.html
# Example of an image viewer with zoom
#
# Created: Thu Feb 25 19:54:49 2010
#      by: PyQt4 UI code generator 4.4.4
#
# Author: Vincent Vande Vyvre <v...@swing.be>
#
# Note: before use, change the line 63

from __future__ import division
from past.builtins import cmp
from builtins import object
from past.utils import old_div
import os
import time
import glob
import sys

from PyQt4 import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(900, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)
        self.verticalLayout.addWidget(self.view)
        self.horizontalLayout = QtGui.QHBoxLayout()
        spacerItem = QtGui.QSpacerItem(150, 20, QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.toolButton_3 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_3.setIconSize(QtCore.QSize(48, 24))
        self.toolButton_3.setText("Previous")
        self.horizontalLayout.addWidget(self.toolButton_3)
        self.toolButton_4 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_4.setIconSize(QtCore.QSize(48, 24))
        self.toolButton_4.setText("Next")
        self.horizontalLayout.addWidget(self.toolButton_4)
        spacerItem1 = QtGui.QSpacerItem(100, 20, QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.toolButton_6 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_6.setText("Quit")
        self.horizontalLayout.addWidget(self.toolButton_6)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("speedyView")
        MainWindow.show()
        QtCore.QCoreApplication.processEvents()

        QtCore.QObject.connect(self.toolButton_3, QtCore.SIGNAL("clicked()"), self.prec)
        QtCore.QObject.connect(self.toolButton_4, QtCore.SIGNAL("clicked()"), self.__next__)
        QtCore.QObject.connect(self.toolButton_6, QtCore.SIGNAL("clicked()"), exit)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.centralwidget.wheelEvent = self.wheel_event

        self.set_view()

    def set_view(self):
        in_folder = "/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtoolbox/tests/"
        chain = in_folder + "/*.jpg"
        self.images = glob.glob(chain)
        self.images.sort(cmp=lambda x, y: cmp(x.lower(), y.lower()))
        self.zoom_step = 0.04
        self.w_vsize = self.view.size().width()
        self.h_vsize = self.view.size().height()
        if self.w_vsize <= self.h_vsize:
            self.max_vsize = self.w_vsize
        else:
            self.max_vsize = self.h_vsize
        self.l_pix = ["", "", ""]

        self.i_pointer = 0
        self.p_pointer = 0
        self.load_current()
        self.p_pointer = 1
        self.load_next()
        self.p_pointer = 2
        self.load_prec()
        self.p_pointer = 0

    def __next__(self):
        self.i_pointer += 1
        if self.i_pointer == len(self.images):
            self.i_pointer = 0
        self.p_view = self.c_view
        self.c_view = self.n_view
        self.view_current()
        if self.p_pointer == 0:
            self.p_pointer = 2
            self.load_next()
            self.p_pointer = 1
        elif self.p_pointer == 1:
            self.p_pointer = 0
            self.load_next()
            self.p_pointer = 2
        else:
            self.p_pointer = 1
            self.load_next()
            self.p_pointer = 0

    def prec(self):
        self.i_pointer -= 1
        if self.i_pointer <= 0:
            self.i_pointer = len(self.images)-1
        self.n_view = self.c_view
        self.c_view = self.p_view
        self.view_current()
        if self.p_pointer == 0:
            self.p_pointer = 1
            self.load_prec()
            self.p_pointer = 2
        elif self.p_pointer == 1:
            self.p_pointer = 2
            self.load_prec()
            self.p_pointer = 0
        else:
            self.p_pointer = 0
            self.load_prec()
            self.p_pointer = 1


    def view_current(self):
        size_img = self.c_view.size()
        wth, hgt = QtCore.QSize.width(size_img), QtCore.QSize.height(size_img)
        self.scene.clear()
        self.scene.setSceneRect(0, 0, wth, hgt)
        self.scene.addPixmap(self.c_view)
        QtCore.QCoreApplication.processEvents()

    def load_current(self):
        self.l_pix[self.p_pointer] = QtGui.QPixmap(self.images[self.i_pointer])
        self.c_view = self.l_pix[self.p_pointer].scaled(self.max_vsize, self.max_vsize,
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.FastTransformation)
        #change the previous line with QtCore.Qt.SmoothTransformation eventually
        self.view_current()

    def load_next(self):
        if self.i_pointer == len(self.images)-1:
            p = 0
        else:
            p = self.i_pointer + 1
        self.l_pix[self.p_pointer] = QtGui.QPixmap(self.images[p])
        self.n_view = self.l_pix[self.p_pointer].scaled(self.max_vsize,
                                            self.max_vsize,
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.FastTransformation)

    def load_prec(self):
        if self.i_pointer == 0:
            p = len(self.images)-1
        else:
            p = self.i_pointer - 1
        self.l_pix[self.p_pointer] = QtGui.QPixmap(self.images[p])
        self.p_view = self.l_pix[self.p_pointer].scaled(self.max_vsize,
                                            self.max_vsize,
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.FastTransformation)

    def wheel_event (self, event):
        numDegrees = old_div(event.delta(), 8)
        numSteps = old_div(numDegrees, 15.0)
        self.zoom(numSteps)
        event.accept()

    def zoom(self, step):
        self.scene.clear()
        w = self.c_view.size().width()
        h = self.c_view.size().height()
        w, h = w * (1 + self.zoom_step*step), h * (1 + self.zoom_step*step)
        self.c_view = self.l_pix[self.p_pointer].scaled(w, h,
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.FastTransformation)
        self.view_current()

def main():
    # this script lets app live preventing it to be garbage collected
    # letting it be instantiated in threading
    global app
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)

    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    app.exit(app.exec_())

def win():
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

if __name__ == "__main__":
    # TODO examine instantiation of the same object
    app = QtGui.QApplication(sys.argv)

    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    """
    MainWindow2 = QtGui.QMainWindow()
    ui2 = Ui_MainWindow()
    ui2.setupUi(MainWindow2)
    MainWindow2.show()"""

    sys.exit(app.exec_())