import sys
from collections import OrderedDict
from pyqtgraph.Qt import QtCore, QtGui # import GUI libraries
from RRtoolFC.lib.rrtoolsNodes import RRtoolConsole, RRtoolFlowchart,isFlowChar  # GUI parts
from pyqtgraph.GraphicsScene.exportDialog import ExportDialog
#from RRtoolFC.lib import RRtoolnamespace
RRtoolnamespace = dict() #RRtoolnamespace.__dict__ # namespace to use inside GUIRRTool as globals
#execfile("../lib/RRtoolnamespace.py",RRtoolnamespace)# loading files

def walkref(obj,lookfor="", level = 0, dipest = 5, ignoreHidden=True):
    """ simple prototype to find references inside an object
    :param obj:
    :param lookfor:
    :param level:
    :param dipest:
    :param ignoreHidden:
    :return:
    """
    for i in dir(obj):
        try:
            if i == lookfor:
                return [i]
            if ignoreHidden and i.startswith("__"):
                continue
            if level<dipest:
                data = walkref(getattr(obj,i),lookfor, level+1, dipest)
                if data:
                    data.append(i)
                    return data
        except:
            pass


menuDescriptors = [(('File', "Open"),
             [dict(label="RR Project",method="openRRProjectAction",short="Ctrl+O",tip="",icon=""),
              dict(label="FlowChart",method="openFCAction",short="",tip="",icon=""),
              dict(label="Library",method="openLIBAction",short="",tip="",icon=""),
              dict(label="Script",method="openScriptAction",short="",tip="",icon="")]),
                   (('File',"Save"),
             [dict(label="RR Project",method="saveRRProjectAction",short="Ctrl+S",tip="",icon=""),
              dict(label="FlowChart",method="saveFCAction",short="",tip="",icon=""),
              dict(label="Library",method="saveLIBAction",short="",tip="",icon=""),
              dict(label="Script",method="saveScriptAction",short="",tip="",icon="")]),
                   (('File',"Save As"),
             [dict(label="RR Project",method="saveAsRRProjectAction",short="Ctrl+Shift+S",tip="",icon=""),
              dict(label="FlowChart",method="saveAsFCAction",short="",tip="",icon=""),
              dict(label="Library",method="saveAsLIBAction",short="",tip="",icon=""),
              dict(label="Script",method="saveAsScriptAction",short="",tip="",icon="")]),
                   (("File","Export"),
             [dict(label="Open Exporter",method="showExporterWin",short="",tip="",icon="")]),
                   (("File",),
             [dict(label="Settings",method="configWin",short="Ctrl+Alt+S",tip="",icon=""),
              dict(label="&Exit",method="exitAction",short="Ctrl+Q",tip="",icon="")]),
                   (('Edit',),
             [dict(label="Lib editor",method="libEditorWin",short="",tip="",icon="")]),
                   (('View',),
             [dict(label="Script Editor",method="scriptEditorWin",short="Ctrl+E",tip="",icon=""),
              dict(label="&Console",method="showConsoleWin",short="Ctrl+C",tip="",icon=""),
              dict(label="&Namespace",method="namespaceWin",short="Ctrl+N",tip="",icon=""),
              dict(label="&Library",method="libraryWin",short="Ctrl+L",tip="",icon=""),
              dict(label="Lo&g",method="logWin",short="Ctrl+G",tip="",icon="")]),
                   (('Tools',),
             [dict(label="Run",method="runAction",short="Ctrl+R",tip="",icon=""),
              dict(label="Install",method="installAction",short="Ctrl+I",tip="",icon=""),
              dict(label="Make Node",method="MakeNodeAction",short="",tip="",icon=""),
              dict(label="Register Node",method="RegNodeAction",short="",tip="",icon="")]),
                   (('Help',),
             [dict(label="Help",method="helpWin",short="Ctrl+H",tip="",icon=""),
              dict(label="About",method="aboutWin",short="Ctrl+A",tip="",icon="")])
                   ]

menuDescriptors = OrderedDict(menuDescriptors)
# menuDescriptors are the instructions to create the GUIRRTool menubar

class GeneralWindow(QtGui.QMainWindow):
    def __init__(self, parent = None, widget = None, title = ""):
        super(GeneralWindow, self).__init__(parent)
        if widget:
            self.setCentralWidget(widget)
            self.resize(widget.size())
        self.setWindowTitle(title)

class Exporter(QtGui.QMainWindow):
    def __init__(self, parent = None, scene = None, contex = None, title = "Exporter"):
        super(Exporter,self).__init__(parent)
        self.setWindowTitle(title)
        self.dialog = ExportDialog(scene)
        self.dialog.ui.closeBtn.clicked.connect(self.close)
        self.setCentralWidget(self.dialog)
        self.contex = contex
    def show(self, contex = None):
        super(Exporter,self).show()
        if contex: self.contex = contex
        self.dialog.show(self.contex)

class GUIRRTool(QtGui.QMainWindow):
    """
    Grafical Interface of RRtoolbox to create custom programs and their executable
    """
    def __init__(self, name ="RRtoolFC", flowchart = None, namespace= None):
        super(GUIRRTool, self).__init__()
        if namespace is None:
            namespace = RRtoolnamespace
        if flowchart is None:
            fcs = [fc for fc in namespace.itervalues() if isFlowChar(fc)]
            if fcs:
                if len(fcs)>1:
                    raise Exception("{} flowcharts not supported: {}".format(len(fcs),fcs))
                flowchart = fcs[0]
            else:
                flowchart = RRtoolFlowchart()
            self.flowchart = flowchart # actual flowchart to edit

        self.isDirectlyClose = True # if True it closes the app regardless of anything (forced close)
        self.projectIsModified = True # if True it asks to save project on close
        self.namespace = namespace # dictionary containing session variables
        self.projectPath = None # the path to project
        self.references = {} # Todo: i need to keep reference of everything so that project saves any detail
        self.setWindowTitle(name)
        self.AppName = name
        #self.resize(500, 400)
        self.initUI()

    def refreshFC(self):
        """
        place Flowchart in Main window
        :return:
        """
        ctrwidget = self.flowchart.widget()
        #ctrwidget = RRtoolCtrlWidget(self.flowchart)
        #fcwidget = RRtoolFCWidget(self.flowchart,ctrwidget)
        fcwidget = ctrwidget.cwWin
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        Hsplitter.addWidget(ctrwidget)
        Hsplitter.addWidget(fcwidget)
        self.FCctrwidget = ctrwidget
        gridlayout = QtGui.QHBoxLayout()
        gridlayout.addWidget(Hsplitter)
        #gridlayout.addWidget(fcwidget)
        cw = QtGui.QWidget()
        cw.setLayout(gridlayout)
        self.scrollArea.setWidget(cw)

    def initUI(self):
        """
        initializes and shows Main window
        :return:
        """
        self.centerWidgeds(self)
        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidgetResizable(True)
        self.setCentralWidget(self.scrollArea)
        #self.setLayout(layout)
        #ACTIONS


        #actions = dict(icon="",label="",short="",tip="",method="",menu="")
        #MENUS
        self.menuHandles,self.actionHandles = self.createMenus(self.menuBar())

        #STATUS
        statusbar = self.statusBar()
        #mm = QtGui.QStatusBar # use this to finde statusbar uses
        statusbar.showMessage("{} is ready".format(self.AppName))
        #toolbar = self.addToolBar('Exit')
        #toolbar.addAction(exitAction)

        #SHOW
        self.refreshFC()
        self.show()

    def createMenus(self, menubar, menuDescriptors = menuDescriptors):
        """
        creates dynamically all actions in the menubar
        :param menubar: Main window menubar
        :param menuDescriptors: dictionary containing construction descriptors
        :return: menuHandles, actionHandles (OrderedDict)
        """
        menuHandles = OrderedDict()
        actionHandles = OrderedDict()
        for menupath,actions in menuDescriptors.iteritems():
            temp = [menubar]
            for i,menu in enumerate(menupath):
                hash = menupath[:i+1]
                if hash in menuHandles:
                    menu = menuHandles[hash]
                else:
                    menu = QtGui.QMenu(menu)
                    menuHandles[hash] = menu
                    temp[i].addMenu(menu)
                temp.append(menu)
            actionH = []
            for act in actions:
                Action = self.createAction(act)
                method = act["method"]
                if isinstance(method,str):
                    if hasattr(self,method):
                        method = getattr(self,method)
                        Action.triggered.connect(method)
                    else:
                        Action.setEnabled(False)
                        print "{} is not implemented!".format(method)
                menu.addAction(Action) # add to handle
                actionH.append(Action)
            actionHandles[hash] = actionH
        return menuHandles,actionHandles

    def createAction(self, act):
        """
        creates custom actions for the menubar
        :param act: dictionary containing instructions
        :return: Action handle
        """
        Action = QtGui.QAction(QtGui.QIcon(act["icon"]), act["label"], self)
        Action.setShortcut(act["short"])
        Action.setStatusTip(act["tip"])
        return Action

    def createConsole(self, parent = None, win="{AppName} console", text="", editor=""):
        win = win.format(**self.__dict__)
        if parent:
            self.console = GeneralWindow(parent, RRtoolConsole(namespace=self.namespace, text=text, editor=editor), win)
        else:
            self.console = GeneralWindow(self, RRtoolConsole(namespace=self.namespace, text=text, editor=editor), win)
        self.centerWidgeds(self.console)

    def createExporter(self,parent = None, win="RRtool exporter"):
        if parent:
            self.exporter = Exporter(parent,self.flowchart.scene,self.FCctrwidget.viewBox(),win)
        else:
            self.exporter = Exporter(self,self.flowchart.scene,self.FCctrwidget.viewBox(),win)
        self.flowchart.scene.exportDialog = self.exporter
        self.centerWidgeds(self.exporter)

    def showExporterWin(self):
        if not hasattr(self,"exporter"):
            self.createExporter()
        self.exporter.show()
        """
        try:
            self.flowchart.scene.showExportDialog() # try from selected context
        except AttributeError as e:
            self.flowchart.scene.exportDialog.show(self.FCctrwidget.viewBox()) # select default Flowchart viewbox
            self.statusBar().showMessage("Selected context from default Flowchart")
        self.centerWidgeds(self.flowchart.scene.exportDialog) # center dialog"""

    def showConsoleWin(self):
        if not hasattr(self,"console"):
            self.createConsole()
        self.console.show()

    def exitAction(self):
        """
        for childQWidget in self.findChildren(QtGui.QWidget):
            try:
                childQWidget.close()
            except:
                pass"""
        #self.isDirectlyClose = True
        return QtGui.QMainWindow.close(self) # this goes to the closeEvent

    def closeEvent (self, eventQCloseEvent):
        # http://stackoverflow.com/a/25887901/5288758
        if self.isDirectlyClose:
            eventQCloseEvent.accept()
        else:
            answer = QtGui.QMessageBox.question (
                self,
                'Task is in progress !',
                'Are you sure you want to quit ?',
                QtGui.QMessageBox.Yes,
                QtGui.QMessageBox.No)
            if (answer == QtGui.QMessageBox.Yes) and self.maybeSave() or (self.isDirectlyClose == True):
                eventQCloseEvent.accept()
            else:
                eventQCloseEvent.ignore()

    ##########################################################

    def installAction(self):
        #output = fc.process(nameOfInputTerminal=newValue)
        pass

    def openFCAction(self):
        self.FCctrwidget.loadClicked()

    def saveFCAction(self):
        try:
            self.FCctrwidget.saveClicked()
            return True
        except:
            return False

    def openScriptAction(self):
        if namespace is None:
            namespace = RRtoolnamespace
        if flowchart is None:
            fcs = [fc for fc in namespace.itervalues() if isFlowChar(fc)]
            if fcs:
                if len(fcs)>1:
                    raise Exception("{} flowcharts not supported: {}".format(len(fcs),fcs))
                flowchart = fcs[0]
            else:
                flowchart = RRtoolFlowchart()

    def saveAsFCAction(self):
        try:
            self.FCctrwidget.saveAsClicked()
            return True
        except:
            return False

    def aboutWin(self):
        QtGui.QMessageBox.about(self, "About RRtoolFC",
            "<p><b>RRtoolFC</b> is a fast prototyping tool ")

    def helpWin(self):
        QtGui.QMessageBox.about(self, "About RRtoolFC",
            "<p><b>RRtoolFC</b> is a fast prototyping tool ")

    def maybeSave(self):
        if self.projectIsModified:
            text =  "{} has been modified.\nDo you want to save your changes?"
            text = text.format(str(self.FCctrwidget.ui.fileNameLabel.text()))
            ret = QtGui.QMessageBox.warning(self, "RRtoolFC",text,
                QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard |
                QtGui.QMessageBox.Cancel)
            if ret == QtGui.QMessageBox.Save:
                return self.saveFCAction()
            elif ret == QtGui.QMessageBox.Cancel:
                return False
        return True

    def saveFile(self, fileFormat):
        initialPath = QtCore.QDir.currentPath() + '/untitled.' + fileFormat

        fileName = QtGui.QFileDialog.getSaveFileName(self, "Save As",
            initialPath,
            "%s Files (*.%s);;All Files (*)" % (fileFormat.upper(), fileFormat))
        if fileName:
            return self.scribbleArea.saveImage(fileName, fileFormat)

        return False

    #########################################################

    @staticmethod
    def centerWidgeds(Widget):
        qr = Widget.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        Widget.move(qr.topLeft())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':

    from RRtoolFC.lib.app import execApp
    file = "Test_main.py"
    #with open(file,"r") as f:
    #    exec f in RRtoolnamespace
    execfile(file,RRtoolnamespace)
    ex = GUIRRTool(namespace=RRtoolnamespace)
    execApp()