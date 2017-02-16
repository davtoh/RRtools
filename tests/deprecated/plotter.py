# -*- coding: utf-8 -*-
"""
    This module holds the plotting and data-visualization tools. Motto: don't know how it is interpreted? i'll show you!

    #Plotim example
    filename = "t2.jpg"
    win = "test"
    img = cv2.resize(cv2.imread(filename), (400, 400))  # (height, width)
    plot = Plotim(win,img)
    plot.show()
"""

# http://docs.opencv.org/2.4.9/modules/highgui/doc/highgui.html
# https://docs.python.org/3/library/string.html
# http://stackoverflow.com/questions/101128/how-do-i-read-text-from-the-windows-clipboard-from-python
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import chr
from builtins import range
from past.builtins import basestring
from builtins import object
from config import FLOAT,FLAG_DEBUG
import copy # copy lists
import sys
import os
import time
import traceback
import cv2
import numpy as np
from arrayops.basic import overlayXY,convertXY,overlay,padVH,anorm,standarizePoints,get_x_space,isnumpy
from arrayops.convert import apply2kp_pairs,dict2keyPoint
from arrayops.filters import sigmoid, bilateralFilter, FilterBase
import matplotlib.animation as animation
from multiprocessing import Process
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from matplotlib import gridspec
from string import Formatter
formater = Formatter() # string formatter str.format
#from fastplt import fastplt as _fastplt
__author__ = 'Davtoh'
wins = [0] # keeps track of image number through different processes
#plt.rcParams['image.cmap'] = 'gray'  # change default colormap

def fastplt(image, cmap = None, title ="visualazor", win = None, block = False, daemon = False):
    """
    Fast plot.

    :param image: image to show
    :param cmap: "gray" or None
    :param title: title of subplot
    :param win: title of window
    :param block: if True it wait for window close, else it detaches (Experimental)
    :param daemon: if True window closes if main thread ends, else windows
                must be closed to main thread to end (Experimental)
    :return: plt

    .. notes:: This is a wrapper of the module fastplt.
    """
    # FIXED: incompatibility with Qt application. It seem the server X works with sockets and can't be accessed at the same time
    # UPDATE 07/04/16: made another library with the same name and images are sent though sockets
    #from RRtoolbox.lib.image import plt2bgr
    #image = plt2bgr(image) # once miniprogram is made this should work -> UPDATED 07/04/16
    wins[-1] += 1
    if FLAG_DEBUG: print("fastplt received image type: ",type(image))
    def myplot():
        if isinstance(image, matplotlib.axes.SubplotBase):
            f = image.figure
        elif isinstance(image, matplotlib.figure.Figure):
            f = image
        else:
            f = plt.figure()
            # Normally this will always be "Figure 1" since it's the first
            # figure created by this process. So do something about it.
            plt.imshow(image,cmap)
            if title: plt.title(title)
            plt.xticks([]), plt.yticks([])
            #plt.colorbar()
        wins[0]+=1
        if win: f.canvas.set_window_title(win)
        else:f.canvas.set_window_title("Figure {}".format(wins[-1]))
        if FLAG_DEBUG: print("showing now...")
        #plt.ion()
        plt.show()
        if FLAG_DEBUG: print("showed...")
    if block:
        myplot()
    elif __name__ == "__main__": # if called from shell or directly
        if FLAG_DEBUG: print("multiprocessing...")
        p = Process(target=myplot) # FIXME i shoud call a miniprogram
        p.daemon = daemon
        p.start()
    else: # Workaround to solve problem multiprocessing with matplotlib this sends to shell
        from serverServices import generateServer,sendPickle
        s,addr = generateServer()
        props = ["python '{script}'".format(script = os.path.abspath(__file__))]
        props.append("{}:{}".format(*addr))
        if FLAG_DEBUG: print("generated server at {}".format(addr))
        d = dict(cmap=cmap,title=title,win=win,num=wins[0])
        props.extend(["--{} '{}'".format(key,val) for key,val in list(d.items()) if val is not None])
        if block: props.append("--block")
        if daemon: props.append("--daemon")
        txt = " ".join(props)
        sendPickle(image,s,timeout=10, threaded = True)
        if FLAG_DEBUG: print("sending",txt)
        def myplot(): os.system(txt)
        p = Process(target=myplot) # FIXME i shoud call a miniprogram
        p.daemon = daemon
        p.start()
    if FLAG_DEBUG: print("left fastplt...")

def graph_filter(filters, levels=None, titles=None, win=None, single = True, legend = True, annotate = True, cols = 3, scale = 0.1, show = True):
    """
    Graph filter with standard data to watch response.

    :param filters: list of filters
    :param levels: numpy array with values. if None tries to fit data or assumes from 0 to 255
    :param titles: list of titles for each filter in filters. if None creates the titles
    :param win: window name
    :param single: True to plot all filters in one plot. else separate each filter in a plot.
    :param legend: True to add legends.
    :param annotate: True to add annotations.
    :param cols: number of columns to create plots
    :param scale: factor from maximum to draw annotations
    :param show: to show the figure
    :return: figure
    """

    def getFilterName(filter):
        try:
            return filter.name
        except:
            return type(filter).__name__

    def formatConsume(s,params):
        keys = [t[1] for t in formater.parse(s) if t[1] is not None]
        txt = s.format(**params)
        for k in keys: del params[k]
        return txt

    def safeReplace(name, params = None):
        if params is None: params = {}
        if isinstance(name, basestring):
            try:
                name = formatConsume(name,params)
                if "*" in name:
                    parts = name.split("*")
                    f = ["{key}: {{{key}}}".format(key=key) for key in list(params.keys())]
                    return safeReplace([[part,f] for part in parts],params)
                return name
            except Exception as e:
                #print e
                return
        else:
            return ", ".join(j for j in [safeReplace(i,params) for i in name] if j)

    def getTitle(name, params):
        if params:
            return name + ", ".join("{}: {}".format(key, val) for key, val in list(params.items()))
        else:
            return name + " filter"

    if isinstance(filters,FilterBase):
        if win is None: win = getFilterName(filters) + " filter"
        filters = [filters]
    elif titles is not None and len(filters) != len(titles):
        raise Exception("Titles len {} does not match with filters len {}".format(len(titles),len(filters)))

    if win is None: win = "Filters Response"

    # calculate filter
    if single:
        if levels is None:
            try:
                level = get_x_space(filters) # try to fit to filters
                if level.size == 0:
                    print("in cero")
                    raise Exception
            except:
                level = np.linspace(0, 256,256) # assume it is for an image
        else:
            level = levels
        xmin,xmax = np.min(level),np.max(level)
        xsize = (xmax+abs(xmin))*scale
    else:
        level = None

    # create grid
    rows = len(filters)//cols
    if rows == 0:
        rows = 1
        cols = len(filters)
    gs = gridspec.GridSpec(rows, cols) # grid for each plot
    # create figure
    fig = plt.figure(win)
    # variables
    lines,axtitles = [],[] # all lines, all titles
    ymins,ymaxs = [],[] # overall ymin and ymax
    for i,f in enumerate(filters):

        if single:
            ax = plt.gca()
            if levels is not None and not isnumpy(levels):
                level = levels[i]
                xmin,xmax = np.min(level),np.max(level)
                xsize = (xmax+abs(xmin))*scale
        else:
            ax = fig.add_subplot(gs[i])
            if levels is None:
                try:
                    level = get_x_space([f]) # try to fit to filters
                    if level.size == 0:
                        print("in cero")
                        raise Exception
                except:
                    level = np.linspace(0, 256,256) # assume it is for an image
            elif isnumpy(levels):
                level = levels
            else:
                level = levels[i]
            xmin,xmax = np.min(level),np.max(level)
            xsize = (xmax+abs(xmin))*scale

        y = f(level)
        title,name = "",""
        ymin,ymax = np.min(y),np.max(y)
        ysize = (ymax+abs(ymin))*scale
        params = {}
        if isinstance(f,FilterBase):
            # get the parameters in a dictionary
            # TODO implement getattr() instead with a list of parameters from annotate
            params = {str(key)[1:]:val for key,val in list(f.__dict__.items()) if val is not None}
            if annotate:
                for key,val in list(params.items()):
                    if key != "alfa":
                        xp,yp = val,f(val)
                        if key == "beta1" and "beta2" in params: #xp < (xmax+abs(xmin))/2: # label position at x
                            lx = xp - xsize - 15 # to the left and add shift for the text box
                        else:
                            lx = xp + xsize # to the right

                        ly = yp # to a side
                        if ly < ymin+ysize: # label position at y
                            ly = yp + ymax*scale # slightly up
                        elif ly > ymax-ysize:
                            ly = yp - ymax*scale # slightly down

                        ax.annotate(u'{}\n{}'.format(key,val).title(), xy=(xp,yp), textcoords='data',
                                    xytext=(lx,ly),
                                    arrowprops=dict(facecolor='black', shrink=0.05))
            name = getFilterName(f) # name of filter
            params["name"] = name
            if titles is None:
                if annotate:
                    if legend:
                        title = name+ " filter" # append filter to name
                        name += ": alfa {}".format(params["alfa"])
                        #name = safeReplace(name+": ",params)
                    else:
                        title = safeReplace(["{name} filter","alfa: {alfa}"],params)
                else:
                    if single:
                        title = name
                        name = safeReplace(["{name}: ","*"],params)
                    else:
                        title = safeReplace(["{name} filter","*"],params)

        if not name:
            name = "Filter {}".format(i)

        if not title:
            if titles is None:
                title = name
            else:
                title = safeReplace(titles[i],params)

        line, = ax.plot(level,y,label=name.title())
        if single:
            lines.append(line)
            axtitles.append(title)
            ymins.append(ymin)
            ymaxs.append(ymax)
        else:
            lines = [line]
            axtitles = [title]
            ymins = [ymin]
            ymaxs = [ymax]
        if not single or single and i == len(filters)-1:
            plt.title("; ".join(axtitles).title())
            plt.xlim([-1+xmin,xmax+1])
            plt.ylim([-0.1+np.min(ymins),0.1+np.max(ymaxs)])
            if legend:
                ax.legend(handles=lines,loc = 'upper left')
                #ax.legend(name, handles=lines,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if show: plt.show()
    return fig

def plotPointsContour(pts, ax= None, lcor="k", pcor=None, deg = None):
    """
    Plots points and joining lines in axes.

    :param pts: points. [(x0,y0)...(xN,yN)]
    :param ax: axes handle to draw points.
    :param lcor: color of joining lines.
    :param pcor: color of points. If specified uses lines, else vectors.
    :param deg: angle of vertex, if True in degrees, if False in radians, if None do not add.
    :return: ax.
    """
    # http://stackoverflow.com/a/12267492/5288758
    from arrayops.basic import relativeVectors,vertexesAngles
    ax = ax or plt.gca() # get axes, creates and show figure if interactive is ON, disable with plt.ioff()
    if pcor is not None:
        ax.plot(pts[:, 0], pts[:, 1], markerfacecolor = pcor , marker='o') # plot points
        ax.plot(pts[:, 0], pts[:, 1], color = lcor) # and lines
        ax.plot([pts[-1, 0],pts[0, 0]], [pts[-1, 1],pts[0, 1]], 'k') # add last line
    if deg is not None:
        for i,((x,y),(u,v),a) in enumerate(zip(pts, relativeVectors(pts), vertexesAngles(pts, deg=deg))): # annotate each point
            if pcor is None: ax.quiver(x,y,u,v,angles='xy',scale_units='xy',scale=1,width=0.004,color = lcor)
            op = u""
            if deg is True:
                op = u"°"
            ax.annotate(u'{}({:1.1f}, {:1.1f}, {:1.1f}{})'.format(i,x,y,a,op), xy=(x,y), textcoords='data')
    else:
        for i,((x,y),(u,v)) in enumerate(zip(pts,relativeVectors(pts))): # annotate each point
            if pcor is None: ax.quiver(x,y,u,v,angles='xy',scale_units='xy',scale=1,width=0.004,color = lcor)
            ax.annotate('{}({:1.1f}, {:1.1f})'.format(i,x,y), xy=(x,y), textcoords='data')
    return ax

def mousefunc(self):
    """
    Decoupled mouse function for Plotim (replace self.mousefunc).

    :param self: Plotim instance
    """
    if self.builtincontrol():
        self.updaterenderer()
    if self.y is not None and self.x is not None:
        self.builtinplot(self.sample[self.y,self.x])

def keyfunc(self):
    """
    Decoupled key function for Plotim (replace self.keyfunc).

    :param self: Plotim instance
    """
    if self.builtincmd():
        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[self.y,self.x])
        else:
            self.builtinplot()

def closefunc(self):
    """
    Decoupled close function for Plotim (replace self.closefunc).

    :param self: Plotim instance
    """
    # do stuff before closing #
    return self.pressedkey == 27 or self.close# close if ESC or Close button

def windowfunc(self):
    """
    Decoupled window function for Plotim (replace self.windowfunc).

    :param self: Plotim instance
    """
    cv2.namedWindow(self.win,self.wintype)  # create window
    #cv2.resizeWindow(self.win,self.rW,self.rH)

def showfunc(self,img=None):
    """
    Decoupled show function for Plotim (replace self.showfunc).

    :param self: Plotim instance
    :param img: image to show
    """
    if img is None:
        cv2.imshow(self.win,self.rimg)
    else:
        cv2.imshow(self.win,img)

def formatcmd(self, cmd, references=("+","-","*","="), lmissing="self."):
    """
    Decoupled cmd formatter for cmdfunc and Plotim.

    :param self: Plotim instance
    :param cmd: command
    :param references:
    :param lmissing: assumed missing part in command
    :return:
    """
    def splitter(cmd,references=("+","-","*","=")):
        for i in references:
            cmd = cmd.replace(i," ")
        return cmd.split()

    def correct(string,lmissing="self."):
        try:
            eval(string)
        except:
            if not string.find(lmissing)!=-1:
                string = lmissing+string
        return string

    def format(cmd,strlist,lmissing):
        if len(strlist)!=1: # ensures toggle functionality for single expression in evalcommand
            for i in range(len(strlist)):
                cmd = cmd.replace(strlist[i],correct(strlist[i],lmissing))
        return cmd

    return format(cmd,splitter(cmd,references),lmissing).replace(" ","")

def echo(obj):
    """
    Printer (used when user wants to print an object from Plotim)
    :param obj: object
    """
    print(obj)

def cmdfunc(self,execute = False):
    """
    Decoupled cmd solver for Plotim. (repalce self.cmdfunc)

    :param self:
    :param execute: True, enable execution of commands, False, disable execution.
    """
    def evalcommand(self,cmd,showresult=True,wait=3):
        if type(cmd) is list: # this configuration lets the user use custom variables
            for command in cmd:
                try:
                    if command.find("=")!=-1 or command.find(" ")!=-1 or command.find("(")!=-1 or command.find(".")!=-1:
                        exec(command, globals(), locals())
                        if showresult: self.plotintime(items=[[command+" executed"]],wait=wait)
                    else:
                        setattr(self,command,not getattr(self,command))
                        if showresult:
                            if getattr(self,command):
                                self.plotintime(items=[[command+" is ON"]],wait=wait)
                            else:
                                self.plotintime(items=[[command+" is OFF"]],wait=wait)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    msg = lines[-1]  # Log it or whatever here
                    self.plotintime(items=[["Error executing "+command+": "+msg]],wait=wait,bgrcolor=self.errorbackground)
                    print(ValueError)
                    return False
            if showresult: self.plotintime(items=[[self.cmd+" executed"]],wait=wait)
            return True
        else:
            return evalcommand(self,[cmd],showresult,wait)
    text = []
    if execute and self.cmd != "": # in execution
        choice = [x for x in self.cmdlist if x.startswith(self.cmd)]
        if choice != []:
            self.cmd = choice[0]
        command = self.cmdeval.get(self.cmd)
        if command is None:
            if self.cmdformatter: command = formatcmd(self,self.cmd)
            else: command = self.cmd
        if evalcommand(self,command):
            command2 = self.cmdeval.get("end")
            if command2 is not None:
                evalcommand(self,command2,False)
                self.cmderror = False
                if not self.cmdcache.count(self.cmd): self.cmdcache.append(command)
                self.cmd = ""
            else:
                self.cmderror = False
        else:
            self.cmderror = True
    else:
        text.extend([["cmd: "+self.cmd]])
        #self.cmdfiltered = [[i] for i in filter(lambda x: x.startswith(self.cmd), self.cmdlist)]
        #if self.cmdfiltered != []:
        #    text.extend(self.cmdfiltered)
    if self.cmderror and self.cmd != "":
        self._cmdbgrcolor = self.errorbackground
    else:
        self.cmderror = False
        self._cmdbgrcolor = self.textbackground
    self._cmditmes = text

def background(color,x=1,y=1,flag=0):
    """
    Creates background rectangle.

    :param color: main color.
    :param x: x pixels in axis x.
    :param y: y pixels in axis y.
    :param flag: Not implemented.
    :return: image of shape y,x and ndim == color.ndim.
    """
    try:
        back = np.zeros((y,x,len(color)), np.uint8)
    except:
        back = np.zeros((y,x), np.uint8)
    back[:,:]= color
    return back

def limitaxis(c,maxc,minc=0):
    """
    Limit value in axis.

    :param c: value
    :param maxc: max c value.
    :param minc: min c value.
    :return: limited c value c E [minc,maxc]
    """
    if c > maxc:
        c = maxc
    if c < minc:
        c = minc
    return c

def convert2bgr(src, bgrcolor = None):
    """
    Tries to convert any image format to BGR.

    :param src: source image.
    :param bgrcolor: background or transparent color.
    :return: BGR array image.
    """
    if not np.max(src)>1:
        im = src*255
        im = im.astype("uint8")
    else:
        im = src.astype("uint8")
    #ERROR: Source image must have 1, 3 or 4 channels in function cvConvertImage
    if len(im.shape)==2:  # 1 channel
        if np.max(im)>1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            im = cv2.cvtColor(im*255, cv2.COLOR_GRAY2BGR)
    elif len(im.shape)==3 and im.shape[2]==2:  # 2 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGR)
    elif len(im.shape)==3 and im.shape[2]==4:  # 4 channels
        if bgrcolor is None:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        else:
            temp=im.shape
            data = np.zeros((temp[0],temp[1],3), np.uint8)
            data[:,:,:] = bgrcolor
            im = overlay(data,im)
    # else 3 channels or error for 2 channels)
    return im

def convert2bgra(src, bgracolor = None,transparency = None):
    """
    Tries to convert any image format to BGRA.

    :param src: source image.
    :param bgracolor: background or transparent color.
    :param transparency: mask or A channel.
            (typically source image has not A channel, so user can provide it)
    :return: BGRA array image.
    """
    if not np.max(src)>1:
        im = src*255
        im = im.astype("uint8")
    else:
        im = src.astype("uint8")
    #ERROR: Source image must have 1, 3 or 4 channels in function cvConvertImage
    if len(im.shape)==2:  # 1 channel
        if np.max(im)>1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        else:
            im = cv2.cvtColor(im*255, cv2.COLOR_GRAY2BGRA)
    elif len(im.shape)==3 and im.shape[2]==2:  # 2 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR5552BGRA)
    elif len(im.shape)==3 and im.shape[2]==3:  # 3 channels
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    elif len(im.shape)==3 and im.shape[2]==4:  # 4 channels
        if bgracolor is not None:
            temp=im.shape
            data = np.zeros((temp[0],temp[1],4), np.uint8)
            data[:,:] = bgracolor
            im = overlay(data,im)
    if transparency is not None:
        im[:,:,3] = transparency
    return im

def onmouse(event, x, y, flags, self):
    """
    Mouse event function for Plotim. (replace self.mousefunc)

    :param event: mouse event
    :param x: x position
    :param y: y postion
    :param flags: mouse flag to use in control (it represents clicks)
    :param self: Plotim object
    :return:
    """
    #print event, x, y, flags, param
    if self.rx!= x or self.ry != y:
        self.mousemoved = True
    self.rx = x  # x rendered position
    self.ry = y  # y rendered positiom

    # correct x,y coordinates from render and zoom
    x = self.rx1+x*(self.rx2-self.rx1)/self.rW
    y = self.ry1+y*(self.ry2-self.ry1)/self.rH

    if self.limitaxes:
        # assign x
        #self.x = limitaxis(x,self.maxX-1,self.minX)
        self.x = limitaxis(x,self.rx2-1,self.rx1)
        # assign y
        #self.y = limitaxis(y,self.maxY-1,self.minY)
        self.y = limitaxis(y,self.ry2-1,self.ry1)
    else:
        self.x = x
        self.y = y

    self.event = event
    self.flags = flags
    self.mousefunc(self)

class plotim(object):
    # FIXME: this code now is buggy, nothing to do. consider replacing it with wxpython or pyqt
    '''
    Show and image with events, animations, controls, internal
    commands and highly customizable by code.

    It implements: self.createfunc, self.pressedkey, self.mousefunc, self.closefunc, self.close.

    .. warning:: Plotim is deprecated and will be replaced in the future (it was made to
                test concepts). Originally it was made for windows but some functions
                were removed to let it be multi-platform.
    '''
    # for rendered visualization
    rWmax = 10000
    rHmax = 10000
    limitrender = True # limit render value
    rx1=0
    ry1=0
    rx = 0
    ry = 0
    rx2=0
    ry2=0
    maxY = 0 # maximum
    maxX = 0
    minY = 0
    minX = 0

    def __init__(self, win, im = np.array([[1]]), bgrcolor = (250,243,238)):
        '''
        :param win: window name
        :param im: image of numpy array
        :param bgrcolor: default color to use for transparent or background color.
        '''
        # clipboard has to be imported internally due to incompatibility with pyplot or Process
        # it just simply creates a weird IOError
        try:
            import pyperclip as clipboard
        except:
            import clipboard
        self.usecontrol = False
        self.clipboard = clipboard
        # for rendered visualization
        self.rW = 600
        self.rH = 400
        self.rWmax = 5000
        self.rHmax = 5000
        self.limitrender = True # limit render value

        ## Image data ##
        self.data = im  # image
        self.sample = im
        self.win = win  # window name
        self.wintype = cv2.WINDOW_NORMAL # window to create
        self.interpolation=cv2.INTER_LINEAR # render's interpolation
        self.bgrcolor = bgrcolor # Plotim background's color
        ## Flow control ##
        self.windowfunc = windowfunc # custom plot on creation function
        self.showfunc = showfunc # custom show function
        self.keyfunc = keyfunc  # only called on keystroke
        self.mousefunc = mousefunc  # only called on mouse event
        self.closefunc = closefunc  # return True to close plot
        self.cmdfunc = cmdfunc # command function
        self.limitaxes = True # correct axes if out of bound
        self.showcoors = True # to show coordinates
        self.showcontrol = True # to show control data
        self.showpixel = True # to show x,y pixel color
        self.showpixelvalue = True # to show x,y pixel value
        self.staticcoors = False # to move coordinates
        self.controlText= [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
        self.cmdlist = ["limitrender","limitaxes","showcoors","showcontrol","usecontrol","toggle all",
                        "showpixel","staticcoors","showpixelvalue","pixels","cmdformatter"] # filter commands
        self.cmdeval = {"pixels":["showpixel","showpixelvalue"],
                        "toggle all":["showpixel","showpixelvalue","showcoors","showcontrol"],
                        "end":"self.updaterenderer()"} # commands with eval operations
        self.cmdbuffer = [] # defines ctr_z behavior
        self.cmdcache = [] # cache successful commands
        self.cmdfiltered = [] # cache filtered list
        self.cmdcache_choice = 0 # selected item in self.cmdcache
        self.cmdfiltered_choice = 0 # selected item in self.cmdfiltered
        ## refresh handles ##
        self.delayplot = 1000 # if delay 0 waits for key infinitely
        #  FIXME for compatibility reason dalyplot can't be 0, if user closes the window it blocks program
        # (it was solved knowing when the window was closed but it was not a multiplatform solution)

        # for Zoom and move
        self.zoomstep = 4 # step to zoom in and out, if 1 zoom is cancelled
        self.minzoom = 2 # minimal zoom
        self.controlkey = cv2.EVENT_FLAG_CTRLKEY # key to press to activate builtincontrol
        self.zoominbutton = cv2.EVENT_LBUTTONDOWN # mouse button for zooming in
        self.zoomoutbutton = cv2.EVENT_RBUTTONDOWN # mouse button for zooming out
        self.movebutton = cv2.EVENT_MBUTTONDOWN # mouse button for moving render
        self.resetbutton = cv2.EVENT_RBUTTONDBLCLK # key for resetting plot

        # for coordinates
        self.fontFace=cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale=self.rH*0.001
        self.thickness=int(self.fontScale)+1 # ensures that thickness is > 1
        self.textcolor = (0,0,0,255)
        self.textbackground = (255,255,255,100)
        self.errorbackground = (0,0,255,100)
        self.coordinateText = [["xy({self.x},{self.y})"]]
        # for time
        self._wait = 0
        self._iswaiting = False
        self._oldt = 0
        self._timeitems = []
        self._timebgrcolor = None
        self._cmditmes = []
        self._cmdbgrcolor = self.textbackground
        self.cmdformatter = True
        # initialize
        self.init()

    def init(self): # Init control variables!
        """
        Pseudo __init__. it is used to restart default
        values without destroying configurations.
        """
        ## Image data ##
        self.img = self.data.copy() # image
        temp = self.img.shape
        self.maxY = temp[0] # maximum
        self.maxX = temp[1]
        self.minY = 0
        self.minX = 0
        ## Flow control ##
        self.close = False # True to close plot
        self.pressedkey = None  # initialize to no key pressed
        self.cmd = ""
        self.iscmd = False
        self.cmderror = False
        # for rendered visualization
        self.rx = None
        self.ry = None
        self.rx2=self.maxX
        self.ry2=self.maxY
        self.rx1=0
        self.ry1=0
        # for Zoom
        self.rxZoom =self.rx2-self.rx1
        self.ryZoom =self.ry2-self.ry1
        # for coordinates
        self.mousemoved = False
        self.event = 0
        self.x = None
        self.y = None
        self.flags = 0
        self.updaterenderer()  # update render

    def show(self, frames = None, block = True, daemon = False):
        """
        Show function. calls buildinwindow, handles key presses and close events.

        :param frames: show number of frames and close.
        :param block: if True it wait for window close, else it detaches (Experimental)
        :param daemon: if True window closes if main thread ends,
                    else windows must be closed to main thread to end (Experimental)
        :return:
        """
        def _show(frames = None):
            self.builtinwindow()  # make window
            while not self.close: # checkLoaded close after processing
                self.pressedkey = cv2.waitKey(self.delayplot) #& 0xFF  # wait for key
                if self.closefunc(self): break  # close by closing function
                self.keyfunc(self)  # do stuff on key
                self.mousemoved = False
                if frames is not None:
                    frames -= 1
                    if frames<1: break
            cv2.destroyWindow(self.win) # close window

        if block:
            _show(frames)
        elif isinstance(self,plotim): # if called from shell or directly #FIXED __name__ == "__main__" does not work when pickled
            if FLAG_DEBUG: print("multiprocessing...")
            p = Process(target=_show,args=(frames,)) # FIXME i shoud call a miniprogram
            p.daemon = daemon
            p.start()
        else: # Workaround to solve problem multiprocessing with matplotlib this sends to shell
            from serverServices import generateServer,sendPickle
            s,addr = generateServer()
            props = ["python '{script}'".format(script = os.path.abspath(__file__))]
            props.append("{}:{}".format(*addr))
            if FLAG_DEBUG: print("generated server at {}".format(addr))
            if block: props.append("--block")
            if daemon: props.append("--daemon")
            if frames: props.append("--frames")
            txt = " ".join(props)
            sendPickle(self,s,timeout=10, threaded = True)
            if FLAG_DEBUG: print("sending",txt)
            def myplot(): os.system(txt)
            p = Process(target=myplot) # FIXME i shoud call a miniprogram
            p.daemon = daemon
            p.start()
        return self

    def plotatpointer(self,items,img=None,x=0,y=0,flag=6,xpad=0,ypad=0,bgrcolor = None,alfa = None,pixel=None):
        """
        Plot message where mouse pointer is.

        :param items: list of items supported by :func:`self.makeoverlay`
        :param img: image to place in items. If None it uses self.remg
        :param x: x position
        :param y: y position
        :param flag: flag for position (default=0).

            * flag==0 : foreground to left up.
            * flag==1 : foreground to left down.
            * flag==2 : foreground to right up.
            * flag==3 : foreground to right down.
            * flag==4 : foreground at center of background.
            * flag==5 : XY 0,0 is at center of background.
            * flag==6 : XY 0,0 is at center of foreground.
            * flag==7 : XY 0,0 is at right down of foreground.
        :param xpad: padding in x
        :param ypad: padding in y
        :param bgrcolor: background color
        :param alfa: alfa mask or value for transparency
        :param pixel: color to add as item im items,
        :return:

        .. olsosee:: :func:`convertXY`, :func:`self.makeoverlay`
        """
        text = copy.deepcopy(items)
        if x is not None and y is not None:
            if img is None:
                img = self.rimg
            if x>self.rW/2:
                if y>self.rH/2:
                    quartile = 2
                    if pixel is not None: text[0].append(pixel)
                else:
                    quartile = 3
                    if pixel is not None: text[0].append(pixel)
            else:
                if y>self.rH/2:
                    quartile = 1
                    if pixel is not None: text[0].insert(0,pixel)
                else:
                    quartile = 4
                    if pixel is not None: text[0].insert(0,pixel)
            foretext = self.makeoverlay(text,xpad,ypad,bgrcolor,alfa)[0]
            tx,ty = convertXY(x,y,img.shape,foretext.shape,flag,quartile=quartile) # translation
        else:
            text[0].append(pixel)
            foretext = self.makeoverlay(text,xpad,ypad,bgrcolor,alfa)[0]
            tx,ty = convertXY(0,0,img.shape,foretext.shape,4) # translation
        self.showfunc(self,overlayXY(tx,ty,back=img,fore=foretext))  # show window

    def plotatxy(self,items,img=None,x=0,y=0,flag=0,xpad=0,ypad=0,bgrcolor = None,alfa = None):
        """
        Plot message in xy position.

        :param items: list of items supported by :func:`makeoverlay`
        :param img: image to place in items. If None it uses self.remg
        :param x: x position
        :param y: y position
        :param flag: flag for position (default=0).

            * flag==0 : foreground to left up.
            * flag==1 : foreground to left down.
            * flag==2 : foreground to right up.
            * flag==3 : foreground to right down.
            * flag==4 : foreground at center of background.
            * flag==5 : XY 0,0 is at center of background.
            * flag==6 : XY 0,0 is at center of foreground.
            * flag==7 : XY 0,0 is at right down of foreground.
        :param xpad: padding in x
        :param ypad: padding in y
        :param bgrcolor: background color
        :param alfa: alfa mask or value for transparency
        :return:
        """
        if img is None:
            img = self.rimg
        foretext = self.makeoverlay(items,xpad,ypad,bgrcolor,alfa)[0]
        tx,ty = convertXY(x,y,img.shape,foretext.shape,flag)
        self.showfunc(self,overlayXY(tx,ty,back=img,fore=foretext))  # show window

    def plotintime(self,items=None,wait=2,img=None,bgrcolor = None):
        """
        plots messages and events.

        :param items: list of items supported by :func:`makeoverlay`
        :param wait: time of message.
        :param img: image to place in items. If None it uses self.remg
        :param bgrcolor: color of message.
        :return:
        """
        if items is None:
            if img is not None and self._iswaiting:
                if self._wait > time.time()-self._oldt:
                    #print self._iswaiting, time.time()-self._oldt
                    if len(self._timeitems)>6: # don't show more than 6 items at a time
                        self._timeitems = self._timeitems[1:]
                    self.plotatxy(self._timeitems,img,flag=4,bgrcolor=self._timebgrcolor)
                else:
                    self._timeitems = []
                    self._iswaiting = False
        else:
            self._wait = wait
            self._iswaiting = True
            self._oldt = time.time()
            self._timeitems.extend(items)
            self._timebgrcolor = bgrcolor

    def updaterenderer(self,img=None,zoom=True):
        """
        update renderer when called.

        :param img: image to update in renderer, if None use self.img
        :param zoom: True to enable zoom, else updates with original img.
        :return: None
        """
        if img is None:
            img = self.img
        if zoom:
            self.rimg = cv2.resize(convert2bgr(img[self.ry1:self.ry2,self.rx1:self.rx2].copy(),self.bgrcolor),
                        (self.rW, self.rH),interpolation=self.interpolation)
        else:
            self.rimg = cv2.resize(convert2bgr(img.copy(),self.bgrcolor),
                        (self.rW, self.rH),interpolation=self.interpolation)
        if self.showcontrol:
            foretext = self.makeoverlay(self.controlText,bgrcolor=self.textbackground)[0]
            overlayXY(*convertXY(0,0,self.rimg.shape,foretext.shape,1),back=self.rimg,fore=foretext)

    def builtinwindow(self):
        """
        loads windowfunc, showfunc, starts window thread and mousecallback.
        """
        self.windowfunc(self)  # user customization
        self.showfunc(self)  # show window
        cv2.startWindowThread()
        cv2.setMouseCallback(self.win,onmouse,self)  # bind mouse events

    def builtincontrol(self,control=False):
        """
        Internal control. use self.usecontrol = True to activate.

        :param control: if True, use control key.
        :return:
        """
        controlled = False
        if self.usecontrol:
            if not control: # if not control wait for controlkey
                control = self.controlkey
            if control is None or self.flags & control:
                # Zoom system #
                if self.event == self.zoominbutton:  # Zoom in
                    mx = my = self.minzoom
                    if self.rxZoom!=mx or self.ryZoom!=my:
                        xdiff = np.min((self.rxZoom,self.ryZoom))/self.zoomstep+1
                        ydiff = xdiff
                        controlled = True

                if self.event == self.zoomoutbutton:  # Zoom out
                    if self.rxZoom!=self.maxX or self.ryZoom!=self.maxY:
                        mx = my = self.minzoom
                        xdiff = np.max((self.rxZoom,self.ryZoom))*self.zoomstep+1
                        ydiff = xdiff
                        controlled = True
                # Move system #
                if self.event == self.movebutton:
                    mx = self.rxZoom
                    my = self.ryZoom
                    xdiff = mx/2
                    ydiff = my/2
                    controlled = True

                if controlled: # update render parameters
                    self.rx2 = limitaxis(self.x+xdiff,self.maxX,self.minX+mx)
                    self.rx1 = limitaxis(self.x-xdiff,self.rx2-mx,self.minX)
                    self.ry2 = limitaxis(self.y+ydiff,self.maxY,self.minY+my)
                    self.ry1 = limitaxis(self.y-ydiff,self.ry2-my,self.minY)

                # resetting plot #
                if self.event == self.resetbutton:
                    self.init()
                    controlled = True

        return controlled

    def builtincmd(self):
        """
        Internal cmd control
        """
        self.iscmd = False
        if self.pressedkey is not None and self.pressedkey != 255 and self.pressedkey != 27: # is normal
            def updatelist():
                if not self.cmdfiltered:
                    if self.cmd == "":
                        self.cmdfiltered = self.cmdlist
                    else:
                        self.cmdfiltered = [i for i in [x for x in self.cmdlist if x.startswith(self.cmd)]]

            if self.pressedkey == 2490368: # if up key
                text = []
                updatelist()
                mylist = self.cmdfiltered
                if mylist:
                    index = self.cmdfiltered_choice
                    index = limitaxis(index -1,len(mylist)-1)
                    self.cmdfiltered_choice = index
                    choice = mylist[index]
                    self.cmd = choice
                    toshow = [[i] for i in mylist]
                    toshow[index] = ["* "+choice]
                    text.extend([["cmd: "+self.cmd]])
                    text.extend(toshow)
                self.cmdfiltered = mylist
                self._cmditmes = text
                self.iscmd = True
                return True
            elif self.pressedkey == 2621440: # if down key
                text = []
                updatelist()
                mylist = self.cmdfiltered
                if mylist:
                    index = self.cmdfiltered_choice
                    index = limitaxis(index +1,len(mylist)-1)
                    self.cmdfiltered_choice = index
                    choice = mylist[index]
                    self.cmd = choice
                    toshow = [[i] for i in mylist]
                    toshow[index] = ["* "+choice]
                    text.extend([["cmd: "+self.cmd]])
                    text.extend(toshow)
                self.cmdfiltered = mylist
                self._cmditmes = text
                self.iscmd = True
                return True
            elif self.pressedkey == 2555904: # if right key
                text = []
                mylist = self.cmdcache
                if mylist:
                    index = self.cmdcache_choice
                    index = limitaxis(index +1,len(mylist)-1)
                    self.cmdcache_choice = index
                    choice = mylist[index]
                    self.cmd = choice
                    toshow = [[i] for i in mylist]
                    toshow[index] = ["* "+choice]
                    text.extend([["cmd: "+self.cmd]])
                    text.extend(toshow)
                self._cmditmes = text
                self.iscmd = True
                return True
            elif self.pressedkey == 2424832: # if left key
                text = []
                mylist = self.cmdcache
                if mylist:
                    index = self.cmdcache_choice
                    index = limitaxis(index -1,len(mylist)-1)
                    self.cmdcache_choice = index
                    choice = mylist[index]
                    self.cmd = choice
                    toshow = [[i] for i in mylist]
                    toshow[index] = ["* "+choice]
                    text.extend([["cmd: "+self.cmd]])
                    text.extend(toshow)
                self._cmditmes = text
                self.iscmd = True
                return True
            elif self.pressedkey == 8: # if backslash
                self.cmd = self.cmd[:-1]
            elif self.pressedkey == 22: # ctrl+v
                buff = self.cmdbuffer
                if buff != [] and self.cmd != self.cmdbuffer[-1]:
                    self.cmdbuffer.append(self.cmd)
                else:
                    self.cmdbuffer.append(self.cmd)
                self.cmd += self.clipboard.paste()
                pass
            elif self.pressedkey == 3: # ctrl+c
                self.clipboard.copy(self.cmd)
            elif self.pressedkey == 24: # ctrl+x
                buff = self.cmdbuffer
                if buff != [] and self.cmd != self.cmdbuffer[-1]:
                    self.cmdbuffer.append(self.cmd)
                else:
                    self.cmdbuffer.append(self.cmd)
                self.clipboard.copy(self.cmd)
                self.cmd = ""
            elif self.pressedkey == 26: # ctrl+z
                if self.cmdbuffer == []:
                    self.cmd = self.cmd[:-1]
                else:
                    self.cmd =self.cmdbuffer.pop()
            elif self.pressedkey != 13 and self.pressedkey != 0 and self.pressedkey<255: # append to cmd and filter commands
                try:
                    self.cmd += chr(self.pressedkey)
                except:
                    pass
                    #print("keystroke not supported...")

            if self.pressedkey == 13: # if enter
                buff = self.cmdbuffer
                if buff != [] and self.cmd != self.cmdbuffer[-1]:
                    self.cmdbuffer.append(self.cmd)
                else:
                    self.cmdbuffer.append(self.cmd)
                self.cmdfunc(self,True)
                self.iscmd = True
                return self.iscmd
            elif self.cmd != "":
                text = []
                text.extend([["cmd: "+self.cmd]])
                mylist = [i for i in [x for x in self.cmdlist if x.startswith(self.cmd)]] # pattern list
                if mylist:
                    toshow = [[i] for i in mylist]
                    text.extend(toshow)
                self._cmditmes = text
                self.iscmd = True
            else:
                self._cmditmes = None
                self.iscmd = True

        return self.iscmd
    def builtinplot(self,pixel=None,useritems = None,flag=1,xpad=0,ypad=0,bgrcolor = None,alfa = None):
        """
        Internal plot.

        :param pixel: pixel color where mouse is placed (placed for better control). Color can be from
                real image, showed image, original image or rendered image, or any color.
        :param useritems: items to show.
        :param flag: flag for position (default=0).

            * flag==0 : foreground to left up.
            * flag==1 : foreground to left down.
            * flag==2 : foreground to right up.
            * flag==3 : foreground to right down.
            * flag==4 : foreground at center of background.
            * flag==5 : XY 0,0 is at center of background.
            * flag==6 : XY 0,0 is at center of foreground.
            * flag==7 : XY 0,0 is at right down of foreground.

        :param xpad: padding in x
        :param ypad: padding in y
        :param bgrcolor: background color
        :param alfa: alfa mask or value for transparency
        :return:
        """
        items = [[]]
        if useritems is None: # assign user items
            if self.showcoors:
                items = copy.deepcopy(self.coordinateText)
        else:
            items = copy.deepcopy(useritems)
        if pixel is not None and self.showpixelvalue: # show pixel values
            items[0].extend([str(pixel)])
        if not self.showpixel: # show pixel color
            pixel = None
        elif pixel is not None and type(pixel) is not int:
            if pixel.shape != ():
                pixel = tuple(pixel)
            else:
                pixel = int(pixel)
        if self.iscmd:  # show commands
            if self._cmditmes: items.extend(self._cmditmes)
            bgrcolor = self._cmdbgrcolor

        # coordinate system #
        if items != [[]] or pixel is not None: # if items to plot or pixel color to show
            img = self.rimg.copy()
            self.plotintime(img=img)
            if self.staticcoors:
                if pixel is not None: items[0].insert(0,pixel)
                if self.showcontrol and flag==1:
                    foretext = self.makeoverlay(self.controlText,xpad,ypad,bgrcolor,alfa)[0]
                    self.plotatxy(items,img,x=foretext.shape[1],flag=flag,xpad=xpad,ypad=ypad,bgrcolor=bgrcolor,alfa=alfa)
                else:
                    self.plotatxy(items,img,flag=flag,xpad=xpad,ypad=ypad,bgrcolor=bgrcolor,alfa=alfa)
            else:
                self.plotatpointer(items,img,self.rx,self.ry,pixel=pixel,xpad=xpad,ypad=ypad,bgrcolor=bgrcolor,alfa=alfa)
        elif self._iswaiting:
            img = self.rimg.copy()
            self.plotintime(img=img)
        else:
            self.showfunc(self)  # show window

    def makeoverlay(self,items,xpad=0,ypad=0,bgrcolor = None,alfa = None):
        """
        overlay items over image.

        :param self:
        :param items:
        :param xpad:
        :param ypad:
        :param bgrcolor:
        :param alfa:
        :return:
        """
        def writetext(text,bgr=None):
            if bgr is None:
                bgr = self.textbackground
            textbox, baseLine = cv2.getTextSize(text, self.fontFace, self.fontScale, self.thickness)  # text dimensions
            bx=(textbox[0],int(textbox[1]*2))
            foretext = cv2.resize(bgr,bx)  # text image
            textOrg= (foretext.shape[1] - textbox[0])//2,(foretext.shape[0]//2 + textbox[1]//2)  # force int, center text
            cv2.putText(foretext,text, textOrg, self.fontFace, self.fontScale, self.textcolor,self.thickness)
            return foretext

        def pixelgraph(pixelval,bgr= None,m=None):
            if bgr is None:
                bgr = self.textbackground
            if m is None:
                textbox, baseLine = cv2.getTextSize("ss", self.fontFace, self.fontScale, self.thickness)  # text dimensions
                m=int(textbox[1]*2)
            n = m/6
            graph = cv2.resize(bgr,(m,m))  # text image
            graph[n:-n,n:-n] = convert2bgra(background(pixelval))[0,0]
            return graph

        def evaluate(imgs,items,bgr = None):
            if isinstance(items,basestring):
                imgs.append(writetext(items.format(self=self),bgr))
            elif isinstance(items,list):
                if items != []:
                    imgs.append([])
                    for i in range(len(items)):
                        evaluate(imgs[-1],items[i],bgr)
            elif isinstance(items,np.ndarray):
                imgs.append(items)
            elif items is not None:
                imgs.append(pixelgraph(items,bgr))

        imgs = []
        r = 0
        for i in range(len(items)): # rows
            if items[i]==[]: # discarding empty lists
                r+=1
            else:
                imgs.append([])
                for j in range(len(items[i])): # columns
                    if type(bgrcolor) is list:
                        evaluate(imgs[i-r],items[i][j],bgrcolor[i][j])
                    else:
                        evaluate(imgs[i-r],items[i][j],bgrcolor)

        if type(bgrcolor) is list:
            return padVH(imgs,ypad,xpad,bgrcolor[-1],alfa)
        else:
            return padVH(imgs,ypad,xpad,bgrcolor,alfa)


    def render2real(self, rx, ry, astype = np.int32):
        """
        from rendered coordinates get real coordinates.

        :param rx: rendered x
        :param ry: rendered y
        :param astype: (np.int32) return as the specified type
        :return: real x, real y
        """
        x = self.rx1+rx*(self.rx2-self.rx1)/self.rW
        y = self.ry1+ry*(self.ry2-self.ry1)/self.rH
        if astype: return astype(rx),astype(ry)
        return x,y

    def real2render(self, x, y, astype = None):
        """
        from real coordinates get rendered coordinates.

        :param x: real x
        :param y: real y
        :param astype: (np.int32) return as the specified type
        :return: rendered x, rendered y
        """
        rx = self.rW*(x-self.rx1)/(self.rx2-self.rx1)
        ry = self.rH*(y-self.ry1)/(self.ry2-self.ry1)
        if astype: return astype(rx),astype(ry)
        return rx,ry

    def save(self,strname=None,ext=".png",name="img"):
        """
        Save image (save image if not Qt backend is installed)
        :param strname: name to save, a label with {win} can be used to be replaced with the plot win name
        :param ext: (".png") extension.
        :param name: ("img") name of image object from self. default is "img" that is self.img
                    (it allows better control to get custom image)
        :return: True if saved, False if not saved (possibly because folder does not exists)
        """
        if not strname:
            strname = self.win
        elif "{win}" in strname:
            strname = strname.format(win=self.win)
        strname+=ext
        r = cv2.imwrite(strname,getattr(self,name))
        if FLAG_DEBUG and r: print(name, "from Plotim saved as",strname)
        return r

    # CONTROL METHODS
    def __setattr__(self, key, value):
        if self.limitrender:
            if key == "rx1":
                super(plotim,self).__setattr__("rx2", limitaxis(self.rx2,min(self.rWmax+value,self.maxX),value))
                super(plotim,self).__setattr__("rxZoom", self.rx2-value)
            if key == "rx2":
                super(plotim,self).__setattr__("rx1", limitaxis(self.rx1,value,max(value-self.rWmax,self.minX)))
                super(plotim,self).__setattr__("rxZoom", value-self.rx1)
            if key == "ry1":
                super(plotim,self).__setattr__("ry2", limitaxis(self.ry2,min(self.rHmax+value,self.maxY),value))
                super(plotim,self).__setattr__("ryZoom", self.ry2-value)
            if key == "ry2":
                super(plotim,self).__setattr__("ry1", limitaxis(self.ry1,value,max(value-self.rHmax,self.minY)))
                super(plotim,self).__setattr__("ryZoom", value-self.ry1)
        if key == "textbackground" or key == "errorbackground":
            try:
                value.copy()
            except:
                value = background(value)
        super(plotim,self).__setattr__(key, value)

def matchExplorer(win, img1, img2, kp_pairs=(), status = None, H = None, show=True, block= True, daemon=True):
    """
    This function draws a set of keypoint pairs obtained on a match method of a descriptor
    on two images imgf and imgb. (backend: Plotim).

    :param win: window's name (str)
    :param img1: image1 (numpy array)
    :param img2: image2 (numpy array)
    :param kp_pairs: zip(keypoint1, keypoint2)
    :param status: obtained from cv2.findHomography
    :param H: obtained from cv2.findHomography (default=None)
    :param show: if True shows Plotim using block and daemon, else do not show
    :param block: if True it wait for window close, else it detaches
    :param daemon: if True window closes if main thread ends, else windows must be closed to main thread to end
    :return: Plotim object with visualization as self.rimg (image with matching result) (default=None)

    .. note:: It supports BGR and gray images.
    """
    # FIXME keypoints visualization wrong
    # functions
    ## GET INITIAL VISUALIZATION
    if len(img1.shape)<3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)<3:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]  # obtaining image1 dimensions
    h2, w2 = img2.shape[:2]  # obtaining image2 dimensions
    # imgf and imgb will be visualized horizontally (left-right)
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)  # making visualization image
    vis[:h1, :w1] = img1  # imgf at the left of vis
    vis[:h2, w1:w1+w2] = img2  # imgf at the right of vis

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)  # making sure every pair of keypoints is graphed

    kp_pairs = [(dict2keyPoint(i),dict2keyPoint(j)) for i,j in kp_pairs]
    p1 = FLOAT([kpp[0].pt for kpp in kp_pairs])  # pair of coordinates for imgf
    p2 = FLOAT([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) # pair of coordinates for imgb

    if H is not None:# does the same as getTransformedCorners
        corners = FLOAT([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))

    def drawline(self):
        vis = self.rimg
        self.thick = int(sigmoid(vis.shape[0] * vis.shape[1], 1723567, 8080000, 5, 1))
        if H is not None:  # enclosing object
            rcorners = np.array([self.real2render(corner[0],corner[1]) for corner in corners],np.int32)
            cv2.polylines(vis, [rcorners], True, self.framecolor) # draw rendered TM encasing

        rp1,rp2 = [],[]
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            rx1,ry1 = self.real2render(x1,y1,np.int32) # real to render
            rx2,ry2 = self.real2render(x2,y2,np.int32) # real to render
            rp1.append((rx1,ry1))
            rp2.append((rx2,ry2))
            r = self.thick
            if inlier and self.showgoods:  # drawing circles (good keypoints)
                col = self.goodcolor
                cv2.circle(vis, (rx1, ry1), r, col, -1)  # for left keypoint (imgf)
                cv2.circle(vis, (rx2, ry2), r, col, -1)  # for right keypoint (imgf)
            elif self.showbads:  # drawing x marks (wrong keypoints)
                col = self.badcolor
                thickness = r+5
                # for left keypoint (imgf)
                cv2.line(vis, (rx1-r, ry1-r), (rx1+r, ry1+r), col, thickness)
                cv2.line(vis, (rx1-r, ry1+r), (rx1+r, ry1-r), col, thickness)
                # for right keypoint (imgf)
                cv2.line(vis, (rx2-r, ry2-r), (rx2+r, ry2+r), col, thickness)
                cv2.line(vis, (rx2-r, ry2+r), (rx2+r, ry2-r), col, thickness)
            # drawing lines for non-onmouse event
        self.rp1 = np.int32(rp1)
        self.rp2 = np.int32(rp2)
        self.vis0 = vis.copy()  # saving state of the visualization for onmouse event
        # get rendered kp_pairs
        self.kp_pairs2 = apply2kp_pairs(kp_pairs,self.real2render,self.real2render)
        # drawing lines for non-onmouse event
        for (rx1, ry1), (rx2, ry2), inlier in zip(rp1, rp2, status):
            if inlier and self.showgoods:
                cv2.line(vis, (rx1, ry1), (rx2, ry2), self.goodcolor,r)
        self.vis = vis#.copy() # visualization with all inliers

    def drawrelation(self):
        if self.flags & cv2.EVENT_FLAG_LBUTTON:
            x,y = self.rx, self.ry
            cur_vis = self.vis0.copy()  # actual visualization
            r = self.thick + 8  # proximity to keypoint
            m = (anorm(self.rp1 - (x, y)) < r) | (anorm(self.rp2 - (x, y)) < r)
            idxs = np.where(m)[0]  # get indexes near pointer
            kp1s, kp2s = [], []
            for i in idxs:  # for all keypints near pointer
                (rx1, ry1), (rx2, ry2) = self.rp1[i], self.rp2[i]  # my keypoint
                col = (self.badcolor, self.goodcolor)[status[i]]  # choosing False=red,True=green
                cv2.line(cur_vis, (rx1,ry1), (rx2,ry2), col, self.thick)  # drawing line
                # keypoints to show on event
                kp1, kp2 = self.kp_pairs2[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            # drawing keypoints near pointer for imgf and imgb
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=self.kpcolor)
            cur_vis = cv2.drawKeypoints(cur_vis, kp2s, flags=4, color=self.kpcolor)
            self.rimg = cur_vis
        else:
            self.rimg = self.vis

        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[self.y,self.x])

    def randomColor():
        return (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

    def mousefunc(self):
        if self.builtincontrol():
            self.updaterenderer()
            drawline(self)

        drawrelation(self)

    def keyfunc(self):
        if self.builtincmd():
            drawline(self)
            drawrelation(self)
            if self.y is not None and self.x is not None:
                self.builtinplot(self.img[self.y,self.x])
            else:
                self.builtinplot()

    self = plotim(win, vis)
    self.mousefunc = mousefunc
    self.keyfunc = keyfunc
    self.showgoods = True
    self.showbads = False
    from image import colors
    self.__dict__.update(colors)
    self.randomColor = randomColor
    self.goodcolor = self.green
    self.badcolor = self.red
    self.kpcolor = self.orange
    self.framecolor = self.blue
    self.cmdlist.extend(["showgoods","showbads","framecolor","kpcolor","badcolor","goodcolor"])
    drawline(self)
    # show window
    if show: self.show(block= block, daemon=daemon)
    return self #self.rimg # return coordinates

def explore_match(win, img1, img2, kp_pairs, status = None, H = None, show=True):
    """
    This function draws a set of keypoint pairs obtained on a match method of a descriptor
    on two images imgf and imgb. (backend: opencv).

    :param win: window's name (str)
    :param img1: image1 (numpy array)
    :param img2: image2 (numpy array)
    :param kp_pairs: zip(keypoint1, keypoint2)
    :param status: obtained from cv2.findHomography
    :param H: obtained from cv2.findHomography (default=None)
    :return: vis (image with matching result) (default=None)

    .. note:: It supports only gray images.
    """
    # colors to use
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)

    if len(img1.shape)<3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)<3:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]  # obtaining image1 dimensions
    h2, w2 = img2.shape[:2]  # obtaining image2 dimensions
    # imgf and imgb will be visualized horizontally (left-right)
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)  # making visualization image
    vis[:h1, :w1] = img1  # imgf at the left of vis
    vis[:h2, w1:w1+w2] = img2  # imgf at the right of vis
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # changing color attribute to background image

    if H is not None:  # enclosing object
        corners = FLOAT([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, red)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)  # making sure every pair of keypoints is graphed

    kp_pairs = [(dict2keyPoint(i),dict2keyPoint(j)) for i,j in kp_pairs]
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  # pair of coordinates for imgf
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) # pair of coordinates for imgb

    thick = int(sigmoid(vis.shape[0] * vis.shape[1], 1723567, 8080000, 5, 1))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:  # drawing circles (good keypoints)
            col = green
            cv2.circle(vis, (x1, y1), thick, col, -1)  # for left keypoint (imgf)
            cv2.circle(vis, (x2, y2), thick, col, -1)  # for right keypoint (imgf)
        else:  # drawing x marks (wrong keypoints)
            col = red
            r = thick
            thickness = thick
            # for left keypoint (imgf)
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            # for right keypoint (imgf)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()  # saving state of the visualization for onmouse event
    # drawing lines for non-onmouse event
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green,thick)

    if show:
        cv2.namedWindow(win,cv2.WINDOW_NORMAL) # Can be resized
        cv2.imshow(win, vis)  # show static image as visualization for non-onmouse event

        def onmouse(event, x, y, flags, param):
            cur_vis = vis  # actual visualization. lines drawed in it
            if flags & cv2.EVENT_FLAG_LBUTTON:  # if onmouse
                cur_vis = vis0.copy() # points and perspective drawed in it
                r = thick+8  # proximity to keypoint
                m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
                idxs = np.where(m)[0]  # get indexes near pointer
                kp1s, kp2s = [], []
                for i in idxs:  # for all keypints near pointer
                     (x1, y1), (x2, y2) = p1[i], p2[i]  # my keypoint
                     col = (red, green)[status[i]]  # choosing False=red,True=green
                     cv2.line(cur_vis, (x1, y1), (x2, y2), col,thick)  # drawing line
                     # keypoints to show on event
                     kp1, kp2 = kp_pairs[i]
                     kp1s.append(kp1)
                     kp2s.append(kp2)
                # drawing keypoints near pointer for imgf and imgb
                cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
                cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

            cv2.imshow(win, cur_vis)  # show visualization
        cv2.setMouseCallback(win, onmouse)
        cv2.waitKey()
        cv2.destroyWindow(win)
    return vis

class edger(plotim):
    """
    Test visualization for edges

    self.edge -> the edges in processed image
    self.img -> the processed image
    self.sample -> the rendered precessed image
    """
    def __init__(self,img,
                 isSIZE = True,
                 isEQUA = False,
                 isCLAHE = False,
                 isBFILTER = False):
        self.load(img,False)
        super(edger, self).__init__("Edger "+self.pathdata[2],self.data2)
        self._th1 = 3000
        self._th2 = 6000
        self._maxth = 10000
        self._showgray = False
        self._isSIZE = isSIZE
        self._isEQUA = isEQUA
        self._isCLAHE = isCLAHE
        self._isBFILTER = isBFILTER
        self._size = (400,400)
        self.windowfunc = edger.windowfunc
        self.edgecolor = (0, 255, 0)
        self.initname = "edge_"
        self.cmdlist.extend(["isSIZE","isEQUA","isBFILTER","isCLAHE","showgray"])
        # best guess: (50,100,10), opencv: (9,75,75), d=-1 is filter distance until sigma
        self.d,self.sigmaColor,self.sigmaSpace =10,20,20
        self.clipLimit,self.tileGridSize=2.0,(8,8)
        self.apertureSize,self.L2gradient= 7,True
        self.computeAll()

    def getParameters(self,params = ("d","sigmaColor","sigmaSpace","clipLimit","tileGridSize",
                                     "isSIZE","isEQUA","isCLAHE","isBFILTER","th1","th2","size",
                                     "apertureSize","L2gradient")):
        if isinstance(params,str):
            params = (params,)
        p = {}
        for param in params:
            p[param]=getattr(self,param)
        return p

    def save(self,strname=None,ext=".png",name="img"):
        if not strname:
            temp = "E"+str(self.th1)+"_"+str(self.th2)+"_"+str(self.apertureSize)+"_"+str(self.L2gradient)+"_"
            name = self.pathdata[:2]+self.initname+self.savename+[temp]+self.pathdata[2:]
            strname = "".join(name)
        return super(edger,self).save(strname,ext,name)

    def load(self,img,compute=True):
        if isinstance(img,str):
            from directory import getData
            self.pathdata = getData(img) # [drive,body,name,header]
            self.data2 = cv2.imread(img)
        else:
            self.pathdata = ["","","img",".png"] # [drive,body,name,header]
            self.data2 = img
        if compute: self.computeAll()

    def computeAll(self):
        self.savename = [""]
        if self.isSIZE:
            size = self._size
            img = cv2.resize(self.data2.copy(),size)
            self.savename.append("SIZE"+str(size[0])+"_"+str(size[1])+"_")
        else:
            img = self.data2.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if self.isEQUA:
            gray = cv2.equalizeHist(gray)
            self.savename.append("EQUA_")
        if self.isCLAHE:
            clipLimit,tileGridSize= self.clipLimit,self.tileGridSize
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            gray = clahe.apply(gray)
            self.savename.append("CLAHE"+str(clipLimit)+"_"+str(tileGridSize[0])+"_"+str(tileGridSize[1])+"_")
        if self.isBFILTER:
            d,sigmaColor,sigmaSpace = self.d,self.sigmaColor,self.sigmaSpace
            gray = bilateralFilter(gray,d,sigmaColor,sigmaSpace)
            self.savename.append("FILTER"+str(d)+"_"+str(sigmaColor)+"_"+str(sigmaSpace)+"_")
        self.bgr = img
        self.gray = gray
        if self.showgray:
            self.sample = self.gray.copy()
            self.data = cv2.cvtColor(self.sample,cv2.COLOR_GRAY2BGR)
        else:
            self.sample = self.bgr.copy()
            self.data = self.sample
        self.init()
        try:
            self.computeEdge()
        except:
            pass

    @property
    def size(self):
        return self.gray.shape
    @size.setter
    def size(self, value):
        if bool(self._size!=value): # on change
            self._size = value
            if self.isSIZE: self.computeAll()
    @size.deleter
    def size(self):
        del self._size

    @property
    def showgray(self):
        return self._showgray
    @showgray.setter
    def showgray(self, value):
        if bool(self._showgray!=value): # on change
            self._showgray = value
            if self._showgray:
                self.sample = self.gray.copy()
                self.data = cv2.cvtColor(self.sample,cv2.COLOR_GRAY2BGR)
            else:
                self.sample = self.bgr.copy()
                self.data = self.sample
            self.init()
            try:
                self.computeEdge()
            except:
                pass
    @showgray.deleter
    def showgray(self):
        del self._showgray

    @property
    def isSIZE(self):
        return self._isSIZE
    @isSIZE.setter
    def isSIZE(self, value):
        if bool(self._isSIZE!=value): # on change
            self._isSIZE = value
            self.computeAll()
    @isSIZE.deleter
    def isSIZE(self):
        del self._isSIZE

    @property
    def isEQUA(self):
        return self._isEQUA

    @isEQUA.setter
    def isEQUA(self, value):
        if bool(self._isEQUA!=value): # on change
            self._isEQUA = value
            self.computeAll()

    @isEQUA.deleter
    def isEQUA(self):
        del self._isEQUA

    @property
    def isCLAHE(self):
        return self._isCLAHE
    @isCLAHE.setter
    def isCLAHE(self, value):
        if bool(self._isCLAHE!=value): # on change
            self._isCLAHE = value
            self.computeAll()
    @isCLAHE.deleter
    def isCLAHE(self):
        del self._isCLAHE

    @property
    def isBFILTER(self):
        return self._isBFILTER
    @isBFILTER.setter
    def isBFILTER(self, value):
        if bool(self._isBFILTER!=value): # on change
            self._isBFILTER = value
            self.computeAll()
    @isBFILTER.deleter
    def isBFILTER(self):
        del self._isBFILTER

    @property
    def th1(self):
        return self._th1
    @th1.setter
    def th1(self, value):
        if bool(self._th1!=value): # on change
            self._th1 = value
            cv2.createTrackbar('th1', self.win, value, self.maxth, self.onTrackbar1)
            self.computeEdge()
    @th1.deleter
    def th1(self):
        del self._th1

    @property
    def th2(self):
        return self._th2
    @th2.setter
    def th2(self, value):
        if bool(self._th2!=value): # on change
            self._th2 = value
            cv2.createTrackbar('th2', self.win, value, self.maxth, self.onTrackbar2)
            self.computeEdge()
    @th2.deleter
    def th2(self):
        del self._th2

    @property
    def maxth(self):
        return self._maxth
    @maxth.setter
    def maxth(self, value):
        if bool(self._maxth!=value): # on change
            self._maxth = value
            if self.th1>value: self.th1 = value
            else: cv2.createTrackbar('th1', self.win, self.th1, self._maxth, self.onTrackbar1)
            if self.th2>value: self.th2 = value
            else: cv2.createTrackbar('th2', self.win, self.th2, self._maxth, self.onTrackbar2)
            self.computeEdge()
    @maxth.deleter
    def maxth(self):
        del self._maxth

    def computeEdge(self):
        edge = cv2.Canny(self.gray, self.th1, self.th2,apertureSize=self.apertureSize,L2gradient=self.L2gradient)
        vis = self.data.copy()
        vis[edge != 0] = self.edgecolor
        self.edge = edge
        self.img = vis
        self.updaterenderer()
        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[self.y,self.x])

    def onTrackbar1(self,*args):
        self.th1 = cv2.getTrackbarPos('th1', self.win)
    def onTrackbar2(self,*args):
        self.th2 = cv2.getTrackbarPos('th2', self.win)

    @staticmethod
    def windowfunc(self):
        cv2.namedWindow(self.win,self.wintype)  # create window
        cv2.createTrackbar('th1', self.win, self.th1, self.maxth, self.onTrackbar1)
        cv2.createTrackbar('th2', self.win, self.th2, self.maxth, self.onTrackbar2)
        cv2.resizeWindow(self.win,self.rW,self.rH)
        self.computeEdge()

if __name__ == "__main__":
    import argparse
    from serverServices import parseString as _parseString
    #import sys
    #if FLAG_DEBUG: print sys.argv
    parser = argparse.ArgumentParser(description='fast plot of images.')
    parser.add_argument('image', metavar='N', #action='append',
                        help='path to image or numpy string',nargs="+")
    parser.add_argument('-m','--cmap', dest='cmap', action='store',
                       help='map to use in matplotlib')
    parser.add_argument('-t','--title', dest='title', action='store',default="visualazor",
                       help='title of subplot')
    parser.add_argument('-w','--win', dest='win', action='store',
                       help='title of window')
    parser.add_argument('-n','--num', dest='num', action='store',type = int, default=0,
                       help='number of Figure')
    parser.add_argument('-f','--frames', dest='frames', action='store',type = int, default=None,
                       help='number of Figure')
    parser.add_argument('-b','--block', dest='block', action='store_true', default=False,
                       help='number of Figure')
    parser.add_argument('-d','--daemon', dest='daemon', action='store_true', default=False,
                       help='number of Figure')
    args = parser.parse_args()
    images = _parseString(args.image)
    wins[-1] = args.num
    for image in images:
        if type(image).__name__ == plotim.__name__: # pickled, so normal comparisons do not work
            image.show(args.frames, args.block, args.daemon)
        else:
            fastplt(image, args.cmap, args.title, args.win, args.block, args.daemon)
    if FLAG_DEBUG: print("leaving plotter module...")