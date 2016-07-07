"""
preamble: use it as base for all the graphs
"""
# http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
# http://matplotlib.org/users/usetex.html
# http://matplotlib.org/users/pgf.html
# http://matplotlib.1069221.n5.nabble.com/LaTeX-text-processing-using-quot-LaTeX-low-level-font-commands-quot-in-texmanager-py-amp-backend-ps-y-td1884.html

# https://www.sharelatex.com/learn/Pgfplots_package
# http://pgfplots.sourceforge.net/gallery.html

#import matplotlib as mpl
#mpl.use("pgf")
#mpl.rc('text', usetex=True)
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#mpl.rc('font',**{'family':'serif','serif':['Palatino']})
import cv2
import numpy as np
import os
print("working dir: {}".format(os.getcwd()))
from matplotlib.pyplot import *
#from sympy import *
root_path = os.path.dirname(os.path.abspath(__file__))

# simulates pytex.context
try:
    pytex.context
except:
    class Pytex:
        def __init__(self):
            self.context = dict(font = "\T1/cmr/m/n/10",shape = "200/200",ext = "pdf",path = "",fn = 'patterns.png')
    pytex = Pytex()

default_context = pytex.context # uses available context

def scape_string(string, listing = "_"):
    lstr = list(string)
    for i,letter in enumerate(lstr):
        if letter in listing:
            lstr[i] = "\\"+lstr[i]
    return "".join(lstr)
    #return "\_".join(string.split("_"))

def xylim_points(points, r=0.1, labellen=0):
    xmin,ymin = np.min(points,0)
    xmax,ymax = np.max(points,0)
    yr = (ymax-ymin)*r
    xr = (xmax-xmin)*r
    ylim([ymin-yr,ymax+yr])
    xlim([xmin-xr,xmax+xr+labellen*xr*0.15])

def points_generator(shape = (10,10), nopoints = None, convex = False, erratic = False, draw = False):
    from RRtoolbox.lib.arrayops.basic import points_generator
    from RRtoolbox.lib.image import getcoors, drawcoorpolyArrow
    if len(shape)>2:
        nopoints = shape[2]
    if draw:
        pts = getcoors(np.ones(shape[:2]),"get pixel coordinates", updatefunc=drawcoorpolyArrow)
    else:
        pts = points_generator(shape=shape,nopoints=nopoints,convex=convex,erratic=erratic)
    return np.array(pts,dtype=np.int32)

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def generateGrid(num,cols=3, invert = False):
    """
    returns an instance of GridSpec.

    :param num: number of figures
    :param cols: number of columns
    :return: gs -> GridSpec instance, use as: ax = fig.add_subplot(gs[i])
    """
    # create grid
    rows = num//cols
    if rows == 0:
        rows = 1
        cols = num
    from matplotlib import gridspec
    if invert:
        return gridspec.GridSpec(cols,rows) # grid for each plot
    return gridspec.GridSpec(rows, cols) # grid for each plot

class LatexGraph(object):
    """
    representation of a subfigure in latex. This class is used in
    :class:`LatexMultipleGraph`.

    .. notes::

        needs to import the subcaption package as \usepackage{subcaption}
    """
    def __init__(self, filename, caption = None, label = None,
                 width = None, position = None):
        """

        :param filename:
        :param caption:
        :param label:
        :param width:
        :param position:
        :return:
        """
        self.filename = filename
        self.caption = caption
        self.label = label
        self.width = width # width of image
        self.position = position # position of subfigure

    @property
    def width(self):
        if self._width is None:
            return r"\textwidth"
        if isinstance(self._width,str):
            return self._width
        return r"{}\textwidth".format(self._width)
    @width.setter
    def width(self,value):
        self._width = value
    @width.deleter
    def width(self):
        self._width = None

    @property
    def position(self):
        if self._width is None:
            return r"t"
        return self._position
    @position.setter
    def position(self,value):
        self._position = value
    @position.deleter
    def position(self):
        self._position = None

    @property
    def filename(self):
        return self._filename
    @filename.setter
    def filename(self,value):
        self._filename = value
    @filename.deleter
    def filename(self):
        self._filename = None

    @property
    def caption(self):
        if self._caption is None:
            return ""
        return self._caption
    @caption.setter
    def caption(self,value):
        self._caption = value
    @caption.deleter
    def caption(self):
        self._caption = None

    @property
    def label(self):
        if self._label is None:
            return ""
        return self._label
    @label.setter
    def label(self,value):
        self._label = value
    @label.deleter
    def label(self):
        self._label = None

    @property
    def formatted_caption(self):
        c,l = self.caption,self.label
        if c == "" and l == "":
            return ""
        if c == "":
            return r"\label{%s}" % (l)
        if l == "":
            return r"\caption{%s}" % (c)
        return r"\caption{%s\label{%s}}" % (c,l)

    def __str__(self):
        space = "   "
        begin = r"\begin{subfigure}[%s]{%s}" % (self.position,self.width)
        fig = r"\includegraphics[width=\textwidth]{%s}" % (self.filename)
        cap = r"%s" % (self.formatted_caption)
        if cap != "":
            cap = "\n{}{}".format(space,cap)
        end = r"\end{subfigure}"
        return "{}\n{}{}{}\n{}".format(begin,space,fig,cap,end)

class LatexMultipleGraph(object):
    """

    .. notes::

        * needs to import the subcaption package as \usepackage{subcaption}
            to use subfigure environment.
        * needs to import the caption package as \usepackage{caption}
            to use captionbox environment.
        * Similar functionality can be found in pylatex python package
            https://jeltef.github.io/PyLaTeX/latest/examples/subfigure.html
    """
    def __init__(self, graphs, caption = None, label = None,
                 position = None, subpositions = None, num_rows = None):
        """

        :param graphs: list of LatexGraph objects
        :param caption:
        :param label:
        :param position:
        :param subpositions:
        :param num_rows:
        :return:
        """
        self._graphs = None
        self.row_limit = 3
        self.num_rows = num_rows
        self.caption = caption
        self.label = label
        self.position = position # position of subfigure
        self.subpositions = subpositions
        self.graphs = graphs

    @property
    def position(self):
        if self._position is None:
            return r"h"
        return self._position
    @position.setter
    def position(self,value):
        self._position = value
    @position.deleter
    def position(self):
        self._position = None

    @property
    def caption(self):
        if self._caption is None:
            return ""
        return self._caption
    @caption.setter
    def caption(self,value):
        self._caption = value
    @caption.deleter
    def caption(self):
        self._caption = None

    @property
    def label(self):
        if self._label is None:
            return ""
        return self._label
    @label.setter
    def label(self,value):
        self._label = value
    @label.deleter
    def label(self):
        self._label = None

    @property
    def formatted_caption(self):
        c,l = self.caption,self.label
        if c == "" and l == "":
            return ""
        if c == "":
            return r"\label{%s}" % (l)
        if l == "":
            return r"\captionbox{%s}[\textwidth][c]{%s}" % (c,"\n%s\n")
        return r"\captionbox{%s\label{%s}}[\textwidth][c]{%s}" % (c,l,"\n%s\n")

    @property
    def num_rows(self):
        return self._num_rows
    @num_rows.setter
    def num_rows(self,value):
        if self.graphs and value is None:
            # calculate num_rows
            self.calculate_rows(self.graphs)
        else:
            self._num_rows = value
    @num_rows.deleter
    def num_rows(self):
        del self._num_rows

    @property
    def graphs(self):
        return self._graphs
    @graphs.setter
    def graphs(self,value):
        if value is not None:
            if self.check_plain(value):
                value = self.formatted_graphs(value)
            elif not self.check_formatted(value):
                raise Exception("graphs are not well formated")

        self._graphs = value

    @graphs.deleter
    def graphs(self):
        del self._graphs

    def calculate_rows(self, graphs):
        """
        calculates self.num_rows according to graphs.
        """
        if len(graphs)<self.row_limit:
            self._num_rows = len(graphs)
        else:
            self._num_rows = self.row_limit

    def check_formatted(self, graphs = None, position = False):
        """
        checks that list of graphs is well formatted.

        :return: If position is False gives True if check is passed, False if not.
                Else if position is True, gives the index at which the check failed.
        """
        if graphs is None:
            graphs = self.graphs
        for i, graph_row in enumerate(graphs):
            for j, graph in enumerate(graph_row):
                if not isinstance(graph,LatexGraph):
                    if position:
                        return i,j
                    return False
        if position:
            return
        return True

    def check_plain(self, graphs = None, position = False):
        """
        checks that list of graphs is plain.

        :return: If position is False gives True if check is passed, False if not.
                Else if position is True, gives the index at which the check failed.
        """
        if graphs is None:
            graphs = self.graphs
        for i, graph in enumerate(graphs):
            if not isinstance(graph,LatexGraph):
                if position:
                    return i
                return False
        if position:
            return
        return True

    def formatted_graphs(self, graphs = None):
        """
        formats list of graphs assuming that they are not
        formatted but plain.

        :param graphs: plain list of graphs
        :return: formatted list of graphs
        """
        if graphs is None:
            graphs = self.graphs

        # calculate num_rows if it is None
        if self.num_rows is None and graphs is not None:
            self.calculate_rows(graphs)

        num = len(graphs) # get limit of graphs
        rows = [] # initialize rows
        i = 0 # index counter
        while num>0:
            cols = [] # initialize columns

            # maximum number of columns
            if num > self.num_rows:
                ncols = self.num_rows
            else:
                ncols = num # remaining

            # append number of columns
            for _ in xrange(ncols):
                cols.append(graphs[i])
                num -= 1 # decrease graphs
                i += 1 # increase index

            rows.append(cols)

        return rows


    def plain_graphs(self, graphs = None):
        """
        gives list of graphs assuming that they are formatted.

        :param graphs: formatted list of graphs
        :return: plain list of graphs
        """
        if graphs is None:
            graphs = self.graphs


        plain = []
        # Method: walk lists
        for graph_row in graphs:
            for graph in graph_row:
                plain.append(graph)

        return plain

    @property
    def str_graphs(self):
        rows = [] # init list of string for rows
        # process rows
        for graph_row in self.graphs:
            # width of each image in a row
            width = 1.0/len(graph_row)
            cols = [] # init list of string for columns

            # process each image in the row
            for graph in graph_row:
                graph.width = width # assign width to graph

                # if global position assign it
                if self.subpositions is not None:
                    graph.position = self.subpositions

                # convert to the string
                cols.append(str(graph))

            # append side by side using r'~'
            rows.append('\n~\n'.join(cols))

        # append under each row using r'\\'
        return '\n\\\\\n'.join(rows)

    def __str__(self):
        begin = r"\begin{figure}[%s]" % (self.position)
        cap = r"%s" % (self.formatted_caption)
        if cap == "":
            cap = self.str_graphs
        else:
            cap = cap % (self.str_graphs)
        end = r"\end{figure}"
        return "{}\n{}\n{}".format(begin,cap,end)

class graph_data(object):
    shows = True
    saves = False
    onlyshow = False
    def __init__(self, pytex = None, unicode =True):
        rcParams['text.latex.unicode'] = unicode
        self.shape=None
        self.ext=None
        self._path=None
        self.fn=None
        self._RGB=None
        self._RGBA = None
        self._gray=None
        self._BGRA=None
        self._BGR=None
        self.showmode = False
        self.context_data = None
        self.context = None
        self.savelog = []
        self.captions = {}
        self.config(pytex)

    @property
    def path(self):
        return self._path
    @path.setter
    def path(self,value):
        try:
            if value[-1] not in ("/","\\"): # ensures path
                value += "/"
        except:
            pass
        self._path = value
    @path.deleter
    def path(self):
        del self._path

    @property
    def BGRA(self):
        if self._BGRA is None:
            self.load()
        return self._BGRA
    @BGRA.setter
    def BGRA(self,value):
        self._BGRA = value
    @BGRA.deleter
    def BGRA(self):
        self._BGRA = None

    @property
    def BGR(self):
        if self._BGR is None:
            self.load()
        return self._BGR
    @BGR.setter
    def BGR(self,value):
        self._BGR = value
    @BGR.deleter
    def BGR(self):
        self._BGR = None

    @property
    def RGB(self):
        if self._RGB is None:
            self._RGB = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2RGB)
        return self._RGB
    @RGB.setter
    def RGB(self,value):
        self._RGB = value
    @RGB.deleter
    def RGB(self):
        self._RGB = None

    @property
    def RGBA(self):
        if self._RGBA is None:
            self._RGBA = cv2.cvtColor(self.BGRA, cv2.COLOR_BGRA2RGBA)
        return self._RGBA
    @RGBA.setter
    def RGBA(self,value):
        self._RGBA = value
    @RGBA.deleter
    def RGBA(self):
        self._RGBA = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2GRAY)
        return self._gray
    @gray.setter
    def gray(self,value):
        self._gray = value
    @gray.deleter
    def gray(self):
        self._gray = None

    def savefig(self, name, caption = None, options = None):
        """
        save to path with extension
        :param name: archive name with no extensions
        :param caption: optional title for captions
        :param options:
        :return:
        """
        if options is None: options = dict(bbox_inches='tight',transparent=True)
        saveto = '{}{}.{}'.format(self.path, name, self.ext)
        print("saving to: {}".format(saveto))
        savefig(saveto, **options)
        self.savelog.append((self.path, name, self.ext)) # keep history of saved images
        self.captions[name] = caption
        # to save pgf install xelatex, in linux sudo apt-get install texlive-xetex

    def output(self, name, caption = None, options = None):
        """
        convenience function used to replace save or show
        :param name: file name with no extensions
        :param caption: optional title for captions
        :param options: options to pass to matplotlib savefig
        :return:
        """
        if graph_data.shows or graph_data.onlyshow:
            show()
        if not graph_data.onlyshow and graph_data.saves:
            self.savefig(name,caption,options)

    def load(self, fn = None, shape = None, path = None):
        if fn is None:
            fn = self.fn
        from RRtoolbox.lib.image import try_loads
        if path is None: path = self.path
        if path is None: path = ""
        img = try_loads([fn],paths=[path], func= lambda x:cv2.imread(x,-1))
        if img is None:
            raise Exception("Image not Loaded")
        if shape is None: shape= self.shape
        if shape is not None: img = cv2.resize(img,shape)

        self._RGB=None
        self._RGBA = None
        self._gray=None

        if img.shape[2] == 3:
            self.BGR = img
            self.BGRA = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        else:
            self.BGRA = img
            self.BGR = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

    def config(self, context = None):
        """
        Loads from context to configurate paramenters
        :param context: pytex from pythontex
        :return:
        """
        if context is None:
            context = default_context
        else:
            for k,v in default_context.iteritems(): # ensures all data to configure
                if k not in context:
                    context[k]=v
        import re
        self.context = context
        font = context.get("font") # font from Latex e.g. \T1/cmr/m/n/10 i.e. \encoding/family/series/shape/size
        shape = context.get("shape")
        ext = context.get("ext") # extension to save images
        path = context.get("path") # path to save images
        fn = context.get("fn") # to load sample image
        pattern = r"\\(?P<encoding>.*)/(?P<family>.*)/(?P<series>.*)/(?P<shape>.*)/(?P<size>.*)"
        context_data = re.match(pattern,font,re.I).groupdict()
        context_data["size"] = float(context_data["size"]) # convert font size to desired float
        shape = tuple([int(i) for i in shape.split("/")]) # convert to usable shape
        self.context_data = context_data
        rc('text', usetex=True)
        rc('font', family=context_data["family"])#serif, Times New Roman, cmr
        rc('font', size=context_data["size"])
        rc('legend', fontsize=context_data["size"])
        rc('font', weight='normal')
        if not ext: raise Exception("ext must not be empty")
        self.shape = shape
        self.ext = ext
        self.path = path
        self.fn = fn

    def wrap_title(self, text, width=None, **kwargs):
        """
        Adequate title by inserting newlines according to width.

        :param text: title text
        :param width: (None) columns width, if None it calculates the width
        :param kwargs: fig (current fig), ax (current axis), adjust (1)
                Ex: adjust = 2 calculates width from current font size as it was double size.
        :return: adecuated title
        """
        import textwrap
        if width is None:
            fig =None
            if "fig" in kwargs: fig = kwargs.pop("fig")
            ax =None
            if "ax" in kwargs: ax = kwargs.pop("ax")
            adjust = 1 # fonts are not precise thus this is to adjust the conversion
            if "adjust" in kwargs: adjust = kwargs.pop("adjust")
            width = self.get_ax_size(fig=fig,ax=ax)[0] # in inches
            width = int(width * adjust * 1.8 * 72.27 / self.context_data["size"]) # to columns of letters
            #size = self.context_data["size"]*1/72.27
        return "\n".join(textwrap.wrap(text, width, **kwargs))

    @staticmethod
    def get_ax_size(fig=None,ax=None, units = "inch"):
        """
        Axes dimensions
        :param fig: matplotlib figure handle
        :param ax: matplotlib axis handle
        :param units: "inches","pixels"
        :return: width, height
        """
        # http://stackoverflow.com/a/19306776/5288758
        if fig is None: fig = gcf()
        if ax is None: ax = gca()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        units = units.lower()
        if units in ("pixels","pixel"):
            width *= fig.dpi
            height *= fig.dpi
        return width, height
# figure(figsize=(mm2inch(163,45))) # 163, 45 mm
# rcParams['text.latex.unicode'] = True
# suptitle("my super title", fontsize= gd.context_data["size"]).set_y(0.8)
# wrap long titles http://stackoverflow.com/a/10634897/5288758
#opts = dict(facecolor=fig.get_facecolor(), edgecolor='none',
#            bbox_inches='tight',transparent=True)
#graph_data.shows = True
#graph_data.saves = False