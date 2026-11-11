# -*- coding: utf-8 -*-
"""
Interactive OpenCV viewer with pan, zoom, HUD overlays, and event loops.
"""

from multiprocessing import Process
import cv2
import numpy as np

from ..config import FLAG_DEBUG
from ..arrayops.basic import overlayXY, convertXY
from .helpers import limitaxis, convert2bgr
from .commands import CommandMixin
from .overlays import OverlayMixin
from .coordinates_mixin import CoordinateMixin

__all__ = ["Plotim"]


class Plotim(CommandMixin, OverlayMixin, CoordinateMixin):
    """
    Show an image with events, animations, controls, internal commands,
    and high runtime customizability.

    :param win: window name
    :param im: numpy array image
    :param bgrcolor: default color to use for background or transparency
    """

    def _init_clipboard(self):
        """Initialize clipboard support safely."""
        try:
            import pyperclip as clipboard
            return clipboard
        except Exception:
            try:
                import clipboard
                return clipboard
            except Exception:
                return None

    def _init_render_dimensions(self):
        """Initialize default display dimensions and font metrics."""
        self._rx1 = self._ry1 = self._rx2 = self._ry2 = 0
        self._textbackground = self._errorbackground = None
        self.usecontrol = False
        self.rW = 600
        self.rH = 400
        self.rWmax = self.rHmax = 5000
        self.limitrender = False
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = self.rH * 0.001
        self.thickness = int(self.fontScale) + 1
        self.textcolor = (0, 0, 0, 255)
        self.textbackground = (255, 255, 255, 100)
        self.errorbackground = (0, 0, 255, 100)
        self.coordinateText = [["xy({self.x},{self.y})"]]

    def _init_control_defaults(self):
        """Initialize flow control toggles and zoom parameters."""
        self.limitaxes = self.showcoors = self.showcontrol = True
        self.showpixel = self.showpixelvalue = True
        self.staticcoors = False
        self.controlText = [["zoom(x{self.rxZoom}({self.rx1}-{self.rx2}), y{self.ryZoom}({self.ry1}-{self.ry2})) "]]
        self.cmdlist = ["limitrender", "limitaxes", "showcoors", "showcontrol", "usecontrol", "toggle all",
                        "showpixel", "staticcoors", "showpixelvalue", "pixels", "cmdformatter"]
        self.cmdeval = {"pixels": ["showpixel", "showpixelvalue"],
                        "toggle all": ["showpixel", "showpixelvalue", "showcoors", "showcontrol"],
                        "reset": ["self.init()"],
                        "end": "self.updaterenderer()"}
        self.cmdbuffer, self.cmdcache, self.cmdfiltered = [], [], []
        self.cmdcache_choice = self.cmdfiltered_choice = 0
        self.delayplot = 1000
        self.zoomstep, self.minzoom = 4, 2
        self.controlkey = cv2.EVENT_FLAG_CTRLKEY
        self.zoominbutton, self.zoomoutbutton = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN
        self.movebutton, self.resetbutton = cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDBLCLK
        self._wait, self._iswaiting, self._oldt = 0, False, 0
        self._timeitems, self._timebgrcolor = [], None
        self._cmditmes, self._cmdbgrcolor = [], self.textbackground
        self.cmdformatter = True

    def __init__(self, win, im=np.array([[1]]), bgrcolor=(250, 243, 238)):
        self.clipboard = self._init_clipboard()
        self._init_render_dimensions()
        self.data = self.sample = im
        self.win = win
        self.wintype = cv2.WINDOW_NORMAL
        self.interpolation = cv2.INTER_LINEAR
        self.bgrcolor = bgrcolor
        self._init_control_defaults()
        self.init()

    def clean(self):
        """Clean object dictionary on window close."""
        for k in list(self.__dict__.keys()):
            try:
                del self.__dict__[k]
            except Exception:
                pass

    def init(self):
        """Pseudo __init__ resetting runtime defaults."""
        self.img = self.data.copy()
        temp = self.img.shape
        self.rxZoom = self.maxY = temp[0]
        self.ryZoom = self.maxX = temp[1]
        self.minY = self.minX = 0
        self.close = False
        self.pressedkey = None
        self.cmd = ""
        self.iscmd = self.cmderror = self.mousemoved = False
        self.event = self.flags = 0
        self.rx = self.ry = self.x = self.y = None
        self.rx2, self.ry2 = self.maxX, self.maxY
        self.rx1 = self.ry1 = 0
        self.updaterenderer()

    def help(self, showAll=False):
        """Print quick help listing available commands."""
        lines = []
        if self.__doc__:
            lines.extend(self.__doc__.splitlines())
        lines.append("\nList of command flags: ")
        lines.extend(self.cmdlist)
        lines.append("\nList of execution commands: ")
        lines.extend(list(self.cmdeval.keys()))
        if showAll:
            lines.append("\nList of hidden command flags:")
            lines.extend([k for k in dir(self) if isinstance(getattr(self, k, None), bool)])
        for line in lines:
            print(line)
        return lines

    def _show_loop(self, frames=None, clean=True):
        """Core waitKey event loop for display."""
        self.builtinwindow()
        while not self.close:
            self.pressedkey = cv2.waitKey(self.delayplot)
            if self.closefunc():
                break
            self.keyfunc()
            self.mousemoved = False
            if frames is not None:
                frames -= 1
                if frames < 1:
                    break
        cv2.destroyWindow(self.win)
        if clean:
            self.clean()

    def show(self, frames=None, block=True, daemon=False, clean=True):
        """Show window, handle keystrokes and close events."""
        if block:
            self._show_loop(frames, clean)
        else:
            p = Process(target=self._show_loop, args=(frames, clean))
            p.daemon = daemon
            p.start()
        return self

    def mousefunc(self):
        """Decoupled mouse callback logic."""
        if self.builtincontrol():
            self.updaterenderer()
        if self.y is not None and self.x is not None:
            self.builtinplot(self.sample[int(self.y), int(self.x)])

    def keyfunc(self):
        """Decoupled key callback logic."""
        if self.builtincmd():
            if self.y is not None and self.x is not None:
                self.builtinplot(self.sample[int(self.y), int(self.x)])
            else:
                self.builtinplot()

    def closefunc(self):
        """Check if ESC key pressed or window marked closed."""
        return self.pressedkey == 27 or self.close

    def windowfunc(self):
        """Create OpenCV window with default settings."""
        cv2.namedWindow(self.win, self.wintype)

    def showfunc(self, img=None):
        """Show current render or provided frame."""
        cv2.imshow(self.win, self.rimg if img is None else img)

    def updaterenderer(self, img=None, zoom=True):
        """Update rendered frame dimensions, zoom slice, and control text."""
        source = self.img if img is None else img
        sub = source[self.ry1:self.ry2, self.rx1:self.rx2].copy() if zoom else source.copy()
        self.rimg = cv2.resize(convert2bgr(sub, self.bgrcolor),
                               (self.rW, self.rH), interpolation=self.interpolation)
        if self.showcontrol:
            foretext = self.makeoverlay(self.controlText, bgrcolor=self.textbackground)[0]
            overlayXY(*convertXY(0, 0, self.rimg.shape, foretext.shape, 1),
                      back=self.rimg, fore=foretext)

    @staticmethod
    def onmouse(event, x, y, flags, self):
        """OpenCV mouse event callback bridge."""
        if self.rx != x or self.ry != y:
            self.mousemoved = True
        self.rx, self.ry = x, y
        corr_x = self.rx1 + x * (self.rx2 - self.rx1) / self.rW
        corr_y = self.ry1 + y * (self.ry2 - self.ry1) / self.rH
        if self.limitaxes:
            self.x = limitaxis(corr_x, self.rx2 - 1, self.rx1)
            self.y = limitaxis(corr_y, self.ry2 - 1, self.ry1)
        else:
            self.x, self.y = corr_x, corr_y
        self.event = event
        self.flags = flags
        self.mousefunc()

    def builtinwindow(self):
        """Initialize window, initial display, and bind mouse callback."""
        self.windowfunc()
        self.showfunc()
        cv2.startWindowThread()
        cv2.setMouseCallback(self.win, self.onmouse, self)

    def _handle_zoom(self):
        """Handle zoom in and zoom out mouse clicks."""
        mx = my = self.minzoom
        if self.event == self.zoominbutton:
            if self.rxZoom != mx or self.ryZoom != my:
                xdiff = np.min((self.rxZoom, self.ryZoom)) / self.zoomstep + 1
                return True, xdiff, xdiff
        elif self.event == self.zoomoutbutton:
            if self.rxZoom != self.maxX or self.ryZoom != self.maxY:
                xdiff = np.max((self.rxZoom, self.ryZoom)) * self.zoomstep + 1
                return True, xdiff, xdiff
        elif self.event == self.movebutton:
            return True, self.rxZoom / 2, self.ryZoom / 2
        return False, 0, 0

    def builtincontrol(self, control=False):
        """Internal pan/zoom control logic."""
        if not self.usecontrol:
            return False
        ctrl = self.controlkey if not control else control
        if ctrl is not None and not (self.flags & ctrl):
            return False

        if self.event == self.resetbutton:
            self.init()
            return True

        controlled, xdiff, ydiff = self._handle_zoom()
        if controlled:
            mx, my = self.minzoom, self.minzoom
            self.rx2 = limitaxis(self.x + xdiff, self.maxX, self.minX + mx)
            self.rx1 = limitaxis(self.x - xdiff, self.rx2 - mx, self.minX)
            self.ry2 = limitaxis(self.y + ydiff, self.maxY, self.minY + my)
            self.ry1 = limitaxis(self.y - ydiff, self.ry2 - my, self.minY)
        return controlled

    def save(self, strname=None, ext=".png", name="img"):
        """Save active image frame to disk."""
        target_name = self.win if not strname else (strname.format(win=self.win) if "{win}" in strname else strname)
        target_name += ext
        saved = cv2.imwrite(target_name, getattr(self, name))
        if FLAG_DEBUG and saved:
            print(name, "from Plotim saved as", target_name)
        return saved