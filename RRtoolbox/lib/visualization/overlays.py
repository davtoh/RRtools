# -*- coding: utf-8 -*-
"""
Overlay generation and on-screen HUD drawing mixin for Plotim.
"""

import copy
import time
import cv2
import numpy as np

from ..arrayops.basic import overlayXY, convertXY, padVH, isnumpy
from .helpers import background, convert2bgra

__all__ = ["OverlayMixin"]


class OverlayMixin(object):
    """Mixin providing text rendering, HUD overlay generation, and pointer plots."""

    def _writetext(self, text, bgr=None):
        """Place text string into an image badge."""
        if bgr is None:
            bgr = self.textbackground
        textbox, _ = cv2.getTextSize(text, self.fontFace, self.fontScale, self.thickness)
        bx = (textbox[0], int(textbox[1] * 2))
        foretext = cv2.resize(bgr, bx)
        textOrg = ((foretext.shape[1] - textbox[0]) // 2,
                   (foretext.shape[0] // 2 + textbox[1] // 2))
        cv2.putText(foretext, text, textOrg, self.fontFace,
                    self.fontScale, self.textcolor, self.thickness)
        return foretext

    def _pixelgraph(self, pixelval, bgr=None, m=None):
        """Draw a colored swatch rectangle representing pixel color."""
        if bgr is None:
            bgr = self.textbackground
        if m is None:
            textbox, _ = cv2.getTextSize("ss", self.fontFace, self.fontScale, self.thickness)
            m = int(textbox[1] * 2)
        n = int(m / 6)
        graph = cv2.resize(bgr, (m, m))
        graph[n:-n, n:-n] = convert2bgra(background(pixelval))[0, 0]
        return graph

    def _evaluate_overlay_item(self, imgs, items, bgr=None):
        """Transform raw items (text, array, list, swatch) into image badges."""
        if isinstance(items, (str, bytes)):
            text = items.decode("utf-8") if isinstance(items, bytes) else items
            imgs.append(self._writetext(text.format(self=self), bgr))
        elif isnumpy(items):
            imgs.append(items)
        elif isinstance(items, list):
            if items:
                imgs.append([])
                for sub in items:
                    self._evaluate_overlay_item(imgs[-1], sub, bgr)
        elif items is not None:
            try:
                imgs.append(self._pixelgraph(items, bgr))
            except Exception:
                imgs.append(self._writetext("{}".format(items), bgr))

    def makeoverlay(self, items, xpad=0, ypad=0, bgrcolor=None, alpha=None):
        """Overlay items in rows and columns over image."""
        imgs = []
        r = 0
        for i in range(len(items)):
            if items[i] == []:
                r += 1
            else:
                imgs.append([])
                for j in range(len(items[i])):
                    col = bgrcolor[i][j] if isinstance(bgrcolor, list) else bgrcolor
                    self._evaluate_overlay_item(imgs[i - r], items[i][j], col)

        bg = bgrcolor[-1] if isinstance(bgrcolor, list) else bgrcolor
        return padVH(imgs, ypad, xpad, bg, alpha)

    def _determine_quartile(self, x, y, img_shape, pixel, text):
        """Determine quartile position and insert pixel swatch."""
        rH, rW = img_shape[:2]
        if x > rW / 2:
            quartile = 2 if y > rH / 2 else 3
            if pixel is not None:
                text[0].append(pixel)
        else:
            quartile = 1 if y > rH / 2 else 4
            if pixel is not None:
                text[0].insert(0, pixel)
        return quartile

    def plotatpointer(self, items, img=None, x=0, y=0, flag=6, xpad=0, ypad=0,
                      bgrcolor=None, alpha=None, pixel=None):
        """Plot HUD message where mouse pointer is positioned."""
        text = copy.deepcopy(items)
        target_img = self.rimg if img is None else img
        if x is not None and y is not None:
            quartile = self._determine_quartile(x, y, (self.rH, self.rW), pixel, text)
            foretext = self.makeoverlay(text, xpad, ypad, bgrcolor, alpha)[0]
            tx, ty = convertXY(x, y, target_img.shape, foretext.shape, flag, quartile=quartile)
        else:
            if pixel is not None:
                text[0].append(pixel)
            foretext = self.makeoverlay(text, xpad, ypad, bgrcolor, alpha)[0]
            tx, ty = convertXY(0, 0, target_img.shape, foretext.shape, 4)

        self.showfunc(overlayXY(tx, ty, back=target_img, fore=foretext))

    def plotatxy(self, items, img=None, x=0, y=0, flag=0, xpad=0, ypad=0, bgrcolor=None, alpha=None):
        """Plot message at specific xy coordinates."""
        target_img = self.rimg if img is None else img
        foretext = self.makeoverlay(items, xpad, ypad, bgrcolor, alpha)[0]
        tx, ty = convertXY(x, y, target_img.shape, foretext.shape, flag)
        self.showfunc(overlayXY(tx, ty, back=target_img, fore=foretext))

    def plotintime(self, items=None, wait=2, img=None, bgrcolor=None):
        """Plot timed transient messages and alerts."""
        if items is None:
            if img is not None and self._iswaiting:
                if self._wait > time.time() - self._oldt:
                    if len(self._timeitems) > 6:
                        self._timeitems = self._timeitems[1:]
                    self.plotatxy(self._timeitems, img, flag=4, bgrcolor=self._timebgrcolor)
                else:
                    self._timeitems = []
                    self._iswaiting = False
        else:
            self._wait = wait
            self._iswaiting = True
            self._oldt = time.time()
            self._timeitems.extend(items)
            self._timebgrcolor = bgrcolor

    def _format_pixel_value(self, pixel):
        """Convert pixel color value to tuple or scalar string."""
        if pixel is not None and not isinstance(pixel, int):
            return tuple(pixel) if getattr(pixel, "shape", None) != () else int(pixel)
        return pixel

    def builtinplot(self, pixel=None, useritems=None, flag=1, xpad=0, ypad=0, bgrcolor=None, alpha=None):
        """Internal plot renderer handling coordinates, commands, and pixel readout."""
        items = copy.deepcopy(useritems) if useritems is not None else (copy.deepcopy(self.coordinateText) if self.showcoors else [[]])
        if pixel is not None and self.showpixelvalue:
            items[0].extend([str(pixel)])
        pixel = self._format_pixel_value(pixel) if self.showpixel else None

        if self.iscmd:
            if self._cmditmes:
                items.extend(self._cmditmes)
            bgrcolor = self._cmdbgrcolor

        if items != [[]] or pixel is not None:
            img = self.rimg.copy()
            self.plotintime(img=img)
            if self.staticcoors:
                if pixel is not None:
                    items[0].insert(0, pixel)
                x_pos = self.makeoverlay(self.controlText, xpad, ypad, bgrcolor, alpha)[0].shape[1] if (self.showcontrol and flag == 1) else 0
                self.plotatxy(items, img, x=x_pos, flag=flag, xpad=xpad, ypad=ypad, bgrcolor=bgrcolor, alpha=alpha)
            else:
                self.plotatpointer(items, img, self.rx, self.ry, pixel=pixel, xpad=xpad, ypad=ypad, bgrcolor=bgrcolor, alpha=alpha)
        elif self._iswaiting:
            self.plotintime(img=self.rimg.copy())
        else:
            self.showfunc()

