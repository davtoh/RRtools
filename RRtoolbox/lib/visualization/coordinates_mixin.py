# -*- coding: utf-8 -*-
"""
Coordinate transformation and property encapsulation mixin for Plotim.
"""

import numpy as np
from .helpers import background, limitaxis

__all__ = ["CoordinateMixin"]


class CoordinateMixin(object):
    """Encapsulates window coordinate bounds, backgrounds, and pixel projection."""

    @property
    def textbackground(self):
        return self._textbackground

    @textbackground.setter
    def textbackground(self, value):
        try:
            value = value.copy()
        except Exception:
            value = background(value)
        self._textbackground = value

    @textbackground.deleter
    def textbackground(self):
        del self._textbackground

    @property
    def errorbackground(self):
        return self._errorbackground

    @errorbackground.setter
    def errorbackground(self, value):
        try:
            value = value.copy()
        except Exception:
            value = background(value)
        self._errorbackground = value

    @errorbackground.deleter
    def errorbackground(self):
        del self._errorbackground

    @property
    def rx1(self):
        return self._rx1

    @rx1.setter
    def rx1(self, value):
        if self.limitrender:
            self._rx2 = limitaxis(self.rx2, min(self.rWmax + value, self.maxX), value)
            self.rxZoom = self.rx2 - value
        self._rx1 = value

    @rx1.deleter
    def rx1(self):
        del self._rx1

    @property
    def ry1(self):
        return self._ry1

    @ry1.setter
    def ry1(self, value):
        if self.limitrender:
            self._ry2 = limitaxis(self.ry2, min(self.rHmax + value, self.maxY), value)
            self.ryZoom = self.ry2 - value
        self._ry1 = value

    @ry1.deleter
    def ry1(self):
        del self._ry1

    @property
    def rx2(self):
        return self._rx2

    @rx2.setter
    def rx2(self, value):
        if self.limitrender:
            self._rx1 = limitaxis(self.rx1, value, max(value - self.rWmax, self.minX))
            self.rxZoom = value - self.rx1
        self._rx2 = value

    @rx2.deleter
    def rx2(self):
        del self._rx2

    @property
    def ry2(self):
        return self._ry2

    @ry2.setter
    def ry2(self, value):
        if self.limitrender:
            self._ry1 = limitaxis(self.ry1, value, max(value - self.rHmax, self.minY))
            self.ryZoom = value - self.ry1
        self._ry2 = value

    @ry2.deleter
    def ry2(self):
        del self._ry2

    def render2real(self, rx, ry, astype=np.int32):
        """Convert render coordinates to real coordinates."""
        x = self.rx1 + rx * (self.rx2 - self.rx1) / self.rW
        y = self.ry1 + ry * (self.ry2 - self.ry1) / self.rH
        return (astype(rx), astype(ry)) if astype else (x, y)

    def real2render(self, x, y, astype=None):
        """Convert real coordinates to render coordinates."""
        rx = self.rW * (x - self.rx1) / (self.rx2 - self.rx1)
        ry = self.rH * (y - self.ry1) / (self.ry2 - self.ry1)
        return (astype(rx), astype(ry)) if astype else (rx, ry)

