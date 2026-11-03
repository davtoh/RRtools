# -*- coding: utf-8 -*-
"""
Unit conversion, magnitude formatting, and metric scaling factors.
"""
import numpy as np

__all__ = ["FactorConvert", "Magnitude"]


class FactorConvert:
    """
    Keep track of factor and converts to any available factor.
    """
    _factors = (("exa", "E", 1000000000000000000),
                ("peta", "P", 1000000000000000),
                ("tera", "T", 1000000000000),
                ("giga", "G", 1000000000),
                ("mega", "M", 1000000),
                ("kilo", "k", 1000),
                ("hecto", "h", 100),
                ("deca", "da", 10),
                (None, "", 1),
                ("deci", "d", 0.1),
                ("centi", "c", 0.01),
                ("milli", "m", 0.001),
                ("micro", "u", 0.000001),
                ("nano", "n", 0.000000001),
                ("pico", "p", 0.000000000001),
                ("femto", "f", 0.000000000000001),
                ("atto", "a", 0.000000000000000001))

    def __init__(self, factor=None, abbreviate=None):
        """
        :param factor: anything to look in factors (i.e. factor list with Factor structures).
        :param abbreviate: index to return from Factor structure when factor is asked.
            if None, it is selected if it match from factor or False if not.

        .. note: A factor structure is of the form ("Name","abbreviation",value)
        """
        self._factor = None
        self._factorsCache = None
        self.abbreviate = abbreviate
        self.factors = self._factors
        self.factor = factor

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, value):
        # itertools.izip is faster but this operation is one time
        self._factorsCache = list(zip(*value))
        self._factors = value

    @property
    def factor(self):
        return self._factor[self.abbreviate]

    @factor.setter
    def factor(self, value):
        self._factor = t = self.getFactor(value)  # transform units
        if self.abbreviate is None and value in t:
            self.abbreviate = t.index(value)

    def convert(self, factor, to=None):
        """
        Convert from actual factor to another factor.

        :param factor: number
        :param to: factor to convert
        :return: converted value, units
        """
        to = self.getFactor(to)
        # converted value, factor
        return factor * self._factor[2] / float(to[2]), to[self.abbreviate]

    def convert2sample(self, factor, to=None):
        """
        Convert to resemble sample.

        :param factor: number
        :param to: sample factor.
        :return: converted value, units
        """
        if to is None or to == 0:
            unit = None
        else:  # calculate as fast as possible
            try:
                unit = factor * self._factor[2] / to
            except (TypeError, ZeroDivisionError):
                to = self.getFactor(to)[2]
                unit = factor * self._factor[2] / to
        return self.convert(factor, unit)  # to units.

    def exactFactorIndex(self, key):
        """
        Find the index of a factor that contains a key.

        :param key: anything to look in factors (i.e. factor name, factor value, abbreviation).
        :return: factor structure, else None.
        """
        for i in self._factorsCache:  # try to find it
            if key in i:
                return i.index(key)

    def nearFactorIndex(self, factor):
        """
        Find the index of nearest factor value.

        :param factor: factor value.
        :return: factor structure near factor value.
        """
        return (np.abs(np.array(self._factorsCache[2]) - factor)).argmin()

    def getFactor(self, key):
        """
        Tries to find factor value in factors.

        :param key: anything to look in factors (i.e. factor name, factor value, abbreviation).
                    If key is a factor value it will look for the nearest factor value.
        :return: factor structure, else raises error.
        """
        if key is None or isinstance(key, str):
            index = self.exactFactorIndex(key)
            if index is None:
                raise Exception("Factor not found for {}".format(key))
        else:
            index = self.nearFactorIndex(key)
        return self._factors[index]

    @staticmethod
    def split(value):
        """
        Get number fraction.

        :param value: number
        :return: integer, fraction
        """
        left = int(value)
        right = value - left
        return left, right

    @staticmethod
    def parts(value, precision=4):
        """
        Get number parts.

        :param value: number
        :param precision: decimal precision
        :return: ([... ,Hundreds, Tens, Ones],[Tenths, ...])
        """
        p = "{{:0.{}f}}".format(precision).format(value).split(".")
        if len(p) > 1:
            return [int(i) for i in p[0]], [int(i) for i in p[1]]
        else:
            return [int(i) for i in p[0]], []


class Magnitude:
    def __init__(self, value=0, factor=None, unit=None, precision=None, abbreviate=None):
        """
        :param value: value of magnitude e.g. 10, 0.00005, etc
        :param factor: factor of magnitude, its name or conversion
            e.g. 1000 or k, 0.001 or milli, giga, etc
        :param unit: unit of magnitude e.g. seconds, meters, etc
        :param precision: precision of value of magnitude
            e.g. 1 for 0.1, 2 for 0.01, etc.
        :param abbreviate: True to abbreviate factor name
            i.e. from "milli" to "m"
        """
        self.value = value
        self.precision = precision
        self.factor = factor
        self.unit = unit
        self.abbreviate = abbreviate

    def format_value(self, value):
        if self.precision is None:
            text = "{:f} {}{}"
        else:
            text = "{{:0.{}f}} {{}}{{}}".format(self.precision)
        if self.factor:
            # force to detect abbreviation if None
            f = FactorConvert(abbreviate=self.abbreviate, factor=self.factor)
            f.factor = None  # real factor which is considered 1
            if isinstance(self.factor, str):
                return text.format(*(f.convert(value, self.factor) +
                                     (self.unit,)))
            else:
                return text.format(*(f.convert2sample(value, self.factor) +
                                     (self.unit,)))
        else:
            return text.format(value, "", self.unit)

    def __str__(self):
        return self.format_value(self.value)

