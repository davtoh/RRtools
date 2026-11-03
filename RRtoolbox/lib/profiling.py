# -*- coding: utf-8 -*-
"""
Profiling tools, execution timers, and code-block chronometers.
"""
import sys
from time import time
import numpy as np

from .unit_conversion import Magnitude

__all__ = ["TimeCode", "Profiler"]


class TimeCode:
    """
    Context to profile code by printing a prelude and prologue with time.

    :param msg: prelude or description message
    :param factor: factor supported by FactorConvert class
    :param precision: number of digits after a float point
    :param abv: if True prints "s", if False "seconds" for time
    :param endmsg: prologue message
    :param enableMsg: (True) A flag specifying if context should be printed.
    :param printfunc: function to print messages. By default sys.stdout.write
    """

    def __init__(self, msg=None, factor=None, precision=None,
                 abv=None, endmsg="{time}\n", enableMsg=True,
                 printfunc=None, profiler=None, profile_point=None,
                 previous_time=None, i=None):
        self.msg = msg
        self.factor = factor
        self.precision = precision
        self.abv = abv
        self.endmsg = endmsg
        self.enableMsg = enableMsg
        self.printfunc = printfunc
        self.time_start = None
        self._time_end = None
        if isinstance(previous_time, TimeCode):
            previous_time = previous_time.time_cummulative
        self.time_cummulative = previous_time
        self.profiler = profiler
        self.profile_point = profile_point

    @property
    def time(self):
        if self.time_start is None:
            return 0
        return time() - self.time_start

    @property
    def time_end(self):
        if self._time_end is None:
            return self.time
        return self._time_end

    @time_end.setter
    def time_end(self, value):
        self._time_end = value

    @time_end.deleter
    def time_end(self):
        del self._time_end

    def start_cummulative(self):
        if self.time_cummulative is None:
            self.time_cummulative = 0
        self.time_start = time()

    def end_cummulative(self):
        self.time_end = t = self.time
        self.time_cummulative += t

    def __enter__(self):
        if self.printfunc is None:
            self.printfunc = sys.stdout.write
        if self.enableMsg and self.msg is not None:
            self.printfunc(self.msg)
        self.time_start = time()
        self.time_end = None
        if self.profiler is not None:
            msg = self.msg if self.profile_point is None else self.profile_point
            if isinstance(msg, str):
                self.profile_point = self.profiler.open_point(msg=msg)
            else:
                self.profile_point = self.profiler.open_point(*msg)
        elif isinstance(self.profile_point, Profiler):
            self.profile_point.time_start = self.time_start
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.profile_point, Profiler):
            self.profile_point.close()
        t = self.time
        if self.time_end is None:
            self.time_end = t
        elif self.time_cummulative is not None:
            t = self.time_cummulative

        if self.enableMsg and self.endmsg is not None:
            u, i = ("s", True) if self.abv is True else ("seconds", False)
            self.printfunc(self.endmsg.format(
                time=Magnitude(value=t, precision=self.precision,
                               unit=u, factor=self.factor, abbreviate=i)))


def _format_field(val, formatter):
    """Helper to format string or callable field."""
    if not val:
        return ""
    if isinstance(formatter, str):
        return formatter.format(val)
    return formatter(val)


def _collapse_structure(children, collapse=None):
    """
    Collapse list of profile structures for repeated tags/messages.
    """
    if collapse is None or not children:
        return children
    new_children = []
    indexes = {}
    for child in children:
        cmp_key = tuple(child[:2])
        should_collapse = (collapse is True or cmp_key[0] in collapse or cmp_key[1] in collapse)
        if new_children and should_collapse and cmp_key in indexes:
            idx = indexes[cmp_key]
            new_children[idx][2] += child[2]
            new_children[idx][3].extend(child[3])
        else:
            indexes[cmp_key] = len(new_children)
            new_children.append(child)
    return new_children


def _walk_unformatted(struct, level=0):
    """Generate (level, tag, msg, time) tuples from structure tree."""
    tag, msg, time_val, children = struct
    yield (level, tag, msg, time_val)
    for child in children:
        yield from _walk_unformatted(child, level + 1)


def _walk_structured(structs, level, formatter_func, format_structure):
    """Helper to generate nested string structure."""
    if not structs:
        return ""
    tag, msg, time_val, children = structs[0]
    space, tag, msg, time_val = formatter_func(level, tag, msg, time_val)
    return format_structure.format(
        space=space, tag=tag, msg=msg, time=time_val,
        child=_walk_structured(children, level + 1, formatter_func, format_structure),
        side=_walk_structured(structs[1:], level, formatter_func, format_structure)
    )


class Profiler:
    r"""
    Profiler for code points.

    :param msg: custom comment for profiling point
    :param tag: classification tag
    """

    def __init__(self, msg=None, tag=None):
        self.time_start = time()
        self.msg = msg
        self.tag = tag
        self.space = " "
        self.format_line = "{space}{tag}{msg}{time}"
        self.format_structure = "\n{space}[{tag}{msg}{time}{child}]{side}"
        self.format_tag = "({}) "
        self.format_time = " -> {:0.4f} secs"
        self.format_msg = "{!r}"
        self.time_end = None
        self.points = []

    @property
    def time(self):
        """Overall time of profiling."""
        try:
            return self.time_end - self.time_start
        except (TypeError, AttributeError):
            points_diff = [i.time for i in self.points if i.time]
            return np.sum(points_diff) if points_diff else None

    def _add_point(self, point):
        """Track and organize profile points."""
        if self.points and point not in self.points:
            child = self.points[-1]
            if child.time_end is None:
                child._add_point(point)
            else:
                self.points.append(point)
        elif not self.points:
            self.points.append(point)

    def _close_point(self, point):
        """Close a point and add to tracked points."""
        point.close()
        self._add_point(point)

    def open_point(self, msg=None, tag=None):
        """Open a profiling point to track time."""
        point = Profiler(msg, tag)
        self._add_point(point)
        return point

    def close(self):
        """Close profiler and all nested points."""
        for p in self.points:
            p.close()
        if self.time_end is None:
            self.time_end = time()

    def formatter(self, level, tag, msg, time_val):
        """Format profiling point arguments."""
        fmt_tag = _format_field(tag, self.format_tag)
        fmt_time = _format_field(time_val, self.format_time)
        fmt_msg = _format_field(msg, self.format_msg)
        return self.space * level, fmt_tag, fmt_msg, fmt_time

    def _collapse_structure(self, children, collapse=None):
        return _collapse_structure(children, collapse)

    def restructure(self, structure, collapse):
        """Reprocess an already created structure."""
        tag, msg, time_val, children = structure
        if children:
            children = _collapse_structure(children, collapse)
            for i, child in enumerate(children):
                children[i] = self.restructure(child, collapse)
            structure[3] = children
        return structure

    def structure(self, collapse=None):
        """Profiling structure with format [tag, msg, time, children]."""
        children = [point.structure(collapse) for point in self.points]
        children = _collapse_structure(children, collapse)
        return [self.tag, self.msg, self.time, children]

    def _helper_lines_unformatted(self, struct, level=0):
        return _walk_unformatted(struct, level)

    def lines_unformatted(self, collapse=None):
        """Generate structure lines as (level, tag, msg, time)."""
        return _walk_unformatted(self.structure(collapse))

    def lines_formatted(self, collapse=None):
        """Generate formatted string lines."""
        lns = []
        for level, tag, msg, time_val in self.lines_unformatted(collapse):
            space, tag, msg, time_val = self.formatter(level, tag, msg, time_val)
            lns.append(self.format_line.format(
                space=space, tag=tag, msg=msg, time=time_val))
        return lns

    def string_lines(self):
        """String with plain structure of profiling."""
        return "\n".join(self.lines_formatted())

    def _helper_string_structured(self, structs, level=0):
        return _walk_structured(structs, level, self.formatter, self.format_structure)

    def string_structured(self, collapse=None, structure=None):
        """String with structured hierarchy of profiling."""
        structs = [self.structure(collapse)] if structure is None else [self.restructure(structure, collapse)]
        return _walk_structured(structs, 0, self.formatter, self.format_structure)

