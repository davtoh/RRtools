# -*- coding: utf-8 -*-
"""
Matplotlib-based plotting utilities and filter response visualizers.
"""

import os
import sys
from multiprocessing import Process
import matplotlib.axes
import matplotlib.figure
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..config import FLAG_DEBUG
from ..root import formatConsume
from ..arrayops.basic import isnumpy, get_x_space, relativeVectors, vertexesAngles
from ..arrayops.filters import FilterBase

__all__ = [
    "wins",
    "servertimeout",
    "fastplt",
    "graph_filter",
    "plotPointsContour",
]

servertimeout = None  # seconds to show an image
wins = [0]  # keeps track of image number through different processes


def _render_fastplt_figure(image, cmap, title, win, kwargs):
    """Render figure inside fastplt."""
    if isinstance(image, matplotlib.axes.SubplotBase):
        f = image.figure
    elif isinstance(image, matplotlib.figure.Figure):
        f = image
    else:
        f = plt.figure()
        plt.imshow(image, cmap, **kwargs)
        if title:
            plt.title(title)
        plt.xticks([])
        plt.yticks([])
    win_title = win if win else "Figure {}".format(wins[-1])
    f.canvas.set_window_title(win_title)
    plt.show()


def _dispatch_socket_plot(image, cmap, title, win, daemon, kwargs):
    """Send plot data to an external process via socket."""
    from ..serverServices import generateServer, sendPickle
    s, addr = generateServer()
    props = ['{py} "{script}"'.format(
        py=sys.executable,
        script=os.path.join(os.path.abspath(os.path.join(
            os.path.realpath(__file__), "../../../../")), "fastplt.py"))]
    props.append("{}:{}".format(*addr))
    d = dict(cmap=cmap, title=title, win=win, num=wins[0])
    props.extend(['--{} "{}"'.format(key, val)
                  for key, val in list(d.items()) if val is not None])
    props.append("--block")
    if daemon:
        props.append("--daemon")
    txt = " ".join(props)
    sendPickle((image, kwargs), s, timeout=servertimeout, threaded=True)

    from subprocess import Popen
    import shlex
    kwargs2 = {}
    if os.name == "nt":
        kwargs2["creationflags"] = 0x00000008  # DETACHED_PROCESS
    Popen(shlex.split(txt), shell=False, stdin=None, stdout=None, stderr=None,
          close_fds=True, **kwargs2)


def fastplt(image, cmap=None, title="visualazor", win=None, block=False, daemon=False,
            mode="socket", **kwargs):
    """Fast plot wrapper."""
    wins[-1] += 1
    if block:
        _render_fastplt_figure(image, cmap, title, win, kwargs)
    elif mode == "socket":
        _dispatch_socket_plot(image, cmap, title, win, daemon, kwargs)
    else:
        p = Process(target=_render_fastplt_figure, args=(image, cmap, title, win, kwargs))
        p.daemon = daemon
        p.start()


def _get_filter_name(filter_obj):
    """Extract descriptive name from a filter object."""
    try:
        return filter_obj.name
    except Exception:
        try:
            return type(filter_obj).__name__
        except Exception:
            try:
                return filter_obj.__name__
            except Exception:
                return ""


def _format_joints(iterator):
    """Join formatted iterator items with colon/comma delimiter."""
    items = list(iterator)
    if len(items) == 1:
        return ", ".join(items)
    if len(items) > 0:
        c = "," if ":" in items[0] else ":"
        return "{}{} ".format(items[0], c) + ", ".join(items[1:])
    return ""


def _safe_replace(name, params=None):
    """Format string templates safely using parameter dictionary."""
    if params is None:
        params = {}
    if isinstance(name, (str, bytes)):
        try:
            name = formatConsume(name, params)
            if "*" in name:
                parts = name.split("*")
                f = ["{key}: {{{key}}}".format(key=key) for key in list(params.keys())]
                return _safe_replace([[part, f] for part in parts], params)
            return name
        except Exception:
            return None
    return _format_joints(j for j in [_safe_replace(i, params) for i in name] if j)


def _annotate_curve(ax, f, params, lxp, lyp, idx, xsize, ysize, scale, xmin, xmax, ymin, ymax):
    """Annotate filter parameters on the plot axes."""
    items = sorted([(k, v) for k, v in list(params.items()) if k.startswith("beta")],
                   key=lambda x: x[1])
    for j, (key, val) in enumerate(items):
        del params[key]
        xp, yp = val, f(val)
        desc = lxp[idx][j] if lxp is not None else not (j % 2)
        lx = (xp + desc) if not isinstance(desc, bool) else ((xp - xsize - 15) if desc else (xp + xsize))
        desc_y = lyp[idx][j] if lyp is not None else (yp < ymin + ysize)
        if isinstance(desc_y, bool):
            ly = (yp + ymax * scale) if desc_y else ((yp - ymax * scale) if yp > ymax - ysize else yp)
        else:
            ly = yp + desc_y
        ax.annotate(u'{}\n{}'.format(key, val).title(), xy=(xp, yp), textcoords='data',
                    xytext=(lx, ly), arrowprops=dict(facecolor='black', shrink=0.05))


def _resolve_level(filters_or_f, levels, idx=None, scale=0.07):
    """Resolve x evaluation range for filters."""
    if levels is None:
        try:
            level = get_x_space(filters_or_f)
            if level.size == 0:
                raise ValueError("Empty level")
        except Exception:
            level = np.linspace(0, 256, 256)
    elif isnumpy(levels):
        level = levels
    elif idx is not None:
        level = levels[idx]
    else:
        level = levels
    xmin, xmax = np.min(level), np.max(level)
    xsize = (xmax + abs(xmin)) * scale
    return level, xmin, xmax, xsize


def _resolve_titles_and_names(f, params, titles, i, annotate, single, legend):
    """Compute title and label name for single filter curve."""
    name = params.get("name", _get_filter_name(f)) if isinstance(f, FilterBase) else _get_filter_name(f)
    if not name:
        name = "Filter {}".format(i)
    title = ""
    if isinstance(f, FilterBase) and titles is None:
        suffix = " filter" if (annotate and legend) or not single else ""
        title = _safe_replace(["{name}" + suffix, "*"], params)
        name = _safe_replace(["{name}", "*"], params)
    if not title:
        if titles is None:
            title = name
        elif not isinstance(titles, (str, bytes)):
            title = _safe_replace(titles[i], params)
    return title, name


def _init_filter_fig(win, filters, cols):
    """Initialize figure and grid for filter plotting."""
    rows = max(1, len(filters) // cols)
    cols = cols if (len(filters) // cols) > 0 else len(filters)
    gs = gridspec.GridSpec(rows, cols)
    if win is False:
        fig = plt.gcf()
    elif isinstance(win, (str, bytes)):
        fig = plt.figure(win)
    else:
        fig = win
    return fig, gs


def _finalize_axes(ax, lines, axtitles, ymins, ymaxs, xmin, xmax, titles, single, legend):
    """Apply titles, limits, and legends to the plotted axes."""
    if single and isinstance(titles, (str, bytes)):
        plt.title(titles.title())
    elif not single or not isinstance(titles, (str, bytes)):
        plt.title("; ".join(axtitles).title())
    plt.xlim([-1 + xmin, xmax + 1])
    plt.ylim([-0.1 + np.min(ymins), 0.1 + np.max(ymaxs)])
    if legend:
        ax.legend(handles=lines, loc='upper left')


def _plot_filter_item(ax, f, level, i, params, titles, annotate, single, legend, scale, xmin, xmax, lxp, lyp):
    """Compute curve, optional annotations, and plot line on axes."""
    y = f(level)
    ymin, ymax = np.min(y), np.max(y)
    if annotate and isinstance(f, FilterBase):
        xsize = (xmax + abs(xmin)) * scale
        _annotate_curve(ax, f, params, lxp, lyp, i, xsize, (ymax + abs(ymin)) * scale, scale, xmin, xmax, ymin, ymax)
    title, name = _resolve_titles_and_names(f, params, titles, i, annotate, single, legend)
    line, = ax.plot(level, y, label=name)
    return line, title, ymin, ymax


def graph_filter(filters, levels=None, titles=None, win=None,
                 single=True, legend=True, annotate=True,
                 cols=3, scale=0.07, show=True, lxp=None, lyp=None):
    """Graph filter responses with optional parameter annotations."""
    if isinstance(filters, FilterBase):
        win = win or (_get_filter_name(filters) + " filter").title()
        filters = [filters]
    elif titles is not None and len(filters) != len(titles) and not isinstance(titles, (str, bytes)):
        raise ValueError("Titles len {} does not match filters len {}".format(len(titles), len(filters)))

    win = win or "Filters Response"
    level_s, xmin_s, xmax_s, xsize_s = _resolve_level(filters, levels, scale=scale) if single else (None, 0, 0, 0)
    fig, gs = _init_filter_fig(win, filters, cols)
    lines, axtitles, ymins, ymaxs = [], [], [], []

    for i, f in enumerate(filters):
        ax = plt.gca() if single else fig.add_subplot(gs[i])
        level, xmin, xmax = (level_s, xmin_s, xmax_s) if (single and (levels is None or isnumpy(levels))) else _resolve_level([f], levels, idx=i, scale=scale)[:3]
        params = {k[1:] if k.startswith("_") else k: v for k, v in list(f.__dict__.items()) if v is not None} if isinstance(f, FilterBase) else {}
        line, title, ymin, ymax = _plot_filter_item(ax, f, level, i, params, titles, annotate, single, legend, scale, xmin, xmax, lxp, lyp)
        lines = (lines if single else []) + [line]
        axtitles = (axtitles if single else []) + [title]
        ymins, ymaxs = (ymins if single else []) + [ymin], (ymaxs if single else []) + [ymax]
        if not single or i == len(filters) - 1:
            _finalize_axes(ax, lines, axtitles, ymins, ymaxs, xmin, xmax, titles, single, legend)

    if show:
        plt.show()
    return fig


def _format_point_labels(pts, deg):
    """Compute vertex angle labels for point contour plotting."""
    if deg is None:
        return [""] * len(pts)
    op = u"°" if deg is True else u""
    return [u"{:1.1f}{}".format(i, op) for i in vertexesAngles(pts, deg=deg)]


def plotPointsContour(pts, ax=None, lcor="k", pcor=None,
                      deg=None, annotate=True, width=0.004,
                      label='pt{pt}({x}, {y}, {a})', arrowprops=None):
    """Plots points and joining lines in axes."""
    label = u"{}".format(label)
    ax = ax or plt.gca()
    if pcor is not None:
        ax.plot(pts[:, 0], pts[:, 1], markerfacecolor=pcor, marker='o')
        ax.plot(pts[:, 0], pts[:, 1], color=lcor)
        ax.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]], 'k')

    deg_list = _format_point_labels(pts, deg)
    for i, ((x, y), (u, v), a) in enumerate(zip(pts, relativeVectors(pts), deg_list)):
        if pcor is None:
            ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=width, color=lcor)
        if annotate:
            ax.annotate(label.format(pt=i, x=u"{:1.1f}".format(x), y=u"{:1.1f}".format(y), a=a),
                        xy=(x, y), textcoords='data', arrowprops=arrowprops)

    r = 0.1
    xmin, ymin = np.min(pts, 0)
    xmax, ymax = np.max(pts, 0)
    plt.ylim([ymin - (ymax - ymin) * r, ymax + (ymax - ymin) * r])
    plt.xlim([xmin - (xmax - xmin) * r, xmax + (xmax - xmin) * r + (xmax - xmin) * r * 0.15 * len(label)])
    return ax