"""Shared chart style: palette tokens, matplotlib defaults, small helpers.

Palette: a validated categorical order (adjacent-pair colour-vision-deficiency
separation >= 8 dE in light and dark). Slots are assigned in fixed order and
follow the entity, never its rank. Text is always ink, never a series colour;
aqua and yellow are below 3:1 on the light surface, so charts that use them
carry a legend and direct labels.

Rules every AeroSim plot follows: one y-axis per panel (small multiples instead
of twin axes), 2 px lines, >= 8 px markers with a surface ring, hairline solid
grid, recessive axes, a legend whenever there are two or more series.
"""

from __future__ import annotations

from contextlib import contextmanager

from aerosim.core import require

THEMES = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                   "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"],
    },
}

_current = {"theme": "light"}


def tokens(theme: str | None = None) -> dict:
    return THEMES[theme or _current["theme"]]


def series(i: int, theme: str | None = None) -> str:
    """Categorical slot ``i`` (0-based). More than 8 series is a design error: facet."""
    s = tokens(theme)["series"]
    if i >= len(s):
        raise ValueError("more than 8 series: fold into 'Other' or use small multiples")
    return s[i]


def _fonts() -> list[str]:
    """Preferred sans faces that are actually installed (avoids findfont warnings)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    pref = ["Inter", "Segoe UI", "Helvetica Neue", "Helvetica", "Arial"]
    return [f for f in pref if f in have] + ["DejaVu Sans"]


def rc(theme: str = "light") -> dict:
    t = THEMES[theme]
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "figure.dpi": 110, "savefig.dpi": 200,
        "font.family": _fonts(),
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": t["ink"], "axes.titlepad": 8,
        "axes.labelcolor": t["ink2"], "axes.labelsize": 9,
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5,
        "ytick.major.pad": 5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "lines.markersize": 8,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": t["ink2"],
        "legend.handlelength": 1.6, "text.color": t["ink"],
        "axes.prop_cycle": __import__("cycler").cycler(color=t["series"]),
    }


@contextmanager
def themed(theme: str = "light"):
    """``with themed("dark"): fig = plots.thrust_curve(...)``"""
    plt = require("matplotlib.pyplot", "plot")
    prev = _current["theme"]
    _current["theme"] = theme
    try:
        with plt.rc_context(rc(theme)):
            yield tokens(theme)
    finally:
        _current["theme"] = prev


def figure(nrows=1, ncols=1, size=(7.0, 3.6), **kw):
    """New figure in the current theme. Applies the theme to matplotlib's rcParams
    (like seaborn's set_theme) so titles, labels and legends added later match."""
    plt = require("matplotlib.pyplot", "plot")
    plt.rcParams.update(rc(_current["theme"]))
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True,
                             squeeze=False, **kw)
    return fig, axes


def figure_legend(fig, axes_with_handles, ncol: int = 4) -> None:
    """One legend for a set of small multiples, top right above the panels."""
    h, labels = axes_with_handles.get_legend_handles_labels()
    fig.legend(h, labels, loc="outside upper right", ncol=ncol)


def restyle(ax) -> None:
    """Apply theme colours to an axes created outside ``figure``."""
    t = tokens()
    ax.set_facecolor(t["surface"])
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0)
    ax.title.set_color(t["ink"])


def end_label(ax, x, y, text, dx=4, dy=0, ha="left") -> None:
    """Direct label at a line end, in secondary ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                va="center", fontsize=8.5, color=tokens()["ink2"])


def event_lines(ax, events: dict, scale: float = 1.0, labels: bool = True,
                y: float = 1.0) -> None:
    """Hairline vertical markers for named events (``{label: x}``), labelled at the top."""
    t = tokens()
    for i, (name, x) in enumerate(events.items()):
        ax.axvline(x * scale, color=t["axis"], linewidth=0.8, zorder=1)
        if labels:
            ax.annotate(name, (x * scale, y), xycoords=("data", "axes fraction"),
                        xytext=(3, -3 - 10 * (i % 2)), textcoords="offset points",
                        fontsize=7.5, color=t["muted"], va="top", ha="left", rotation=0)


def event_axis(ax, events: dict) -> None:
    """Name events on a secondary top axis (rotated), so labels never sit on the data."""
    t = tokens()
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(list(events.values()), labels=list(events.keys()))
    sec.tick_params(labelsize=7.5, labelcolor=t["muted"], length=0, labelrotation=90, pad=2)
    sec.spines["top"].set_visible(False)


def ringed_points(ax, x, y, color, label=None, size=7.0, zorder=5):
    """Markers >= 8 px with a 2 px surface ring so they read over lines."""
    return ax.plot(x, y, "o", markersize=size, color=color, markeredgecolor=tokens()["surface"],
                   markeredgewidth=2.0, linestyle="none", label=label, zorder=zorder)


def compact(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e3:
        return f"{v/1e3:.1f}K" if a < 1e4 else f"{v/1e3:.0f}K"
    return f"{v:.0f}" if a >= 10 else f"{v:.1f}"
