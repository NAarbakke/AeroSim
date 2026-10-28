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
        "font": '"Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif', "mono": "ui-monospace,Consolas,monospace",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"],
        "font": '"Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif', "mono": "ui-monospace,Consolas,monospace",
    },
    # pure black (Vercel-style carbon) with pgfplots-like charts: framed axes, inward ticks, no
    # grid, thin lines told apart by dash as well as hue (the LaTeX typeface is one menu away)
    "carbon": {
        "surface": "#0a0a0a", "page": "#000000", "ink": "#ededed", "ink2": "#a1a1a1",
        "muted": "#6e6e6e", "grid": "#1c1c1c", "axis": "#3d3d3d",
        "series": ["#ededed", "#8ab4f8", "#f28b82", "#fdd663", "#81c995", "#c58af9",
                   "#78d9ec", "#a1a1a1"],
        "font": '"Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif', "mono": "ui-monospace,Consolas,monospace",
        "academic": True,
    },
    # white paper: the look of LaTeX publication figures (pgfplots / matplotlib-pgf at column
    # width) - Computer Modern, black first series, grey dashed second, frame with inward ticks
    # on all four sides, no grid, a frameless key inside the plot
    "paper": {
        "surface": "#ffffff", "page": "#ffffff", "ink": "#000000", "ink2": "#404040",
        "muted": "#737373", "grid": "#e0e0e0", "axis": "#999999",
        "series": ["#000000", "#737373", "#000000", "#a6a6a6", "#000000", "#737373",
                   "#404040", "#a6a6a6"],
        "font": '"Latin Modern Roman","LM Roman 10","CMU Serif","Computer Modern",serif',
        "mono": '"Latin Modern Mono","LM Mono 10",monospace',
        "academic": True, "key_inside": True, "weights": [1.4, 0.9],
    },
}

# themes the site's pages can switch between at run time (nav toggle)
SITE_THEMES = ("dark", "carbon", "paper")

# typefaces, chosen independently of the colour theme (until one is chosen, a theme uses its own):
# the site's original sans, LaTeX's Latin Modern, or B612 - the typeface Airbus drew for cockpit
# displays - for a technical look
FONTS = {
    "original": {"label": "Original", "font": THEMES["dark"]["font"], "mono": THEMES["dark"]["mono"]},
    "serif": {"label": "LaTeX serif", "font": THEMES["paper"]["font"], "mono": THEMES["paper"]["mono"]},
    "technical": {"label": "Technical (B612)", "font": '"B612","Segoe UI",system-ui,sans-serif',
                  "mono": '"B612",ui-monospace,Consolas,monospace'},   # + tabular figures
}
DARK_THEMES = ("dark", "carbon")


def default_font(theme: str) -> str:
    """The ``FONTS`` key a theme uses until the reader picks a typeface."""
    return next((k for k, v in FONTS.items() if v["font"] == THEMES[theme]["font"]), "original")

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


def rc(theme: str = "light", pgf: bool = False) -> dict:
    """matplotlib rcParams for a theme. ``carbon`` gives LaTeX-style figures (Computer Modern
    mathtext, framed axes, inward ticks on all sides, no grid); with ``pgf=True`` they are set
    by LaTeX itself through the pgf backend (lualatex, Latin Modern), for .pgf / .pdf output."""
    t = THEMES[theme]
    if t.get("academic"):
        from cycler import cycler
        r = {"figure.facecolor": t["page"], "axes.facecolor": t["page"], "savefig.facecolor": t["page"],
             "figure.dpi": 110, "savefig.dpi": 200,
             "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
             "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
             "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "normal",
             "axes.labelcolor": t["ink"], "axes.labelsize": 10, "text.color": t["ink"],
             "axes.edgecolor": t["ink"], "axes.linewidth": 0.6, "axes.grid": False,
             "axes.spines.top": True, "axes.spines.right": True,
             "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
             "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.visible": True,
             "ytick.minor.visible": True, "xtick.minor.size": 2, "ytick.minor.size": 2,
             "xtick.color": t["ink"], "ytick.color": t["ink"], "xtick.major.width": 0.6,
             "ytick.major.width": 0.6, "lines.linewidth": 1.0, "legend.frameon": False,
             "legend.loc": "upper right", "legend.handlelength": 1.8,
             "legend.fontsize": 9, "legend.labelcolor": t["ink"],
             "axes.prop_cycle": cycler(color=t["series"][:4],
                                       linestyle=["-", "--", ":", "-."])}
        if pgf:
            r |= {"pgf.texsystem": "lualatex", "pgf.rcfonts": False, "text.usetex": True,
                  "pgf.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}"}
        return r
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
