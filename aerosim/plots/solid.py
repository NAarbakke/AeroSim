"""Solid motor plots: thrust curves, ballistics history, grain burnback."""

from __future__ import annotations

import numpy as np

from aerosim.core import ThrustCurve, require
from aerosim.plots.style import end_label, figure, series, tokens


def thrust_curve(curves: ThrustCurve | list[ThrustCurve], title: str | None = None):
    """Thrust vs time for one or more motors (legend + end labels for >= 2)."""
    curves = curves if isinstance(curves, list) else [curves]
    fig, ax = figure(size=(7.0, 3.4))
    ax = ax[0, 0]
    for i, c in enumerate(curves):
        ax.plot(c.time, c.thrust, color=series(i), label=c.name)
        if len(curves) == 1:                         # wash only for a single series
            ax.fill_between(c.time, c.thrust, color=series(i), alpha=0.10, linewidth=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    if len(curves) == 1:
        c = curves[0]
        ax.set_title(title or f"{c.name}: {c.total_impulse:,.0f} N·s, {c.burn_time:.2f} s burn, "
                     f"Isp {c.average_isp:.0f} s")
    else:
        ax.set_title(title or "Thrust curves")
        ax.legend(loc="upper right")
    return fig


def motor_history(sim: dict, name: str = "Motor"):
    """Small multiples from ``GrainMotor.simulate()``: thrust, chamber pressure, Kn, mass."""
    fig, axes = figure(2, 2, size=(7.4, 4.6), sharex=True)
    t = sim["time"]
    panels = [("Thrust", sim["thrust"], "N", 1.0), ("Chamber pressure", sim["pressure"], "MPa", 1e-6),
              ("Kn (burn area / throat area)", sim["kn"], "", 1.0),
              ("Propellant remaining", sim["mass"], "kg", 1.0)]
    for ax, (label, y, unit, k) in zip(axes.flat, panels):
        ax.plot(t, y * k, color=series(0))
        ax.set_title(label + (f" ({unit})" if unit else ""))
        ax.set_ylim(bottom=0)
        i = int(np.argmax(y))
        if label != "Propellant remaining":
            end_label(ax, t[i], y[i] * k, f"peak {y[i]*k:,.2f}" if k != 1 else f"peak {y[i]:,.0f}",
                      dx=6, dy=-10)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(name, x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=tokens()["ink"])
    return fig


def grain_burnback(grain, webs: int = 5, title: str | None = None):
    """Cross-section of a map grain at successive webs (left) and burn area vs web (right)."""
    from aerosim.propulsion.solid.grains import MapGrain

    t = tokens()
    fig, axes = figure(1, 2, size=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax_map, ax_ab = axes[0]
    wmax = grain.web_max
    levels = np.linspace(0, wmax * 0.999, webs)
    R = grain.diameter / 2
    th = np.linspace(0, 2 * np.pi, 400)
    ax_map.plot(R * np.cos(th) * 1e3, R * np.sin(th) * 1e3, color=t["axis"], linewidth=1.0)
    if isinstance(grain, MapGrain):
        grain.build()
        from scipy.ndimage import gaussian_filter
        skfmm = require("skfmm", "grains")
        n = 301
        ax_ = np.linspace(-R, R, n)
        x, y = np.meshgrid(ax_, ax_)
        inside = x**2 + y**2 <= R**2
        port = grain.port_mask(x, y) & inside
        phi = np.ma.MaskedArray(0.5 - gaussian_filter(port.astype(float), 1.5), mask=~inside)
        d = np.ma.filled(skfmm.distance(phi, dx=float(ax_[1] - ax_[0])), np.nan)
        cs = ax_map.contour(x * 1e3, y * 1e3, d, levels=levels,
                            colors=[series(0)], linewidths=1.2, alpha=1.0)
        for i, c in enumerate(cs.collections if hasattr(cs, "collections") else []):
            c.set_alpha(1.0 - 0.75 * i / max(webs - 1, 1))
    else:   # circular port grains
        for i, w in enumerate(levels):
            r = (getattr(grain, "core_diameter", 0.0) / 2 + w)
            if r < R:
                ax_map.plot(r * np.cos(th) * 1e3, r * np.sin(th) * 1e3, color=series(0),
                            linewidth=1.2, alpha=1.0 - 0.75 * i / max(webs - 1, 1))
    ax_map.set_aspect("equal")
    ax_map.set_title("Burning surface at equal web steps")
    ax_map.set_xlabel("mm")
    ax_map.grid(False)
    w = np.linspace(0, wmax, 200)
    ab = np.array([grain.burn_area(x) for x in w])
    ax_ab.plot(w * 1e3, ab * 1e4, color=series(0))
    ax_ab.fill_between(w * 1e3, ab * 1e4, color=series(0), alpha=0.10, linewidth=0)
    ax_ab.set_xlabel("Web burned (mm)")
    ax_ab.set_title("Burning area (cm²)")
    ax_ab.set_ylim(bottom=0)
    fig.suptitle(title or type(grain).__name__, x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=t["ink"])
    return fig
