"""Flight plots: sounding-rocket profiles and multi-stage orbital ascents."""

from __future__ import annotations

import numpy as np

from aerosim.core import FlightResult
from aerosim.plots.style import (
    end_label,
    event_axis,
    event_lines,
    figure,
    figure_legend,
    ringed_points,
    series,
    tokens,
)


def flight_profile(results: FlightResult | dict[str, FlightResult], title: str = "Flight profile"):
    """Altitude, speed and ground track (east vs north) for one or more flights."""
    if isinstance(results, FlightResult):
        results = {results.meta.get("simulator", "flight"): results}
    fig, axes = figure(1, 3, size=(9.0, 3.2))
    a_alt, a_spd, a_trk = axes[0]
    for i, (name, r) in enumerate(results.items()):
        c = series(i)
        a_alt.plot(r.time, r.position[:, 2], color=c, label=name)
        a_spd.plot(r.time, np.linalg.norm(r.velocity, axis=1), color=c, label=name)
        a_trk.plot(r.position[:, 0], r.position[:, 1], color=c, label=name)
        ringed_points(a_trk, [r.position[-1, 0]], [r.position[-1, 1]], c)
        k = int(np.argmax(r.position[:, 2]))
        if len(results) == 1:
            end_label(a_alt, r.time[k], r.position[k, 2], f"apogee {r.apogee:,.0f} m", dy=8)
    a_alt.set(title="Altitude (m)", xlabel="Time (s)")
    a_spd.set(title="Speed (m/s)", xlabel="Time (s)")
    a_trk.set(title="Ground track (m)", xlabel="East", ylabel="North")
    a_trk.set_aspect("equal", adjustable="datalim")
    for ax in (a_alt, a_spd):
        ax.set_ylim(bottom=0)
    if len(results) > 1:
        figure_legend(fig, a_alt)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=tokens()["ink"])
    return fig


_ASCENT_EVENTS = {"max_q": "Max q", "Stage 1_burnout": "S1 sep", "fairing_jettison": "Fairing",
                  "Stage 2_burnout": "S2 sep", "Breeze-KM_burn1_end": "Burn 1 end"}


def _events(res, names=None, until=None):
    names = names or _ASCENT_EVENTS
    out = {}
    for k, label in names.items():
        if k in res.events and (until is None or res.events[k] <= until):
            out[label] = res.events[k]
    return out


def ascent_overview(res, reference: dict | None = None, until: float | None = None,
                    title: str | None = "Ascent"):
    """2 x 3 small multiples vs time: altitude, relative speed, flight-path angle,
    dynamic pressure, axial acceleration, mass. ``reference`` = {event: {"time", "altitude",
    "speed", "fpa", "q"}} adds published waypoints as ringed points (legend shown)."""
    t_end = until or res.events.get("Breeze-KM_burn1_end", res.time[-1]) + 60
    m = res.time <= t_end
    t = res.time[m]
    fig, axes = figure(2, 3, size=(9.6, 5.4), sharex=True)
    panels = [
        ("Altitude (km)", res.altitude[m] / 1e3, "altitude", 1e-3),
        ("Speed, Earth-relative (km/s)", res.speed[m] / 1e3, "speed", 1e-3),
        ("Flight-path angle (deg)", res.flight_path_angle[m], "fpa", 1.0),
        ("Dynamic pressure (kPa)", res.dynamic_pressure[m] / 1e3, "q", 1e-3),
        ("Acceleration (g)", res.acceleration[m] / 9.80665, None, None),
        ("Mass (t)", res.mass[m] / 1e3, None, None),
    ]
    ev = _events(res, until=t_end)
    for ax, (label, y, key, k) in zip(axes.flat, panels):
        ax.plot(t, y, color=series(0), label="AeroSim model")
        event_lines(ax, ev, labels=False)
        if ax in axes[0]:
            event_axis(ax, ev)
        ax.set_title(label)
        if reference and key:
            pts = [(g["time"], g[key] * k) for g in reference.values() if key in g]
            if pts:
                xs, ys = zip(*pts)
                ringed_points(ax, xs, ys, series(1), label="User's Guide waypoints")
    for ax in axes[1]:
        ax.set_xlabel("Time from lift-off (s)")
    if reference:
        figure_legend(fig, axes[0, 1])
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold",
                     color=tokens()["ink"])
    return fig


def ascent_trajectory(res, until: float | None = None, title: str = "Altitude vs downrange"):
    """Altitude against ground range with stage segments labelled."""
    t_end = until or res.events.get("Breeze-KM_burn1_end", res.time[-1])
    m = res.time <= t_end
    fig, ax = figure(size=(8.0, 3.2))
    ax = ax[0, 0]
    ax.plot(res.downrange[m] / 1e3, res.altitude[m] / 1e3, color=series(0))
    t = tokens()
    for label, te in _events(res, until=t_end).items():
        x = np.interp(te, res.time, res.downrange) / 1e3
        y = np.interp(te, res.time, res.altitude) / 1e3
        ringed_points(ax, [x], [y], series(0), size=6)
        high = y > 0.3 * res.altitude[m].max() / 1e3        # below-right of high points
        ax.annotate(label, (x, y), xytext=(6, -12) if high else (6, 4),
                    textcoords="offset points", fontsize=8, color=t["ink2"])
    ax.set(xlabel="Downrange (km)", ylabel="Altitude (km)", title=title)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    return fig


def ground_track(res, site: tuple[float, float] | None = None, title: str = "Ground track"):
    """Latitude/longitude trace (equirectangular, no basemap). Wraps at +-180 deg."""
    fig, ax = figure(size=(8.0, 3.8))
    ax = ax[0, 0]
    lon, lat = res.longitude, res.latitude
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]       # split at the date line
    seg = np.split(np.arange(lon.size), jumps + 1)
    for s in seg:
        ax.plot(lon[s], lat[s], color=series(0))
    if site:
        ringed_points(ax, [site[1]], [site[0]], series(1), label="Launch site")
        ax.annotate("Plesetsk" if abs(site[0] - 62.8) < 0.5 else "Launch site", (site[1], site[0]),
                    xytext=(6, -10), textcoords="offset points", fontsize=8,
                    color=tokens()["ink2"])
    ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=title,
           xlim=(-180, 180), ylim=(-90, 90))
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    return fig
