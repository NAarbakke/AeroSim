"""Flight data of an orbital ascent, gathered once for the data dashboard (``ascent_dashboard``)
and the LaTeX report (``ascent_report``), so both show the same numbers.

    data = ascent_dataset({"First principles": (res, vehicle), "Calibrated": (res2, veh2)},
                          site=(62.8, 40.3), target="650 km circular, 86.6°")

Per model: the full-flight time series (``series``), forces and velocity losses through the
powered ascent (``physics``, from ``ascent_closeup.ascent_physics``), the ground trajectory about
the launch site (``enu``), the event timeline, the osculating orbit at
each staging event and summary figures. Units are the ones plotted: km, km/s, kN, t, kPa, g (heat flux in W/m²).
"""

from __future__ import annotations

import math

import numpy as np

from aerosim.flight import orbits
from aerosim.plots.ascent_closeup import ascent_physics, mach1_time

EVENT_LABELS = {"liftoff": "Lift-off", "max_q": "Max-Q", "fairing_jettison": "Fairing jettison"}
ORBIT_LABELS = {"transfer": "Transfer orbit", "final": "Final orbit"}


def event_label(key: str) -> str:
    """'Stage 1_burnout' -> 'Stage 1 burnout and separation', 'Breeze-KM_burn1_end' -> ..."""
    if key in EVENT_LABELS:
        return EVENT_LABELS[key]
    s = key.replace("_", " ")
    return (s.replace("burnout", "burnout and separation").replace("burn1 start", "ignition")
            .replace("burn1 end", "cut-off (transfer orbit)").replace("burn2 start", "second burn")
            .replace("burn2 end", "cut-off (final orbit)"))


def _r(a, d):
    return np.round(np.asarray(a, float), d).tolist()


def _thin(n: int, max_points: int) -> np.ndarray:
    return np.unique(np.r_[np.linspace(0, n - 1, min(n, max_points)).astype(int), n - 1])


def _insertion_time(res) -> float:
    """End of the first burn of the last stage (transfer-orbit insertion), else the last sample."""
    ends = [t for k, t in res.events.items() if k.endswith("burn1_end")]
    return float(ends[0]) if ends else float(res.time[-1])


def _enu(res, site, t_end: float, max_points: int) -> dict:
    """Trajectory up to ``t_end`` over the ground (km): an azimuthal equidistant projection about
    the launch site (great-circle range along the true bearing, so it holds over the pole), with
    altitude up."""
    keep = res.time <= t_end + 1e-9
    lat, lon = np.radians(res.latitude[keep]), np.radians(res.longitude[keep])
    lat0, lon0 = map(math.radians, site)
    dlon = lon - lon0
    c = np.clip(math.sin(lat0) * np.sin(lat) + math.cos(lat0) * np.cos(lat) * np.cos(dlon), -1, 1)
    rng = orbits.RE * np.arccos(c) / 1e3
    az = np.arctan2(np.sin(dlon) * np.cos(lat),
                    math.cos(lat0) * np.sin(lat) - math.sin(lat0) * np.cos(lat) * np.cos(dlon))
    i = _thin(int(keep.sum()), max_points)
    return {"t": _r(res.time[keep][i], 2), "e": _r((rng * np.sin(az))[i], 2), "n": _r((rng * np.cos(az))[i], 2),
            "u": _r(res.altitude[keep][i] / 1e3, 3)}


def _model(name, res, vehicle, site, max_points: int) -> dict:
    t = res.time
    t_ins = _insertion_time(res)
    i = _thin(t.size, max_points)
    air = res.altitude < 200e3
    series = {"t": _r(t[i], 2), "h": _r(res.altitude[i] / 1e3, 3), "v": _r(res.speed[i] / 1e3, 4),
              "vi": _r(res.inertial_speed[i] / 1e3, 4), "fpa": _r(res.flight_path_angle[i], 3),
              "dr": _r(res.downrange[i] / 1e3, 2), "m": _r(res.mass[i] / 1e3, 4),
              "lat": _r(res.latitude[i], 3), "lon": _r(res.longitude[i], 3),
              "acc": _r(res.acceleration[i] / 9.80665, 3), "q": _r(res.dynamic_pressure[i] / 1e3, 3),
              "mach": _r(np.where(air, res.mach, np.nan)[i], 3), "phase": res.phase[i].astype(int).tolist()}
    d = ascent_physics(res, vehicle, t_end=t_ins + 30)
    j = _thin(d["t"].size, max_points)
    physics = {"t": _r(d["t"][j], 2), "thrust": _r(d["thrust"][j] / 1e3, 2), "drag": _r(d["drag"][j] / 1e3, 3),
               "weight": _r(d["weight"][j] / 1e3, 2), "heat": _r(d["heat"][j], 1), "cd": _r(d["cd"][j], 4),
               "loss": {k: _r(v[j], 2) for k, v in d["losses"].items()}}
    ev = []
    for key, te in sorted(res.events.items(), key=lambda e: e[1]):
        ev.append({"key": key, "label": event_label(key), "t": round(float(te), 1),
                   "h": round(float(np.interp(te, t, res.altitude)) / 1e3, 2),
                   "v": round(float(np.interp(te, t, res.speed)) / 1e3, 3),
                   "fpa": round(float(np.interp(te, t, res.flight_path_angle)), 2),
                   "dr": round(float(np.interp(te, t, res.downrange)) / 1e3, 1),
                   "m": round(float(np.interp(te, t, res.mass)) / 1e3, 3)})
    orbs = [{"key": k, "label": ORBIT_LABELS.get(k, event_label(k)), "perigee": round(float(o.perigee_alt) / 1e3, 1),
             "apogee": round(float(o.apogee_alt) / 1e3, 1), "inc": round(o.inclination, 3),
             "ecc": round(o.eccentricity, 5), "period": round(o.period / 60, 2)}
            for k, o in res.orbits.items()]
    pw = t <= d["t"][-1]
    i_q, i_g = int(np.argmax(res.dynamic_pressure)), int(np.argmax(np.where(pw, res.acceleration, 0)))
    t_m1 = mach1_time(t, np.where(air, res.mach, np.nan))
    fo = res.final_orbit
    losses = {k: float(v[-1]) for k, v in d["losses"].items()}
    summary = {"liftoff_mass": float(res.mass[0] / 1e3), "payload": float(getattr(vehicle, "payload_mass", 0) / 1e3),
               "max_q": float(res.dynamic_pressure[i_q] / 1e3), "t_max_q": float(t[i_q]),
               "h_max_q": float(res.altitude[i_q] / 1e3), "max_g": float(res.acceleration[i_g] / 9.80665),
               "t_max_g": float(t[i_g]), "t_mach1": t_m1, "t_insertion": t_ins,
               "t_end": float(t[-1]), "propellant_left": float(res.meta.get("propellant_left", 0.0)),
               "perigee": float(fo.perigee_alt) / 1e3, "apogee": float(fo.apogee_alt) / 1e3, "inc": fo.inclination,
               "period": fo.period / 60, "losses": losses,
               "dv_ideal": float(sum(_stage_dv(vehicle, res)))}
    return {"name": name, "series": series, "physics": physics, "enu": _enu(res, site, t_ins, max_points),
            "events": ev, "orbits": orbs, "phases": list(res.phases), "summary": summary,
            "stages": [{"name": s.name, "thrust_vac": s.thrust_vac / 1e3} for s in vehicle.stages]}


def _stage_dv(vehicle, res) -> list[float]:
    """Ideal (vacuum rocket-equation) velocity gained in each powered phase: the propellant burned
    is the stage's mass flow times the phase duration, so the fairing dropped mid-burn does not count."""
    out = []
    for k, name in enumerate(res.phases):
        st = next((s for s in vehicle.stages if name == s.name or name.startswith(s.name + " burn")), None)
        sel = np.nonzero(res.phase == k)[0]
        if st is None or sel.size < 2:
            continue
        m0 = res.mass[sel[0]]
        burned = st.mass_flow * (res.time[sel[-1]] - res.time[sel[0]])
        if 0 < burned < m0:
            out.append(st.isp_vac * 9.80665 * math.log(m0 / (m0 - burned)))
    return out


def ascent_dataset(models: dict, site: tuple[float, float], target: str = "", title: str = "",
                   max_points: int = 1500) -> dict:
    """``models``: {name: (AscentResult, LaunchVehicle)}, the first is the reference (solid line)."""
    return {"title": title, "target": target, "site": {"lat": site[0], "lon": site[1]},
            "models": [_model(n, r, v, site, max_points) for n, (r, v) in models.items()]}
