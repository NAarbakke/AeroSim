"""Rockot / Breeze-KM briefing: fly the mission, write figures + data.json for the deck.

    python docs/briefings/rockot/make_briefing.py     # then: node build/deck.js

Everything quoted in the deck comes from this run (``build/data.json``) or from
the Rockot User's Guide (docs/references/rockot/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "examples"))

import rockot_ascent as ra

from aerosim import plots

BUILD = HERE / "build"
FIG = BUILD / "figures"


LABELS = {"max_q": "Max q", "Stage 1_burnout": "S1 sep", "fairing_jettison": "Fairing",
          "Stage 2_burnout": "S2 sep", "Breeze-KM_burn1_end": "Burn 1 end"}


def comparison_bars(rows):
    """Model vs guide at each event: speed and altitude as two grouped-bar panels
    (same chart as the deck's native one, for the PDF)."""
    from aerosim.plots.style import figure, figure_legend, series

    fig, axes = figure(1, 2, size=(9.0, 3.3))
    x = np.arange(len(rows))
    w = 0.36
    for ax, key, k, title in ((axes[0, 0], "speed", 1e-3, "Earth-relative speed (km/s)"),
                              (axes[0, 1], "altitude", 1e-3, "Altitude (km)")):
        ax.bar(x - w / 2 - 0.01, [r["model"][key] * k for r in rows], w, color=series(0),
               label="AeroSim model")
        ax.bar(x + w / 2 + 0.01, [r["guide"][key] * k for r in rows], w, color=series(1),
               label="User's Guide")
        ax.set_xticks(x, [LABELS[r["event"]] for r in rows])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    figure_legend(fig, axes[0, 0], ncol=2)
    return fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    _cfg, veh, f = ra.load()
    res = ra.fly(veh, f)
    ref = {k: v for k, v in ra.GUIDE.items() if isinstance(v, dict) and "time" in v}

    plots.ascent_overview(res, reference=ref, until=950,
                          title="Rockot ascent: AeroSim model vs User's Guide (Fig. 3-6)"
                          ).savefig(FIG / "ascent_overview.png")
    plots.ascent_trajectory(res, until=res.events["Breeze-KM_burn1_end"],
                            title="Altitude vs downrange, lift-off to end of Breeze-KM burn 1"
                            ).savefig(FIG / "trajectory.png")
    plots.ground_track(res, site=(f["latitude"], f["longitude"]),
                       title="Ground track, lift-off to end of recorded coast"
                       ).savefig(FIG / "ground_track.png")

    comparison_bars(ra.comparison(res)).savefig(FIG / "comparison.png")

    # downsampled series for native PowerPoint charts (lift-off to burn 1 end + 60 s)
    m = res.time <= res.events["Breeze-KM_burn1_end"] + 60
    t = res.time[m]
    pick = np.unique(np.concatenate([np.searchsorted(t, np.arange(0, t[-1], 10.0)), [t.size - 1]]))
    series = {"time": t[pick].round(1).tolist(),
              "altitude_km": (res.altitude[m][pick] / 1e3).round(2).tolist(),
              "speed_kms": (res.speed[m][pick] / 1e3).round(3).tolist(),
              "q_kpa": (res.dynamic_pressure[m][pick] / 1e3).round(2).tolist()}

    comp = ra.comparison(res)
    o = {k: {"perigee_km": v.perigee_alt / 1e3, "apogee_km": v.apogee_alt / 1e3,
             "inclination": v.inclination, "period_min": v.period / 60}
         for k, v in res.orbits.items() if v.perigee_alt > -1e6}
    data = {
        "vehicle": {"liftoff_mass": veh.liftoff_mass, "payload": veh.payload_mass,
                    "fairing": veh.fairing_mass,
                    "stages": [{"name": s.name, "propellant": s.propellant_mass, "dry": s.dry_mass,
                                "thrust_vac_kN": s.thrust_vac / 1e3, "isp_vac": s.isp_vac,
                                "isp_sl": s.isp_sl, "burn_time": s.burn_time}
                               for s in veh.stages]},
        "site": {"lat": f["latitude"], "lon": f["longitude"]},
        "program": {k: f["program"][k] for k in f["program"]},
        "events": res.events,
        "max_q": {"time": res.events["max_q"],
                  "q_kpa": float(res.dynamic_pressure.max() / 1e3)},
        "peak_accel_g": {ph: float(res.acceleration[res.phase == i].max() / 9.80665)
                         for i, ph in enumerate(res.phases) if "coast" not in ph},
        "comparison": comp, "orbits": o,
        "propellant_left": res.meta["propellant_left"], "warnings": res.meta["warnings"],
        "series": series,
        "calibration": json.loads((ROOT / "outputs" / "rockot" / "calibration.json").read_text())
        if (ROOT / "outputs" / "rockot" / "calibration.json").exists() else None,
    }
    (BUILD / "data.json").write_text(json.dumps(data, indent=2, default=float))
    print(json.dumps({k: data[k] for k in ("max_q", "peak_accel_g", "orbits",
                                           "propellant_left", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
