"""Rockot / Breeze-KM note: fly the mission, write the figure + data.json for make_pdf.py and deck.js.

    python docs/briefings/rockot/make_briefing.py     # then make_pdf.py and node build/deck.js

Everything quoted in the note and deck comes from this run (``build/data.json``) or from
the Rockot User's Guide (docs/references/rockot/).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "examples"))

import rockot_ascent as ra

from aerosim import plots

BUILD = HERE / "build"
FIG = BUILD / "figures"
G0 = 9.80665


def ideal_dv(veh):
    """Ideal vacuum delta-v per stage (m/s); fairing carried to stage-2 burnout."""
    m0, out = veh.liftoff_mass, []
    for i, s in enumerate(veh.stages):
        mf = m0 - s.propellant_mass
        out.append(s.isp_vac * G0 * math.log(m0 / mf))
        m0 = mf - s.dry_mass - (veh.fairing_mass if i == 1 else 0)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    _cfg, veh, f = ra.load()
    res = ra.fly(veh, f)
    ref = {k: v for k, v in ra.GUIDE.items() if isinstance(v, dict) and "time" in v}

    plots.ascent_overview(res, reference=ref, until=950, title=None
                          ).savefig(FIG / "report_ascent.png")

    comp = ra.comparison(res)
    o = {k: {"perigee_km": v.perigee_alt / 1e3, "apogee_km": v.apogee_alt / 1e3,
             "inclination": v.inclination, "period_min": v.period / 60}
         for k, v in res.orbits.items() if v.perigee_alt > -1e6}
    s1 = veh.stages[0]
    data = {
        "vehicle": {"liftoff_mass": veh.liftoff_mass, "payload": veh.payload_mass,
                    "liftoff_tw": s1.mass_flow * s1.isp_sl / veh.liftoff_mass,
                    "ideal_dv": ideal_dv(veh),
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
        "calibration": json.loads((ROOT / "outputs" / "rockot" / "calibration.json")
                                  .read_text(encoding="utf-8"))
        if (ROOT / "outputs" / "rockot" / "calibration.json").exists() else None,
    }
    (BUILD / "data.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8",
                                     newline="\n")
    print(json.dumps({k: data[k] for k in ("max_q", "peak_accel_g", "orbits",
                                           "propellant_left", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
