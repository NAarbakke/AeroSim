"""Strela, Start-1, Long March 2C / CTS and Dnepr: calibrate each mass model and flight
programme to its user's manual, then fly the full mission (Rockot: examples/rockot_ascent.py).

    python examples/slv_ascent.py strela            # calibrate + fly, writes outputs/strela/
    python examples/slv_ascent.py dnepr --no-fit    # fly the values already in the config
    python examples/slv_ascent.py all

Strela, Start-1 and LM-2C publish a reference trajectory: the free mass and steering values
are fitted to its waypoints by least squares, as for Rockot. Dnepr publishes none, so its
steering is optimised for the most propellant left on the guide's headline mission and the
result is the largest payload the model can carry there. The fitted values are printed and
saved in outputs/<vehicle>/calibration.json; copy them into the config by hand. Profiles
and the reasoning behind each value: docs/references/<vehicle>/profile/.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml

from aerosim.flight.rocket.orbital_ascent import (
    AscentProgram,
    OrbitalAscentSim,
    optimize_program,
    solve_program,
)
from aerosim.flight.vehicles import LaunchVehicle

ROOT = Path(__file__).resolve().parents[1]


def _lm_fpa(fpa, range_km):
    """LM-2C Table 3-2 gives the "ballistic inclination": the velocity's angle above the launch
    point's horizontal plane. Above the local horizontal it is larger by the central angle flown."""
    return fpa + math.degrees(range_km / 6378.14)


# Guide waypoints: model event -> label and guide values (time s, altitude m, speed m/s
# Earth-relative, flight-path angle deg above the local horizontal; None = not given)
GUIDES = {
    "strela": [                                   # UM Fig. 3.4
        ("max_q", "Max q", {"time": 51, "altitude": None, "speed": 566, "fpa": 49, "q": 65.77e3}),
        ("Stage 1_separation", "Stage 1 separation",
         {"time": 126.1, "altitude": 70e3, "speed": 3555, "fpa": 20}),
        ("fairing_jettison", "Fairing jettison", {"time": 164.25, "altitude": 114e3, "speed": 3992, "fpa": 15}),
        ("Stage 2_burnout", "Stage 2 / MIS separation",
         {"time": 309.1, "altitude": 240e3, "speed": 7727, "fpa": 7}),
        ("MIS_burn2_start", "MIS ignition at apogee", {"time": 1575, "altitude": 1000e3, "speed": 6894, "fpa": 0}),
        ("MIS_burn2_end", "SC separation", {"time": 1760, "altitude": 1000e3, "speed": 7371, "fpa": 0}),
    ],
    "start1": [                                   # UH Fig. 2-3
        ("Stage 1_separation", "Stage 1 separation", {"time": 87, "altitude": 39.7e3, "speed": 1283, "fpa": 27.6}),
        ("Stage 2_separation", "Stage 2 separation", {"time": 148, "altitude": 86.5e3, "speed": 3347, "fpa": 18.1}),
        ("Stage 3_burnout", "Stage 3 separation", {"time": 211, "altitude": 167.9e3, "speed": 5535, "fpa": 19.2}),
        ("fairing_jettison", "Fairing release", {"time": 295, "altitude": 303.7e3, "speed": 5306, "fpa": 15.0}),
        ("Stage 4_burn2_start", "Stage 4 ignition", {"time": 488, "altitude": 477.8e3, "speed": 5012, "fpa": 4.7}),
        ("Stage 4_burn2_end", "Stage 4 burnout", {"time": 538, "altitude": 478.7e3, "speed": 7704, "fpa": 0.0}),
    ],
    "long_march_2c": [                            # UM Table 3-2
        ("Stage 1_burnout", "Stage 1 shutdown",
         {"time": 120.27, "altitude": 47.052e3, "speed": 2035.9, "fpa": _lm_fpa(22.765, 61.755)}),
        ("Stage 1_separation", "Stage 1/2 separation",
         {"time": 121.77, "altitude": 48.257e3, "speed": 2043.8, "fpa": _lm_fpa(22.403, 64.549)}),
        ("fairing_jettison", "Fairing jettison",
         {"time": 231.67, "altitude": 117.618e3, "speed": 3698.2, "fpa": _lm_fpa(4.263, 352.768)}),
        ("Stage 2_burnout", "Stage 2 main engine shutdown",
         {"time": 301.184, "altitude": 146.895e3, "speed": 6379.4, "fpa": _lm_fpa(-2.540, 679.624)}),
        ("Stage 2 verniers_burnout", "Vernier shutdown",
         {"time": 613.333, "altitude": 181.142e3, "speed": 7917.7, "fpa": _lm_fpa(-25.629, 2825.723)}),
        ("CTS_burn2_start", "CTS ignition at apogee",
         {"time": 2888.347, "altitude": 637.804e3, "speed": 7402.7, "fpa": None}),
        ("CTS_burn2_end", "CTS velocity trim end",
         {"time": 2928.347, "altitude": 639.455e3, "speed": 7512.4, "fpa": None}),
    ],
    "dnepr": [],
}


def _ev(name, key):
    k = {"fpa": "flight_path_angle"}.get(key, key)
    return lambda res, veh: res.at(name)[k]


def _waypoints(guide, *specs):
    """Least-squares targets on guide waypoints: (event, key, scale) for each spec."""
    g = {e: v for e, _lab, v in guide}
    return [(_ev(e, k), g[e][k], s) for e, k, s in specs]


def _transfer(key):
    return lambda res, veh: getattr(res.orbits["transfer"], key)


def _event_time(name):
    return lambda res, veh: res.events.get(name, res.time[-1] + 600.0)   # missed: far off


def _liftoff(res, veh):
    return veh.liftoff_mass


S = GUIDES["strela"]
T = GUIDES["start1"]
L = GUIDES["long_march_2c"]
VEHICLES = {
    "strela": {
        "config": "strela.yaml", "mode": "fit",
        "targets": _waypoints(S, ("Stage 1_separation", "speed", 25.0),
                              ("Stage 1_separation", "altitude", 2e3), ("Stage 1_separation", "fpa", 1.0),
                              ("Stage 2_burnout", "speed", 25.0), ("Stage 2_burnout", "altitude", 4e3),
                              ("Stage 2_burnout", "fpa", 1.0), ("max_q", "fpa", 3.0))
        + [(_transfer("apogee_alt"), 1000e3, 5e3),
           (_event_time("fairing_jettison"), 164.25, 5.0),
           (_liftoff, 105_000.0, 1500.0)],                   # UM s 2: ~105 t
        "free": {
            "kick_angle": (18.0, 5.0, 30.0),
            "upper_pitch.0.0": (22.0, 0.0, 60.0),
            "upper_pitch.0.1": (0.0, -20.0, 40.0),
            "fairing_heat_flux": (1135.0, 100.0, 5000.0),
            "vehicle.stages.0.propellant_mass": (80_000.0, 65_000.0, 95_000.0),
            "vehicle.stages.1.propellant_mass": (14_000.0, 9_000.0, 20_000.0),
        },
    },
    "start1": {
        "config": "start1.yaml", "mode": "fit",
        "targets": _waypoints(T, ("Stage 1_separation", "speed", 25.0),
                              ("Stage 1_separation", "altitude", 2e3), ("Stage 1_separation", "fpa", 1.0),
                              ("Stage 2_separation", "speed", 25.0), ("Stage 2_separation", "altitude", 3e3),
                              ("Stage 2_separation", "fpa", 1.0), ("Stage 3_burnout", "speed", 25.0),
                              ("Stage 3_burnout", "altitude", 4e3), ("Stage 3_burnout", "fpa", 1.0))
        + [(_transfer("apogee_alt"), 479e3, 3e3),            # stage 4 burns at 478-479 km
           (_event_time("fairing_jettison"), 295.0, 10.0),
           (_liftoff, 47_000.0, 1000.0)],                    # UH s 1.1
        "free": {
            "kick_angle": (25.2, 5.0, 35.0),
            "upper_pitch.0.0": (25.3, 0.0, 60.0),
            "upper_pitch.0.1": (26.6, -10.0, 50.0),
            "upper_pitch.1.0": (26.8, 0.0, 50.0),
            "upper_pitch.1.1": (33.3, -10.0, 50.0),
            "fairing_heat_flux": (1.62, 0.05, 100.0),
            "vehicle.stages.0.propellant_mass": (26_400.0, 18_000.0, 32_000.0),
            "vehicle.stages.1.propellant_mass": (11_300.0, 6_000.0, 14_000.0),
            "vehicle.stages.2.propellant_mass": (4_800.0, 2_500.0, 7_000.0),
            "vehicle.stages.0.isp_vac": (280.0, 255.0, 285.0),      # the UH gives no Isp
            "vehicle.stages.1.isp_vac": (295.0, 265.0, 300.0),
        },
    },
    "long_march_2c": {
        "config": "long_march_2c.yaml", "mode": "fit",
        "targets": _waypoints(L, ("Stage 1_burnout", "speed", 25.0), ("Stage 1_burnout", "altitude", 2e3),
                              ("Stage 1_burnout", "fpa", 1.0), ("Stage 2_burnout", "speed", 25.0),
                              ("Stage 2_burnout", "altitude", 3e3), ("Stage 2_burnout", "fpa", 1.0),
                              ("Stage 2 verniers_burnout", "speed", 25.0),
                              ("Stage 2 verniers_burnout", "altitude", 3e3),
                              ("Stage 2 verniers_burnout", "fpa", 1.0))
        + [(_transfer("apogee_alt"), 637.8e3, 3e3)],
        "free": {                                            # lift-off mass left free: the UM
            "kick_angle": (8.92, 3.0, 30.0),                 # quotes both 233 t and 213 t
            "upper_pitch.0.0": (28.2, -10.0, 50.0),
            "upper_pitch.0.1": (1.6, -20.0, 40.0),
            "upper_pitch.1.0": (14.9, -20.0, 30.0),
            "upper_pitch.1.1": (-5.4, -30.0, 30.0),
            "vehicle.stages.0.dry_mass": (11_250.0, 4_000.0, 40_000.0),
            "vehicle.stages.2.dry_mass": (4_300.0, 1_000.0, 10_000.0),
            "vehicle.stages.0.isp_vac": (289.0, 275.0, 300.0),      # outside the UM
        },
    },
    "dnepr": {
        "config": "dnepr.yaml", "mode": "optimise",
        "free": {
            "kick_angle": (10.0, 3.0, 45.0),
            "upper_pitch.0.0": (5.0, -20.0, 40.0),
            "upper_pitch.0.1": (-15.0, -40.0, 20.0),
            "insertion_climb.0": (0.0, -100.0, 200.0),
            "insertion_climb.1": (0.0, -200.0, 100.0),
        },
        "constraints": [
            (lambda res, veh: res.orbits["transfer"].perigee_alt, 150e3, None, 5e3),   # no re-entry
            (lambda res, veh: res.orbits["final"].perigee_alt, 295e3, None, 2e3),      # orbit reached
            (lambda res, veh: res.orbits["final"].apogee_alt, None, 310e3, 2e3),       # and no higher
        ],
    },
}


def load(name):
    cfg = yaml.safe_load((ROOT / "configs" / "orbital" / VEHICLES[name]["config"])
                         .read_text(encoding="utf-8"))
    veh = LaunchVehicle(**cfg["vehicle"])
    f = dict(cfg["flight"])
    f.pop("backend")
    return cfg, veh, f


def _sim(f, prog, rtol=1e-8):
    return OrbitalAscentSim(f["latitude"], f["longitude"], f.get("site_height", 0.0),
                            program=prog, rtol=rtol)


def calibrate(name, veh, f, maxfev=200):
    v = VEHICLES[name]
    if v["mode"] == "fit":
        prog = AscentProgram(**f["program"])
        prog.circularise = not prog.insertion_burn    # kick stage: the coast holds targets
        prog, veh, _res, opt = solve_program(_sim(f, prog, 1e-7), veh, v["free"], v["targets"],
                                             verbose=1, max_nfev=maxfev)
    else:
        prog, veh, _res, opt = optimize_program(_sim(f, AscentProgram(**f["program"]), 1e-7), veh,
                                                v["free"], constraints=v["constraints"],
                                                maxfev=maxfev, verbose=True)
    prog.circularise = True
    return prog, veh, opt


def fly(veh, f, prog=None, coast_after=1200.0):
    prog = copy.deepcopy(prog) if prog else AscentProgram(**f["program"])
    prog.coast_after = coast_after
    return _sim(f, prog).run(veh)


def max_payload(veh, f, prog=None, lo=0.0, hi=6000.0, tol=10.0):
    """Payload at which the last stage just reaches the final orbit, on the same programme."""
    target = AscentProgram(**f["program"]).final_perigee or AscentProgram(**f["program"]).transfer_apogee

    def margin(m):
        v = copy.deepcopy(veh)
        v.payload_mass = m
        r = fly(v, f, prog, coast_after=0.0)
        ok = not r.meta["warnings"] and r.orbits["final"].perigee_alt > target - 2e3
        return r.meta["propellant_left"] if ok else -1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin(mid) > 0 else (lo, mid)
    return lo


def comparison(name, res) -> list[dict]:
    rows = []
    for ev, label, g in GUIDES[name]:
        if ev not in res.events:
            continue
        a = res.at(ev)
        rows.append({"event": ev, "label": label, "guide": g,
                     "model": {"time": a["time"], "altitude": a["altitude"], "speed": a["speed"],
                               "fpa": a["flight_path_angle"], "q": a["dynamic_pressure"]}})
    return rows


def _fmt(x, scale=1.0, nd=0):
    return "  -  " if x is None else f"{x / scale:.{nd}f}"


def run(name, fit=True, out=None, maxfev=200):
    out = Path(out or ROOT / "outputs" / name)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, veh, f = load(name)
    prog = None
    if fit:
        prog, veh, opt = calibrate(name, veh, f, maxfev)
        p = {k: getattr(prog, k) for k in ("kick_angle", "upper_pitch", "insertion_climb",
                                           "fairing_heat_flux")}
        cal = {"program": p, "stages": [{"name": s.name, "propellant_mass": s.propellant_mass,
                                         "dry_mass": s.dry_mass, "isp_vac": s.isp_vac,
                                         "isp_sl": s.isp_sl} for s in veh.stages],
               "liftoff_mass": veh.liftoff_mass, "cost": float(opt.fun if VEHICLES[name]["mode"]
                                                               == "optimise" else opt.cost)}
        (out / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        print(json.dumps(cal, indent=2))
    res = fly(veh, f, prog)
    rows = comparison(name, res)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{name}: lift-off {veh.liftoff_mass / 1e3:.1f} t   (model / guide)")
    for r in rows:
        m, g = r["model"], r["guide"]
        print(f"  {r['label']:30s} t {m['time']:6.0f}/{_fmt(g['time']):<6} h {m['altitude'] / 1e3:6.1f}/"
              f"{_fmt(g['altitude'], 1e3, 1):<6} km  v {m['speed']:5.0f}/{_fmt(g['speed']):<5}  "
              f"fpa {m['fpa']:5.1f}/{_fmt(g['fpa'], 1, 1)}")
    for k, o in res.orbits.items():
        print(f"  {k:26s} {o.perigee_alt / 1e3:7.1f} x {o.apogee_alt / 1e3:7.1f} km  i {o.inclination:.3f}")
    print(f"  {veh.stages[-1].name} propellant left: {res.meta['propellant_left']:.0f} kg",
          res.meta["warnings"])
    if VEHICLES[name]["mode"] == "optimise":
        print(f"  largest payload on this programme: {max_payload(veh, f, prog):.0f} kg "
              f"(guide {veh.payload_mass:.0f} kg)")
    return res, veh


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vehicle", choices=[*VEHICLES, "all"])
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--maxfev", type=int, default=200)
    args = ap.parse_args(argv)
    for name in VEHICLES if args.vehicle == "all" else [args.vehicle]:
        run(name, not args.no_fit, maxfev=args.maxfev)


if __name__ == "__main__":
    main()
