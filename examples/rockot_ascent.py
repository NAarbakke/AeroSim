"""Rockot / Breeze-KM from Plesetsk: calibrate the mass model and flight programme to the
Rockot User's Guide reference mission (Figure 3-6), then fly the full mission.

    python examples/rockot_ascent.py            # calibrate + fly, writes outputs/rockot/
    python examples/rockot_ascent.py --no-fit   # fly the values already in the config

The guide publishes engines, burn times, Breeze-KM masses and lift-off mass, but not
stage 1/2 propellant masses; Encyclopedia Astronautica's values do not reproduce the
guide's own trajectory (stage 1 burns out ~300 m/s slow). Stage 1/2 propellant is
therefore fitted, together with the pitch programme, to the guide's waypoints and
its 107 t lift-off mass. Needs nothing beyond the base install.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim, solve_program
from aerosim.flight.vehicles import LaunchVehicle

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "orbital" / "rockot.yaml"

# Rockot User's Guide, Figure 3-6 (650 km circular, i = 86.583 deg)
GUIDE = {
    "max_q":               {"time": 50, "altitude": 10.7e3, "speed": 567, "fpa": 47.9,
                            "q": 61.29e3},
    "Stage 1_burnout":     {"time": 122, "altitude": 68.3e3, "speed": 3163, "fpa": 23.3},
    "fairing_jettison":    {"time": 173, "altitude": 125.6e3, "speed": 3599, "fpa": 16.0},
    "Stage 2_burnout":     {"time": 305, "altitude": 235.7e3, "speed": 5634, "fpa": 7.3},
    "Breeze-KM_burn1_end": {"time": 882, "altitude": 250.7e3, "speed": 7832, "fpa": -1.5},
    "transfer_orbit":      {"perigee": 160e3, "apogee": 635.3e3},
    "liftoff_mass": 107_000.0,
}


def _ev(name, key):
    k = {"fpa": "flight_path_angle"}.get(key, key)
    return lambda res, veh: res.at(name)[k]


TARGETS = [
    (_ev("Stage 1_burnout", "speed"), GUIDE["Stage 1_burnout"]["speed"], 25.0),
    (_ev("Stage 1_burnout", "altitude"), GUIDE["Stage 1_burnout"]["altitude"], 2e3),
    (_ev("Stage 1_burnout", "fpa"), GUIDE["Stage 1_burnout"]["fpa"], 1.0),
    (_ev("Stage 2_burnout", "speed"), GUIDE["Stage 2_burnout"]["speed"], 25.0),
    (_ev("Stage 2_burnout", "altitude"), GUIDE["Stage 2_burnout"]["altitude"], 4e3),
    (_ev("Stage 2_burnout", "fpa"), GUIDE["Stage 2_burnout"]["fpa"], 1.0),
    (_ev("max_q", "fpa"), GUIDE["max_q"]["fpa"], 3.0),
    (lambda res, veh: res.orbits["transfer"].perigee_alt, GUIDE["transfer_orbit"]["perigee"], 3e3),
    (lambda res, veh: res.events["Breeze-KM_burn1_end"], GUIDE["Breeze-KM_burn1_end"]["time"], 15.0),
    (lambda res, veh: veh.liftoff_mass, GUIDE["liftoff_mass"], 1500.0),
]

FREE = {
    "kick_angle": (19.0, 5.0, 30.0),
    "upper_pitch.0.0": (32.0, 0.0, 60.0),
    "upper_pitch.0.1": (14.0, -10.0, 40.0),
    "insertion_climb.0": (20.0, -50.0, 150.0),
    "insertion_climb.1": (-10.0, -100.0, 50.0),
    "vehicle.stages.0.propellant_mass": (79_000.0, 65_000.0, 90_000.0),
    "vehicle.stages.1.propellant_mass": (14_900.0, 9_000.0, 18_000.0),
}


def load():
    cfg = yaml.safe_load(CFG.read_text())
    veh = LaunchVehicle(**cfg["vehicle"])
    f = dict(cfg["flight"])
    f.pop("backend")
    return cfg, veh, f


def calibrate(veh, f):
    prog = AscentProgram(**{**f["program"], "circularise": False})
    sim = OrbitalAscentSim(f["latitude"], f["longitude"], program=prog, rtol=1e-7)
    prog, veh, _res, opt = solve_program(sim, veh, FREE, TARGETS, verbose=1, max_nfev=200)
    return prog, veh, opt


def fly(veh, f, prog=None):
    prog = prog or AscentProgram(**f["program"])
    prog.circularise, prog.coast_after = True, 1200.0
    return OrbitalAscentSim(f["latitude"], f["longitude"], program=prog).run(veh)


def comparison(res) -> list[dict]:
    rows = []
    for ev in ("max_q", "Stage 1_burnout", "fairing_jettison", "Stage 2_burnout",
               "Breeze-KM_burn1_end"):
        if ev not in res.events:
            continue
        a, g = res.at(ev), GUIDE[ev]
        rows.append({"event": ev, "guide": g,
                     "model": {"time": a["time"], "altitude": a["altitude"],
                               "speed": a["speed"], "fpa": a["flight_path_angle"],
                               "q": a["dynamic_pressure"]}})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "rockot"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, veh, f = load()
    prog = None
    if not args.no_fit:
        prog, veh, opt = calibrate(veh, f)
        fit = {"program": {k: getattr(prog, k) for k in ("kick_angle", "upper_pitch",
                                                        "insertion_climb")},
               "stage_propellant": [s.propellant_mass for s in veh.stages],
               "liftoff_mass": veh.liftoff_mass, "cost": float(opt.cost),
               "residuals": opt.fun.tolist()}
        (out / "calibration.json").write_text(json.dumps(fit, indent=2))
        print(json.dumps(fit, indent=2))
    res = fly(veh, f, prog)
    (out / "comparison.json").write_text(json.dumps(comparison(res), indent=2))
    for r in comparison(res):
        m, g = r["model"], r["guide"]
        print(f"{r['event']:22s} t {m['time']:6.0f}/{g['time']:<4}  h {m['altitude']/1e3:6.1f}/"
              f"{g['altitude']/1e3:<6} km  v {m['speed']:5.0f}/{g['speed']:<5}  "
              f"fpa {m['fpa']:5.1f}/{g['fpa']}")
    for k, o in res.orbits.items():
        print(f"{k:18s} {o.perigee_alt/1e3:7.1f} x {o.apogee_alt/1e3:7.1f} km  i {o.inclination:.3f}")
    print("Breeze-KM propellant left:", round(res.meta["propellant_left"]), "kg",
          res.meta["warnings"])
    return res, veh


if __name__ == "__main__":
    main()
