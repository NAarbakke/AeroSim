"""Rockot / Breeze-KM from first principles: predict the reference mission, then check it.

    python examples/rockot_first_principles.py            # optimise + payload sweep (~10 min)
    python examples/rockot_first_principles.py --no-opt   # fly the programme in the config

Unlike examples/rockot_ascent.py (which fits stage masses and steering to the User's Guide
trajectory), nothing here is tuned to the guide's trajectory:

- engines   NASA CEA at published chamber pressure / mixture ratio, textbook losses
- masses    propellant = engine mass flow x published burn time; dry masses as published
- drag      component build-up from the vehicle's length, diameter and nose
- steering  trajectory optimisation: maximise Breeze-KM propellant left in the 650 km orbit
            (transfer perigee >= 150 km); the guide's waypoints are compared afterwards

Writes outputs/rockot/first_principles.json. Needs the [chem] extra (NASA CEA).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim, optimize_program
from aerosim.flight.vehicles import LaunchVehicle

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "examples" / "rocket" / "orbital" / "rockot_first_principles.yaml"

FREE = {
    "kick_angle": (15.0, 5.0, 30.0),
    "upper_pitch.0.0": (25.0, -5.0, 60.0),
    "upper_pitch.0.1": (5.0, -20.0, 40.0),
    "insertion_climb.0": (20.0, -100.0, 200.0),
    "insertion_climb.1": (-50.0, -200.0, 100.0),
}
CONSTRAINTS = [
    (lambda res, veh: res.orbits["transfer"].perigee_alt, 150e3, None, 5e3),   # no re-entry
    (lambda res, veh: res.orbits["final"].perigee_alt, 645e3, None, 2e3),      # orbit reached
]


def load():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    veh = LaunchVehicle(**cfg["vehicle"])
    f = dict(cfg["flight"])
    f.pop("backend")
    return cfg, veh, f


def optimise(veh, f, maxfev=300):
    sim = OrbitalAscentSim(f["latitude"], f["longitude"], program=AscentProgram(**f["program"]),
                           rtol=1e-7)
    prog, _veh, res, opt = optimize_program(sim, veh, FREE, constraints=CONSTRAINTS,
                                            maxfev=maxfev, verbose=True)
    return prog, res, opt


def fly(veh, f, prog=None, coast_after=1200.0):
    prog = copy.deepcopy(prog) if prog else AscentProgram(**f["program"])
    prog.coast_after = coast_after
    return OrbitalAscentSim(f["latitude"], f["longitude"], program=prog).run(veh)


def max_payload(veh, f, prog, lo=500.0, hi=4000.0, tol=10.0):
    """Payload at which Breeze-KM just reaches the final orbit, flying the same programme
    (burn 1 is closed-loop, so it adapts to the heavier stack)."""
    def margin(m):
        v = copy.deepcopy(veh)
        v.payload_mass = m
        r = fly(v, f, prog, coast_after=0.0)
        return -1.0 if r.meta["warnings"] else r.meta["propellant_left"]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin(mid) > 0 else (lo, mid)
    return lo


def engines_table(veh):
    rows = []
    for s in veh.stages:
        for e in getattr(s, "engines", []):
            rows.append({k: e.get(k) for k in ("name", "count", "thrust_vac", "mdot", "isp_vac",
                                                "isp_sl", "expansion_ratio", "chamber_pressure",
                                                "mixture_ratio", "t_chamber", "isp_ideal",
                                                "isp_ideal_equilibrium", "source")} | {"stage": s.name})
    return rows


def main(argv=None):
    import rockot_ascent as ra  # guide waypoints + comparison helper

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-opt", action="store_true")
    ap.add_argument("--maxfev", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "rockot"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, veh, f = load()
    prog, opt = None, None
    if not args.no_opt:
        prog, _res, opt = optimise(veh, f, args.maxfev)
    prog = prog or AscentProgram(**f["program"])
    res = fly(veh, f, prog)
    payload = max_payload(veh, f, prog)
    comp = ra.comparison(res)
    summary = {
        "program": {k: getattr(prog, k) for k in ("kick_angle", "upper_pitch", "insertion_climb")},
        "optimiser": None if opt is None else {"nfev": int(opt.nfev), "cost": float(opt.fun)},
        "engines": engines_table(veh),
        "stages": [{"name": s.name, "propellant": s.propellant_mass, "dry": s.dry_mass,
                    "isp_vac": s.isp_vac, "isp_sl": s.isp_sl, "burn_time": s.burn_time,
                    "thrust_vac": s.thrust_vac} for s in veh.stages],
        "liftoff_mass": veh.liftoff_mass, "drag_curve": veh.drag_curve,
        "propellant_left": res.meta["propellant_left"], "warnings": res.meta["warnings"],
        "max_payload_650km": payload,
        "orbits": {k: {"perigee_km": o.perigee_alt / 1e3, "apogee_km": o.apogee_alt / 1e3,
                       "inclination": o.inclination} for k, o in res.orbits.items()
                   if o.perigee_alt > -1e6},
        "max_q": {"time": res.events["max_q"], "q": float(res.dynamic_pressure.max())},
        "comparison": comp,
    }
    (out / "first_principles.json").write_text(json.dumps(summary, indent=2, default=float),
                                               encoding="utf-8")
    for r in comp:
        m, g = r["model"], r["guide"]
        print(f"{r['event']:22s} t {m['time']:6.0f}/{g['time']:<4}  h {m['altitude']/1e3:6.1f}/"
              f"{g['altitude']/1e3:<6} km  v {m['speed']:5.0f}/{g['speed']:<5}  "
              f"fpa {m['fpa']:5.1f}/{g['fpa']}")
    for k, o in summary["orbits"].items():
        print(f"{k:18s} {o['perigee_km']:7.1f} x {o['apogee_km']:7.1f} km  i {o['inclination']:.3f}")
    print(f"lift-off {veh.liftoff_mass/1e3:.1f} t   propellant left {res.meta['propellant_left']:.0f} kg"
          f"   max payload to 650 km {payload:.0f} kg (guide 1570)   program {summary['program']}")
    return res, veh, prog


if __name__ == "__main__":
    main()
