"""Command line: ``aerosim backends`` | ``aerosim run mission.yaml`` | ``aerosim atm 10000``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aerosim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backends", help="list registered backends per layer")
    run = sub.add_parser("run", help="run a mission YAML")
    run.add_argument("config")
    run.add_argument("--eng", help="also export the thrust curve as a RASP .eng file")
    atm = sub.add_parser("atm", help="print standard-atmosphere state at an altitude (m)")
    atm.add_argument("altitude", type=float)
    args = ap.parse_args(argv)

    if args.cmd == "backends":
        import aerosim.missions  # noqa: F401  registers everything
        from aerosim.core import registry
        print(json.dumps(registry.available(), indent=2))

    elif args.cmd == "atm":
        from aerosim.atmosphere import StandardAtmosphere
        s = StandardAtmosphere().at(args.altitude)
        print(json.dumps(s.__dict__, indent=2))

    elif args.cmd == "run":
        from aerosim.core import ThrustCurve
        from aerosim.missions import run_mission
        out = run_mission(args.config)
        r = out.flight
        if r.meta.get("simulator") == "jsbsim":           # aircraft: where it ended up
            m = r.meta
            summary = {"duration_s": round(float(r.time[-1]), 1),
                       "final_altitude_m": round(float(r.position[-1, 2]), 1),
                       "final_mach": round(float(m["mach"][-1]), 3),
                       "final_heading_deg": round(float(m["heading_deg"][-1]), 1),
                       "fuel_used_kg": round(m["fuel_used"], 1), "targets": m["targets"]}
        else:
            summary = {"apogee_m": round(r.apogee, 1), "max_speed_m_s": round(r.max_speed, 1),
                       "events_s": {k: round(v, 2) for k, v in r.events.items()}}
        orbit = getattr(r, "final_orbit", None)
        if orbit is not None:
            summary |= {"final_orbit_km": [round(orbit.perigee_alt / 1e3, 1),
                                           round(orbit.apogee_alt / 1e3, 1)],
                        "inclination_deg": round(orbit.inclination, 3),
                        "max_q_kPa": round(float(r.dynamic_pressure.max()) / 1e3, 1)}
        if isinstance(out.propulsion, ThrustCurve):
            p = out.propulsion
            summary |= {"total_impulse_Ns": round(p.total_impulse, 1),
                        "burn_time_s": round(p.burn_time, 2), "avg_isp_s": round(p.average_isp, 1)}
            if args.eng:
                from aerosim.core import write_eng
                write_eng(p, args.eng)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
