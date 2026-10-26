"""Build the AeroSim results site: 3D replays and the engines page, all offline HTML.

    python examples/build_site.py            # -> docs/site/index.html (about 15 minutes)
    python examples/build_site.py --reuse    # reuse cached engines data (about 2 minutes)

Pages: Rockot ascent (globe replay), booster landing, F-16 flight, hybrid sounding rocket
(local-frame 3D replays) and engines (liquid cycles, regen cooling, hybrid burn, ramjet /
scramjet, pyCycle gas turbines). Every number on them is computed here from AeroSim models;
published figures are the ones the tests check against.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "site"
sys.path.insert(0, str(ROOT / "examples"))

from aerosim.atmosphere.standard import isa
from aerosim.core.units import G0
from aerosim.plots import site
from aerosim.plots.ascent_closeup import mach1_time
from aerosim.plots.web import nav

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


def write(name: str, text: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    log(f"wrote {name} ({len(text) / 1e6:.2f} MB)")


# ------------------------------------------------------------------ replays
def rockot():
    import rockot_ascent as ra
    import rockot_first_principles as fp

    from aerosim.plots.replay import replay_html
    _c, veh_cal, f = ra.load()
    _c, veh_fp, f_fp = fp.load()
    res = {"Calibrated to the guide": ra.fly(veh_cal, f), "First principles": fp.fly(veh_fp, f_fp)}
    write("rockot.html", replay_html(res, (f["latitude"], f["longitude"]),
                                     "Rockot / Breeze-KM ascent from Plesetsk",
                                     "650 km circular, 86.6°  ·  AeroSim orbital_ascent",
                                     nav=nav(site.PAGES, "rockot.html")))
    rockot_closeup(res["First principles"], veh_fp)
    return res["First principles"]


def rockot_closeup(res, veh):
    from aerosim.flight.stability import ROCKOT_LAYOUT
    from aerosim.plots.ascent_closeup import ascent_closeup_html
    t_end = res.events.get("Breeze-KM_burn1_end", res.time[-1]) + 30
    write("rockot-closeup.html", ascent_closeup_html(
        res, veh, title="Rockot through the atmosphere",
        subtitle="First-principles Rockot / Breeze-KM from Plesetsk, up close  ·  AeroSim orbital_ascent",
        nav=nav(site.PAGES, "rockot-closeup.html"), t_end=t_end, layout=ROCKOT_LAYOUT))


def landing():
    from aerosim.gnc.landing import Booster, simulate_landing
    b = Booster(dry_mass=25e3, propellant_mass=4e3, thrust_max=845e3, throttle_min=0.4,
                isp=282, diameter=3.66)
    models, stats = [], {}
    for name, wind in (("No wind", (0.0, 0.0)), ("Wind 8 m/s east, 3 m/s north", (8.0, 3.0))):
        f = simulate_landing(b, [200, 0, 1800], [-15, 0, -170], wind=wind)
        F = np.linalg.norm(f.thrust, axis=1)
        thr = F / b.thrust_max
        d = np.where(F[:, None] > 0, f.thrust / np.maximum(F, 1e-9)[:, None], [0, 0, 1])
        ev = [("Landing burn", 0.0), ("Full throttle", float(f.time[np.argmax(thr > 0.95)])),
              ("Touchdown", float(f.time[-1]))]
        models.append({"name": name, "t": f.time, "pos": f.position,
                       "channels": {"alt": f.position[:, 2], "v": np.linalg.norm(f.velocity, axis=1),
                                    "vz": f.velocity[:, 2], "thr": thr * 100, "m": f.mass / 1e3,
                                    "dist": np.linalg.norm(f.position[:, :2], axis=1)},
                       "events": ev, "throttle": thr, "thrust_dir": d})
        stats[name] = (f.touchdown_speed, f.touchdown_miss, f.propellant_used, f.replans)
        log(f"landing {name}: touchdown {f.touchdown_speed:.2f} m/s, miss {f.touchdown_miss:.2f} m")
    write("landing.html", site.scene_replay_html(
        models, "Booster landing burn: convex guidance, replanned with drag and wind",
        cards=[{"k": "alt", "label": "altitude, m", "d": 0}, {"k": "vz", "label": "vertical speed, m/s", "d": 1},
               {"k": "thr", "label": "throttle, %", "d": 0}, {"k": "m", "label": "mass, t", "d": 2},
               {"k": "dist", "label": "distance to pad, m", "d": 0}, {"k": "v", "label": "speed, m/s", "d": 1}],
        charts=[{"k": "alt", "label": "Altitude (m)"}, {"k": "thr", "label": "Throttle (% of 845 kN)"}],
        body="booster", vehicle={"length": 47.0, "diameter": 3.66},
        subtitle="Falcon 9-class first stage, one engine, 40 % minimum throttle  ·  gnc/landing.py",
        rates=(0.25, 0.5, 1, 2), rate=1, current="landing.html", zero=("alt", "thr"),
        phase0="Landing burn"))
    s = stats["Wind 8 m/s east, 3 m/s north"]
    return {"touchdown, m/s": f"{s[0]:.1f}", "miss, m": f"{s[1]:.1f}", "propellant, t": f"{s[2] / 1e3:.2f}"}


def f16():
    from aerosim.missions.runner import run_mission
    out = run_mission(ROOT / "configs" / "aircraft" / "turbojet.yaml")
    r, m = out.flight, out.flight.meta
    h = r.position[:, 2]
    tgt = m["targets"]
    t_alt = float(r.time[np.argmax(np.abs(h - tgt["altitude"]) < 15)])
    t_hdg = float(r.time[np.argmax(np.abs(m["heading_deg"] - tgt["heading"]) < 2)])
    t_m = float(r.time[np.argmax((np.abs(m["mach"] - tgt["mach"]) < 0.005) & (r.time > 30))])
    ev = sorted([("Altitude captured", t_alt), ("Heading captured", t_hdg), ("Mach captured", t_m)],
                key=lambda e: e[1])
    model = {"name": "F-16 on AeroSim turbojet deck", "t": r.time, "pos": r.position,
             "channels": {"alt": h, "mach": m["mach"], "pwr": m["power"] * 100, "F": m["thrust"] / 1e3,
                          "hdg": m["heading_deg"], "mass": r.mass / 1e3},
             "events": ev, "attitude": {"pitch": m["pitch_deg"], "roll": m["roll_deg"],
                                        "heading": m["heading_deg"]},
             "throttle": m["power"], "thrust_dir": r.velocity / np.linalg.norm(r.velocity, axis=1)[:, None]}
    write("f16.html", site.scene_replay_html(
        [model], "F-16 airframe (JSBSim) on an AeroSim turbojet deck",
        cards=[{"k": "alt", "label": "altitude, m", "d": 0}, {"k": "mach", "label": "Mach", "d": 3},
               {"k": "hdg", "label": "heading, deg", "d": 1}, {"k": "pwr", "label": "power, %", "d": 0},
               {"k": "F", "label": "thrust, kN", "d": 1}, {"k": "mass", "label": "mass, t", "d": 3}],
        charts=[{"k": "alt", "label": "Altitude (m)"}, {"k": "mach", "label": "Mach"}],
        body="aircraft", vehicle={"length": 15.0, "diameter": 1.2},
        subtitle=f"autopilot: {tgt['altitude']:.0f} m, Mach {tgt['mach']}, heading {tgt['heading']}°  ·  flight/adapters/jsbsim_sim.py",
        rates=(2, 5, 10, 30), rate=10, current="f16.html", exag=5, phase0="Climbing turn"))
    log("f16 done")
    return {"final altitude, m": f"{h[-1]:.0f}", "final Mach": f"{m['mach'][-1]:.3f}",
            "fuel used, kg": f"{m['fuel_used']:.0f}"}


def terrain_stats() -> dict:
    """Hub tile for the terrain pages, which ``build_terrain_site.py`` writes into this site."""
    import json
    path = ROOT / "outputs" / "site" / "terrain.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"regions": "2", "flights": "4", "build": "build_terrain_site.py"}


def hybrid():
    from aerosim.missions.runner import run_mission
    out = run_mission(ROOT / "configs" / "hybrid" / "n2o_htpb_sounding.yaml")
    tc, r = out.propulsion, out.flight
    F = np.interp(r.time, tc.time, tc.thrust, right=0.0)
    pc = np.interp(r.time, tc.time, tc.chamber_pressure, right=0.0)
    v = np.linalg.norm(r.velocity, axis=1)
    model = {"name": "N2O / HTPB sounding rocket", "t": r.time, "pos": r.position,
             "channels": {"alt": r.position[:, 2], "v": v, "F": F / 1e3, "pc": pc / 1e6, "m": r.mass,
                          "dr": np.linalg.norm(r.position[:, :2], axis=1)},
             "events": sorted(((k.capitalize(), float(t)) for k, t in r.events.items()), key=lambda e: e[1]),
             "throttle": F / F.max(), "thrust_dir": r.velocity / np.maximum(v, 1e-9)[:, None]}
    write("hybrid.html", site.scene_replay_html(
        [model], "Hybrid sounding rocket: N2O self-pressurising tank over an HTPB grain",
        cards=[{"k": "alt", "label": "altitude, m", "d": 0}, {"k": "v", "label": "speed, m/s", "d": 0},
               {"k": "F", "label": "thrust, kN", "d": 2}, {"k": "pc", "label": "chamber, MPa", "d": 2},
               {"k": "m", "label": "mass, kg", "d": 1}, {"k": "dr", "label": "downrange, m", "d": 0}],
        charts=[{"k": "alt", "label": "Altitude (m)"}, {"k": "F", "label": "Vacuum thrust (kN)"}],
        body="rocket", vehicle={"length": 3.2, "diameter": 0.12},
        subtitle="configs/hybrid/n2o_htpb_sounding.yaml  ·  point-mass flight, ISA, no wind",
        rates=(1, 2, 5, 10), rate=2, current="hybrid.html", zero=("alt", "F"), phase0="Boost"))
    return tc, r


# ------------------------------------------------------------------ engines page
def cycles():
    from aerosim.propulsion.liquid.cycle import CycleEngine
    f1 = CycleEngine("gas_generator", "LOX", "RP1", 2.27, 7.0e6, 16, 7.77e6,
                     turbine_inlet_temperature=1089, turbine_pressure_ratio=16)
    rs = CycleEngine("staged_combustion", "LOX", "LH2", 6.03, 20.64e6, 69, 2.09e6,
                     turbine_inlet_temperature=1000, pump_efficiency=0.75, turbine_efficiency=0.75,
                     tank_pressure_ox=2.9e6, tank_pressure_fuel=1.9e6)
    rd = CycleEngine("staged_combustion", "LOX", "RP1", 2.72, 26.7e6, 36.87, 4.15e6, rich="oxidizer",
                     turbine_inlet_temperature=772, pump_efficiency=0.75, turbine_efficiency=0.8,
                     injector_dp=0.15)
    rl = CycleEngine("expander", "LOX", "LH2", 5.0, 3.275e6, 61, 73.4e3,
                     regen={"material": "ss347", "channel_height": 4e-3, "l_star": 0.9})
    engines = [("F-1", "gas generator", f1), ("RS-25", "staged, fuel-rich", rs),
               ("RD-180", "staged, ox-rich", rd), ("RL10-class", "expander (pyskyfire jacket)", rl)]
    sol = {n: e.solve() for n, _, e in engines}
    log("cycles solved")
    pct = lambda a, b: 100 * (a / b - 1)
    dev = [("F-1 turbopump power", sol["F-1"]["turbine_power"] / 1e6, 41, "MW", 0),
           ("F-1 vacuum Isp", sol["F-1"]["isp_vac"], 304, "s", 0),
           ("F-1 sea-level Isp", sol["F-1"]["isp_sl"], 263, "s", 0),
           ("RS-25 fuel pump discharge", sol["RS-25"]["fuel_discharge_pressure"] / 1e6, 45, "MPa", 1),
           ("RS-25 turbopump power", sol["RS-25"]["turbine_power"] / 1e6, 70.4, "MW", 1),
           ("RS-25 vacuum Isp", sol["RS-25"]["isp_vac"], 452.3, "s", 1),
           ("RD-180 vacuum Isp", sol["RD-180"]["isp_vac"], 338, "s", 2),
           ("RD-180 sea-level Isp", sol["RD-180"]["isp_sl"], 311, "s", 2),
           ("RL10 vacuum Isp", sol["RL10-class"]["isp_vac"], 444, "s", 3)]
    rows = [{"name": n, "cycle": c, "pc": e.chamber_pressure / 1e6, "isp": sol[n]["isp_vac"],
             "tf": sol[n].get("turbine_flow_fraction", 0.0), "pr": sol[n].get("turbine_pressure_ratio", 1.0),
             "pw": sol[n]["turbine_power"] / 1e6, "pf": sol[n]["fuel_discharge_pressure"] / 1e6}
            for n, c, e in engines]
    return {"dev": [{"label": lab, "value": pct(m, p), "c": c,
                     "note": f"model {m:.1f} {u}  ·  published {p:g} {u}"
                             + ("  (HPFTP + HPOTP)" if "RS-25 turbopump" in lab else "")}
                    for lab, m, p, u, c in dev],
            "rows": rows,
            "note": ("Published: Wikipedia engine pages (F-1, RS-25, RD-180; RL10A-3C 444 s). Turbopump power "
                     "runs low because the model has no valve, line or boost-pump losses; Isp uses textbook "
                     "c* and nozzle efficiencies (0.98 each), which are pessimistic for the RS-25.")}, sol


def regen():
    from aerosim.propulsion.adapters.regen import regen_cooling
    r = regen_cooling("LOX", "LCH4", 3.4, 10e6, 100e3, 30)
    o = np.argsort(r.x)
    s = r.summary()
    log("regen solved")
    return {"x": np.round(r.x, 4).tolist(), "t_hot": np.round(r.t_hot_wall, 1).tolist(),
            "t_cold": np.round(r.t_cold_wall, 1).tolist(), "t_cool": np.round(r.t_coolant, 1).tolist(),
            "q": np.round(r.heat_flux / 1e6, 2).tolist(),
            "contour_x": np.round(r.x[o], 4).tolist(), "contour_r": np.round(r.radius[o], 4).tolist(),
            "caption": (f"pyskyfire solution for a 100 kN LOX/methane chamber (10 MPa, O/F 3.4, expansion 30, "
                        f"GRCop-42 wall, {r.chamber['n_channels']} channels): methane enters at the nozzle exit at "
                        f"{r.t_coolant[0]:.0f} K and leaves at the injector at {s['coolant_outlet_temperature']:.0f} K, "
                        f"picking up {s['heat_load'] / 1e6:.1f} MW and losing {s['pressure_drop'] / 1e5:.1f} bar. "
                        f"The hot wall peaks at {s['max_wall_temperature']:.0f} K just upstream of the throat, "
                        f"where heat flux reaches {s['max_heat_flux'] / 1e6:.0f} MW/m².")}


def hybrid_burn(tc):
    from aerosim.propulsion.hybrid import HybridMotor
    m = HybridMotor(fuel="HTPB", oxidizer="N2O", grain_length=1.0, port_radius=0.015, outer_radius=0.045,
                    regression_a=5.3e-5, regression_n=0.5, throat_area=5.3e-4,
                    tank={"volume": 0.012, "mass": 8.0, "temperature": 293.15}, injector_area=2.8e-5)
    h = m.simulate()
    i = np.unique(np.r_[np.linspace(0, h.time.size - 1, 400).astype(int), h.time.size - 1])
    return {"t": np.round(h.time[i], 2).tolist(), "thrust": np.round(h.thrust[i] / 1e3, 3).tolist(),
            "pc": np.round(h.chamber_pressure[i] / 1e6, 3).tolist(),
            "p_tank": np.round(h.tank_pressure[i] / 1e6, 3).tolist(), "of": np.round(h.of_ratio[i], 3).tolist(),
            "r": np.round(h.port_radius[i] * 1e3, 2).tolist(),
            "tiles": {"total impulse": f"{tc.total_impulse / 1e3:.1f} kN·s", "burn time": f"{tc.burn_time:.1f} s",
                      "average Isp": f"{tc.average_isp:.0f} s", "peak thrust": f"{h.thrust.max() / 1e3:.2f} kN",
                      "N2O used": f"{h.oxidizer_used:.2f} kg", "HTPB burned": f"{h.fuel_used:.2f} kg"},
            "caption": ("8 kg of N2O at 20 °C blows down through the injector into a 1 m HTPB grain. The port "
                        "opens as it burns, so the oxidizer flux and the regression rate fall and O/F drifts "
                        "fuel-rich; once the tank's liquid is gone the vapour tail carries the burn down to "
                        f"cut-off ({h.end_reason}). Regression coefficients are assumed, not measured.")}


def q_altitude(mach, q=50e3):
    p = 2 * q / (1.4 * mach * mach)
    return brentq(lambda h: isa(h)[0] - p, 0.0, 80e3)


def ram():
    from aerosim.propulsion.airbreathing import Ramjet
    rj, sj = Ramjet(fuel="JetA"), Ramjet(mode="scramjet", fuel="H2")
    out = []
    for name, eng, machs in (("Kerosene ramjet", rj, np.arange(1.5, 5.51, 0.25)),
                             ("Hydrogen scramjet", sj, np.arange(5.0, 14.01, 0.5))):
        pts = [eng.operating_point(q_altitude(mn), mn, 1.0) for mn in machs]
        ok = [math.isfinite(p["t4"]) for p in pts]      # scramjet inlet cannot reach burner T at low Mach
        val = lambda k, p, good: round(p[k], 1) if good else None
        out.append({"name": name, "mach": np.round(machs, 2).tolist(),
                    "isp": [val("isp", p, g) for p, g in zip(pts, ok)],
                    "sf": [val("specific_thrust", p, g) for p, g in zip(pts, ok)],
                    "t4": [val("t4", p, g) for p, g in zip(pts, ok)]})
    log("ramjet / scramjet done")
    return out


def gas_turbines():
    from aerosim.core import AerosimError
    from aerosim.propulsion.adapters.pycycle_engine import PyCycleEngine
    res, val = {}, []
    # Mach ranges where each engine flies at that altitude; lines are solved in serpentine order
    # so every point starts from its converged neighbour
    ranges = {"turbojet": {0.0: [0.0, 0.3, 0.6, 0.9, 1.2, 1.5], 6e3: [0.3, 0.6, 0.9, 1.2, 1.5],
                           11e3: [0.6, 0.9, 1.2, 1.5]},
              "turbofan": {0.0: [0.0, 0.2, 0.4, 0.6, 0.8], 6e3: [0.3, 0.5, 0.7, 0.85],
                           11e3: [0.6, 0.7, 0.8, 0.85, 0.9]}}
    for arch in ("turbojet", "turbofan"):
        e = PyCycleEngine(arch)
        d = e.design_point()
        lines = []
        for n, (name, alt) in enumerate((("Sea level", 0.0), ("6 km", 6e3), ("11 km", 11e3))):
            machs = ranges[arch][alt]
            order = range(len(machs)) if n % 2 == 0 else reversed(range(len(machs)))
            th, sf = [None] * len(machs), [None] * len(machs)
            for i in order:
                try:
                    o = e.operating_point(alt, machs[i], 0.85)
                    th[i], sf[i] = round(o["thrust"] / 1e3, 2), round(o["tsfc_lb_lbf_h"], 4)
                except AerosimError:
                    pass
            lines.append({"name": name, "mach": machs, "thrust": th, "tsfc": sf})
            log(f"pyCycle {arch} {name}")
        res[arch] = {"lines": lines, "design": d}
    tj, tf = res["turbojet"]["design"], res["turbofan"]["design"]
    val = [{"engine": "J79-GE-17", "q": "air flow, kg/s", "model": f"{tj['air_flow']:.1f}", "pub": "77"},
           {"engine": "J79-GE-17", "q": "TSFC, lb/(lbf·h)", "model": f"{tj['tsfc_lb_lbf_h']:.3f}", "pub": "0.84"},
           {"engine": "CFM56-7B27", "q": "overall pressure ratio", "model": f"{tf['opr']:.2f}", "pub": "32.7"},
           {"engine": "CFM56-7B27", "q": "air flow, kg/s", "model": f"{tf['air_flow']:.0f}", "pub": "307-355 (7B)"},
           {"engine": "CFM56-7B27", "q": "takeoff TSFC, lb/(lbf·h)", "model": f"{tf['tsfc_lb_lbf_h']:.3f}",
            "pub": "0.356-0.386"}]
    for v in res.values():
        v.pop("design")
    return {**res, "validation": val,
            "caption": ("pyCycle turbojet (J79-like: pressure ratio 13.5, 1210 K) and two-spool separate-flow "
                        "turbofan (CFM56-7B-like: bypass 5.1, OPR 32.7, 1640 K), sized at sea-level static and "
                        "run off-design on pyCycle's component maps at 85 % power. Points that do not converge "
                        "on the maps are left blank. Thrust falls with altitude as air density does; TSFC "
                        "rises with Mach as ram drag grows.")}


def propulsion_map(gt, ram_data, cyc):
    lb = 0.45359237 / 3600 / 4.4482216152605
    out = []
    for arch, name in (("turbofan", "Turbofan (11 km)"), ("turbojet", "Turbojet (11 km)")):
        ln = next(s for s in gt[arch]["lines"] if s["name"] == "11 km")
        out.append({"name": name, "mach": ln["mach"],
                    "isp": [round(1 / (s * lb * G0)) if s else None for s in ln["tsfc"]]})
    for s in ram_data:
        out.append({"name": s["name"], "mach": s["mach"], "isp": s["isp"]})
    out.append({"name": "LOX/LH2 rocket (RS-25, vac)", "mach": [0, 14], "isp": [round(cyc["RS-25"]["isp_vac"])] * 2})
    out.append({"name": "LOX/RP-1 rocket (RD-180, vac)", "mach": [0, 14], "isp": [round(cyc["RD-180"]["isp_vac"])] * 2})
    return out


def engines_data(tc, reuse: bool):
    """All engines-page data; cached in outputs/site/engines.json (``--reuse`` skips the solves)."""
    import json
    cache = ROOT / "outputs" / "site" / "engines.json"
    if reuse and cache.exists():
        log("engines data from cache")
        return json.loads(cache.read_text(encoding="utf-8"))
    cyc, sol = cycles()
    rg = regen()
    rm = ram()
    gt = gas_turbines()
    data = {"map": propulsion_map(gt, rm, sol), "cycles": cyc, "regen": rg, "hybrid": hybrid_burn(tc),
            "ram": rm, "gt": gt,
            "ram_caption": ("Stream-tube cycles on Cantera (propulsion/airbreathing/ramjet.py) along "
                            "a 50 kPa dynamic-pressure climb, stoichiometric fuel. The kerosene "
                            "ramjet burns subsonically after a MIL-E-5007D inlet; the hydrogen "
                            "scramjet burns supersonically at constant pressure (Heiser & Pratt "
                            "efficiencies). Dissociation caps burner temperature, so specific thrust "
                            "and Isp fall as flight speed rises."),
            "stats": {"F-1 Isp, s": f"{sol['F-1']['isp_vac']:.0f}",
                      "RD-180 Isp, s": f"{sol['RD-180']['isp_vac']:.0f}",
                      "max wall, K": f"{max(rg['t_hot']):.0f}"}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def engine_figures(reuse: bool = False):
    """3D figures of every engine model (engines3d.html); cached in outputs/site/engines3d.json."""
    import json
    import types

    from aerosim.plots.engines3d import engines3d_html
    cache = ROOT / "outputs" / "site" / "engines3d.json"
    if reuse and cache.exists():
        write("engines3d.html", engines3d_html(json.loads(cache.read_text(encoding="utf-8"))))
        return
    import warnings

    import yaml

    from aerosim.missions.runner import _backend
    from aerosim.plots import engine_models as em
    from aerosim.propulsion.adapters.pycycle_engine import PyCycleEngine
    from aerosim.propulsion.adapters.regen import regen_cooling
    from aerosim.propulsion.airbreathing import Ramjet
    from aerosim.propulsion.hybrid import HybridMotor
    from aerosim.propulsion.liquid.cycle import CycleEngine
    from aerosim.propulsion.liquid.design import engine_point
    warnings.simplefilter("ignore")
    figs = []
    liquids = [
        ("F-1 (gas generator)", CycleEngine("gas_generator", "LOX", "RP1", 2.27, 7.0e6, 16, 7.77e6,
                                            turbine_inlet_temperature=1089, turbine_pressure_ratio=16), 1.0, 1,
         ("Rocketdyne F-1 sized from AeroSim's gas-generator cycle (7.77 MN vacuum, 70 bar, O/F 2.27, "
          "expansion 16): throat and exit diameters follow from the cycle's mass flow and c*.")),
        ("RS-25 (staged, fuel-rich)", CycleEngine("staged_combustion", "LOX", "LH2", 6.03, 20.64e6, 69, 2.09e6,
                                                  turbine_inlet_temperature=1000, pump_efficiency=0.75,
                                                  turbine_efficiency=0.75, tank_pressure_ox=2.9e6,
                                                  tank_pressure_fuel=1.9e6), 0.9, 1,
         ("Space Shuttle main engine: 206 bar, O/F 6.03, expansion 69. Gas temperature and pressure "
          "fall through the bell as the flow expands; switch the field to Mach to see it.")),
        ("RD-180 (staged, ox-rich)", CycleEngine("staged_combustion", "LOX", "RP1", 2.72, 26.7e6, 36.87, 4.15e6,
                                                 rich="oxidizer", turbine_inlet_temperature=772,
                                                 pump_efficiency=0.75, turbine_efficiency=0.8, injector_dp=0.15),
         1.0, 2, "RD-180: one of its two chambers (267 bar, O/F 2.72, expansion 36.9)."),
    ]
    for name, ce, ls, n, cap in liquids:
        fig = em.liquid_engine(name, ce, ce.solve(), l_star=ls, chambers=n)
        figs.append(fig | {"caption": cap})
        log(f"figure {name}")
    rl = CycleEngine("expander", "LOX", "LH2", 5.0, 3.275e6, 61, 73.4e3,
                     regen={"material": "ss347", "channel_height": 4e-3, "l_star": 0.9})
    o = rl.solve()
    rr = rl.regen_result
    figs.append(em.liquid_engine("RL10-class (expander)", rl, o, regen=rr, contour=(rr.x, rr.radius))
                | {"caption": "Expander cycle closed on pyskyfire's cooling-jacket solution: hydrogen warmed "
                              f"to {o['turbine_inlet_temperature']:.0f} K in the jacket drives the turbine."})
    rg = regen_cooling("LOX", "LCH4", 3.4, 10e6, 100e3, 30)
    ep = engine_point(1.0, "LOX", "LCH4", 3.4, 10e6, expansion_ratio=30)
    ch = types.SimpleNamespace(oxidizer="LOX", fuel="LCH4", mixture_ratio=3.4, chamber_pressure=10e6,
                               expansion_ratio=30)
    figs.append(em.liquid_engine("LOX/methane, 100 kN sea level (regen-cooled)", ch,
                                 {"thrust_vac": 100e3 / ep["isp_sl"] * ep["isp_vac"], "isp_vac": ep["isp_vac"]},
                                 regen=rg, contour=(rg.x, rg.radius))
                | {"caption": "pyskyfire regen solution, GRCop-42 wall: the hot wall peaks just upstream of the "
                              "throat where heat flux is highest."})
    log("liquid figures done")
    cfg = yaml.safe_load((ROOT / "configs" / "solid" / "finocyl_grain_motor.yaml").read_text(encoding="utf-8"))
    figs.append(em.solid_motor("54 mm KNSB (finocyl + BATES)", _backend(cfg["propulsion"], "rocket_engine"))
                | {"caption": "Each grain's port regresses normal to the burning surface (fast-marching map for "
                              "the finocyl); press Play to burn it. Section cut to show the cores."})
    hm = HybridMotor(fuel="HTPB", oxidizer="N2O", grain_length=1.0, port_radius=0.015, outer_radius=0.045,
                     regression_a=5.3e-5, regression_n=0.5, throat_area=5.3e-4,
                     tank={"volume": 0.012, "mass": 8.0, "temperature": 293.15}, injector_area=2.8e-5)
    figs.append(em.hybrid_motor("N2O / HTPB hybrid", hm, hm.simulate())
                | {"caption": "The N2O tank drains (blue: liquid) while the HTPB port opens; press Play."})
    log("motor figures done")
    figs.append(em.ramjet("Kerosene ramjet, Mach 3", Ramjet().operating_point(18e3, 3.0, 1.0), 0.05)
                | {"caption": "Subsonic burner: the inlet slows the flow, kerosene burns at about 2,400 K, the "
                              "convergent-divergent nozzle re-expands it."})
    figs.append(em.scramjet("Hydrogen scramjet, Mach 8", Ramjet(mode="scramjet", fuel="H2").operating_point(30e3, 8.0, 1.0), 0.05)
                | {"caption": "Supersonic burner: the ramp compresses the air to about 1,600 K at 1.7 km/s and "
                              "hydrogen burns without slowing the flow to subsonic."})
    for arch, name in (("turbojet", "Turbojet (J79-like)"), ("turbofan", "Turbofan (CFM56-7B-like)")):
        e = PyCycleEngine(arch)
        figs.append(em.gas_turbine(name, e, e.design_point())
                    | {"caption": "pyCycle design point, sea-level static: total temperature rises through the "
                                  "compressor and burner and falls through the turbine."})
    log("gas-turbine figures done")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(figs, default=float), encoding="utf-8")
    write("engines3d.html", engines3d_html(figs))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse cached engines data (no pyCycle runs)")
    args = ap.parse_args(argv)
    first = rockot()
    land = landing()
    f16s = f16()
    tc, hy = hybrid()
    data = engines_data(tc, args.reuse)
    write("engines.html", site.engines_html(data))
    engine_figures(args.reuse)
    orbit = first.orbits["final"]
    write("index.html", site.hub_html([
        {"href": "rockot.html", "title": "Rockot ascent", "text": "Calibrated and first-principles models of the "
         "Rockot / Breeze-KM reference mission, on a globe.", "stats": {
             "final orbit, km": f"{orbit.perigee_alt / 1e3:.0f} × {orbit.apogee_alt / 1e3:.0f}",
             "inclination": f"{orbit.inclination:.1f}°", "models": "2"}},
        {"href": "rockot-closeup.html", "title": "Rockot close-up", "text": "Follow the rocket up through the "
         "layers of the atmosphere: drag, max-Q, Mach, heating, thrust gain and where the Δv goes.",
         "stats": {"max-Q, kPa": f"{first.dynamic_pressure.max() / 1e3:.0f}",
                   "Mach 1, s": f"{mach1_time(first.time, first.mach):.0f}", "layers": "8"}},
        {"href": "landing.html", "title": "Booster landing", "text": "Fuel-optimal landing burn by convex "
         "optimisation, re-solved in flight against drag and wind.", "stats": land},
        {"href": "f16.html", "title": "F-16 flight", "text": "JSBSim F-16 airframe flown on an AeroSim "
         "turbojet deck by an altitude, Mach and heading autopilot.", "stats": f16s},
        {"href": "terrain.html", "title": "Terrain flights", "text": "The JSBSim F-16 terrain-following at "
         "100 m through Lysefjorden and across Novaya Zemlya, on Kartverket and Copernicus terrain.",
         "stats": terrain_stats()},
        {"href": "hybrid.html", "title": "Hybrid sounding rocket", "text": "N2O / HTPB hybrid with tank "
         "blowdown and O/F shift, flown point-mass.", "stats": {
             "apogee, km": f"{hy.apogee / 1e3:.1f}", "impulse, kN·s": f"{tc.total_impulse / 1e3:.1f}",
             "Isp, s": f"{tc.average_isp:.0f}"}},
        {"href": "engines.html", "title": "Engines", "text": "Liquid cycles, regen cooling, the hybrid burn, "
         "ramjet / scramjet and pyCycle gas turbines.", "stats": data["stats"]},
        {"href": "engines3d.html", "title": "Engine models", "text": "3D cutaways of every engine, coloured by "
         "gas temperature, pressure or Mach; solid and hybrid motors burn on a time slider.",
         "stats": {"engines": "11", "liquid": "5", "air-breathing": "4"}},
    ]))


if __name__ == "__main__":
    main()
