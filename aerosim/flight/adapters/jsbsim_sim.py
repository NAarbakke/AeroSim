"""JSBSim backend for aircraft (LGPL-2.1, used as an unmodified library).

An AeroSim ``EngineDeck`` replaces the aircraft's own engines: the JSBSim aircraft folder is
copied to a scratch root and an ``<external_reactions>`` body-axis force is added at each
thruster location; the aircraft's engines stay in the model (its systems read their
properties) but are shut down and given no throttle, so they make no thrust and burn no fuel. Every
step the deck gives thrust and fuel flow at the current altitude, Mach and throttle; the
force magnitude is set and the fuel drawn from the tanks, so mass and CG follow the burn.
The aircraft's own aerodynamics, flight-control system and ground reactions are untouched.

A simple autopilot flies the aircraft: altitude hold (altitude error -> pitch attitude ->
elevator), heading hold (heading error -> bank -> aileron) and speed hold (Mach error ->
throttle, i.e. deck power). Winds from any AtmosphereModel are pushed every step.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from aerosim.core import EngineDeck, FlightResult, registry, require
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import AircraftVehicle

FT = 0.3048
LBF = 4.4482216152605
LB = 0.45359237
_TO_IN = {"IN": 1.0, "FT": 12.0, "M": 1 / 0.0254, "CM": 1 / 2.54, "MM": 1 / 25.4}


def _location(el) -> tuple[float, float, float]:
    loc = el.find("location")
    k = _TO_IN[(loc.get("unit") or "IN").upper()]
    return tuple(float(loc.find(a).text) * k for a in "xyz")


def prepare_aircraft(model: str, root_dir: str | Path | None = None) -> tuple[Path, int, int]:
    """Copy ``model`` to a scratch JSBSim root with an external thrust force per engine.

    Returns (root, number of thrust forces, number of fuel tanks). Force i's magnitude is
    the property ``external_reactions/aerosim_thrust<i>/magnitude`` (lbf).
    """
    jsb = require("jsbsim", "aircraft")
    src = Path(root_dir or jsb.get_default_root_dir())
    if not (src / "aircraft" / model).is_dir():
        raise FileNotFoundError(f"no JSBSim aircraft {model!r} under {src / 'aircraft'}")
    root = Path(tempfile.mkdtemp(prefix="aerosim_jsbsim_"))
    shutil.copytree(src / "aircraft" / model, root / "aircraft" / model)
    for d in ("engine", "systems"):
        if (src / d).is_dir():
            shutil.copytree(src / d, root / d)
    path = root / "aircraft" / model / f"{model}.xml"
    tree = ET.parse(path)
    fdm = tree.getroot()
    prop = fdm.find("propulsion")
    engines = prop.findall("engine") if prop is not None else []
    if not engines:
        raise ValueError(f"{model} has no <engine> to replace")
    locs = []
    for e in engines:
        th = e.find("thruster")
        locs.append(_location(th) if th is not None and th.find("location") is not None else (0, 0, 0))
    ext = fdm.find("external_reactions")
    if ext is None:
        ext = ET.Element("external_reactions")
        fdm.insert(list(fdm).index(prop) + 1, ext)
    for i, (x, y, z) in enumerate(locs):
        f = ET.SubElement(ext, "force", name=f"aerosim_thrust{i}", frame="BODY")
        loc = ET.SubElement(f, "location", unit="IN")
        for a, v in zip("xyz", (x, y, z)):
            ET.SubElement(loc, a).text = f"{v:.3f}"
        d = ET.SubElement(f, "direction")
        for a, v in zip("xyz", (1, 0, 0)):
            ET.SubElement(d, a).text = str(v)
    tree.write(path)
    return root, len(locs), len(prop.findall("tank"))


def apply_deck(fdm, deck: EngineDeck, h: float, mach: float, power: float, n_thrust: int,
               n_tanks: int, dt: float) -> tuple[float, float]:
    """Set each external thrust force from the deck and draw dt of fuel from the tanks
    (first tank first). Returns (thrust per engine N, fuel flow per engine kg/s); a dry
    tank set flames out (zero thrust)."""
    F, wf = deck.lookup(h, mach, power)
    for i in range(n_thrust):
        fdm[f"external_reactions/aerosim_thrust{i}/magnitude"] = max(F, 0.0) / LBF
    burn = max(wf, 0.0) * n_thrust * dt / LB
    for i in range(n_tanks):
        c = fdm[f"propulsion/tank[{i}]/contents-lbs"]
        if c > 0 and burn > 0:
            take = min(c, burn)
            fdm[f"propulsion/tank[{i}]/contents-lbs"] = c - take
            burn -= take
    if burn > 0:                                  # tanks dry: flame-out
        for i in range(n_thrust):
            fdm[f"external_reactions/aerosim_thrust{i}/magnitude"] = 0.0
        return 0.0, 0.0
    return F, wf


def engine_deck_to_jsbsim_xml(deck: EngineDeck, path: str | Path) -> Path:
    """Write the deck as JSBSim <function> tables (thrust, fuel flow vs Mach and altitude per
    power setting), for use in a hand-built JSBSim engine; the simulator below does not need it."""
    root = ET.Element("aerosim_engine_deck", name=deck.name)
    for qty, arr, unit in (("thrust", deck.thrust / LBF, "LBS"), ("fuel_flow", deck.fuel_flow / LB, "LBS/SEC")):
        for k, p in enumerate(deck.power):
            fn = ET.SubElement(root, "function", name=f"{qty}_power_{p:g}", unit=unit)
            tbl = ET.SubElement(fn, "table")
            ET.SubElement(tbl, "independentVar", lookup="row").text = "velocities/mach"
            ET.SubElement(tbl, "independentVar", lookup="column").text = "atmosphere/density-altitude"
            rows = ["\t" + "  ".join(f"{h / FT:.0f}" for h in deck.altitude)]
            for j, m in enumerate(deck.mach):
                rows.append(f"{m:g}\t" + "  ".join(f"{arr[i, j, k]:.2f}" for i in range(len(deck.altitude))))
            ET.SubElement(tbl, "tableData").text = "\n" + "\n".join(rows) + "\n"
    path = Path(path)
    ET.ElementTree(root).write(path)
    return path


@registry.register("flight", "jsbsim")
class JSBSimSim:
    def __init__(self, dt: float = 1 / 120, duration: float = 60.0, root_dir: str | None = None,
                 autopilot: dict | None = None, record_every: int = 12) -> None:
        """autopilot: {"altitude": m, "mach": -, "heading": deg} targets (default: hold the
        initial values); gains via "k_alt", "k_vs", "k_pitch", "k_q", "k_hdg", "k_bank",
        "k_p", "k_mach", "ki_mach". None of altitude/mach/heading = that loop is off."""
        self.dt, self.duration, self.root_dir = dt, duration, root_dir
        self.autopilot = {"k_alt": 0.004, "k_vs": 0.02, "k_pitch": 2.0, "k_q": 0.5,
                          "k_hdg": 1.5, "k_bank": 2.0, "k_p": 0.3, "k_mach": 4.0,
                          "ki_mach": 0.5, **(autopilot or {})}
        self.record_every = record_every

    def run(self, vehicle: AircraftVehicle, propulsion: EngineDeck | None,
            atmosphere: AtmosphereModel, **options) -> FlightResult:
        jsb = require("jsbsim", "aircraft")
        if propulsion is not None:
            root, n_thrust, n_tanks = prepare_aircraft(vehicle.jsbsim_model, self.root_dir)
        else:
            root, n_thrust, n_tanks = self.root_dir, 0, 0
        try:
            return self._fly(jsb, root, vehicle, propulsion, atmosphere, n_thrust, n_tanks, options)
        finally:
            if propulsion is not None:
                shutil.rmtree(root, ignore_errors=True)

    def _fly(self, jsb, root, vehicle, deck, atmosphere, n_thrust, n_tanks, options):
        fdm = jsb.FGFDMExec(str(root) if root else None)
        fdm.set_debug_level(0)
        fdm.set_dt(self.dt)
        if not fdm.load_model(vehicle.jsbsim_model):
            raise RuntimeError(f"JSBSim could not load {vehicle.jsbsim_model}")
        ic = {"ic/h-sl-ft": 10000.0, "ic/vc-kts": 300.0, "ic/psi-true-deg": 0.0, **vehicle.initial}
        for k, v in ic.items():
            fdm[k] = v
        fdm.run_ic()
        for i in range(vehicle.engine_count):     # deck flights: native engines off
            fdm[f"propulsion/engine[{i}]/set-running"] = int(deck is None)
            fdm[f"fcs/throttle-cmd-norm[{i}]"] = options.get("throttle", 1.0) if deck is None else 0.0
        fdm["gear/gear-cmd-norm"] = 0.0
        ap = dict(self.autopilot)
        tgt_h = ap.get("altitude", fdm["position/h-sl-ft"] * FT)
        tgt_m = ap.get("mach", fdm["velocities/mach"])
        tgt_psi = ap.get("heading", fdm["attitude/psi-deg"])
        p0 = options.get("power", 0.5)                 # throttle bias of the speed hold
        power, mach_int = p0, 0.0
        rows, step = [], 0
        while fdm.get_sim_time() < self.duration:
            h = fdm["position/h-sl-ft"] * FT
            mach = fdm["velocities/mach"]
            atm = atmosphere.at(h)
            fdm["atmosphere/wind-north-fps"] = atm.wind_v / FT
            fdm["atmosphere/wind-east-fps"] = atm.wind_u / FT
            # autopilot
            if tgt_h is not None:
                vs = -fdm["velocities/v-down-fps"] * FT
                theta_cmd = np.clip(ap["k_alt"] * (tgt_h - h) - ap["k_vs"] * vs, -0.26, 0.26)
                e = -(ap["k_pitch"] * (theta_cmd - fdm["attitude/theta-rad"]) - ap["k_q"] * fdm["velocities/q-rad_sec"])
                fdm["fcs/elevator-cmd-norm"] = float(np.clip(e, -1, 1))
            if tgt_psi is not None:
                err = (tgt_psi - fdm["attitude/psi-deg"] + 180) % 360 - 180
                phi_cmd = np.clip(ap["k_hdg"] * math.radians(err), -0.5, 0.5)
                a = ap["k_bank"] * (phi_cmd - fdm["attitude/phi-rad"]) - ap["k_p"] * fdm["velocities/p-rad_sec"]
                fdm["fcs/aileron-cmd-norm"] = float(np.clip(a, -1, 1))
            if deck is not None:
                if tgt_m is not None:
                    err = tgt_m - mach
                    raw = p0 + ap["k_mach"] * err + ap["ki_mach"] * (mach_int + err * self.dt)
                    if 0.0 < raw < 1.0 or raw * err < 0:      # anti-windup: hold when saturated
                        mach_int += err * self.dt
                    power = float(np.clip(p0 + ap["k_mach"] * err + ap["ki_mach"] * mach_int, 0, 1))
                F, _ = apply_deck(fdm, deck, h, mach, power, n_thrust, n_tanks, self.dt)
            if not fdm.run():
                break
            if step % self.record_every == 0:
                rows.append([fdm.get_sim_time(),
                             fdm["position/distance-from-start-lon-mt"],
                             fdm["position/distance-from-start-lat-mt"], h,
                             fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT,
                             -fdm["velocities/v-down-fps"] * FT,
                             fdm["inertia/weight-lbs"] * LB, mach, power,
                             F * n_thrust if deck is not None else float("nan"),
                             fdm["attitude/theta-deg"], fdm["attitude/phi-deg"], fdm["attitude/psi-deg"],
                             fdm["aero/alpha-deg"]])
            step += 1
        a = np.asarray(rows)
        return FlightResult(a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7],
                            meta={"simulator": "jsbsim", "model": vehicle.jsbsim_model,
                                  "mach": a[:, 8], "power": a[:, 9], "thrust": a[:, 10],
                                  "pitch_deg": a[:, 11], "roll_deg": a[:, 12],
                                  "heading_deg": a[:, 13], "alpha_deg": a[:, 14],
                                  "fuel_used": float(a[0, 7] - a[-1, 7]),
                                  "targets": {"altitude": tgt_h, "mach": tgt_m, "heading": tgt_psi}})
