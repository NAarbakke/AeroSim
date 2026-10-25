"""Multi-grain solid motor: any stack of grain shapes, time-stepped internal ballistics.

Each step (fixed ``timestep``):

1. burning area of every grain at its current web -> Kn = sum(Ab) / At
2. quasi-steady chamber pressure from the mass balance (``solve_pressure``)
3. burn rate r(Pc); every live grain regresses by r * dt
4. thrust F = eta * lambda * Cf(gamma, Ae/At, Pc, Pa) * Pc * At, with
   lambda = (1 + cos(divergence)) / 2 the conical-nozzle divergence loss
5. throat diameter changes by erosion (grows with Pc) and slag (shrinks with 1/Pc)

Propellant: library name / recipe (NASA CEA), manual ``Propellant``, or a
``TabbedPropellant`` (what openMotor .ric files store). ``load_ric`` in ``ric.py``
builds a GrainMotor straight from a .ric file; the ``grain_motor`` backend accepts
``ric_file:`` in mission YAML.

Not modelled: erosive burning, ignition transient, two-phase flow losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerosim.core import ThrustCurve, registry
from aerosim.propulsion.nozzle import thrust_coefficient
from aerosim.propulsion.solid.grains import Grain, make_grain
from aerosim.propulsion.solid.propellant import make_model, solve_pressure


@dataclass
class Nozzle:
    throat_diameter: float         # m
    exit_diameter: float           # m
    efficiency: float = 1.0        # thrust efficiency (multiplies Cf)
    divergence_deg: float = 15.0   # half-angle; sets the divergence loss
    erosion_coeff: float = 0.0     # m/(s Pa): dDt/dt = +erosion_coeff * Pc
    slag_coeff: float = 0.0        # m Pa/s:    dDt/dt = -slag_coeff / Pc

    @property
    def divergence_factor(self) -> float:
        return (1 + math.cos(math.radians(self.divergence_deg))) / 2


@dataclass
class MotorLimits:
    """Design checks; violations become ``GrainMotor.warnings``."""

    max_pressure: float = 10.34e6      # Pa
    max_mass_flux: float = 1406.0      # kg/(m^2 s), core flux at a grain's aft end
    min_port_throat: float = 2.0       # port area / throat area at ignition


@registry.register("rocket_engine", "grain_motor")
class GrainMotor:
    def __init__(self, grains: list | None = None, nozzle: Nozzle | dict | None = None,
                 propellant=None, dry_mass: float = 0.0, ambient_pressure: float | None = None,
                 timestep: float | None = None, burnout_web: float | None = None,
                 cstar_efficiency: float = 1.0, limits: MotorLimits | dict | None = None,
                 name: str | None = None, ric_file: str | None = None) -> None:
        """
        grains     : list of Grain or dicts ``{"type": "bates", "diameter": ..., ...}``,
                     forward to aft
        nozzle     : Nozzle or dict of its fields
        propellant : see ``propellant.make_model``
        ambient_pressure, timestep, burnout_web : default 101325 Pa, 0.01 s, 0.25 mm
        ric_file   : load grains / nozzle / propellant / settings from an openMotor
                     file; any argument given explicitly overrides it
        """
        spec = {"settings": {}}
        if ric_file is not None:
            from aerosim.propulsion.solid.ric import read_ric
            spec = read_ric(ric_file)
        grains = grains or spec.get("grains")
        nozzle = nozzle or spec.get("nozzle")
        propellant = propellant or spec.get("propellant")
        limits = limits or spec.get("limits")
        name = name or spec.get("name", "grain motor")
        cfg = {"ambient_pressure": 101_325.0, "timestep": 0.01, "burnout_web": 2.5e-4,
               **spec["settings"]}
        ambient_pressure = cfg["ambient_pressure"] if ambient_pressure is None else ambient_pressure
        timestep = cfg["timestep"] if timestep is None else timestep
        burnout_web = cfg["burnout_web"] if burnout_web is None else burnout_web
        if not grains or nozzle is None:
            raise ValueError("GrainMotor needs grains and a nozzle (or a ric_file)")
        self.grains: list[Grain] = [make_grain(g) for g in grains]
        self.nozzle = nozzle if isinstance(nozzle, Nozzle) else Nozzle(**nozzle)
        eps = (self.nozzle.exit_diameter / self.nozzle.throat_diameter) ** 2
        self.model = make_model(propellant, eps, cstar_efficiency)
        self.limits = limits if isinstance(limits, MotorLimits) else MotorLimits(**(limits or {}))
        self.dry_mass, self.pa, self.dt = dry_mass, ambient_pressure, timestep
        self.burnout_web, self.name = burnout_web, name
        self.warnings: list[str] = []

    @property
    def propellant_mass(self) -> float:
        return self.model.density * sum(g.volume(0.0) for g in self.grains)

    @property
    def kn_initial(self) -> float:
        At = math.pi / 4 * self.nozzle.throat_diameter**2
        return sum(g.burn_area(0.0) for g in self.grains) / At

    def simulate(self) -> dict[str, np.ndarray]:
        """Full time history: time, thrust, pressure, mass_flow, mass, kn, throat, and webs
        (n, grains): web burned per grain."""
        nz, m, rho = self.nozzle, self.model, self.model.density
        webs = np.zeros(len(self.grains))
        dt_ = nz.throat_diameter
        t, pc = 0.0, 2e6
        out = {k: [] for k in ("time", "thrust", "pressure", "mass_flow", "mass", "kn",
                               "throat", "mass_flux", "webs")}
        mass = self.propellant_mass
        for _ in range(1_000_000):
            Ab = np.array([g.burn_area(w) for g, w in zip(self.grains, webs)])
            if Ab.sum() <= 0:
                break
            At = math.pi / 4 * dt_**2
            kn = Ab.sum() / At
            pc = solve_pressure(m, kn, pc)
            r = m.rate(pc)
            _, gamma = m.cstar_gamma(pc)
            cf = thrust_coefficient(gamma, (nz.exit_diameter / dt_) ** 2, pc, self.pa)
            F = max(nz.efficiency * nz.divergence_factor * cf * pc * At, 0.0)

            # regress, consume propellant
            new = np.array([min(w + r * self.dt, g.web_max) if a > 0 else w
                            for g, w, a in zip(self.grains, webs, Ab)])
            for i, g in enumerate(self.grains):   # drop slivers thinner than burnout_web
                if g.web_max - new[i] < self.burnout_web:
                    new[i] = g.web_max
            v_old = [g.volume(w) for g, w in zip(self.grains, webs)]
            v_new = [g.volume(w) for g, w in zip(self.grains, new)]
            dm = rho * (np.array(v_old) - np.array(v_new))
            flux = np.cumsum(dm / self.dt) / np.maximum(
                [g.port_area(w) for g, w in zip(self.grains, webs)], 1e-12)

            for k, v in (("time", t), ("thrust", F), ("pressure", pc), ("kn", kn),
                         ("mass_flow", dm.sum() / self.dt), ("mass", mass), ("throat", dt_),
                         ("mass_flux", flux.max()), ("webs", webs.copy())):
                out[k].append(v)
            mass -= dm.sum()
            webs = new
            dt_ = max(dt_ + (nz.erosion_coeff * pc - nz.slag_coeff / pc) * self.dt, 1e-6)
            t += self.dt
        for k, v in (("time", t), ("thrust", 0.0), ("pressure", self.pa), ("kn", 0.0),
                     ("mass_flow", 0.0), ("mass", 0.0), ("throat", dt_), ("mass_flux", 0.0),
                     ("webs", np.array([gr.web_max for gr in self.grains]))):
            out[k].append(v)
        res = {k: np.asarray(v) for k, v in out.items()}
        self._check(res)
        return res

    def _check(self, res) -> None:
        L, w = self.limits, []
        At0 = math.pi / 4 * self.nozzle.throat_diameter**2
        pt = min(g.port_area(0.0) for g in self.grains) / At0
        if pt < L.min_port_throat:
            w.append(f"initial port/throat ratio {pt:.3f} is below {L.min_port_throat}")
        if res["pressure"].max() > L.max_pressure:
            w.append(f"peak chamber pressure {res['pressure'].max()/1e6:.2f} MPa exceeds "
                     f"{L.max_pressure/1e6:.2f} MPa")
        if res["mass_flux"].max() > L.max_mass_flux:
            w.append(f"peak core mass flux {res['mass_flux'].max():.0f} kg/m2/s exceeds "
                     f"{L.max_mass_flux:.0f}")
        rng = getattr(self.model, "pressure_range", None)
        p = res["pressure"][:-1]
        if rng and (p.min() < rng[0] or p.max() > rng[1]):
            w.append(f"Pc {p.min()/1e6:.2f}-{p.max()/1e6:.2f} MPa leaves the propellant's "
                     f"burn-rate range {rng[0]/1e6:.2f}-{rng[1]/1e6:.2f} MPa")
        self.warnings = w

    def thrust_curve(self) -> ThrustCurve:
        r = self.simulate()
        At = math.pi / 4 * self.nozzle.throat_diameter**2
        eps = (self.nozzle.exit_diameter / self.nozzle.throat_diameter) ** 2
        # step-wise samples: rescale so trapezoidal integration (as flight sims do)
        # consumes exactly the loaded propellant
        mdot = r["mass_flow"] * self.propellant_mass / np.trapezoid(r["mass_flow"], r["time"])
        return ThrustCurve(r["time"], r["thrust"], mdot, self.propellant_mass,
                           self.dry_mass, self.name, exit_area=eps * At,
                           chamber_pressure=r["pressure"])
