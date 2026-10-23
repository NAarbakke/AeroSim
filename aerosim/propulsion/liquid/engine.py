"""Liquid engine performance model built on any CombustionModel (CEA backends).

Level 0 (this file): chamber performance from CEA + throttle schedule -> ThrustCurve.
Level 1 (propulsion/adapters/feed_system.py): pressure-fed / pump-fed network that sets Pc and MR.
Level 2: regen cooling & full cycle closure via pyskyfire.
"""

from __future__ import annotations

import numpy as np

from aerosim.core import ThrustCurve, registry
from aerosim.core.interfaces import CombustionModel
from aerosim.core.units import G0


@registry.register("rocket_engine", "liquid")
class LiquidEngine:
    def __init__(self, oxidizer, fuel, mixture_ratio: float,
                 chamber_pressure: float, expansion_ratio: float, throat_area: float,
                 burn_time: float, combustion: CombustionModel | str | dict = "nasa_cea",
                 throttle: list[tuple[float, float]] | None = None,
                 cstar_efficiency: float = 0.97, cf_efficiency: float = 0.98,
                 dry_mass: float = 0.0, name: str = "liquid engine") -> None:
        """oxidizer/fuel: alias ("LOX"), CEA species, or blend {"Ethanol": 0.75, "Water": 0.25}.
        combustion: backend name, an instance, or {"backend": "nasa_cea", "frozen": "throat"}."""
        if isinstance(combustion, str):
            combustion = {"backend": combustion}
        if isinstance(combustion, dict):
            import aerosim.thermo  # noqa: F401  (registers combustion backends)
            cfg = dict(combustion)
            combustion = registry.create("combustion", cfg.pop("backend", "nasa_cea"), **cfg)
        self.cm = combustion
        self.ox, self.fuel, self.mr = oxidizer, fuel, float(mixture_ratio)
        self.pc, self.eps = float(chamber_pressure), float(expansion_ratio)
        self.At, self.burn_time = float(throat_area), float(burn_time)
        self.throttle = throttle or [(0.0, 1.0), (burn_time, 1.0)]
        self.eta_cstar, self.eta_cf = cstar_efficiency, cf_efficiency
        self.dry_mass = dry_mass
        self.name = name

    def performance(self) -> dict[str, float]:
        perf = self.cm.chamber(self.ox, self.fuel, self.mr, self.pc, self.eps)
        cstar = perf["cstar"] * self.eta_cstar
        mdot = self.pc * self.At / cstar
        isp_vac = perf["isp_vac"] * self.eta_cstar * self.eta_cf
        return {**perf, "mdot": mdot, "thrust_vac": isp_vac * G0 * mdot, "isp_vac_delivered": isp_vac}

    def thrust_curve(self, samples: int = 200) -> ThrustCurve:
        perf = self.performance()
        t = np.linspace(0.0, self.burn_time, samples)
        tt, frac = zip(*self.throttle)
        k = np.interp(t, tt, frac)          # Isp held constant across throttle (first order)
        F = perf["thrust_vac"] * k
        mdot = perf["mdot"] * k
        F[-1] = mdot[-1] = 0.0
        prop = float(np.trapezoid(mdot, t))
        return ThrustCurve(t, F, mdot, prop, self.dry_mass, self.name,
                           exit_area=self.eps * self.At, vacuum=True)
