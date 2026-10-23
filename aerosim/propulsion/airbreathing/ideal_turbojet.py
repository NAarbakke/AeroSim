"""Built-in single-spool turbojet cycle with component efficiencies.

Two gas models:
- ``"constant"`` (default, dependency-free): textbook cold/hot cp and gamma.
- ``"cantera"``: enthalpy/entropy of real air and Jet-A combustion products from
  Cantera, so cp and gamma vary with temperature and fuel-air ratio. The burner
  solves an energy balance for the fuel-air ratio instead of using a mean cp.

Constant corrected airflow, convergent-divergent nozzle assumed fully expanded.
Used for quick studies and as a cross-check for the pyCycle backend.
"""

from __future__ import annotations

import math

import numpy as np

from aerosim.atmosphere.standard import isa
from aerosim.core import EngineDeck, registry

_R = 287.05


@registry.register("airbreathing", "turbojet")
class Turbojet:
    def __init__(self, pressure_ratio: float = 12.0, tt4_max: float = 1500.0,
                 tt4_idle: float = 900.0, airflow_sls: float = 20.0,
                 eta_compressor: float = 0.85, eta_turbine: float = 0.90,
                 eta_burner: float = 0.98, eta_nozzle: float = 0.98,
                 inlet_recovery: float = 0.98, burner_pressure_ratio: float = 0.95,
                 fuel_lhv: float = 43.0e6, gas_model: str = "constant",
                 name: str = "turbojet") -> None:
        self.pic, self.tt4_max, self.tt4_idle = pressure_ratio, tt4_max, tt4_idle
        self.m_ref = airflow_sls
        self.etac, self.etat, self.etab, self.etan = eta_compressor, eta_turbine, eta_burner, eta_nozzle
        self.pid, self.pib = inlet_recovery, burner_pressure_ratio
        self.h = fuel_lhv
        self.cpc, self.gc = 1004.5, 1.4
        self.cph, self.gh = 1156.0, 1.33
        if gas_model not in ("constant", "cantera"):
            raise ValueError("gas_model must be 'constant' or 'cantera'")
        self.gas_model = gas_model
        if gas_model == "cantera":
            from aerosim.thermo.gas import AirFuelGas
            self.gas = AirFuelGas(lhv=fuel_lhv)
        self.name = name

    def _tt4(self, power: float) -> float:
        return self.tt4_idle + float(np.clip(power, 0, 1)) * (self.tt4_max - self.tt4_idle)

    def operating_point(self, altitude: float, mach: float, power: float) -> dict[str, float]:
        if self.gas_model == "cantera":
            return self._operating_point_cantera(altitude, mach, power)
        p0, t0, _ = isa(altitude)
        gc, gh, cpc, cph = self.gc, self.gh, self.cpc, self.cph
        v0 = mach * math.sqrt(gc * _R * t0)

        ram = 1 + (gc - 1) / 2 * mach**2
        tt2, pt2 = t0 * ram, p0 * ram ** (gc / (gc - 1)) * self.pid
        m0 = self.m_ref * (pt2 / 101_325.0) / math.sqrt(tt2 / 288.15)

        tt3 = tt2 * (1 + (self.pic ** ((gc - 1) / gc) - 1) / self.etac)
        pt3 = pt2 * self.pic
        tt4 = self._tt4(power)
        f = max(cph * tt4 - cpc * tt3, 0.0) / (self.etab * self.h - cph * tt4)
        pt4 = pt3 * self.pib

        tt5 = tt4 - cpc * (tt3 - tt2) / ((1 + f) * cph)
        tt5_ideal = tt4 - (tt4 - tt5) / self.etat
        pt5 = pt4 * (tt5_ideal / tt4) ** (gh / (gh - 1))

        if pt5 > p0:
            v9 = math.sqrt(2 * cph * tt5 * self.etan * (1 - (p0 / pt5) ** ((gh - 1) / gh)))
        else:
            v9 = 0.0
        thrust = m0 * ((1 + f) * v9 - v0)
        wf = f * m0
        return {
            "thrust": thrust, "fuel_flow": wf, "airflow": m0, "fuel_air_ratio": f,
            "tsfc": wf / thrust if thrust > 0 else float("inf"),
            "Tt3": tt3, "Tt4": tt4, "Tt5": tt5, "Pt3": pt3, "Pt5": pt5, "V9": v9,
        }

    def _operating_point_cantera(self, altitude: float, mach: float, power: float) -> dict:
        g = self.gas
        p0, t0, _ = isa(altitude)
        st = g.set(t0, p0)
        a0 = math.sqrt(st.cp_mass / st.cv_mass * p0 / st.density)
        v0 = mach * a0
        h0, s0 = g.state(t0, p0)

        # inlet: stagnation via the isentrope, then recovery loss
        ht2 = h0 + 0.5 * v0**2
        pt0 = g.P_from_hs(ht2, s0, 0.0, p0 * 0.999, p0 * 60.0) if mach > 0 else p0
        pt2 = pt0 * self.pid
        tt2 = g.T_from_h(ht2, pt2)
        m0 = self.m_ref * (pt2 / 101_325.0) / math.sqrt(tt2 / 288.15)
        s2 = g.state(tt2, pt2)[1]

        # compressor
        pt3 = pt2 * self.pic
        h3s = g.isentropic_h(s2, pt3)
        ht3 = ht2 + (h3s - ht2) / self.etac
        tt3 = g.T_from_h(ht3, pt3)

        # burner: solve the energy balance for fuel-air ratio
        tt4 = self._tt4(power)
        pt4 = pt3 * self.pib
        f = g.far_for_T(tt4, ht3, pt4, self.etab) if tt4 > tt3 else 0.0
        ht4, s4 = g.state(tt4, pt4, f)

        # turbine drives compressor: (1+f)(h4 - h5) = h3 - h2
        ht5 = ht4 - (ht3 - ht2) / (1 + f)
        h5s = ht4 - (ht4 - ht5) / self.etat
        pt5 = g.P_from_hs(h5s, s4, f, 100.0, pt4)
        tt5 = g.T_from_h(ht5, pt5, f)
        s5 = g.state(tt5, pt5, f)[1]

        # nozzle: expand to ambient
        v9 = math.sqrt(2 * self.etan * max(ht5 - g.isentropic_h(s5, p0, f), 0.0)) if pt5 > p0 else 0.0
        thrust = m0 * ((1 + f) * v9 - v0)
        wf = f * m0
        return {
            "thrust": thrust, "fuel_flow": wf, "airflow": m0, "fuel_air_ratio": f,
            "tsfc": wf / thrust if thrust > 0 else float("inf"),
            "Tt3": tt3, "Tt4": tt4, "Tt5": tt5, "Pt3": pt3, "Pt5": pt5, "V9": v9,
            "gamma_t4": g.gamma(tt4, pt4, f), "cp_t4": g.cp(tt4, pt4, f),
        }

    def deck(self, altitudes, machs, powers) -> EngineDeck:
        A, M, P = (np.asarray(x, float) for x in (altitudes, machs, powers))
        F = np.empty((len(A), len(M), len(P)))
        W = np.empty_like(F)
        for i, a in enumerate(A):
            for j, m in enumerate(M):
                for k, p in enumerate(P):
                    op = self.operating_point(a, m, p)
                    F[i, j, k], W[i, j, k] = op["thrust"], op["fuel_flow"]
        return EngineDeck(A, M, P, F, W, self.name)
