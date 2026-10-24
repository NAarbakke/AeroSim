"""Engine design point from chemistry: chamber pressure, mixture ratio and nozzle -> delivered
Isp, mass flow and nozzle size, for building launch-vehicle stages from first principles.

Ideal performance is the NASA CEA infinite-area-combustor solution. Real-engine losses are
applied as textbook factors rather than tuned to the engine:

- kinetics: the delivered expansion lies between shifting equilibrium and frozen-at-throat
  composition; ``kinetics="mean"`` takes the average of the two ideal Isp values
- combustion: c* efficiency (injector mixing, 0.97-0.99 for staged combustion)
- nozzle: thrust-coefficient efficiency (divergence and boundary layer, 0.97-0.99)

The expansion ratio comes from one of: the value itself, a published sea-level / vacuum Isp
pair (the difference is p_a * A_e / (mdot g0), so A_e/mdot and hence eps follow directly), or
the nozzle exit diameter (eps and throat area solved together for the rated thrust).
"""

from __future__ import annotations

import math

from aerosim.core.units import G0

P_SL = 101_325.0
_MODELS: dict = {}


def _cea(frozen):
    if frozen not in _MODELS:
        from aerosim.thermo.combustion.nasa_cea import NasaCEAModel
        _MODELS[frozen] = NasaCEAModel(frozen=frozen)
    return _MODELS[frozen]


def _ideal(ox, fuel, mr, pc, eps, kinetics):
    eq = _cea(None).chamber(ox, fuel, mr, pc, eps)
    if kinetics == "equilibrium":
        return eq, eq["isp_vac"], eq["cf_vac"]
    fr = _cea("throat").chamber(ox, fuel, mr, pc, eps)
    if kinetics == "frozen":
        return eq, fr["isp_vac"], fr["isp_vac"] * G0 / eq["cstar"]
    isp = 0.5 * (eq["isp_vac"] + fr["isp_vac"])
    return eq, isp, isp * G0 / eq["cstar"]


def engine_point(thrust_vac: float, oxidizer: str | None = None, fuel: str | None = None,
                 mixture_ratio: float | None = None, chamber_pressure: float | None = None,
                 expansion_ratio: float | None = None, isp_split: tuple | None = None,
                 exit_diameter: float | None = None, isp_vac: float | None = None,
                 count: int = 1, kinetics: str = "mean", cstar_efficiency: float = 0.98,
                 cf_efficiency: float = 0.98, name: str = "engine") -> dict:
    """Delivered performance of ``count`` identical engines at full thrust (vacuum).

    Give ``isp_vac`` alone (with ``thrust_vac``) for an element taken as published, such as a
    vernier set; otherwise propellants, ``mixture_ratio``, ``chamber_pressure`` and one of
    ``expansion_ratio`` / ``isp_split`` (published (vac, sea-level) Isp, s) / ``exit_diameter``.
    Returns per-set totals: thrust_vac (N), mdot (kg/s), exit_area (m^2), isp_vac and isp_sl (s),
    plus the per-engine design (eps, throat area, c*, chamber temperature, ideal Isp).
    """
    if isp_vac is not None:
        mdot = thrust_vac / (isp_vac * G0)
        return {"name": name, "count": count, "thrust_vac": thrust_vac * count,
                "mdot": mdot * count, "exit_area": 0.0, "isp_vac": isp_vac, "isp_sl": None,
                "source": "published"}
    pc, mr = float(chamber_pressure), float(mixture_ratio)
    eta = cstar_efficiency * cf_efficiency
    cstar0 = _cea(None).chamber(oxidizer, fuel, mr, pc, 30.0)["cstar"]    # chamber-only
    cstar = cstar0 * cstar_efficiency
    if expansion_ratio is not None:
        eps = float(expansion_ratio)
    elif isp_split is not None:
        dv = (isp_split[0] - isp_split[1]) * G0                 # = p_a A_e / mdot
        eps = dv * pc / (P_SL * cstar)                          # A_e/mdot = eps c* / pc
    elif exit_diameter is not None:
        ae, eps = math.pi / 4 * exit_diameter**2, 50.0
        for _ in range(30):                                     # A_t = F / (Cf pc), eps = A_e/A_t
            _, _, cf = _ideal(oxidizer, fuel, mr, pc, eps, kinetics)
            new = ae * cf * cf_efficiency * pc / thrust_vac
            if abs(new - eps) < 1e-3 * eps:
                break
            eps = new
    else:
        raise ValueError(f"{name}: give expansion_ratio, isp_split or exit_diameter")
    eq, isp_ideal, _ = _ideal(oxidizer, fuel, mr, pc, eps, kinetics)
    isp = isp_ideal * eta
    mdot = thrust_vac / (isp * G0)
    at = mdot * cstar / pc
    ae = eps * at
    return {"name": name, "count": count, "thrust_vac": thrust_vac * count, "mdot": mdot * count,
            "exit_area": ae * count, "isp_vac": isp, "isp_sl": isp - P_SL * ae / (mdot * G0),
            "expansion_ratio": eps, "throat_area": at, "cstar": cstar,
            "isp_ideal_equilibrium": eq["isp_vac"], "isp_ideal": isp_ideal,
            "t_chamber": eq["t_chamber"], "mixture_ratio": mr, "chamber_pressure": pc,
            "source": "cea"}
