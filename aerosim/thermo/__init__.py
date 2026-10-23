"""Thermo layer. Three tools, three jobs (see docs/architecture.md, "Who does which chemistry"):

- combustion : NASA CEA rocket-chamber equilibrium + nozzle expansion,
               propellant aliases, solid propellant recipes          (burned, rocket)
- gas        : Cantera gas properties along a flow path and
               finite-rate chemistry; equilibrium cross-check         (burned, airbreathing)
- fluids     : CoolProp real-fluid properties                         (unburned propellants)
"""

from aerosim.thermo import combustion, fluids, gas  # noqa: F401  (registers backends)
from aerosim.thermo.combustion import NasaCEAModel

__all__ = ["NasaCEAModel"]
