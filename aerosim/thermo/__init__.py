"""Thermo layer. Three tools, three jobs (see docs/architecture.md, "Who does which chemistry"):

- nasa_cea   : rocket-chamber equilibrium + nozzle expansion       (burned, rocket)
- gas, kinetics : Cantera gas properties along a flow path and
                  finite-rate chemistry; equilibrium cross-check     (burned, airbreathing)
- fluids     : CoolProp real-fluid properties                      (unburned propellants)
"""

from aerosim.thermo import (  # noqa: F401  (registers backends)
    fluids,
    gas,
    kinetics,
    nasa_cea,
    propellants,
    rocketcea,
)
from aerosim.thermo.nasa_cea import NasaCEAModel

__all__ = ["NasaCEAModel"]
