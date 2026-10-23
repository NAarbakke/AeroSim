"""Adapters to external propulsion libraries. Interface-only or partly done; see README status.

openmotor       openMotor solid grains (GPL-3.0, user-installed)
pycycle_engine  NASA pyCycle gas-turbine cycles
turbomachinery  TurboFlow meanline -> ComponentMap
feed_system     FullFlow / pyskyfire feed networks and regen cooling
"""

from aerosim.propulsion.adapters import feed_system, openmotor, pycycle_engine  # noqa: F401
from aerosim.propulsion.adapters.turbomachinery import ComponentMap

__all__ = ["ComponentMap"]
