"""Adapters to external propulsion libraries. Interface-only or partly done; see README status.

openmotor       openMotor solid grains (GPL-3.0, user-installed)
pycycle_engine  NASA pyCycle turbojet / two-spool turbofan, design + off-design
regen           pyskyfire regenerative cooling of a thrust chamber
turbomachinery  TurboFlow meanline -> ComponentMap
feed_system     FullFlow feed networks (interface only)
"""

from aerosim.propulsion.adapters import feed_system, openmotor, pycycle_engine, regen  # noqa: F401
from aerosim.propulsion.adapters.turbomachinery import ComponentMap

__all__ = ["ComponentMap"]
