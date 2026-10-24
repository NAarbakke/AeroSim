"""Solid rocket motors.

bates        BatesMotor: fast 0-D BATES model, CEA propellant library
grain_motor  GrainMotor: any grain stack (BATES, finocyl, star, C, D, X, moon,
             rod-and-tube, end burner, custom outline), time-stepped ballistics
grains       grain geometries (fast-marching regression for non-circular cores)
propellant   manual / CEA / tabbed propellant models
ric          read openMotor .ric files (format only)
"""

from aerosim.propulsion.solid import bates, grain_motor  # noqa: F401  (registers backends)
from aerosim.propulsion.solid.bates import BatesMotor
from aerosim.propulsion.solid.grain_motor import GrainMotor, MotorLimits, Nozzle
from aerosim.propulsion.solid.propellant import Propellant, TabbedPropellant
from aerosim.propulsion.solid.ric import load_ric, read_ric

__all__ = ["BatesMotor", "GrainMotor", "MotorLimits", "Nozzle", "Propellant",
           "TabbedPropellant", "load_ric", "read_ric"]
