"""Guidance, navigation and control.

blocks       PID, Actuator, Sensor, Controller protocol (no dependencies)
controllers  AirbrakeApogeeController, AttitudeHoldTVC
design       python-control: linear plants, pole placement, LQR, margins
landing      convex powered-descent guidance for booster landing (cvxpy; import directly)
"""

from aerosim.gnc.blocks import PID, Actuator, Controller, Sensor
from aerosim.gnc.controllers import AirbrakeApogeeController, AttitudeHoldTVC

__all__ = ["PID", "Actuator", "AirbrakeApogeeController", "AttitudeHoldTVC", "Controller", "Sensor"]
