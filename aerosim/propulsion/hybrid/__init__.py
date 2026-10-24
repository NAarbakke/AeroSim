"""Hybrid rocket motors: liquid / gaseous oxidizer over a solid fuel grain."""

from aerosim.propulsion.hybrid import motor  # noqa: F401  (registers the backend)
from aerosim.propulsion.hybrid.motor import HybridHistory, HybridMotor

__all__ = ["HybridHistory", "HybridMotor"]
