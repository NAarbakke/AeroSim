"""Flight layer: rocket (point-mass, RocketPy) and aircraft (JSBSim) simulators -> FlightResult."""

from aerosim.flight import aircraft, rocket  # noqa: F401  (registers backends)
from aerosim.flight.vehicles import AircraftVehicle, RocketVehicle

__all__ = ["AircraftVehicle", "RocketVehicle"]
