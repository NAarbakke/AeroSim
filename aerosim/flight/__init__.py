"""Flight layer: rocket simulators (point-mass, planar) and adapters (RocketPy, JSBSim) -> FlightResult."""

from aerosim.flight import adapters, rocket  # noqa: F401  (registers backends)
from aerosim.flight.vehicles import AircraftVehicle, RocketVehicle

__all__ = ["AircraftVehicle", "RocketVehicle"]
