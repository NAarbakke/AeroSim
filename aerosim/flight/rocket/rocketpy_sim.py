"""RocketPy backend: 6-DOF, variable mass, parachutes, native weather support.

Translation layer only: aerosim ThrustCurve -> rocketpy GenericMotor,
RocketVehicle -> rocketpy Rocket, any AtmosphereModel -> rocketpy custom_atmosphere.
RocketPy can also fetch GFS etc. itself (pass ``rocketpy_weather={...}``).
"""

from __future__ import annotations

import numpy as np

from aerosim.core import FlightResult, ThrustCurve, registry, require
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import RocketVehicle


def _environment(rp, atmosphere: AtmosphereModel, lat, lon, elevation, date, native=None):
    env = rp.Environment(latitude=lat, longitude=lon, elevation=elevation, date=date)
    if native:  # e.g. {"type": "Forecast", "file": "GFS"}
        env.set_atmospheric_model(**native)
        return env
    z = np.linspace(0.0, 30_000.0, 121)
    s = [atmosphere.at(h) for h in z]
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=list(zip(z, [x.pressure for x in s])),
        temperature=list(zip(z, [x.temperature for x in s])),
        wind_u=list(zip(z, [x.wind_u for x in s])),
        wind_v=list(zip(z, [x.wind_v for x in s])),
    )
    return env


@registry.register("flight", "rocketpy")
class RocketPySim:
    def __init__(self, latitude: float = 0.0, longitude: float = 0.0, elevation: float = 0.0,
                 date=None, rocketpy_weather: dict | None = None) -> None:
        self.site = (latitude, longitude, elevation, date)
        self.native = rocketpy_weather

    def run(self, vehicle: RocketVehicle, propulsion: ThrustCurve,
            atmosphere: AtmosphereModel, **options) -> FlightResult:
        rp = require("rocketpy", "rocket")
        env = _environment(rp, atmosphere, *self.site, native=self.native)  # noqa: F841

        # TODO: confirm GenericMotor signature against the pinned RocketPy version
        motor = rp.GenericMotor(  # noqa: F841
            thrust_source=np.column_stack([propulsion.time, propulsion.thrust]).tolist(),
            burn_time=propulsion.burn_time,
            propellant_initial_mass=propulsion.propellant_mass,
            dry_mass=propulsion.dry_mass,
            **options.get("motor_geometry", {}),
        )
        # TODO: build rp.Rocket from vehicle (inertia, drag curves, nose, fins, parachutes)
        raise NotImplementedError("TODO: RocketVehicle -> rocketpy.Rocket, then rp.Flight")

    @staticmethod
    def _to_result(flight) -> FlightResult:
        t = np.asarray(flight.time)
        pos = np.column_stack([flight.x(t), flight.y(t), flight.z(t) - flight.env.elevation])
        vel = np.column_stack([flight.vx(t), flight.vy(t), flight.vz(t)])
        return FlightResult(t, pos, vel, np.asarray(flight.rocket.total_mass(t)),
                            events={"apogee": float(flight.apogee_time)},
                            meta={"simulator": "rocketpy"})
