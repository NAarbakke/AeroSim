"""RocketPy backend: 6-DOF, variable mass, parachutes, native weather support.

Translation layer only: aerosim ThrustCurve -> rocketpy GenericMotor,
RocketVehicle -> rocketpy Rocket, any AtmosphereModel -> rocketpy custom_atmosphere.
RocketPy can also fetch GFS etc. itself (pass ``rocketpy_weather={...}``).

Geometry is measured from the nose tip, positive aft (RocketPy "nose_to_tail").
The vehicle's own drag (``drag_coefficient`` / ``drag_curve``) is used as RocketPy's
power-off and power-on drag, so point-mass and 6-DOF fly the same drag; nose and
fins only set lift and stability. Missing geometry gets defaults sized from the
diameter so any vehicle that flies in ``point_mass`` also flies here:

    length  15 D          nose     {"length": 3 D, "kind": "von karman"}
    cg      0.55 length   fins     {"n": 4, "root_chord": 2 D, "tip_chord": D,
                                    "span": 1.2 D, "position": length - root_chord}
    parachutes [{"name": "main", "cd_s": 1.0, "trigger": "apogee", "lag": 1.0}, ...]
"""

from __future__ import annotations

import math

import numpy as np

from aerosim.core import FlightResult, ThrustCurve, registry, require
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import RocketVehicle

_PROPELLANT_DENSITY = 1700.0   # kg/m^3, only to size a default motor chamber


def _environment(rp, atmosphere: AtmosphereModel, lat, lon, elevation, date, native=None):
    env = rp.Environment(latitude=lat, longitude=lon, elevation=elevation, date=date)
    if native:  # e.g. {"type": "Forecast", "file": "GFS"}
        env.set_atmospheric_model(**native)
        return env
    # AtmosphereModel altitudes are above the launch site (as in point_mass). Dense
    # near the ground: power-law wind rises steeply there and sets rail-exit weathercocking
    z = np.unique(np.concatenate([np.linspace(0.0, 20.0, 41), np.geomspace(20.0, 1000.0, 60),
                                  np.linspace(1000.0, 30_000.0, 117)]))
    s = [atmosphere.at(h) for h in z]
    h = z + elevation
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=list(zip(h, [x.pressure for x in s])),
        temperature=list(zip(h, [x.temperature for x in s])),
        wind_u=list(zip(h, [x.wind_u for x in s])),
        wind_v=list(zip(h, [x.wind_v for x in s])),
    )
    return env


def _motor(rp, tc: ThrustCurve, diameter: float, geometry: dict):
    g = dict(geometry)
    radius = g.pop("chamber_radius", 0.45 * diameter)
    height = g.pop("chamber_height", max(
        tc.propellant_mass / (_PROPELLANT_DENSITY * math.pi * radius**2), 2 * radius))
    thrust = np.column_stack([tc.time - tc.time[0], tc.thrust])
    return rp.GenericMotor(
        thrust_source=thrust.tolist(),
        burn_time=float(thrust[-1, 0]),
        chamber_radius=radius,
        chamber_height=height,
        chamber_position=g.pop("chamber_position", height / 2),
        propellant_initial_mass=tc.propellant_mass,
        nozzle_radius=g.pop("nozzle_radius", math.sqrt((tc.exit_area or 0.5 * radius**2)
                                                         / math.pi)),
        dry_mass=tc.dry_mass,
        center_of_dry_mass_position=g.pop("center_of_dry_mass_position", height / 2),
        dry_inertia=g.pop("dry_inertia", (0.0, 0.0, 0.0)),
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        **g,
    ), height


def build_rocket(rp, vehicle: RocketVehicle, motor) -> object:
    D = vehicle.diameter
    L = vehicle.length or 15 * D
    m, r = vehicle.dry_mass, D / 2
    if vehicle.pitch_inertia:
        i_pitch = vehicle.pitch_inertia
    else:
        i_pitch = m * (L**2 / 12 + r**2 / 4)     # slender cylinder about its CG
    drag = ([list(p) for p in vehicle.drag_curve] if vehicle.drag_curve
            else vehicle.drag_coefficient)
    rocket = rp.Rocket(
        radius=r, mass=m, inertia=(i_pitch, i_pitch, m * r**2 / 2),
        power_off_drag=drag, power_on_drag=drag,
        center_of_mass_without_motor=vehicle.cg_position or 0.55 * L,
        coordinate_system_orientation="nose_to_tail",
    )
    rocket.add_motor(motor, position=L)
    nose = {"length": 3 * D, "kind": "von karman", **vehicle.nose}
    rocket.add_nose(position=0.0, **nose)
    fins = {"n": 4, "root_chord": 2 * D, "tip_chord": D, "span": 1.2 * D, **vehicle.fins}
    fins.setdefault("position", L - fins["root_chord"])
    rocket.add_trapezoidal_fins(**fins)
    for p in vehicle.parachutes:
        rocket.add_parachute(**{"trigger": "apogee", "sampling_rate": 100, "lag": 0.0, **p})
    return rocket


@registry.register("flight", "rocketpy")
class RocketPySim:
    def __init__(self, latitude: float = 0.0, longitude: float = 0.0, elevation: float = 0.0,
                 date=None, rocketpy_weather: dict | None = None, max_time: float = 600.0,
                 terminate_on_apogee: bool = False, motor_geometry: dict | None = None,
                 rtol: float = 1e-6) -> None:
        self.site = (latitude, longitude, elevation, date)
        self.native = rocketpy_weather
        self.max_time, self.terminate_on_apogee, self.rtol = max_time, terminate_on_apogee, rtol
        self.motor_geometry = motor_geometry or {}
        self.flight = None   # the last rocketpy.Flight, for RocketPy's own plots / analysis

    def run(self, vehicle: RocketVehicle, propulsion: ThrustCurve,
            atmosphere: AtmosphereModel, **options) -> FlightResult:
        rp = require("rocketpy", "rocket")
        env = _environment(rp, atmosphere, *self.site, native=self.native)
        motor, _ = _motor(rp, propulsion, vehicle.diameter,
                          {**self.motor_geometry, **options.get("motor_geometry", {})})
        rocket = build_rocket(rp, vehicle, motor)
        self.flight = rp.Flight(
            rocket=rocket, environment=env, rail_length=vehicle.rail_length,
            inclination=vehicle.elevation_deg, heading=vehicle.heading_deg,
            terminate_on_apogee=self.terminate_on_apogee, max_time=self.max_time,
            rtol=self.rtol, verbose=False)
        return self._to_result(self.flight, propulsion)

    @staticmethod
    def _to_result(flight, propulsion: ThrustCurve) -> FlightResult:
        t = np.asarray(flight.time, float)
        pos = np.column_stack([flight.x(t), flight.y(t), flight.z(t) - flight.env.elevation])
        vel = np.column_stack([flight.vx(t), flight.vy(t), flight.vz(t)])
        mass = np.array([flight.rocket.total_mass(ti) for ti in t])
        events = {"rail_exit": float(flight.out_of_rail_time),
                  "burnout": float(propulsion.time[-1] - propulsion.time[0]),
                  "apogee": float(flight.apogee_time), "impact": float(t[-1])}
        for p in getattr(flight, "parachute_events", []):
            events[f"deploy_{p[1].name}"] = float(p[0])
        return FlightResult(t, pos, vel, mass, events=events,
                            meta={"simulator": "rocketpy",
                                  "static_margin_cal": float(flight.rocket.static_margin(0)),
                                  "max_mach": float(flight.max_mach_number)})
