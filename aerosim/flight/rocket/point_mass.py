"""Built-in 3-DOF point-mass rocket simulator with wind. Fast enough for
Monte Carlo and optimization inner loops; use RocketPy for 6-DOF fidelity."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from aerosim.core import FlightResult, ThrustCurve, registry
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import RocketVehicle

G0 = 9.80665


@registry.register("flight", "point_mass")
class PointMassRocketSim:
    """Open loop: adaptive solve_ivp. With ``controller``: fixed-step RK4, commands
    held between controller updates (zero-order hold) and passed through an actuator.
    Supported command: ``airbrake`` (0..1, scales vehicle.airbrake_cd_area)."""

    def __init__(self, max_time: float = 600.0, rtol: float = 1e-6, controller=None,
                 sensor=None, dt: float = 0.005) -> None:
        self.max_time, self.rtol = max_time, rtol
        self.controller, self.sensor, self.dt = controller, sensor, dt

    def run(self, vehicle: RocketVehicle, propulsion: ThrustCurve,
            atmosphere: AtmosphereModel, **_) -> FlightResult:
        if self.controller is not None:
            return self._run_closed_loop(vehicle, propulsion, atmosphere)
        tc = propulsion
        burned = np.concatenate([[0.0], cumulative_trapezoid(tc.mass_flow, tc.time)])
        m_full = vehicle.dry_mass + tc.dry_mass + tc.propellant_mass
        el, hd = math.radians(vehicle.elevation_deg), math.radians(vehicle.heading_deg)
        rail_dir = np.array([math.cos(el) * math.sin(hd), math.cos(el) * math.cos(hd), math.sin(el)])
        A = vehicle.reference_area

        def mass(t):
            return m_full - float(np.interp(t, tc.time, burned, right=burned[-1]))

        def rhs(t, s):
            r, v = s[:3], s[3:]
            atm = atmosphere.at(max(r[2], 0.0))
            v_air = v - np.array([atm.wind_u, atm.wind_v, 0.0])
            speed = float(np.linalg.norm(v_air))
            on_rail = float(np.dot(r, rail_dir)) < vehicle.rail_length
            direction = rail_dir if on_rail or speed < 1e-6 else v_air / speed
            m = mass(t)
            thrust = tc.thrust_at(t, atm.pressure)
            drag = 0.5 * atm.density * speed * vehicle.cd(speed / atm.speed_of_sound) * A * v_air
            a = (thrust * direction - drag) / m - np.array([0, 0, G0])
            if on_rail:  # constrain to rail; no sinking back before liftoff
                a = max(float(np.dot(a, rail_dir)), 0.0) * rail_dir
            return np.concatenate([v, a])

        def impact(t, s):
            return s[2] if t > tc.time[-1] or s[5] < 0 else 1.0
        impact.terminal, impact.direction = True, -1

        def apogee(t, s):
            return s[5] if t > 0.5 else 1.0
        apogee.direction = -1

        sol = solve_ivp(rhs, (0.0, self.max_time), np.zeros(6), events=(impact, apogee),
                        rtol=self.rtol, atol=1e-6, max_step=0.05, dense_output=False)
        events = {"burnout": float(tc.time[-1])}
        if len(sol.t_events[1]):
            events["apogee"] = float(sol.t_events[1][0])
        if len(sol.t_events[0]):
            events["impact"] = float(sol.t_events[0][0])
        return FlightResult(
            time=sol.t, position=sol.y[:3].T, velocity=sol.y[3:].T,
            mass=np.array([mass(t) for t in sol.t]), events=events,
            meta={"simulator": "point_mass", "motor": tc.name},
        )

    # -- closed loop -------------------------------------------------------------------
    def _run_closed_loop(self, vehicle, tc, atmosphere) -> FlightResult:
        from aerosim.gnc.blocks import Actuator

        ctrl, dt = self.controller, self.dt
        ctrl.reset()
        n_sub = max(1, round(ctrl.dt / dt))
        brake = Actuator(vehicle.airbrake_tau, vehicle.airbrake_rate, 0.0, 1.0)
        burned = np.concatenate([[0.0], cumulative_trapezoid(tc.mass_flow, tc.time)])
        m_full = vehicle.dry_mass + tc.dry_mass + tc.propellant_mass
        el, hd = math.radians(vehicle.elevation_deg), math.radians(vehicle.heading_deg)
        rail_dir = np.array([math.cos(el) * math.sin(hd), math.cos(el) * math.cos(hd), math.sin(el)])
        A = vehicle.reference_area

        def mass(t):
            return m_full - float(np.interp(t, tc.time, burned, right=burned[-1]))

        def rhs(t, s, b):
            r, v = s[:3], s[3:]
            atm = atmosphere.at(max(r[2], 0.0))
            v_air = v - np.array([atm.wind_u, atm.wind_v, 0.0])
            speed = float(np.linalg.norm(v_air))
            on_rail = float(np.dot(r, rail_dir)) < vehicle.rail_length
            direction = rail_dir if on_rail or speed < 1e-6 else v_air / speed
            cda = vehicle.cd(speed / atm.speed_of_sound) * A + b * vehicle.airbrake_cd_area
            a = (tc.thrust_at(t, atm.pressure) * direction - 0.5 * atm.density * speed * cda * v_air) \
                / mass(t) - np.array([0, 0, G0])
            if on_rail:
                a = max(float(np.dot(a, rail_dir)), 0.0) * rail_dir
            return np.concatenate([v, a])

        t, s = 0.0, np.zeros(6)
        T, S, B, events, cmd, k = [t], [s.copy()], [0.0], {"burnout": float(tc.time[-1])}, {}, 0
        while t < self.max_time:
            if k % n_sub == 0:
                atm = atmosphere.at(max(s[2], 0.0))
                speed = float(np.linalg.norm(s[3:]))
                state = {"t": t, "altitude": s[2], "vz": s[5], "speed": speed,
                         "position": s[:3].copy(), "velocity": s[3:].copy(), "mass": mass(t),
                         "density": atm.density, "burnout": t >= tc.time[-1],
                         "cd_area": vehicle.cd(speed / atm.speed_of_sound) * A,
                         "brake_cd_area": vehicle.airbrake_cd_area, "airbrake": brake.position}
                cmd = ctrl.update(t, self.sensor.measure(state) if self.sensor else state)
            b = brake.step(cmd.get("airbrake", 0.0), dt)
            k1 = rhs(t, s, b)
            k2 = rhs(t + dt / 2, s + dt / 2 * k1, b)
            k3 = rhs(t + dt / 2, s + dt / 2 * k2, b)
            k4 = rhs(t + dt, s + dt * k3, b)
            s_new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if "apogee" not in events and t > 0.5 and s[5] > 0 >= s_new[5]:
                events["apogee"] = t + dt * s[5] / (s[5] - s_new[5])
            t, s, k = t + dt, s_new, k + 1
            T.append(t), S.append(s.copy()), B.append(b)
            if t > tc.time[-1] and s[2] <= 0:
                events["impact"] = t
                break
        S = np.asarray(S)
        return FlightResult(np.asarray(T), S[:, :3], S[:, 3:], np.array([mass(x) for x in T]),
                            events=events, meta={"simulator": "point_mass", "motor": tc.name,
                                                 "closed_loop": True, "airbrake": np.asarray(B)})
