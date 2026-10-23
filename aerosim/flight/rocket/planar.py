"""Planar (pitch-plane) rigid-body rocket: x, z, vx, vz, theta, q.

The step between point-mass and full 6-DOF: it has attitude, so it can model
weathercocking, aerodynamic (in)stability and thrust-vector control, but motion
is restricted to the vertical plane containing the launch heading.

Conventions: theta is pitch from horizontal (pi/2 = vertical), q = theta'.
Gimbal delta > 0 produces a nose-up moment T*l_t*sin(delta).
Aerodynamics: drag along -v_air; normal force N = qbar*S*CNa*alpha acting at the
CP, which sits ``static_margin`` aft of the CG, so M_aero = -N*static_margin;
pitch damping M = -qbar*S*d*Cmq*q*d/(2V).
Wind: the horizontal wind component along the heading.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid

from aerosim.core import FlightResult, ThrustCurve, registry
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import RocketVehicle

G0 = 9.80665


@registry.register("flight", "planar")
class PlanarRocketSim:
    def __init__(self, controller=None, sensor=None, dt: float = 0.002, max_time: float = 600.0,
                 initial_pitch_deg: float | None = None, gust: tuple[float, float, float] | None = None):
        """gust: (t_start, duration, speed m/s) extra horizontal wind pulse for disturbance tests."""
        self.controller, self.sensor, self.dt, self.max_time = controller, sensor, dt, max_time
        self.initial_pitch = initial_pitch_deg
        self.gust = gust

    def run(self, vehicle: RocketVehicle, propulsion: ThrustCurve,
            atmosphere: AtmosphereModel, **_) -> FlightResult:
        from aerosim.gnc.blocks import Actuator

        tc, v, dt = propulsion, vehicle, self.dt
        if v.pitch_inertia is None:
            raise ValueError("planar simulator needs vehicle.pitch_inertia")
        burned = np.concatenate([[0.0], cumulative_trapezoid(tc.mass_flow, tc.time)])
        m_full = v.dry_mass + tc.dry_mass + tc.propellant_mass
        S, d = v.reference_area, v.diameter
        hd = math.radians(v.heading_deg)
        lim = math.radians(v.gimbal_limit_deg)
        gimbal = Actuator(v.gimbal_tau, math.radians(v.gimbal_rate_deg), -lim, lim)
        ctrl = self.controller
        if ctrl:
            ctrl.reset()
        n_sub = max(1, round(ctrl.dt / dt)) if ctrl else 1

        def mass(t):
            return m_full - float(np.interp(t, tc.time, burned, right=burned[-1]))

        def wind(t, z):
            atm = atmosphere.at(max(z, 0.0))
            w = atm.wind_u * math.sin(hd) + atm.wind_v * math.cos(hd)   # along heading
            if self.gust and self.gust[0] <= t < self.gust[0] + self.gust[1]:
                w += self.gust[2]
            return atm, w

        def aero(t, s):
            _x, z, vx, vz, th, _q = s
            atm, w = wind(t, z)
            vax, vaz = vx - w, vz
            V = math.hypot(vax, vaz)
            alpha = 0.0 if V < 1.0 else (th - math.atan2(vaz, vax) + math.pi) % (2 * math.pi) - math.pi
            return atm, vax, vaz, V, alpha

        def rhs(t, s, delta, on_rail):
            _x, _z, vx, vz, th, q = s
            atm, vax, vaz, V, alpha = aero(t, s)
            m = mass(t)
            inertia = v.pitch_inertia * m / m_full
            T = tc.thrust_at(t, atm.pressure)
            qbar = 0.5 * atm.density * V**2
            eb = (math.cos(th), math.sin(th))          # body axis
            nb = (-math.sin(th), math.cos(th))         # body normal
            fx = T * (math.cos(delta) * eb[0] - math.sin(delta) * nb[0])
            fz = T * (math.cos(delta) * eb[1] - math.sin(delta) * nb[1])
            if V > 1e-6:
                D = qbar * S * v.cd(V / atm.speed_of_sound)
                fx -= D * vax / V
                fz -= D * vaz / V
            N = qbar * S * v.cn_alpha * alpha
            fx += N * nb[0]
            fz += N * nb[1]
            M = T * v.cg_to_nozzle * math.sin(delta) - N * v.static_margin
            if V > 1.0:
                M -= qbar * S * d * v.pitch_damping * q * d / (2 * V)
            ax, az = fx / m, fz / m - G0
            if on_rail:  # along the rail only, no rotation, no sinking
                a_along = max(ax * eb[0] + az * eb[1], 0.0)
                return np.array([vx, vz, a_along * eb[0], a_along * eb[1], 0.0, 0.0])
            return np.array([vx, vz, ax, az, q, M / inertia])

        th0 = math.radians(self.initial_pitch if self.initial_pitch is not None else v.elevation_deg)
        s = np.array([0.0, 0.0, 0.0, 0.0, th0, 0.0])
        t, k, cmd = 0.0, 0, {}
        rows, events = [], {"burnout": float(tc.time[-1])}
        while t < self.max_time:
            on_rail = math.hypot(s[0], s[1]) < v.rail_length
            if "rail_exit" not in events and not on_rail:
                events["rail_exit"] = t
            atm, _vax, _vaz, V, alpha = aero(t, s)
            thrust = tc.thrust_at(t, atm.pressure)
            if ctrl and k % n_sub == 0:
                state = {"t": t, "theta": s[4], "q": s[5], "alpha": alpha, "altitude": s[1],
                         "vx": s[2], "vz": s[3], "speed": V, "thrust": thrust, "mass": mass(t),
                         "burnout": t >= tc.time[-1], "on_rail": on_rail}
                cmd = ctrl.update(t, self.sensor.measure(state) if self.sensor else state)
            delta = gimbal.step(cmd.get("gimbal", 0.0), dt) if ctrl else 0.0
            rows.append((t, *s, delta, alpha, mass(t)))
            k1 = rhs(t, s, delta, on_rail)
            k2 = rhs(t + dt / 2, s + dt / 2 * k1, delta, on_rail)
            k3 = rhs(t + dt / 2, s + dt / 2 * k2, delta, on_rail)
            k4 = rhs(t + dt, s + dt * k3, delta, on_rail)
            s_new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if "apogee" not in events and t > 0.5 and s[3] > 0 >= s_new[3]:
                events["apogee"] = t
            t, s, k = t + dt, s_new, k + 1
            if t > tc.time[-1] and s[1] <= 0:
                events["impact"] = t
                break
        a = np.asarray(rows)
        pos = np.column_stack([a[:, 1] * math.sin(hd), a[:, 1] * math.cos(hd), a[:, 2]])
        vel = np.column_stack([a[:, 3] * math.sin(hd), a[:, 3] * math.cos(hd), a[:, 4]])
        return FlightResult(a[:, 0], pos, vel, a[:, 9], events=events,
                            meta={"simulator": "planar", "theta": a[:, 5], "q": a[:, 6],
                                  "gimbal": a[:, 7], "alpha": a[:, 8], "downrange": a[:, 1]})
