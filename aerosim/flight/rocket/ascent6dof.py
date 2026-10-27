"""Six-degree-of-freedom orbital ascent: the 3-DOF ascent (``orbital_ascent``) with the
vehicle's rotation, a thrust-vector-control autopilot and wind.

    sim = SixDofAscentSim(62.9, 40.6, program=prog, layout=ROCKOT_LAYOUT, wind=WindProfile())
    res = sim.run(vehicle)            # AscentResult plus attitude, angle of attack, gimbal ...

Same flight programme, staging, fairing jettison and stop conditions as ``OrbitalAscentSim``:
its steering laws now give the *commanded* thrust direction, and the autopilot turns the
vehicle to follow it. State: ECI position, velocity, mass, attitude quaternion (body to ECI,
scalar first), body rates, and the two lateral thrust components of the gimbal (sines of
the deflection in the body y and z planes).

Body axes: x along the vehicle towards the nose; on the pad z points down the launch
azimuth and y = z cross x, so pitch-over is a rotation about y. Roll is not modelled (an
axisymmetric vehicle with a roll-hold loop: the roll rate stays zero).

Forces: J2 gravity; thrust ``stage.thrust(p)`` along the gimballed nozzle axis; axial drag
q A C_D(M) along -x; normal force q A C_Nα sin α at the Barrowman centre of pressure, in
the plane of the air-relative velocity (``aerosim.flight.stability``). The air co-rotates
with the Earth and moves with ``wind``.

Moments about the CG (mass, CG and pitch inertia from the ``Layout`` as tanks drain and
parts drop): normal force at x_cp, thrust at the gimbal point, aerodynamic pitch damping
-q A Σ C_Nα,i (x_i - x_cg)^2 / V and jet damping -ṁ (x_exit - x_cg)^2, times the body rate.
I ω' = M for pitch and yaw; the İω term of a variable-mass body is left out (jet damping
carries the dominant part of it).

Autopilot (continuous, gain-scheduled on dynamic pressure, as launch-vehicle autopilots
are): attitude error e = x_body × u_cmd in body axes; wanted angular acceleration
Kp e - Kd (ω - ω_cmd), ω_cmd the turn rate of the command itself (so a turning command is
tracked without lag; without it the gravity turn lofts by a few kilometres), built as an
attitude loop feeding a rate loop whose rate command is capped (``max_rate_deg``), so large
errors - a guidance hand-over at staging - are slewed out instead of overshooting. Kp = ω_c^2 + max(M_α / I, 0) - the extra term cancels the airframe's
aerodynamic instability - and Kd = 2 ζ ω_c. The torque I (Kp e - Kd ω) maps to lateral
thrust F_z = τ_y / l, F_y = -τ_z / l (l: gimbal point aft of the CG), limited to the
maximum deflection, through a first-order actuator with a slew-rate limit. Guidance uses
the navigation (ground-relative) velocity, not the wind, as real open-loop pitch
programmes do; so wind shows up as angle of attack and the gimbal holds it against the
aerodynamic moment. Without thrust (coasts) a reaction-control loop points the vehicle
prograde. No load relief, no slosh, no structural bending. The law is continuous because
it runs inside SciPy's adaptive integrator (trial steps are evaluated and discarded), where
a stateful discrete controller such as ``gnc.controllers.AttitudeHoldTVC`` cannot live.
``autopilot_margins`` checks the same loop's gain and phase margins with python-control.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from aerosim.core import registry
from aerosim.flight import orbits
from aerosim.flight.orbits import OMEGA_VEC
from aerosim.flight.rocket.orbital_ascent import AscentResult, OrbitalAscentSim
from aerosim.flight.stability import Layout, _stage_of_phase, barrowman, mass_properties


@dataclass
class WindProfile:
    """Horizontal wind against altitude: a surface value plus a jet stream (Gaussian in
    altitude) and optional 1 - cos gusts ``(centre altitude m, thickness m, peak m/s)``,
    all from one direction (meteorological: where it blows *from*, deg from north)."""

    surface: float = 5.0                    # m/s
    peak: float = 35.0                      # m/s, jet-stream core
    peak_altitude: float = 10.5e3           # m
    width: float = 3.5e3                    # m (Gaussian sigma)
    from_deg: float = 270.0                 # westerly
    gusts: list = field(default_factory=list)

    def speed(self, h: float) -> float:
        s = self.surface * math.exp(-max(h, 0.0) / 2e3) + self.peak * math.exp(
            -0.5 * ((h - self.peak_altitude) / self.width) ** 2)
        for hc, thick, amp in self.gusts:
            if abs(h - hc) < thick / 2:
                s += amp * 0.5 * (1 - math.cos(2 * math.pi * (h - hc + thick / 2) / thick))
        return s

    def vector(self, r: np.ndarray, h: float) -> np.ndarray:
        """Wind velocity (ECI components, relative to the rotating Earth) at position r."""
        s = self.speed(h)
        if s == 0.0:
            return np.zeros(3)
        e, n, _ = orbits.enu(r)
        to = math.radians(self.from_deg + 180.0)
        return s * (math.sin(to) * e + math.cos(to) * n)


class RocketPyWind:
    """Wind from a RocketPy ``Environment`` (any of its atmospheric models: custom profiles,
    soundings, forecasts, reanalysis): its east and north components against height above
    sea level.

        env = rocketpy.Environment(latitude=62.9, longitude=40.6)
        env.set_atmospheric_model(type="Wyoming_sounding", file=url)
        sim = SixDofAscentSim(..., wind=RocketPyWind(env))
    """

    def __init__(self, environment) -> None:
        self.env = environment

    def components(self, h: float) -> tuple[float, float]:
        return float(self.env.wind_velocity_x(h)), float(self.env.wind_velocity_y(h))

    def speed(self, h: float) -> float:
        return math.hypot(*self.components(h))

    def vector(self, r: np.ndarray, h: float) -> np.ndarray:
        u, v = self.components(h)
        if u == 0.0 and v == 0.0:
            return np.zeros(3)
        e, n, _ = orbits.enu(r)
        return u * e + v * n


@dataclass
class Autopilot:
    bandwidth: float = 3.0                  # rad/s, closed-loop attitude frequency ω_c
    damping: float = 0.7
    max_gimbal_deg: float = 7.0
    gimbal_rate_deg: float = 20.0           # deg/s slew limit
    lag: float = 0.05                       # s, actuator time constant
    max_rate_deg: float = 5.0               # deg/s, cap on the rate the attitude loop asks for
    rcs_bandwidth: float = 1.0              # rad/s, coast attitude loop
    rcs_max_accel_deg: float = 3.0          # deg/s^2


@dataclass
class Ascent6DofResult(AscentResult):
    """``AscentResult`` plus the rotational state. Angles in degrees."""

    attitude: np.ndarray = None             # (n, 4) quaternion body -> ECI, scalar first
    body_axis: np.ndarray = None            # (n, 3) ECI unit vector of the body x axis
    body_rates: np.ndarray = None           # (n, 3) deg/s, body axes
    air_velocity: np.ndarray = None         # (n, 3) ECI, relative to the moving air
    wind: np.ndarray = None                 # (n, 3) ECI wind vector
    wind_speed: np.ndarray = None           # m/s
    alpha: np.ndarray = None                # pitch-plane angle of attack (body x-z)
    beta: np.ndarray = None                 # yaw-plane angle (body x-y)
    alpha_total: np.ndarray = None          # angle between the body axis and the airflow
    gimbal_pitch: np.ndarray = None         # nozzle deflection in the body x-z plane
    gimbal_yaw: np.ndarray = None           # in the body x-y plane
    gimbal_total: np.ndarray = None
    q_alpha: np.ndarray = None              # Pa deg: the aerodynamic load index
    x_cg: np.ndarray = None                 # m from the nose tip
    x_cp: np.ndarray = None
    inertia: np.ndarray = None              # kg m^2, pitch, about the CG


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_matrix(q):
    w, x, y, z = q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _matrix_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


@registry.register("flight", "orbital_ascent_6dof")
class SixDofAscentSim(OrbitalAscentSim):
    """``OrbitalAscentSim`` with rotational dynamics, TVC autopilot and wind (see module)."""

    NX = 16                                  # r 3, v 3, m, q 4, ω 3, gimbal 2

    def __init__(self, latitude: float, longitude: float = 0.0, site_height: float = 0.0,
                 program=None, *, layout: Layout, wind: WindProfile | RocketPyWind | None = None,
                 autopilot: Autopilot | None = None, rtol: float = 1e-7, max_step: float = 1.0,
                 dt_out: float = 0.5) -> None:
        super().__init__(latitude, longitude, site_height, program, rtol, max_step)
        self.layout = layout
        self.wind = wind if wind is not None else WindProfile(surface=0.0, peak=0.0)
        self.autopilot = autopilot or Autopilot()
        self.dt_out = dt_out
        self._aero = {on: barrowman(layout, on) for on in (True, False)}

    # --- state ------------------------------------------------------------------
    def _initial_state(self, r0, v0, mass, n):
        up = r0 / np.linalg.norm(r0)
        z = self._launch_direction(r0, n)
        R = np.column_stack([up, np.cross(z, up), z])
        return np.concatenate([r0, v0, [mass], _matrix_to_quat(R), np.zeros(3), np.zeros(2)])

    def _segment(self, ctx, t0, y0, t_end, stage, steer, stop=None, phase="", dt_out=1.0):
        y0 = np.array(y0, float)
        y0[14:16] = 0.0                     # a new engine (or none) starts centred
        return super()._segment(ctx, t0, y0, t_end, stage, steer, stop, phase,
                                min(dt_out, self.dt_out) if stage is not None else dt_out)

    # --- mass properties and aerodynamics --------------------------------------
    def _props(self, m: float, k: int, gone) -> tuple[float, float, float]:
        """(x_cg, inertia, stage index) at mass m while stage k burns (or last burned):
        earlier stages are empty, later ones full, stage k holds the rest."""
        stages = self._ctx["veh"].stages
        full = {s.name: s.propellant_mass for s in stages}
        present = lambda p: p.until is None or p.until not in gone
        fixed = sum(p.mass for p in self.layout.parts if present(p))
        later = sum(s.propellant_mass for s in stages[k + 1:])
        left = {s.name: (0.0 if i < k else s.propellant_mass) for i, s in enumerate(stages)}
        left[stages[k].name] = max(m - fixed - later, 0.0)
        _, xcg, inertia = mass_properties(self.layout, left, full, present)
        return xcg, inertia

    def _air(self, t, r, v, h, R, stage, m, k, gone, fairing_on, atm):
        """Everything the dynamics and the outputs share at one state."""
        xcg, inertia = self._props(m, k, gone)
        vnav = v - np.cross(OMEGA_VEC, r)                 # ground-relative (navigation)
        wind = self.wind.vector(r, h) if h < 60e3 else np.zeros(3)
        vair = vnav - wind
        out = {"xcg": xcg, "I": inertia, "vnav": vnav, "wind": wind, "vair": vair,
               "rho": 0.0, "p": 0.0, "a_snd": np.inf, "q": 0.0, "qnav": 0.0}
        if h < 200e3:
            s = atm.at(max(h, 0.0))
            va = float(np.linalg.norm(vair))
            out.update(rho=s.density, p=s.pressure, a_snd=s.speed_of_sound,
                       q=0.5 * s.density * va * va,
                       qnav=0.5 * s.density * float(np.dot(vnav, vnav)))
        comps = self._aero[fairing_on]
        cna = sum(c for c, _ in comps)
        out["cna"], out["xcp"] = cna, sum(c * x for c, x in comps) / cna
        out["vb"] = R.T @ vair
        return out

    # --- dynamics ---------------------------------------------------------------
    def _rhs(self, t, y, vehicle, atm, stage, steer):
        ctx, L, ap = self._ctx, self.layout, self.autopilot
        r, v, m, q, w, gim = y[:3], y[3:6], y[6], y[7:11], y[11:14], y[14:16]
        stages = vehicle.stages
        k = stages.index(stage) if stage is not None else len(stages) - 1
        gone = ctx["events"]
        h = orbits.altitude(r)
        R = _quat_to_matrix(q)
        xb = R[:, 0]
        S = self._air(t, r, v, h, R, stage, m, k, gone, ctx["fairing_on"], atm)
        xcg, inertia, vb = S["xcg"], S["I"], S["vb"]
        A = math.pi / 4 * L.diameter ** 2
        acc = orbits.gravity(r)
        torque = np.zeros(3)
        va = float(np.linalg.norm(vb))
        cna, xcp = S["cna"], S["xcp"]
        if S["q"] > 0 and va > 1.0:
            cd = vehicle.cd(va / S["a_snd"])
            fb = np.array([-S["q"] * vehicle.reference_area * cd * np.sign(vb[0]),
                           -S["q"] * A * cna * vb[1] / va, -S["q"] * A * cna * vb[2] / va])
            acc = acc + (R @ fb) / m
            rx = xcg - xcp
            torque += np.array([0.0, -rx * fb[2], rx * fb[1]])
            mq = -S["q"] * A * sum(c * (x - xcg) ** 2 for c, x in self._aero[ctx["fairing_on"]]) / va
            torque[1:] += mq * w[1:]
        dm = 0.0
        dgim = -gim / ap.lag
        thrust = stage.thrust(S["p"]) if stage is not None and m > 0 else 0.0
        if thrust > 0:
            dm = -stage.mass_flow
            l_arm = L.gimbal[stage.name] - xcg
            torque[1:] += -stage.mass_flow * (L.nozzle_exit[stage.name] - xcg) ** 2 * w[1:]
            a_, b_ = gim
            c_ = math.sqrt(max(1.0 - a_ * a_ - b_ * b_, 0.0))
            fb = thrust * np.array([c_, a_, b_])
            acc = acc + (R @ fb) / m
            torque += np.array([0.0, l_arm * fb[2], -l_arm * fb[1]])
            # autopilot: attitude error, and the command's own turn rate (feed-forward: the
            # steering direction a short step ahead along the current acceleration)
            u = steer(t, r, v, S["vnav"])
            e = R.T @ np.cross(xb, u)
            dt_ff = 0.05
            u2 = steer(t + dt_ff, r + v * dt_ff, v + acc * dt_ff,
                       S["vnav"] + (acc - np.cross(OMEGA_VEC, v)) * dt_ff)
            w_cmd = R.T @ np.cross(u, (u2 - u) / dt_ff)
            m_alpha = S["qnav"] * A * cna * (xcg - xcp)
            wc = ap.bandwidth
            kp = wc * wc + max(m_alpha / inertia, 0.0)
            # attitude loop -> rate command (capped) -> rate loop; for small errors this is
            # Kp e - Kd (ω - ω_cmd), for large ones it slews at the capped rate
            kr = 2 * ap.damping * wc
            w_des = kp / kr * e[1:]
            nw = math.hypot(*w_des)
            if nw > math.radians(ap.max_rate_deg):
                w_des *= math.radians(ap.max_rate_deg) / nw
            want = inertia * kr * (w_des + w_cmd[1:] - w[1:])                  # τ_y, τ_z
            cmd = np.array([-want[1], want[0]]) / (l_arm * thrust)          # (F_y, F_z) / T
            smax = math.sin(math.radians(ap.max_gimbal_deg))
            nc = math.hypot(*cmd)
            if nc > smax:
                cmd *= smax / nc
            rate = math.radians(ap.gimbal_rate_deg)
            dgim = np.clip((cmd - gim) / ap.lag, -rate, rate)
        else:
            vv = v / np.linalg.norm(v)                        # coast: RCS holds prograde
            e = R.T @ np.cross(xb, vv)
            wc = ap.rcs_bandwidth
            alpha_d = wc * wc * e[1:] - 2 * ap.damping * wc * w[1:]
            amax = math.radians(ap.rcs_max_accel_deg)
            na = math.hypot(*alpha_d)
            if na > amax:
                alpha_d *= amax / na
            torque[1:] += inertia * alpha_d
        dw = np.array([0.0, torque[1] / inertia, torque[2] / inertia])
        dq = 0.5 * _quat_mul(q, np.array([0.0, *w]))
        return np.concatenate([v, acc, [dm], dq, dw, dgim])

    # --- results ----------------------------------------------------------------
    def _result(self, t, Y, ph, names, events, orbs, final, atm, r0, prop_left):
        base = super()._result(t, Y, ph, names, events, orbs, final, atm, r0, prop_left)
        tt, idx = np.unique(t, return_index=True)
        Y, ph = Y[:, idx], ph[idx]
        veh = self._ctx["veh"]
        stages = veh.stages
        n = tt.size
        cols = {k: np.zeros((n, 3)) for k in ("body_axis", "air_velocity", "wind")}
        sc = {k: np.full(n, np.nan) for k in ("alpha", "beta", "alpha_total", "gimbal_pitch",
                                               "gimbal_yaw", "q_alpha", "x_cg", "x_cp", "inertia")}
        t_fair = events.get("fairing_jettison", np.inf)
        for i in range(n):
            y = Y[:, i]
            gone = {e for e, te in events.items() if te < tt[i]}
            st = _stage_of_phase(veh, names[ph[i]])
            k = stages.index(st) if st is not None else len(stages) - 1
            R = _quat_to_matrix(y[7:11])
            h = float(base.altitude[i])
            S = self._air(tt[i], y[:3], y[3:6], h, R, st, y[6], k, gone, tt[i] < t_fair, atm)
            vb = S["vb"]
            va = float(np.linalg.norm(vb))
            cols["body_axis"][i], cols["air_velocity"][i], cols["wind"][i] = R[:, 0], S["vair"], S["wind"]
            sc["x_cg"][i], sc["x_cp"][i], sc["inertia"][i] = S["xcg"], S["xcp"], S["I"]
            if va > 1.0 and S["q"] > 200.0:         # airflow angle only where the air carries load
                sc["alpha"][i] = math.degrees(math.atan2(vb[2], vb[0]))
                sc["beta"][i] = math.degrees(math.atan2(vb[1], vb[0]))
                sc["alpha_total"][i] = math.degrees(math.acos(min(max(vb[0] / va, -1.0), 1.0)))
                sc["q_alpha"][i] = S["q"] * sc["alpha_total"][i]
            a_, b_ = y[14], y[15]
            sc["gimbal_yaw"][i] = math.degrees(math.asin(a_))
            sc["gimbal_pitch"][i] = math.degrees(math.asin(b_))
        sc["gimbal_total"] = np.degrees(np.arcsin(np.minimum(np.hypot(Y[14], Y[15]), 1.0)))
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        return Ascent6DofResult(
            **fields, attitude=Y[7:11].T / np.linalg.norm(Y[7:11], axis=0)[:, None],
            body_rates=np.degrees(Y[11:14].T), wind_speed=np.linalg.norm(cols["wind"], axis=1),
            **cols, **sc)


def autopilot_margins(res: Ascent6DofResult, vehicle, layout: Layout, autopilot: Autopilot | None = None,
                      atmosphere=None, dt: float = 2.0, t_end: float | None = None,
                      delay: float = 0.03) -> dict[str, np.ndarray]:
    """Classical stability margins of the pitch autopilot along the flight (python-control).

    At each time the rigid-body pitch dynamics are linearised about the flight (attitude =
    angle of attack, the flight path held fixed over the short period):
        theta / delta = T l / (I s^2 - M_q s - M_alpha)
    with the actuator 1 / (tau s + 1) and the autopilot (I / T l)(K_p + K_d s), the same
    gain-scheduled law the simulation flies. The loop is broken at the actuator. With
    M_alpha > 0 the airframe has an unstable pole, so there are two gain margins: the *upper*
    (rigid-body) margin, how much more gain the loop takes before the actuator lag makes it
    oscillate, and the *lower* (aerodynamic) margin, how far the gain may fall before the
    airframe's instability wins (reported where q > 1 kPa). ``delay`` (s) adds the sensing and computation delay of a real
    flight computer (second-order Pade), which the simulation idealises away; without it the
    upper margin is infinite. Rate limits and saturation are left out (linear analysis).
    Returns arrays: t, pm (deg), gm_upper and gm_lower (dB), wc (rad/s, gain crossover),
    stable (closed-loop poles in the left half plane)."""
    import control

    if atmosphere is None:
        from aerosim.atmosphere import StandardAtmosphere
        atmosphere = StandardAtmosphere()
    ap = autopilot or Autopilot()
    A = math.pi / 4 * layout.diameter ** 2
    t_stop = t_end if t_end is not None else res.time[-1]
    out = {k: [] for k in ("t", "pm", "gm_upper", "gm_lower", "wc", "stable")}
    t_next = res.time[0]
    for i, t in enumerate(res.time):
        if t < t_next or t > t_stop:
            continue
        st = _stage_of_phase(vehicle, res.phases[res.phase[i]])
        if st is None:
            continue
        t_next = t + dt
        h, xcg, inertia = res.altitude[i], res.x_cg[i], res.inertia[i]
        comps = barrowman(layout, t < res.events.get("fairing_jettison", np.inf))
        cna = sum(c for c, _ in comps)
        xcp = sum(c * x for c, x in comps) / cna
        v = max(float(res.speed[i]), 1.0)
        q = 0.5 * atmosphere.at(max(h, 0.0)).density * v * v if h < 200e3 else 0.0
        m_alpha = q * A * cna * (xcg - xcp)
        m_q = (-q * A * sum(c * (x - xcg) ** 2 for c, x in comps) / v
               - st.mass_flow * (layout.nozzle_exit[st.name] - xcg) ** 2)
        wc = ap.bandwidth
        kp, kd = wc * wc + max(m_alpha / inertia, 0.0), 2 * ap.damping * wc
        L = (control.tf([kd, kp], [1.0]) * control.tf([1.0], [ap.lag, 1.0])
             * control.tf([1.0], [1.0, -m_q / inertia, -m_alpha / inertia]))
        if delay > 0:
            L = L * control.tf(*control.pade(delay, 2))
        gms, pms, _, _, wgcs, _ = control.stability_margins(L, returnall=True)
        gms, pms, wgcs = (np.atleast_1d(np.asarray(a, float)) for a in (gms, pms, wgcs))
        up = [g for g in gms if np.isfinite(g) and g > 1]
        lo = [g for g in gms if np.isfinite(g) and 0 < g < 1]
        out["t"].append(t)
        out["gm_upper"].append(20 * math.log10(min(up)) if up else np.inf)
        out["gm_lower"].append(20 * math.log10(max(lo)) if lo and q > 1e3 else np.nan)
        k = int(np.argmin(np.abs(pms))) if pms.size else None
        out["pm"].append(float(pms[k]) if k is not None else np.nan)
        out["wc"].append(float(wgcs[k]) if k is not None else np.nan)
        out["stable"].append(bool(np.all(np.real(control.poles(control.feedback(L, 1))) < 0)))
    return {k: np.asarray(v) for k, v in out.items()}
