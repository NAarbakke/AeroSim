"""Multi-stage orbital ascent: 3-DOF point mass over a rotating, oblate Earth.

Equations in an Earth-centred inertial frame (``aerosim.flight.orbits``): J2
gravity, thrust with back-pressure loss, drag on the Earth-relative velocity
(atmosphere co-rotates), constant mass flow per stage, instantaneous staging.

Flight programme (``AscentProgram``), the usual open-loop shape for a launcher:

1. vertical rise for ``vertical_time``
2. pitch kick: tilt ``kick_angle`` from vertical towards the launch azimuth
3. hold that attitude until the relative velocity has tilted as far, then gravity
   turn (thrust along the relative velocity) to the end of the first stage
4. intermediate stages: pitch above local horizontal linear in time,
   ``upper_pitch = (start, end)`` per stage (deg)
5. last stage, burn 1, until the orbit apogee reaches ``transfer_apogee``: either
   closed loop (``insertion_mode="climb_rate"``, default) - pitch that holds a climb
   rate ramping linearly over the burn from ``insertion_climb[0]`` to ``[1]`` m/s,
   compensating gravity minus centrifugal acceleration - or open loop
   (``"pitch"``): ``insertion_pitch = (start deg, rate deg/s)``
6. coast to apogee, burn 2 prograde until perigee reaches ``final_perigee``
   (default: circularise at the transfer apogee); skip with ``final_perigee=None``
   and ``circularise=False``

The fairing is jettisoned when the free-molecular heat flux 0.5 rho v^3 falls
below ``fairing_heat_flux`` (W/m^2), the criterion launch-vehicle guides quote.
Steering is in the inertial plane of ``target_inclination``: the kick heads along the
rotation-corrected launch azimuth, later stages yaw out residual out-of-plane velocity.
``solve_program`` tunes free parameters to hit reference targets (calibration);
``optimize_program`` chooses them to maximise delivered performance (prediction).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from aerosim.core import FlightResult, registry
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight import orbits
from aerosim.flight.orbits import OMEGA_VEC, Orbit
from aerosim.flight.vehicles import LaunchVehicle, Stage

G0 = 9.80665


@dataclass
class AscentProgram:
    target_inclination: float               # deg
    northbound: bool = True                 # azimuth branch for the inclination
    vertical_time: float = 10.0             # s
    kick_angle: float = 4.0                 # deg from vertical
    kick_time: float = 8.0                  # s
    upper_pitch: list = field(default_factory=lambda: [[25.0, 8.0]])   # per middle stage
    insertion_mode: str = "climb_rate"      # "climb_rate" (closed loop) | "pitch" (open loop)
    insertion_pitch: tuple = (5.0, 0.0)     # "pitch": deg, deg/s for the last stage's first burn
    insertion_climb: tuple = (0.0, 0.0)     # "climb_rate": target dh/dt m/s at start, at end
    insertion_tau: float = 40.0             # s, climb-rate loop time constant
    transfer_apogee: float = 400e3          # m above RE
    final_perigee: float | None = None      # m; None -> circularise at transfer apogee
    circularise: bool = True
    fairing_heat_flux: float = 1135.0       # W/m^2
    coast_after: float = 0.0                # s of coast recorded after the last burn


@dataclass
class AscentResult(FlightResult):
    """ECI ``position``/``velocity``; derived series below. Events map name -> time."""

    altitude: np.ndarray = None             # m above ellipsoid
    speed: np.ndarray = None                # m/s, Earth-relative
    inertial_speed: np.ndarray = None
    flight_path_angle: np.ndarray = None    # deg, relative velocity above horizon
    dynamic_pressure: np.ndarray = None     # Pa
    mach: np.ndarray = None
    acceleration: np.ndarray = None         # m/s^2, non-gravitational (what the payload feels)
    downrange: np.ndarray = None            # m, great circle from the launch site
    latitude: np.ndarray = None             # deg, geocentric
    longitude: np.ndarray = None            # deg, Earth-fixed
    phase: np.ndarray = None                # int phase index per sample
    phases: list = field(default_factory=list)       # phase names
    orbits: dict = field(default_factory=dict)       # event -> Orbit
    final_orbit: Orbit | None = None

    @property
    def apogee(self) -> float:
        return float(self.altitude.max())

    def at(self, event: str) -> dict:
        """Interpolated state summary at an event time."""
        t = self.events[event]
        g = {k: float(np.interp(t, self.time, getattr(self, k))) for k in
             ("altitude", "speed", "inertial_speed", "flight_path_angle", "dynamic_pressure",
              "downrange", "mass", "acceleration")}
        return {"time": t, **g}


@registry.register("flight", "orbital_ascent")
class OrbitalAscentSim:
    def __init__(self, latitude: float, longitude: float = 0.0, site_height: float = 0.0,
                 program: AscentProgram | dict | None = None, rtol: float = 1e-8,
                 max_step: float = 2.0) -> None:
        self.lat, self.lon, self.h0 = latitude, longitude, site_height
        if program is None or isinstance(program, dict):
            program = AscentProgram(**(program or {}))
        self.program = program
        self.rtol, self.max_step = rtol, max_step

    # --- geometry -----------------------------------------------------------------
    def _plane_normal(self, r0) -> np.ndarray:
        az = math.radians(orbits.inertial_azimuth(self.lat, self.program.target_inclination,
                                                  self.program.northbound))
        e, n, _ = orbits.enu(r0)
        d = math.sin(az) * e + math.cos(az) * n
        nv = np.cross(r0, d)
        return nv / np.linalg.norm(nv)

    # --- dynamics -----------------------------------------------------------------
    def _rhs(self, t, y, vehicle, atm, stage: Stage | None, steer):
        r, v, m = y[:3], y[3:6], y[6]
        h = orbits.altitude(r)
        a = orbits.gravity(r)
        dm = 0.0
        vrel = v - np.cross(OMEGA_VEC, r)
        if h < 200e3:
            s = atm.at(max(h, 0.0))
            vr = float(np.linalg.norm(vrel))
            if vr > 0:
                cd = vehicle.cd(vr / s.speed_of_sound)
                a = a - 0.5 * s.density * vr * cd * vehicle.reference_area / m * vrel
            p = s.pressure
        else:
            p = 0.0
        if stage is not None and m > 0:
            u = steer(t, r, v, vrel)
            a = a + stage.thrust(p) / m * u
            dm = -stage.mass_flow
        return np.concatenate([v, a, [dm]])

    def _launch_direction(self, r0, n, v_orbit: float = 7_600.0) -> np.ndarray:
        """Ground-relative launch heading whose inertial velocity lies in the target plane:
        the in-plane direction at orbital speed minus the site's rotation velocity."""
        up0 = r0 / np.linalg.norm(r0)
        d = v_orbit * np.cross(n, up0) - np.cross(OMEGA_VEC, r0)
        d -= np.dot(d, up0) * up0
        return d / np.linalg.norm(d)

    def _steer_first(self, t0, n, d_launch):
        P = self.program

        def steer(t, r, v, vrel):
            up = r / np.linalg.norm(r)
            hz = d_launch - np.dot(d_launch, up) * up
            hz /= np.linalg.norm(hz)
            if t < P.vertical_time:
                return up
            kick = math.radians(P.kick_angle)
            if t < P.vertical_time + P.kick_time:
                ang = kick * (t - P.vertical_time) / P.kick_time
                return math.cos(ang) * up + math.sin(ang) * hz
            # hold the kick attitude until the relative velocity has tilted as far,
            # then zero angle of attack (gravity turn)
            vr = np.linalg.norm(vrel)
            if vr < 1e-6 or math.acos(min(np.dot(vrel, up) / vr, 1.0)) < kick:
                return math.cos(kick) * up + math.sin(kick) * hz
            return vrel / vr
        return steer

    @staticmethod
    def _steer_pitch(n, theta_of_t, t_end, accel, max_yaw_deg: float = 17.0):
        """Pitch programme in the target plane plus yaw that removes the out-of-plane
        inertial velocity by ``t_end``: a_n = -(v . n) / t_go, |yaw| <= max_yaw_deg."""
        smax = math.sin(math.radians(max_yaw_deg))

        def steer(t, r, v, vrel):
            up = r / np.linalg.norm(r)
            hz = np.cross(n, up)
            th = math.radians(theta_of_t(t))
            t_go = max(t_end - t, 5.0)
            sy = max(-smax, min(smax, -float(np.dot(v, n)) / t_go / max(accel(t), 1e-3)))
            cy = math.sqrt(1 - sy * sy)
            return cy * (math.cos(th) * hz + math.sin(th) * up) + sy * n
        return steer

    def _steer_climb(self, n, ts, t_burn, accel, max_pitch_deg: float = 45.0):
        """Closed-loop pitch holding a commanded climb rate (gravity-compensated)."""
        P = self.program
        h0, h1 = P.insertion_climb
        yaw = self._steer_pitch(n, lambda t: 0.0, ts + t_burn, accel)

        def steer(t, r, v, vrel):
            rn = np.linalg.norm(r)
            up = r / rn
            hdot = float(np.dot(v, up))
            vh2 = float(np.dot(v, v)) - hdot**2
            cmd = h0 + (h1 - h0) * min(max((t - ts) / t_burn, 0.0), 1.0)
            need = orbits.MU / rn**2 - vh2 / rn + (cmd - hdot) / P.insertion_tau
            sp = max(-math.sin(math.radians(max_pitch_deg)),
                     min(math.sin(math.radians(max_pitch_deg)), need / accel(t)))
            base = yaw(t, r, v, vrel)                 # in-plane horizontal + yaw
            horiz = base - np.dot(base, up) * up
            horiz /= np.linalg.norm(horiz)
            return math.sqrt(1 - sp * sp) * horiz + sp * up
        return steer

    @staticmethod
    def _steer_prograde(t, r, v, vrel):
        return v / np.linalg.norm(v)

    # --- integration helpers ------------------------------------------------------
    def _segment(self, ctx, t0, y0, t_end, stage, steer, stop=None, phase="", dt_out=1.0):
        """Integrate [t0, t_end], handling fairing jettison inside; ``stop(t, y)`` is a
        terminal event function (crossing from - to +). Returns (t, y, stopped)."""
        while True:
            evs = []
            if ctx["fairing_on"]:
                def fairing(t, y, *_):
                    r, v = y[:3], y[3:6]
                    h = orbits.altitude(r)
                    if h < 60e3:
                        return 1.0
                    rho = ctx["atm"].at(h).density
                    vrel = np.linalg.norm(v - np.cross(OMEGA_VEC, r))
                    return 0.5 * rho * vrel**3 - self.program.fairing_heat_flux
                fairing.terminal, fairing.direction = True, -1
                evs.append(fairing)
            if stop is not None:
                def stop_ev(t, y, *_, f=stop):
                    return f(t, y)
                stop_ev.terminal, stop_ev.direction = True, 1
                evs.append(stop_ev)
            evs_t = np.arange(t0, t_end, dt_out)
            sol = solve_ivp(self._rhs, (t0, t_end), y0, args=(ctx["veh"], ctx["atm"], stage, steer),
                            rtol=self.rtol, atol=1e-6, max_step=self.max_step, events=evs or None,
                            t_eval=np.append(evs_t[evs_t > t0], t_end), dense_output=False)
            if sol.status < 0:
                raise RuntimeError(f"{phase}: integration failed at t={t0:.1f}: {sol.message}")
            if len(sol.t):
                ctx["record"](np.asarray(sol.t), np.asarray(sol.y).reshape(7, -1), phase)
                t1, y1 = sol.t[-1], sol.y[:, -1]
            else:
                t1, y1 = t0, y0
            hit = sol.status == 1
            if hit and ctx["fairing_on"] and sol.t_events[0].size:
                te, ye = sol.t_events[0][0], sol.y_events[0][0].copy()
                ctx["record"](np.array([te]), ye[:, None], phase)
                ctx["fairing_on"] = False
                ctx["events"]["fairing_jettison"] = float(te)
                ye[6] -= ctx["veh"].fairing_mass
                t0, y0 = te, ye
                continue
            if hit and stop is not None:
                k = 1 if len(evs) == 2 else 0
                te, ye = float(sol.t_events[k][0]), sol.y_events[k][0]
                ctx["record"](np.array([te]), ye[:, None], phase)
                return te, ye, True
            return float(t1), y1, False

    # --- main ---------------------------------------------------------------------
    def run(self, vehicle: LaunchVehicle, propulsion=None, atmosphere: AtmosphereModel | None = None,
            **_) -> AscentResult:
        if atmosphere is None:
            from aerosim.atmosphere import StandardAtmosphere
            atmosphere = StandardAtmosphere()
        P = self.program
        r0, v0 = orbits.site_state(self.lat, self.lon, self.h0)
        n = self._plane_normal(r0)
        T, Y, PH, names = [], [], [], []
        events: dict[str, float] = {"liftoff": 0.0}
        orbs: dict[str, Orbit] = {}

        def record(t, y, phase):
            if phase not in names:
                names.append(phase)
            T.append(t)
            Y.append(y)
            PH.append(np.full(t.size, names.index(phase)))

        ctx = {"veh": vehicle, "atm": atmosphere, "fairing_on": vehicle.fairing_mass > 0,
               "events": events, "record": record}
        y = np.concatenate([r0, v0, [vehicle.liftoff_mass]])
        t = 0.0
        stages = vehicle.stages
        # lower stages
        for i, st in enumerate(stages[:-1]):
            if i == 0:
                steer = self._steer_first(t, n, self._launch_direction(r0, n))
            else:
                a, b = (P.upper_pitch[i - 1] if i - 1 < len(P.upper_pitch) else P.upper_pitch[-1])
                ts, m0 = t, y[6]
                steer = self._steer_pitch(
                    n, lambda tt, a=a, b=b, ts=ts, T_=st.burn_time: a + (b - a) * (tt - ts) / T_,
                    t + st.burn_time,
                    lambda tt, ts=ts, m0=m0, st=st: st.thrust_vac / (m0 - st.mass_flow * (tt - ts)))
            t, y, _ = self._segment(ctx, t, y, t + st.burn_time, st, steer, phase=st.name)
            events[f"{st.name}_burnout"] = t
            orbs[f"{st.name}_burnout"] = Orbit.from_state(y[:3], y[3:6])
            y = y.copy()
            y[6] -= st.dry_mass
        # last stage
        st = stages[-1]
        prop_left = st.propellant_mass
        th0, rate = P.insertion_pitch
        ts = t

        def apogee_reached(tt, yy, *_):
            return Orbit.from_state(yy[:3], yy[3:6]).apogee_alt - P.transfer_apogee

        m0, t_burn = y[6], prop_left / st.mass_flow

        def accel(tt):
            return st.thrust_vac / (m0 - st.mass_flow * (tt - ts))
        if P.insertion_mode == "pitch":
            steer = self._steer_pitch(n, lambda tt: th0 + rate * (tt - ts), t + t_burn, accel)
        else:
            steer = self._steer_climb(n, ts, t_burn, accel)
        t1, y, stopped = self._segment(
            ctx, t, y, t + t_burn, st, steer,
            stop=apogee_reached, phase=f"{st.name} burn 1")
        prop_left -= (t1 - t) * st.mass_flow
        events[f"{st.name}_burn1_start"], events[f"{st.name}_burn1_end"] = t, t1
        warnings = [] if stopped else [
            f"{st.name} ran out of propellant before the transfer apogee was reached"]
        orbs["transfer"] = Orbit.from_state(y[:3], y[3:6])
        t = t1
        if P.circularise and prop_left > 0:
            def at_apogee(tt, yy, *_):
                return -float(np.dot(yy[:3], yy[3:6]))          # radial velocity + -> -
            t, y, _ = self._segment(ctx, t, y, t + orbs["transfer"].period, None,
                                    None, stop=at_apogee, phase="coast", dt_out=10.0)
            target = P.final_perigee if P.final_perigee is not None else orbs["transfer"].apogee_alt

            def perigee_reached(tt, yy, *_):
                return Orbit.from_state(yy[:3], yy[3:6]).perigee_alt - target
            t2, y, _ = self._segment(ctx, t, y, t + prop_left / st.mass_flow, st,
                                     self._steer_prograde, stop=perigee_reached,
                                     phase=f"{st.name} burn 2")
            prop_left -= (t2 - t) * st.mass_flow
            events[f"{st.name}_burn2_start"], events[f"{st.name}_burn2_end"] = t, t2
            t = t2
        if P.coast_after > 0:
            t, y, _ = self._segment(ctx, t, y, t + P.coast_after, None, None,
                                    phase="final coast", dt_out=10.0)
        final = Orbit.from_state(y[:3], y[3:6])
        orbs["final"] = final
        res = self._result(np.concatenate(T), np.concatenate(Y, axis=1),
                           np.concatenate(PH), names, events, orbs, final, atmosphere,
                           r0, prop_left)
        res.meta["warnings"] = warnings
        return res

    def _result(self, t, Y, ph, names, events, orbs, final, atm, r0, prop_left) -> AscentResult:
        t, idx = np.unique(t, return_index=True)
        Y, ph = Y[:, idx], ph[idx]
        r, v, m = Y[:3].T, Y[3:6].T, Y[6]
        vrel = v - np.cross(OMEGA_VEC, r)
        alt = np.array([orbits.altitude(x) for x in r])
        spd = np.linalg.norm(vrel, axis=1)
        up = r / np.linalg.norm(r, axis=1)[:, None]
        fpa = np.degrees(np.arcsin(np.clip(np.sum(vrel * up, axis=1) / np.maximum(spd, 1e-9),
                                           -1, 1)))
        st = [atm.at(max(h, 0.0)) if h < 200e3 else None for h in alt]
        rho = np.array([s.density if s else 0.0 for s in st])
        a_snd = np.array([s.speed_of_sound if s else np.inf for s in st])
        q = 0.5 * rho * spd**2
        # site position rotating with the Earth -> ground range
        ang = OMEGA_VEC[2] * t
        site = np.column_stack([r0[0] * np.cos(ang), r0[0] * np.sin(ang), np.full(t.size, r0[2])])
        cosang = np.sum(up * site, axis=1) / np.linalg.norm(site, axis=1)
        down = orbits.RE * np.arccos(np.clip(cosang, -1, 1))
        lat = np.degrees(np.arcsin(up[:, 2]))
        lon = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - ang) + self.lon + 180) % 360 - 180
        # non-gravitational acceleration from the velocity history (thrust + drag)
        grav = np.array([orbits.gravity(x) for x in r])
        acc = np.linalg.norm(np.gradient(v, t, axis=0) - grav, axis=1)
        for k in range(1, t.size):          # gradient smears across staging; use one-sided
            if ph[k] != ph[k - 1]:
                acc[k] = acc[min(k + 1, t.size - 1)]
                acc[k - 1] = acc[max(k - 2, 0)]
        powered_low = t <= events.get(f"{names[0]}_burnout", t[-1]) + 1.0   # max-q in first stage
        i_q = int(np.argmax(np.where(powered_low, q, -1.0)))
        events = {**events, "max_q": float(t[i_q])}
        return AscentResult(
            time=t, position=r, velocity=v, mass=m, events=events,
            meta={"simulator": "orbital_ascent", "propellant_left": prop_left,
                  "program": dataclasses.asdict(self.program)},
            altitude=alt, speed=spd, inertial_speed=np.linalg.norm(v, axis=1),
            flight_path_angle=fpa, dynamic_pressure=q, mach=spd / a_snd, acceleration=acc,
            downrange=down, latitude=lat, longitude=lon, phase=ph, phases=names,
            orbits=orbs, final_orbit=final)


def _set_path(obj, path: list[str], value: float):
    for part in path[:-1]:
        obj = obj[part] if isinstance(obj, dict) else (
            obj[int(part)] if isinstance(obj, list) else getattr(obj, part))
    last = path[-1]
    if isinstance(obj, dict):
        obj[last] = value
    elif isinstance(obj, list):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _builder(sim: OrbitalAscentSim, vehicle: LaunchVehicle, names: list[str], atmosphere=None):
    """build(x) -> (program, vehicle) and run(x) -> (result, vehicle) for dotted parameter paths
    (a "vehicle." prefix addresses the LaunchVehicle instead of the programme)."""
    import copy

    base = dataclasses.asdict(sim.program)

    def build(x):
        p = copy.deepcopy(base)
        p["upper_pitch"] = [list(u) for u in p["upper_pitch"]]
        tuples = [k for k, v in p.items() if isinstance(v, tuple)]
        for k in tuples:
            p[k] = list(p[k])
        veh = copy.deepcopy(vehicle)
        for k, val in zip(names, x):
            parts = k.split(".")
            if parts[0] == "vehicle":
                _set_path(veh, parts[1:], float(val))
            else:
                _set_path(p, parts, float(val))
        for k in tuples:
            p[k] = tuple(p[k])
        return AscentProgram(**p), veh

    def run(x):
        prog, veh = build(x)
        s = OrbitalAscentSim(sim.lat, sim.lon, sim.h0, prog, sim.rtol, sim.max_step)
        return s.run(veh, atmosphere=atmosphere), veh

    return build, run


def solve_program(sim: OrbitalAscentSim, vehicle: LaunchVehicle, free: dict, targets: list,
                  atmosphere=None, **kw):
    """Tune programme (and optionally vehicle) parameters by least squares.

    free    : {"kick_angle": (x0, lo, hi), "upper_pitch.0.1": ..., "insertion_pitch.0": ...,
               "vehicle.stages.0.propellant_mass": ...}; dotted paths index into fields,
              a "vehicle." prefix addresses the LaunchVehicle instead of the programme
    targets : [(fn(result, vehicle) -> value, goal, scale), ...]
    Returns (program, vehicle, result, scipy OptimizeResult).
    """
    from scipy.optimize import least_squares

    names = list(free)
    x0 = np.array([free[k][0] for k in names], float)
    lo = np.array([free[k][1] for k in names], float)
    hi = np.array([free[k][2] for k in names], float)
    build, run = _builder(sim, vehicle, names, atmosphere)

    def residuals(x):
        try:
            res, veh = run(x)
            return np.array([(fn(res, veh) - goal) / scale for fn, goal, scale in targets])
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full(len(targets), 1e3)

    opt = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3, **kw)
    prog, veh = build(opt.x)
    res, _ = run(opt.x)
    return prog, veh, res, opt


def optimize_program(sim: OrbitalAscentSim, vehicle: LaunchVehicle, free: dict,
                     objective=None, constraints: list = (), atmosphere=None,
                     maxfev: int = 400, verbose: bool = False):
    """Choose the programme by trajectory optimisation, not by fitting a reference trajectory.

    Maximises ``objective(result, vehicle)`` (default: last-stage propellant left in the final
    orbit, i.e. delivered performance) subject to ``constraints`` = [(fn, lower, upper, scale)]
    (None for an open side), as an exterior quadratic penalty. A run whose last stage cannot
    reach the transfer apogee scores by its apogee shortfall, so the search is pulled towards
    feasible programmes. Bounded Nelder-Mead in coordinates normalised to the ``free`` bounds.
    Returns (program, vehicle, result, scipy OptimizeResult).
    """
    from scipy.optimize import minimize

    names = list(free)
    lo = np.array([free[k][1] for k in names], float)
    hi = np.array([free[k][2] for k in names], float)
    z0 = (np.array([free[k][0] for k in names], float) - lo) / (hi - lo)
    build, run = _builder(sim, vehicle, names, atmosphere)
    objective = objective or (lambda res, veh: res.meta["propellant_left"])
    best = {"J": np.inf}

    def cost(z):
        x = lo + np.clip(z, 0, 1) * (hi - lo)
        try:
            res, veh = run(x)
        except (RuntimeError, ValueError, FloatingPointError):
            return 1e6
        if res.meta["warnings"]:                          # transfer apogee not reached
            short = sim.program.transfer_apogee - res.orbits["transfer"].apogee_alt
            J = 1e3 + max(short, 0.0) / 1e3
        else:
            J = -objective(res, veh) / 1e3
            for fn, a, b, scale in constraints:
                v = fn(res, veh)
                viol = (a - v if a is not None and v < a else 0.0) +                        (v - b if b is not None and v > b else 0.0)
                J += 10.0 * (viol / scale) ** 2
        if J < best["J"]:
            best.update(J=J, x=x)
            if verbose:
                print(f"J {J:10.4f}  " + "  ".join(f"{n}={v:.3f}" for n, v in zip(names, x)))
        return J

    opt = minimize(cost, z0, method="Nelder-Mead", bounds=[(0, 1)] * len(names),
                   options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True})
    x = lo + np.clip(opt.x, 0, 1) * (hi - lo)
    opt.x_phys = x
    prog, veh = build(x)
    res, _ = run(x)
    return prog, veh, res, opt
