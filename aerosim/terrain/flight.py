"""Fly a JSBSim aircraft along a Track at low level, following a terrain Profile.

    res = fly_route(grid, track, profile, deck, mach=0.5)   # FlightResult, meta has agl etc.

Each simulator step (default 120 Hz):

1. Navigation: geodetic position -> grid x, y (the grid's own projection); the terrain
   height there is written to JSBSim (``position/terrain-elevation-asl-ft``), so its
   above-ground height and ground contact use the real terrain. The aircraft is matched to
   the nearest point on the track: arc length s, cross-track error e (right +).
2. Lateral guidance (horizontal acceleration, right +):
       a_lat = V^2 kappa(s + V tau) - w^2 e - 2 zeta w de/dt
   curvature feed-forward (looked ahead by the roll response time tau) plus a
   second-order cross-track loop.
3. Vertical guidance (acceleration normal to the velocity, up +):
       gamma_cmd = gamma_ref(s) + k_h (h_ref(s) - h) / V
       a_up = V (V dgamma_ref/ds + k_gamma (gamma_cmd - gamma)) + g cos(gamma)
4. The two make one specific-force vector normal to the velocity: bank angle
   phi_cmd = atan2(a_lat, a_up) and load factor n_cmd = |a| / g. The vertical part is held
   between ``n_up_min`` g (the push-over the profile was planned with; it keeps the bank
   law continuous) and ``n_max``, and keeps priority: the horizontal part gets what the g
   and bank limits leave.
5. Inner loops on the F-16 flight-control system: roll-rate command (``aileron-cmd-norm``,
   full scale = 3.14 rad/s) from the bank error; pitch stick from a PI loop on load factor
   (the F-16 FCS itself blends pitch rate and g); deck power from a PI Mach hold.

Engine thrust and fuel come from an AeroSim EngineDeck as in ``JSBSimSim`` (the airframe's
own engine is shut down). The flight stops at the end of the track, on ground contact
(above-ground height < 0) or on leaving the terrain grid.
"""

from __future__ import annotations

import math
import shutil

import numpy as np

from aerosim.core import EngineDeck, FlightResult, require
from aerosim.core.units import G0
from aerosim.flight.adapters.jsbsim_sim import FT, LB, apply_deck, prepare_aircraft
from aerosim.terrain.dem import TerrainGrid
from aerosim.terrain.route import Profile, Track

GAINS = {
    "tau": 1.0,          # s, curvature look-ahead (roll response)
    "w_lat": 0.22,       # rad/s, cross-track loop natural frequency
    "zeta_lat": 0.9,
    "k_h": 0.4,          # 1/s, altitude error -> flight-path angle (times V)
    "k_gamma": 0.9,      # 1/s, flight-path angle error -> normal acceleration (times V)
    "k_phi": 2.5,        # 1/s, bank error -> roll-rate command
    "p_max": 2.4,        # rad/s, roll-rate command limit
    "k_n": 0.12,         # stick per g of load-factor error
    "ki_n": 0.35,        # stick per g s
    "n_rate": 6.0,       # g/s, onset limit on the load-factor command
    "k_mach": 4.0,
    "ki_mach": 0.5,
    "power0": 0.3,       # Mach-hold bias (about cruise power at low level)
}


class _Matcher:
    """Nearest point on a polyline, searched in a window around the previous match."""

    def __init__(self, track: Track):
        self.x, self.y, self.s = track.x, track.y, track.s
        self.course = track.course
        self.i = 0

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        n = self.x.size
        lo, hi = max(self.i - 20, 0), min(self.i + 200, n - 1)
        d = (self.x[lo:hi + 1] - x) ** 2 + (self.y[lo:hi + 1] - y) ** 2
        i = lo + int(np.argmin(d))
        j = min(i, n - 2)
        ax, ay = self.x[j + 1] - self.x[j], self.y[j + 1] - self.y[j]
        L2 = ax * ax + ay * ay
        u = min(max(((x - self.x[j]) * ax + (y - self.y[j]) * ay) / L2, 0.0), 1.0) if L2 > 0 else 0.0
        if u <= 0.0 and j > 0:                    # before this segment: try the previous one
            j -= 1
            ax, ay = self.x[j + 1] - self.x[j], self.y[j + 1] - self.y[j]
            L2 = ax * ax + ay * ay
            u = min(max(((x - self.x[j]) * ax + (y - self.y[j]) * ay) / L2, 0.0), 1.0)
        self.i = j
        px, py = self.x[j] + u * ax, self.y[j] + u * ay
        L = math.sqrt(L2) or 1.0
        e = ((x - px) * ay - (y - py) * ax) / L     # right of track +
        return self.s[j] + u * (self.s[j + 1] - self.s[j]), e


def fly_route(grid: TerrainGrid, track: Track, profile: Profile, deck: EngineDeck, *,
              mach: float, model: str = "f16", engine_count: int = 1, dt: float = 1 / 120,
              record_every: int = 6, n_max: float = 7.0, n_up_min: float = 0.3,
              bank_max_deg: float = 80.0, max_time: float | None = None,
              gains: dict | None = None, root_dir: str | None = None) -> FlightResult:
    jsb = require("jsbsim", "aircraft")
    root, n_thrust, n_tanks = prepare_aircraft(model, root_dir)
    try:
        fdm = jsb.FGFDMExec(str(root))
        fdm.set_debug_level(0)
        fdm.set_dt(dt)
        if not fdm.load_model(model):
            raise RuntimeError(f"JSBSim could not load {model}")
        return _fly(fdm, grid, track, profile, deck, mach, engine_count, n_thrust, n_tanks, dt,
                    record_every, n_max, n_up_min, math.radians(bank_max_deg), max_time,
                    {**GAINS, **(gains or {})})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fly(fdm, grid, track, prof, deck, mach_cmd, engine_count, n_thrust, n_tanks, dt,
         record_every, n_max, n_up_min, bank_max, max_time, k):
    S, X, Y = track.s, track.x, track.y
    course = track.course
    kappa = track.curvature
    h_ref, g_ref, dg_ref = prof.altitude, prof.gamma, prof.dgamma_ds
    lat0, lon0 = grid.to_latlon(X[0], Y[0])
    to_xy = grid._tr(False).transform
    h0 = float(h_ref[0])
    fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"] = float(lat0), float(lon0)
    fdm["ic/h-sl-ft"] = h0 / FT
    fdm["ic/terrain-elevation-ft"] = grid.elevation_at(X[0], Y[0]) / FT
    fdm["ic/mach"] = mach_cmd
    fdm["ic/psi-true-deg"] = math.degrees(course[0] - grid.convergence(lat0, lon0)) % 360
    fdm["ic/gamma-deg"] = math.degrees(g_ref[0])
    fdm.run_ic()
    for i in range(engine_count):                 # deck flight: the airframe's engine is off
        fdm[f"propulsion/engine[{i}]/set-running"] = 0
        fdm[f"fcs/throttle-cmd-norm[{i}]"] = 0.0
    fdm["gear/gear-cmd-norm"] = 0.0
    matcher = _Matcher(track)
    w, z = k["w_lat"], k["zeta_lat"]
    t_end = max_time or 3.0 * track.length / max(1.0, mach_cmd * 300.0)
    p0 = k["power0"]
    power, mach_int, n_int, n_prev = p0, 0.0, 0.0, 1.0
    rows, step, status = [], 0, "end of route"
    while True:
        t = fdm.get_sim_time()
        lat, lon = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"]
        x, y = to_xy(lon, lat)
        if not grid.contains(x, y, 2 * grid.spacing):
            status = "left the terrain grid"
            break
        terr = grid.elevation_at(x, y)
        fdm["position/terrain-elevation-asl-ft"] = terr / FT
        h = fdm["position/h-sl-ft"] * FT
        agl = h - terr
        if agl < 0.0:
            status = "ground contact"
            break
        s, e = matcher(x, y)
        if s >= S[-1] - 1.0:
            break
        if t > t_end:
            status = "time limit"
            break
        ve, vn = fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT
        vu = -fdm["velocities/v-down-fps"] * FT
        vg = math.hypot(ve, vn)
        V = math.sqrt(vg * vg + vu * vu)
        gam = math.atan2(vu, vg)
        conv = float(grid.convergence(lat, lon))
        chi = math.atan2(ve, vn) + conv               # course in the grid frame
        cc, sc = math.cos(conv), math.sin(conv)
        ve, vn = ve * cc + vn * sc, vn * cc - ve * sc     # velocity in the grid frame
        mach = fdm["velocities/mach"]
        # lateral: curvature feed-forward + cross-track PD
        kap = float(np.interp(s + vg * k["tau"], S, kappa))
        chi_t = float(np.interp(s, S, course))
        e_dot = vg * math.sin(chi - chi_t)
        a_lat = vg * vg * kap - w * w * e - 2 * z * w * e_dot
        # vertical: profile tracking
        hr = float(np.interp(s, S, h_ref))
        gr = float(np.interp(s, S, g_ref))
        dgr = float(np.interp(s + vg * 0.5, S, dg_ref))
        g_cmd = gr + max(min(k["k_h"] * (hr - h) / V, 0.35), -0.35)
        a_up = V * (V * dgr + k["k_gamma"] * (g_cmd - gam)) + G0 * math.cos(gam)
        # one specific-force vector normal to the velocity -> bank and load factor
        a_up = max(min(a_up, n_max * G0), n_up_min * G0)
        a_lat_max = min(math.sqrt(max((n_max * G0) ** 2 - a_up * a_up, 0.0)), a_up * math.tan(bank_max))
        a_lat = max(min(a_lat, a_lat_max), -a_lat_max)
        phi_cmd = math.atan2(a_lat, a_up)
        phi = fdm["attitude/phi-rad"]
        # load factor for the bank the aircraft has now, so a roll reversal (lift passing
        # through vertical) does not balloon it: the vertical part is held, the turn follows
        n_cmd = min(a_up / (G0 * max(math.cos(phi), math.cos(bank_max))), n_max)
        n_cmd = n_prev = min(max(n_cmd, n_prev - k["n_rate"] * dt), n_prev + k["n_rate"] * dt)
        n_meas = fdm["accelerations/Nz"]
        # inner loops
        p_cmd = max(min(k["k_phi"] * (phi_cmd - phi), k["p_max"]), -k["p_max"])
        fdm["fcs/aileron-cmd-norm"] = max(min(p_cmd * 0.31821, 1.0), -1.0)
        n_err = n_cmd - n_meas
        stick = -(k["k_n"] * n_err + k["ki_n"] * n_int) - 0.11 * (n_cmd - 1.0)
        if -1.0 < stick < 0.44 or stick * n_err > 0:     # anti-windup
            n_int += n_err * dt
        fdm["fcs/elevator-cmd-norm"] = max(min(stick, 0.44), -1.0)
        m_err = mach_cmd - mach
        raw = p0 + k["k_mach"] * m_err + k["ki_mach"] * (mach_int + m_err * dt)
        if 0.0 < raw < 1.0 or raw * m_err < 0:
            mach_int += m_err * dt
        power = float(np.clip(p0 + k["k_mach"] * m_err + k["ki_mach"] * mach_int, 0.0, 1.0))
        F, wf = apply_deck(fdm, deck, h, mach, power, n_thrust, n_tanks, dt)
        if step % record_every == 0:
            rows.append([t, x, y, h, ve, vn, vu, fdm["inertia/weight-lbs"] * LB, terr, agl, s, e, hr,
                         mach, V, n_meas, n_cmd, math.degrees(phi), math.degrees(phi_cmd),
                         fdm["attitude/theta-deg"], fdm["attitude/psi-deg"] + math.degrees(conv),
                         fdm["aero/alpha-deg"],
                         power, F * n_thrust, wf * n_thrust, math.degrees(gam)])
        if not fdm.run():
            status = "simulator stopped"
            break
        step += 1
    a = np.asarray(rows)
    names = ["terrain", "agl", "s", "cross_track", "h_ref", "mach", "airspeed", "nz", "nz_cmd",
             "roll_deg", "roll_cmd_deg", "pitch_deg", "heading_deg", "alpha_deg", "power",
             "thrust", "fuel_flow", "gamma_deg"]
    meta = {nm: a[:, 8 + i] for i, nm in enumerate(names)}
    meta |= {"simulator": "jsbsim", "model": track.name, "status": status,
             "fuel_used": float(a[0, 7] - a[-1, 7]), "min_agl": float(a[:, 9].min()),
             "route_length": track.length, "distance_flown": float(a[-1, 10])}
    return FlightResult(a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7], meta=meta)
