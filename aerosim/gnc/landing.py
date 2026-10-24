"""Powered-descent landing guidance for a reusable booster by convex optimisation (cvxpy).

    b = Booster(dry_mass=25e3, propellant_mass=4e3, thrust_max=845e3, throttle_min=0.4,
                isp=282, diameter=3.66)
    plan = solve_descent(b, r0=[0, 0, 1500], v0=[30, 0, -180])      # fuel-optimal plan
    flight = simulate_landing(b, r0, v0, replan_every=2.0)          # closed loop, with drag

Guidance: lossless convexification of the fuel-optimal powered-descent problem (Acikmese &
Ploen, J. Guid. Control Dyn. 30(5), 2007; Acikmese, Carson & Blackmore, IEEE TCST 21(6), 2013).
With thrust-acceleration u = T/m, slack sigma >= |u| and z = ln m the problem becomes a
second-order cone programme:

    maximise z(t_f)                          (least propellant)
    r'' = u + g,  z' = -sigma / (Isp g0)
    |u| <= sigma,  rho1 e^-z0 [1 - dz + dz^2/2] <= sigma <= rho2 e^-z0 [1 - dz]   (throttle band)
    u_up >= cos(theta_max) sigma             (thrust within theta_max of vertical)
    altitude >= tan(gamma_gs) |horizontal offset from pad|   (glide slope)
    r(t_f) = pad, v(t_f) = 0

The non-convex lower thrust bound is convexified exactly by sigma (the "lossless" result);
the mass bounds use the Taylor expansion about the max-thrust mass profile z0(t). Time of
flight is chosen by golden-section search on propellant used. Flat, non-rotating Earth with
constant gravity, no aerodynamics inside the optimisation: ``simulate_landing`` flies the
plan with drag in a point-mass model and re-solves from the current state every
``replan_every`` seconds (model-predictive control).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aerosim.atmosphere.standard import isa
from aerosim.core import AerosimError, require
from aerosim.core.units import G0

GRAVITY = np.array([0.0, 0.0, -G0])


@dataclass
class Booster:
    dry_mass: float                  # kg, empty stage at landing
    propellant_mass: float           # kg, available for the landing burn
    thrust_max: float                # N, landing engine(s) at full throttle (sea level)
    throttle_min: float              # fraction of thrust_max the engine cannot go below
    isp: float                       # s, sea level
    diameter: float = 3.66           # m, drag reference
    drag_coefficient: float = 1.0    # body falling tail-first, engines leading
    pointing_limit_deg: float = 15.0
    glide_slope_deg: float = 5.0     # minimum elevation of the approach above the pad

    @property
    def mass(self) -> float:
        return self.dry_mass + self.propellant_mass


@dataclass
class DescentPlan:
    time: np.ndarray
    position: np.ndarray             # (n, 3) m, x east, y north, z up, pad at origin
    velocity: np.ndarray
    mass: np.ndarray
    thrust: np.ndarray               # (n, 3) N
    time_of_flight: float
    propellant_used: float
    status: str
    meta: dict = field(default_factory=dict)

    @property
    def throttle(self) -> np.ndarray:
        return np.linalg.norm(self.thrust, axis=1)


_PROBLEMS: dict = {}


def _problem(n: int, cos_t: float, tan_g: float):
    """The landing SOCP on an n-interval grid, built once; numbers enter as cvxpy Parameters
    (disciplined parametrised programming), so re-solving skips problem compilation.
    Positions are relative to the pad."""
    key = (n, cos_t, tan_g)
    if key in _PROBLEMS:
        return _PROBLEMS[key]
    cp = require("cvxpy", "landing")
    P = {k: cp.Parameter(shape) for k, shape in (("r0", 3), ("v0", 3), ("z_init", ()),
                                                    ("z_dry", ()), ("z0", n + 1))}
    for k in ("dt", "dt2_12", "adt"):
        P[k] = cp.Parameter(nonneg=True)
    P["mu1"], P["mu2"] = cp.Parameter(n + 1, nonneg=True), cp.Parameter(n + 1, nonneg=True)
    r, v, z = cp.Variable((n + 1, 3)), cp.Variable((n + 1, 3)), cp.Variable(n + 1)
    u, sg = cp.Variable((n + 1, 3)), cp.Variable(n + 1)
    g = np.tile(GRAVITY, (n, 1))
    dz = cp.Variable(n + 1)               # z - z0 as its own variable keeps the problem DPP
    cons = [dz == z - P["z0"], r[0] == P["r0"], v[0] == P["v0"], z[0] == P["z_init"], r[n] == 0, v[n] == 0,
            z[n] >= P["z_dry"],
            # trapezoidal velocity, exact position for linearly varying acceleration
            v[1:] == v[:-1] + P["dt"] / 2 * (u[:-1] + u[1:]) + P["dt"] * g,
            r[1:] == r[:-1] + P["dt"] / 2 * (v[:-1] + v[1:]) - P["dt2_12"] * (u[1:] - u[:-1]),
            z[1:] == z[:-1] - P["adt"] * (sg[:-1] + sg[1:]),
            cp.norm(u, axis=1) <= sg, u[:, 2] >= cos_t * sg,
            sg >= cp.multiply(P["mu1"], 1 - dz + cp.square(dz) / 2),
            sg <= cp.multiply(P["mu2"], 1 - dz),
            r[:, 2] >= tan_g * cp.norm(r[:, :2], axis=1)]
    prob = cp.Problem(cp.Maximize(z[n]), cons)
    _PROBLEMS[key] = (prob, P, (r, v, z, u))
    return _PROBLEMS[key]


def _socp(b: Booster, r0, v0, m0, tf, n, target):
    cp = require("cvxpy", "landing")
    cos_t = math.cos(math.radians(b.pointing_limit_deg))
    tan_g = math.tan(math.radians(b.glide_slope_deg))
    prob, P, (r, v, z, u) = _problem(n, cos_t, tan_g)
    dt = tf / n
    alpha = 1.0 / (b.isp * G0)
    rho1, rho2 = b.throttle_min * b.thrust_max, b.thrust_max
    tk = np.arange(n + 1) * dt
    z0 = np.log(np.maximum(m0 - alpha * rho2 * tk, b.dry_mass))
    vals = {"r0": r0 - target, "v0": v0, "z_init": math.log(m0), "z_dry": math.log(b.dry_mass),
            "z0": z0, "dt": dt, "dt2_12": dt * dt / 12, "adt": alpha * dt / 2,
            "mu1": rho1 * np.exp(-z0), "mu2": rho2 * np.exp(-z0)}
    for k, val in vals.items():
        P[k].value = val
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or z.value is None:
        return None
    m = np.exp(z.value)
    return DescentPlan(tk, r.value + target, v.value, m, u.value * m[:, None], tf, m0 - m[-1],
                       prob.status)


def solve_descent(b: Booster, r0, v0, m0: float | None = None, target=(0.0, 0.0, 0.0), n: int = 60,
                  tf_bounds: tuple[float, float] | None = None, tol: float = 0.25) -> DescentPlan:
    """Fuel-optimal landing from (r0, v0); the flight time is searched within ``tf_bounds``."""
    r0, v0, target = (np.asarray(a, float) for a in (r0, v0, target))
    m0 = m0 or b.mass
    if tf_bounds is None:
        a_max = b.thrust_max / m0 - G0                        # net deceleration at full thrust
        t_min = max(np.linalg.norm(v0) / max(b.thrust_max / b.dry_mass, 1e-6), 1.0)
        t_max = b.propellant_mass * b.isp * G0 / (b.throttle_min * b.thrust_max)
        tf_bounds = (t_min, max(t_max, t_min + 1)) if a_max > 0 else (t_min, t_max)
    lo, hi = tf_bounds
    # coarse scan for a feasible bracket, then golden section on propellant used
    grid = np.linspace(lo, hi, 12)
    plans = {t: _socp(b, r0, v0, m0, t, n, target) for t in grid}
    feas = [t for t in grid if plans[t] is not None]
    if not feas:
        raise AerosimError(f"no feasible landing for flight times {lo:.1f}-{hi:.1f} s "
                           "(too fast, too low, or not enough propellant / thrust)")
    best = min(feas, key=lambda t: plans[t].propellant_used)
    i = list(grid).index(best)
    a, c = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    cache = dict(plans)

    def cost(t):
        if t not in cache:
            cache[t] = _socp(b, r0, v0, m0, t, n, target)
        p = cache[t]
        return p.propellant_used if p is not None else float("inf")

    x1, x2 = c - gr * (c - a), a + gr * (c - a)
    while c - a > tol:
        if cost(x1) < cost(x2):
            c, x2 = x2, x1
            x1 = c - gr * (c - a)
        else:
            a, x1 = x1, x2
            x2 = a + gr * (c - a)
    t_best = min((t for t in cache if cache[t] is not None), key=lambda t: cache[t].propellant_used)
    plan = cache[t_best]
    plan.meta["evaluations"] = len(cache)
    return plan


@dataclass
class LandingFlight:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    thrust: np.ndarray
    replans: int
    touchdown_speed: float
    touchdown_miss: float
    propellant_used: float
    plans: list = field(default_factory=list)


def simulate_landing(b: Booster, r0, v0, replan_every: float = 2.0, dt: float = 0.05,
                     n: int = 40, target=(0.0, 0.0, 0.0), final_replan_altitude: float = 5.0,
                     wind: tuple[float, float] = (0.0, 0.0)) -> LandingFlight:
    """Fly the guidance in a point-mass model with drag (and optional constant wind, m/s east,
    north). The plan is re-solved from the current state every ``replan_every`` s, more often
    as time-to-go shrinks (at most 30 % of it), down to ``final_replan_altitude``. The engine
    shuts down when the plan is complete or the booster stops descending near the pad, as a
    real landing burn does at touchdown. Stops at touchdown (z <= target z)."""
    r, v = np.asarray(r0, float).copy(), np.asarray(v0, float).copy()
    m = b.mass
    target = np.asarray(target, float)
    area = math.pi / 4 * b.diameter**2
    t, rows, plans = 0.0, [], []
    plan, t_plan = None, -1e9
    w = np.array([wind[0], wind[1], 0.0])
    while True:
        tau = t - t_plan
        to_go = plan.time_of_flight - tau if plan is not None else 0.0
        due = plan is None or tau >= min(replan_every, max(0.25, 0.3 * to_go)) - 1e-9
        if r[2] - target[2] > final_replan_altitude and due and to_go >= 0.0:
            try:
                bounds = None if plan is None else (max(to_go - 6, 0.5), max(to_go + 6, 2.0))
                plan, t_plan = solve_descent(b, r, v, m0=m, target=target, n=n, tf_bounds=bounds), t
                plans.append(plan)
            except AerosimError:
                if plan is None:
                    raise                         # keep flying the previous plan otherwise
        tau = t - t_plan
        T = np.array([plan.thrust[:, i] for i in range(3)])
        F = np.array([np.interp(tau, plan.time, T[i]) for i in range(3)])
        landed = tau >= plan.time_of_flight or (v[2] >= 0.0 and r[2] - target[2] < 10.0)
        if landed or m <= b.dry_mass + 1e-6:
            F = np.zeros(3)                       # plan complete / hovering at the pad: shut down
        mag = np.linalg.norm(F)
        _p, _T, rho = isa(max(r[2], 0.0))
        vrel = v - w
        D = -0.5 * rho * b.drag_coefficient * area * np.linalg.norm(vrel) * vrel
        a = (F + D) / m + GRAVITY
        rows.append((t, *r, *v, m, *F))
        r = r + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        m = max(m - mag / (b.isp * G0) * dt, b.dry_mass)
        t += dt
        if r[2] <= target[2] or t > 600:
            rows.append((t, *r, *v, m, 0, 0, 0))
            break
    A = np.asarray(rows)
    return LandingFlight(A[:, 0], A[:, 1:4], A[:, 4:7], A[:, 7], A[:, 8:11], len(plans),
                         float(np.linalg.norm(A[-1, 4:7])),
                         float(np.linalg.norm(A[-1, 1:3] - target[:2])), b.mass - A[-1, 7], plans)
