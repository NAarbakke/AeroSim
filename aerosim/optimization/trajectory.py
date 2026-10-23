"""Optimal-control trajectories with Dymos (pseudospectral collocation on OpenMDAO).

Dymos answers "what is the best *command history*?" - e.g. the throttle profile
that maximises apogee - in contrast to gnc/, which makes the vehicle *follow* a
command in the presence of disturbances. Typical pipeline: Dymos designs the
throttle/pitch schedule offline, the flight sim + gnc controller fly it.

Implemented: vertical ascent (Goddard problem) with drag, throttle limits and an
optional max-dynamic-pressure path constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerosim.core import require

G0 = 9.80665


def _cs_abs(x):
    """|x| that keeps the complex-step perturbation (np.abs would discard it)."""
    return np.where(np.real(x) < 0, -x, x)


def _density(h):
    """Complex-step-safe ISA density (troposphere law, isothermal stratosphere)."""
    h = np.asarray(h)
    low = np.real(h) < 11_000.0
    h_trop = np.where(low, h, 0.0)                # keep the power law in its valid range
    trop = 1.225 * (1 - 2.25577e-5 * h_trop) ** 4.25588
    strat = 0.36392 * np.exp(-(h - 11_000.0) / 6341.6)
    return np.where(low, trop, strat)


def _ode_class():
    om = require("openmdao.api", "optim")

    class VerticalAscentODE(om.ExplicitComponent):
        def initialize(self):
            self.options.declare("num_nodes", types=int)
            self.options.declare("thrust_max", types=float)
            self.options.declare("isp", types=float)
            self.options.declare("cd_area", types=float)

        def setup(self):
            nn = self.options["num_nodes"]
            for name, units in (("h", "m"), ("v", "m/s"), ("m", "kg"), ("throttle", None)):
                self.add_input(name, np.ones(nn), units=units)
            self.add_output("h_dot", np.zeros(nn), units="m/s")
            self.add_output("v_dot", np.zeros(nn), units="m/s**2")
            self.add_output("m_dot", np.zeros(nn), units="kg/s")
            self.add_output("q", np.zeros(nn), units="Pa")
            self.add_output("thrust", np.zeros(nn), units="N")
            ar = np.arange(nn)
            self.declare_partials("*", "*", rows=ar, cols=ar, method="cs")
            self.declare_coloring(wrt="*", method="cs")

        def compute(self, i, o):
            T = self.options["thrust_max"] * i["throttle"]
            rho = _density(i["h"])
            v = i["v"]
            drag = 0.5 * rho * v * _cs_abs(v) * self.options["cd_area"]
            o["h_dot"] = v
            o["v_dot"] = (T - drag) / i["m"] - G0
            o["m_dot"] = -T / (self.options["isp"] * G0)
            o["q"] = 0.5 * rho * v**2
            o["thrust"] = T

    return VerticalAscentODE


@dataclass
class AscentSolution:
    time: np.ndarray
    altitude: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    throttle: np.ndarray
    dynamic_pressure: np.ndarray
    apogee: float
    success: bool


def optimal_vertical_ascent(m0: float, propellant_mass: float, thrust_max: float, isp: float,
                            cd_area: float, max_q: float | None = None, min_throttle: float = 0.0,
                            num_segments: int = 15, order: int = 3, throttle_order: int | None = 6, tol: float = 1e-6,
                            maxiter: int = 500, initial_throttle: float | None = None,
                            optimizer: str = "SLSQP") -> AscentSolution:
    """Throttle history u(t) in [min_throttle, 1] that maximises apogee (Goddard problem).

    Two linked Radau phases: a powered phase that ends when the propellant is
    gone (m = m_dry, free duration, throttle is the control), then an unpowered
    coast to apogee (v = 0). The optimiser shapes the burn - typically full
    thrust, then a reduced-thrust "singular arc" that avoids wasting propellant
    on drag at high speed. ``min_throttle=1`` reproduces a fixed full-thrust burn.

    ``throttle_order``: the throttle is a single polynomial of this order over the
    burn (few design variables, smooth, robust with SLSQP, flyable by a real
    engine). ``None`` = a free value at every collocation node (sharper arcs;
    best with IPOPT via pyoptsparse).
    """
    om = require("openmdao.api", "optim")
    dm = require("dymos", "optim")
    ODE = _ode_class()
    m_dry = m0 - propellant_mass
    t_full = propellant_mass * isp * G0 / thrust_max          # full-throttle burn time
    # initial guess: a part-throttle burn at ~2x liftoff weight (or min_throttle), which
    # sits much closer to the drag-optimal solution than a full-thrust guess
    u0 = float(np.clip(initial_throttle if initial_throttle is not None else 2.0 * m0 * G0 / thrust_max,
                       max(min_throttle, 0.05), 1.0))
    t_burn = t_full / u0
    v_bo = max(isp * G0 * np.log(m0 / m_dry) - G0 * t_burn, 20.0) * 0.6
    h_bo = 0.5 * v_bo * t_burn
    kw = {"thrust_max": float(thrust_max), "isp": float(isp), "cd_area": float(cd_area)}

    import tempfile
    # keep OpenMDAO reports/coloring files out of the caller's working directory
    p = om.Problem(reports=False, work_dir=tempfile.mkdtemp(prefix="aerosim_dymos_"))
    traj = p.model.add_subsystem("traj", dm.Trajectory())

    burn = dm.Phase(ode_class=ODE, transcription=dm.Radau(num_segments=num_segments, order=order),
                    ode_init_kwargs=kw)
    traj.add_phase("burn", burn)
    burn.set_time_options(fix_initial=True, duration_bounds=(0.9 * t_full, 30 * t_full),
                          duration_ref=t_burn, units="s")
    burn.add_state("h", rate_source="h_dot", fix_initial=True, units="m", ref=max(h_bo, 1.0))
    burn.add_state("v", rate_source="v_dot", fix_initial=True, units="m/s", ref=v_bo)
    burn.add_state("m", rate_source="m_dot", fix_initial=True, fix_final=True, units="kg", ref=m0)
    lo = max(min_throttle, 1e-3)
    if throttle_order is not None:
        burn.add_control("throttle", control_type="polynomial", order=throttle_order,
                         lower=lo, upper=1.0)
    else:
        burn.add_control("throttle", lower=lo, upper=1.0, continuity=True, rate_continuity=False)
    if max_q is not None:
        burn.add_path_constraint("q", upper=max_q, ref=max_q, units="Pa")
    burn.add_timeseries_output("q")

    coast = dm.Phase(ode_class=ODE, transcription=dm.Radau(num_segments=8, order=3),
                     ode_init_kwargs=kw)
    traj.add_phase("coast", coast)
    t_coast = v_bo / G0
    coast.set_time_options(initial_bounds=(0.9 * t_full, 30 * t_full), duration_bounds=(0.1, 10 * t_coast),
                           duration_ref=t_coast, units="s")
    coast.add_state("h", rate_source="h_dot", units="m", ref=max(h_bo, 1.0))
    coast.add_state("v", rate_source="v_dot", units="m/s", ref=v_bo)
    coast.add_state("m", rate_source="m_dot", units="kg", ref=m0)
    coast.add_parameter("throttle", val=0.0, opt=False, static_target=False)
    if max_q is not None:
        coast.add_path_constraint("q", upper=max_q, ref=max_q, units="Pa")
    coast.add_boundary_constraint("v", loc="final", equals=0.0, units="m/s")
    coast.add_timeseries_output("q")
    coast.add_objective("h", loc="final", ref=-max(h_bo + v_bo**2 / (2 * G0), 1.0))

    traj.link_phases(["burn", "coast"], vars=["time", "h", "v", "m"])

    p.driver = om.ScipyOptimizeDriver(optimizer=optimizer, maxiter=maxiter, tol=tol, disp=False)
    p.driver.declare_coloring()
    p.setup(force_alloc_complex=True)

    burn.set_time_val(initial=0.0, duration=t_burn)
    burn.set_state_val("h", [0.0, h_bo])
    burn.set_state_val("v", [0.0, v_bo])
    burn.set_state_val("m", [m0, m_dry])
    burn.set_control_val("throttle", [u0, u0])
    coast.set_time_val(initial=t_burn, duration=0.5 * t_coast)
    coast.set_state_val("h", [h_bo, h_bo + 0.3 * v_bo * t_coast])
    coast.set_state_val("v", [v_bo, 0.0])
    coast.set_state_val("m", [m_dry, m_dry])

    failed = dm.run_problem(p, run_driver=True, simulate=False)

    def series(phase, key):
        return np.asarray(p.get_val(f"traj.{phase}.timeseries.{key}")).ravel()

    cat = lambda k: np.concatenate([series("burn", k), series("coast", k)])
    thr = np.concatenate([series("burn", "throttle"), np.zeros_like(series("coast", "time"))])
    h = cat("h")
    return AscentSolution(cat("time"), h, cat("v"), cat("m"), thr, cat("q"),
                          float(h[-1]), success=not failed)


def simulate_vertical_ascent(m0, propellant_mass, thrust_max, isp, cd_area, throttle=lambda t: 1.0):
    """Reference forward simulation (scipy) of the same model for a given throttle law."""
    from scipy.integrate import solve_ivp

    m_dry = m0 - propellant_mass

    def rhs(t, s):
        h, v, m = s
        u = throttle(t) if m > m_dry else 0.0
        T = thrust_max * u
        rho = float(np.real(_density(h)))
        return [v, (T - 0.5 * rho * v * abs(v) * cd_area) / m - G0, -T / (isp * G0)]

    apo = lambda t, s: s[1]
    apo.terminal, apo.direction = True, -1
    sol = solve_ivp(rhs, (0, 2000), [0.0, 1e-6, m0], events=apo, max_step=0.05, rtol=1e-8)
    return float(sol.y[0, -1]), sol


def optimal_ascent_for(engine, vehicle, **options) -> AscentSolution:
    """Goddard problem for an aerosim LiquidEngine + RocketVehicle.

    Uses the engine's delivered vacuum Isp and full-throttle thrust (from NASA CEA),
    its propellant load (from ``thrust_curve()``) and the vehicle's Cd*A.
    """
    perf = engine.performance()
    tc = engine.thrust_curve()
    m0 = vehicle.dry_mass + tc.dry_mass + tc.propellant_mass
    return optimal_vertical_ascent(m0=m0, propellant_mass=tc.propellant_mass,
                                   thrust_max=perf["thrust_vac"], isp=perf["isp_vac_delivered"],
                                   cd_area=vehicle.drag_coefficient * vehicle.reference_area,
                                   **options)
