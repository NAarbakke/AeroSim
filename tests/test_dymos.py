"""Dymos optimal ascent (Goddard problem), checked against an independent scipy replay."""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("dymos")

from aerosim.optimization.trajectory import (
    _density,
    _ode_class,
    optimal_vertical_ascent,
    simulate_vertical_ascent,
)

ARGS = {"m0": 20.0, "propellant_mass": 8.0, "thrust_max": 2500.0, "isp": 230.0, "cd_area": 0.5 * 0.018}


def _solve(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return optimal_vertical_ascent(**ARGS, **kw)


def _replay(sol):
    burn = sol.throttle > 0
    tb, ub = sol.time[burn], sol.throttle[burn]
    return simulate_vertical_ascent(**ARGS, throttle=lambda t: float(np.interp(t, tb, ub, right=0.0)))[0]


@pytest.fixture(scope="module")
def optimum():
    return _solve()


def test_density_matches_isa():
    from aerosim.atmosphere import isa
    for h in (0.0, 5_000.0, 15_000.0):
        assert float(np.real(_density(h))) == pytest.approx(isa(h)[2], rel=0.01)


def test_ode_partials_are_exact(tmp_path):
    """Complex-step partials must match finite differences (regression: np.abs broke them)."""
    import openmdao.api as om
    p = om.Problem(reports=False, work_dir=str(tmp_path))
    p.model.add_subsystem("ode", _ode_class()(num_nodes=4, thrust_max=2500.0, isp=230.0, cd_area=0.01))
    p.setup(force_alloc_complex=True)
    p.set_val("ode.h", [0, 3000, 9000, 20000])
    p.set_val("ode.v", [5, 200, -50, 300])
    p.set_val("ode.m", 15.0)
    p.set_val("ode.throttle", 0.5)
    data = p.check_partials(method="fd", out_stream=None)
    for (of, wrt), d in data["ode"].items():
        assert d["abs error"].forward < 1e-3 * max(1.0, np.abs(d["J_fwd"]).max()), (of, wrt)


def test_optimum_beats_full_throttle_and_replays(optimum):
    full, _ = simulate_vertical_ascent(**ARGS)
    assert optimum.success
    assert optimum.apogee > 1.4 * full                         # drag-limited: throttling wins big
    assert _replay(optimum) == pytest.approx(optimum.apogee, rel=5e-3)
    assert optimum.mass[-1] == pytest.approx(ARGS["m0"] - ARGS["propellant_mass"], rel=1e-6)


def test_goddard_structure(optimum):
    burn = optimum.throttle[optimum.throttle > 0]
    assert burn[0] > 0.9                                       # starts near full thrust
    assert np.median(burn) < 0.3                               # then a low-throttle singular arc


def test_independent_of_initial_guess(optimum):
    assert _solve(initial_throttle=1.0).apogee == pytest.approx(optimum.apogee, rel=1e-3)


def test_max_q_constraint_binds(optimum):
    capped = _solve(max_q=15e3)
    assert capped.success
    assert capped.dynamic_pressure.max() <= 15e3 * 1.001
    assert capped.apogee < optimum.apogee                      # a binding constraint costs altitude
