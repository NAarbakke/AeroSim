"""Powered-descent landing guidance (convex) and its closed-loop flight with drag."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from aerosim.gnc.landing import Booster, simulate_landing, solve_descent

F9_LIKE = Booster(dry_mass=25e3, propellant_mass=4e3, thrust_max=845e3, throttle_min=0.4,
                  isp=282, diameter=3.66)


def test_fuel_optimal_plan_is_min_max():
    p = solve_descent(F9_LIKE, [200, 0, 1800], [-15, 0, -170])
    assert np.allclose(p.position[-1], 0, atol=1e-3) and np.allclose(p.velocity[-1], 0, atol=1e-3)
    thr = p.throttle / F9_LIKE.thrust_max
    assert thr.min() == pytest.approx(0.4, abs=0.02) and thr.max() == pytest.approx(1.0, abs=0.02)
    assert thr[0] < 0.5 and thr[-1] > 0.9                        # coast-like low thrust, then full
    assert p.propellant_used < F9_LIKE.propellant_mass
    # thrust stays within the pointing cone
    up = p.thrust[:, 2] / np.linalg.norm(p.thrust, axis=1)
    assert up.min() >= np.cos(np.radians(15)) - 1e-3


@pytest.mark.parametrize("wind", [(0.0, 0.0), (8.0, 3.0)])
def test_closed_loop_landing_with_drag(wind):
    f = simulate_landing(F9_LIKE, [200, 0, 1800], [-15, 0, -170], wind=wind)
    assert f.touchdown_speed < 3.0
    assert f.touchdown_miss < 1.0
    assert f.propellant_used < F9_LIKE.propellant_mass
    assert f.replans > 5
