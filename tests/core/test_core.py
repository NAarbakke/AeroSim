import numpy as np
import pytest

from aerosim.core import EngineDeck, OptionalDependencyError, ThrustCurve, registry, require


def test_thrust_curve_integrals():
    t = np.array([0.0, 1.0, 2.0])
    tc = ThrustCurve(t, np.array([100.0, 100.0, 0.0]), np.array([0.1, 0.1, 0.0]), 0.15)
    assert tc.total_impulse == pytest.approx(150.0)
    assert tc.burn_time == 2.0
    assert tc.thrust_at(5.0) == 0.0


def test_thrust_curve_validation():
    with pytest.raises(ValueError):
        ThrustCurve(np.array([0.0, 0.0]), np.zeros(2), np.zeros(2), 1.0)


def test_engine_deck_lookup():
    a, m, p = np.array([0.0, 1000.0]), np.array([0.0, 0.5]), np.array([0.0, 1.0])
    F = np.ones((2, 2, 2)) * 1000.0
    deck = EngineDeck(a, m, p, F, F * 1e-3)
    assert deck.lookup(500, 0.2, 0.5) == pytest.approx((1000.0, 1.0))


def test_registry_unknown_backend():
    with pytest.raises(KeyError, match="available"):
        registry.create("flight", "does_not_exist")


def test_require_message():
    with pytest.raises(OptionalDependencyError, match=r"\.\[x\]"):
        require("definitely_not_a_module_xyz", "x")
