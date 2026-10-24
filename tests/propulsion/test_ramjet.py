"""Ramjet and scramjet cycles with Cantera equilibrium chemistry."""

import pytest

pytest.importorskip("cantera")

from aerosim.propulsion.airbreathing import Ramjet


def test_lossless_ramjet_without_fuel_makes_no_thrust():
    r = Ramjet(inlet_recovery=1.0, burner_pressure_ratio=1.0, nozzle_efficiency=1.0)
    assert r.operating_point(5000, 0.9, 0.0)["thrust"] == pytest.approx(0.0, abs=1e-3)   # N, of ~6 kN ram drag


def test_kerosene_ramjet():
    r = Ramjet(fuel="JetA", capture_area=0.05)
    pts = [r.operating_point(h, m, 1.0) for h, m in ((15e3, 2.0), (18e3, 3.0), (20e3, 4.0))]
    for o in pts:
        assert 1200 < o["isp"] < 2000                        # textbook hydrocarbon ramjet band
        assert 2000 < o["t4"] < 2700                         # dissociation caps T4 below ~2700 K
    part = r.operating_point(18e3, 3.0, 0.5)
    assert part["isp"] > pts[1]["isp"] and part["thrust"] < pts[1]["thrust"]


def test_hydrogen_scramjet():
    s = Ramjet(mode="scramjet", fuel="H2", capture_area=0.05)
    isp = [s.operating_point(h, m, 1.0)["isp"] for h, m in ((25e3, 6), (30e3, 8), (36e3, 12))]
    assert isp[0] > isp[1] > isp[2] > 1000
    deck = s.deck([25e3, 30e3], [6, 8], [0.5, 1.0])
    assert deck.thrust.shape == (2, 2, 2) and (deck.fuel_flow > 0).all()
