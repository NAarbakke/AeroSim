"""Hybrid motor: conservation, O/F shift, and flying it as a sounding rocket."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cea")
pytest.importorskip("CoolProp")

from aerosim.missions.runner import run_mission
from aerosim.propulsion.hybrid import HybridMotor

CFG = Path(__file__).parents[2] / "configs" / "hybrid" / "n2o_htpb_sounding.yaml"
N2O_HTPB = {"fuel": "HTPB", "oxidizer": "N2O", "grain_length": 1.0, "port_radius": 0.015,
            "outer_radius": 0.045, "regression_a": 5.3e-5, "regression_n": 0.5,
            "throat_area": 5.3e-4}


def test_self_pressurising_n2o_htpb():
    m = HybridMotor(**N2O_HTPB, tank={"volume": 0.012, "mass": 8.0, "temperature": 293.15},
                    injector_area=2.8e-5)
    h = m.simulate()
    # fuel burned equals the volume the port opened up
    fuel_geom = 920 * np.pi * (h.port_radius[-1] ** 2 - 0.015**2) * 1.0
    assert h.fuel_used == pytest.approx(fuel_geom, rel=2e-3)
    assert 7.0 < h.oxidizer_used <= 8.0
    # the port opens, flux falls and O/F shifts fuel-rich; tank blows down
    assert h.of_ratio[0] > h.of_ratio[len(h.time) // 2] > h.of_ratio[-1]
    assert h.tank_pressure[0] > h.tank_pressure[-1]
    assert h.chamber_pressure.max() < h.tank_pressure[0]
    assert 200 < m.average_isp < 280


def test_regulated_gox_paraffin():
    m = HybridMotor(fuel="Paraffin", oxidizer="GOX", grain_length=0.4, port_radius=0.01,
                    outer_radius=0.03, regression_a=1.5e-4, regression_n=0.6, throat_area=2e-4,
                    oxidizer_mass_flow=0.3, oxidizer_mass=3.0)
    h = m.simulate()
    assert np.allclose(h.mdot_ox, 0.3)
    assert h.end_reason in ("web burned through", "oxidizer exhausted")


def test_hybrid_sounding_rocket_config():
    out = run_mission(CFG)
    assert out.propulsion.total_impulse == pytest.approx(21.8e3, rel=0.05)
    assert 5e3 < out.flight.apogee < 15e3
