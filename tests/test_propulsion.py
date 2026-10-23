import numpy as np
import pytest

from aerosim.propulsion.airbreathing import Turbojet
from aerosim.propulsion.liquid import LiquidEngine
from aerosim.propulsion.nozzle import area_ratio, exit_mach
from aerosim.propulsion.solid import BatesMotor


def test_nozzle_roundtrip():
    assert area_ratio(exit_mach(8.0, 1.2), 1.2) == pytest.approx(8.0)


def test_bates_conserves_propellant_and_isp_is_sane():
    m = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0)
    tc = m.thrust_curve()
    assert np.trapezoid(tc.mass_flow, tc.time) == pytest.approx(m.propellant_mass, rel=1e-6)
    assert 100 < tc.average_isp < 180          # KNSB-class motor
    assert tc.thrust[-1] == 0.0


class FakeCEA:
    def chamber(self, *a):
        return {"isp_vac": 280.0, "isp_sl": 240.0, "cstar": 1700.0,
                "t_chamber": 3200.0, "gamma": 1.2, "mw": 22.0}


def test_liquid_engine_with_injected_combustion_model():
    e = LiquidEngine("LOX", "Ethanol", 1.3, 2e6, 4.0, 3e-4, 10.0, combustion=FakeCEA())
    perf = e.performance()
    assert perf["mdot"] == pytest.approx(2e6 * 3e-4 / (1700 * 0.97))
    tc = e.thrust_curve()
    assert tc.vacuum and tc.burn_time == pytest.approx(10.0)


def test_turbojet_trends():
    tj = Turbojet()
    sls = tj.operating_point(0, 0, 1.0)
    alt = tj.operating_point(10_000, 0.8, 1.0)
    idle = tj.operating_point(0, 0, 0.0)
    assert sls["thrust"] > alt["thrust"] > 0          # lapse with altitude
    assert sls["thrust"] > idle["thrust"]
    assert 1.5e-5 < sls["tsfc"] < 4e-5                # kg/(N s): typical turbojet range
    deck = tj.deck([0, 5000], [0, 0.5], [0.5, 1.0])
    assert deck.thrust.shape == (2, 2, 2)
