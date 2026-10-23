"""Cantera gas model, turbojet with real-gas properties, and CEA cross-check."""

import pytest

pytest.importorskip("cantera")

from aerosim.propulsion.airbreathing import Turbojet
from aerosim.thermo.gas import AirFuelGas
from aerosim.thermo.gas.kinetics import chamber_equilibrium, ignition_delay


def test_air_properties_vs_tables():
    g = AirFuelGas()
    assert g.cp(300.0) == pytest.approx(1005.0, rel=3e-3)     # air tables: 1005 J/kg K
    assert g.cp(1500.0) == pytest.approx(1211.0, rel=5e-3)    # 1211 J/kg K
    assert g.gamma(300.0) == pytest.approx(1.400, abs=2e-3)


def test_jet_a_stoichiometry_and_burner():
    g = AirFuelGas()
    assert g.far_stoich == pytest.approx(0.068, rel=0.01)
    f = g.far_for_T(1500.0, g.state(600.0, 12e5)[0], 12e5)
    assert 0.024 < f < 0.028
    # products are hotter-cp than air: gamma drops with fuel
    assert g.gamma(1500, 12e5, f) < g.gamma(1500, 12e5, 0.0)


def test_turbojet_cantera_close_to_constant_cp():
    const = Turbojet(gas_model="constant").operating_point(0, 0, 1.0)
    real = Turbojet(gas_model="cantera").operating_point(0, 0, 1.0)
    assert real["thrust"] == pytest.approx(const["thrust"], rel=0.05)
    assert real["fuel_air_ratio"] == pytest.approx(const["fuel_air_ratio"], rel=0.10)
    assert real["Tt4"] == 1500.0


def test_turbojet_cantera_deck_lapse():
    tj = Turbojet(gas_model="cantera")
    deck = tj.deck([0, 10_000], [0.0, 0.8], [1.0])
    assert deck.thrust[1, 1, 0] < deck.thrust[0, 0, 0]


@pytest.mark.parametrize("fuel, mr, pc", [("CH4", 3.2, 10e5), ("CH4", 2.5, 50e5), ("H2", 6.0, 20e5)])
def test_cantera_and_cea_agree_on_chamber(fuel, mr, pc):
    cea = pytest.importorskip("cea")  # noqa: F841
    from aerosim.thermo import NasaCEAModel
    c = chamber_equilibrium(fuel, "O2", mr, pc)
    s = NasaCEAModel().solve("O2", fuel, mr, pc, 10.0)
    assert c["T"] == pytest.approx(s["T"][0], rel=3e-3)
    assert c["MW"] == pytest.approx(s["MW"][0], rel=3e-3)


def test_ignition_delay_is_finite_rate():
    hot, cool = ignition_delay(T=1300.0), ignition_delay(T=1100.0)
    assert 0 < hot < cool                       # Arrhenius: hotter ignites faster
