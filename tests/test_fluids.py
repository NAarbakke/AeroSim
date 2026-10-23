"""CoolProp-backed fluids and tanks. Reference values from NIST REFPROP/Webbook."""

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from aerosim.propulsion.liquid.tanks import PropellantTank, SelfPressurisingTank
from aerosim.thermo.fluids import Fluid, blend_density


@pytest.mark.parametrize("name, P, rho, T", [
    ("LOX", 101_325.0, 1141.2, 90.19),     # NIST: 1141.2 kg/m3 at NBP 90.19 K
    ("LCH4", 101_325.0, 422.4, 111.67),    # NIST: 422.4 kg/m3 at NBP 111.67 K
    ("LH2", 101_325.0, 70.85, 20.27),      # para-H2: 70.85 kg/m3 at 20.27 K
])
def test_cryogen_saturation(name, P, rho, T):
    s = Fluid(name).saturation(P=P)
    assert s.rho_liquid == pytest.approx(rho, rel=2e-3)
    assert s.temperature == pytest.approx(T, abs=0.05)


def test_n2o_vapour_pressure():
    assert Fluid("N2O").saturation(T=293.15).pressure == pytest.approx(50.5e5, rel=5e-3)


def test_tank_mass_and_sizing_roundtrip():
    tank = PropellantTank("LOX", volume=0.02, pressure=3e5)
    assert tank.temperature == pytest.approx(Fluid("LOX").saturation(P=3e5).temperature)
    v = PropellantTank.volume_for("LOX", tank.propellant_mass, 3e5)
    assert v == pytest.approx(0.02, rel=1e-9)


def test_storable_is_subcooled_at_room_temperature():
    assert PropellantTank("Ethanol", 0.01, 30e5).temperature == 293.15


def test_blend_between_pure_densities():
    rho = blend_density({"Ethanol": 0.75, "Water": 0.25}, 293.15, 30e5)
    assert Fluid("Ethanol").density(293.15, 30e5) < rho < Fluid("Water").density(293.15, 30e5)


def test_rp1_is_flagged_surrogate():
    assert Fluid("RP1").surrogate


def test_n2o_blowdown_physics():
    r = SelfPressurisingTank(fluid="N2O", volume=0.005, mass=3.0, temperature=293.15).run()
    assert r.pressure[0] == pytest.approx(50.5e5, rel=5e-3)           # starts at vapour pressure
    # monotonic blowdown; allow one tiny step artefact where the liquid runs out
    assert np.all(np.diff(r.pressure) <= 1e-3 * r.pressure[:-1])
    assert r.temperature[-1] < r.temperature[0] - 20                   # evaporative cooling
    assert r.liquid_depletion is not None                              # liquid runs out first
    expelled = r.mass[0] - r.mass[-1]
    assert np.trapezoid(r.mass_flow, r.time) == pytest.approx(expelled, rel=0.02)
