"""openMotor adapter. Skipped unless openMotor is on the path (GPL-3.0, user-installed)."""

from pathlib import Path

import pytest

motorlib = pytest.importorskip("motorlib.motor")

from aerosim.propulsion.adapters.openmotor import OpenMotorModel

RIC = Path(motorlib.__file__).parents[1] / "test" / "data" / "real" / "n2950" / "motor.ric"


@pytest.mark.skipif(not RIC.exists(), reason="openMotor sample motors not found")
def test_n2950_matches_its_designation():
    c = OpenMotorModel(RIC, dry_mass=5.0).thrust_curve()
    assert 10_240 < c.total_impulse < 20_480                    # N class
    assert c.total_impulse / c.burn_time == pytest.approx(2950, rel=0.03)
    assert c.dry_mass == 5.0 and c.propellant_mass > 0
