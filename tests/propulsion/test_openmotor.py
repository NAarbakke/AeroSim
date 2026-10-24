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


CUSTOM = Path(motorlib.__file__).parents[1] / "test" / "data" / "regression" / "custom" / "motor.ric"


@pytest.mark.skipif(not CUSTOM.exists(), reason="openMotor sample motors not found")
def test_grain_motor_agrees_with_openmotor_on_custom_outline():
    from aerosim.propulsion.solid import load_ric
    ours = load_ric(CUSTOM).thrust_curve()
    theirs = OpenMotorModel(CUSTOM).thrust_curve()
    assert ours.propellant_mass == pytest.approx(theirs.propellant_mass, rel=0.01)
    on = ours.time[ours.thrust > 0.001 * ours.thrust.max()]   # openMotor stops at 0.1 % of peak
    assert on[-1] - on[0] == pytest.approx(theirs.burn_time, rel=0.03)
    assert -0.02 < ours.total_impulse / theirs.total_impulse - 1 < 0.08
