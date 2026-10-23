import pytest

from aerosim.atmosphere import ProfileAtmosphere, StandardAtmosphere, isa


@pytest.mark.parametrize("z, p, t", [
    (0.0, 101_325.0, 288.15),
    (11_000.0, 22_700.0, 216.77),     # US76 table (geometric altitude)
    (20_000.0, 5_529.0, 216.65),
])
def test_isa_against_us76(z, p, t):
    pp, tt, _ = isa(z)
    assert pp == pytest.approx(p, rel=5e-3)
    assert tt == pytest.approx(t, abs=0.2)


def test_wind_direction_from_west_blows_east():
    s = StandardAtmosphere(wind_speed=10, wind_from_deg=270).at(10.0)
    assert s.wind_u == pytest.approx(10.0)
    assert s.wind_v == pytest.approx(0.0, abs=1e-9)


def test_profile_interpolates_and_extends():
    prof = ProfileAtmosphere([0, 1000], [101_325, 89_875], [288.15, 281.65], [0, 10], [0, 0])
    assert prof.at(500).wind_u == pytest.approx(5.0)
    assert prof.at(5000).pressure < prof.at(1000).pressure
