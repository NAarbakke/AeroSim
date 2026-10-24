"""Every plot renders (Agg backend) and returns a Figure."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from aerosim import plots
from aerosim.atmosphere import StandardAtmosphere
from aerosim.flight import RocketVehicle
from aerosim.flight.rocket import PointMassRocketSim
from aerosim.propulsion.solid import BatesMotor, GrainMotor
from aerosim.propulsion.solid.grains import Bates


def test_solid_and_flight_plots(tmp_path):
    curve = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0, name="4-grain").thrust_curve()
    gm = GrainMotor(grains=[{"type": "bates", "diameter": 0.05, "length": 0.09,
                             "core_diameter": 0.02}],
                    nozzle={"throat_diameter": 0.008, "exit_diameter": 0.02})
    flight = PointMassRocketSim().run(RocketVehicle(dry_mass=3, diameter=0.066), curve,
                                      StandardAtmosphere())
    figs = [plots.thrust_curve(curve), plots.thrust_curve([curve, gm.thrust_curve()]),
            plots.motor_history(gm.simulate()), plots.grain_burnback(Bates(0.05, 0.09,
                                                                           core_diameter=0.02)),
            plots.flight_profile(flight)]
    for i, f in enumerate(figs):
        f.savefig(tmp_path / f"{i}.png")
        assert (tmp_path / f"{i}.png").stat().st_size > 10_000


def test_series_order_is_fixed():
    assert plots.series(0) != plots.series(1)
    with pytest.raises(ValueError):
        plots.series(8)
    with plots.themed("dark") as t:
        assert t["surface"] == "#1a1a19"
        assert plots.series(0) == "#3987e5"
    assert np.all([plots.tokens()["surface"] == "#fcfcfb"])


def test_ascent_replay_page(tmp_path):
    import json
    import re

    from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
    from aerosim.flight.vehicles import LaunchVehicle, Stage
    from aerosim.plots.replay import write_replay

    veh = LaunchVehicle(stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
                        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6,
                         transfer_apogee=300e3, coast_after=100)
    res = OrbitalAscentSim(45.0, 63.0, program=prog).run(veh)
    out = tmp_path / "replay.html"
    write_replay({"A": res, "B <test>": res}, out, site=(45.0, 63.0), title="T & t")
    page = out.read_text(encoding="utf-8")
    assert "<title>T &amp; t</title>" in page
    imap = json.loads(re.search(r'type="importmap">(.*?)</script>', page).group(1))["imports"]
    assert all(u.startswith("data:text/javascript;base64,") for u in imap.values())  # offline
    assert "http" not in page.split('id="data"')[0]
    data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    assert [m["name"] for m in data["models"]] == ["A", "B <test>"]
    m = data["models"][0]
    assert len(m["t"]) == len(m["lat"]) == len(m["alt"]) and m["events"]
    assert m["color"] != data["models"][1]["color"]
    assert "__" not in re.sub(r'id="data">.*?</script>', "", page, flags=re.DOTALL)  # all filled
