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
    models = data["vehicles"][0]["models"]
    assert [m["name"] for m in models] == ["A", "B <test>"]
    m = models[0]
    assert len(m["t"]) == len(m["lat"]) == len(m["alt"]) and m["events"]
    assert m["color"] != models[1]["color"]
    assert "__" not in re.sub(r'id="data">.*?</script>', "", page, flags=re.DOTALL)  # all filled

    from aerosim.plots.replay import ascent_html  # several vehicles behind a selector
    page = ascent_html([{"name": "One", "site": (45.0, 63.0), "results": {"A": res}},
                        {"name": "Two & co", "site": (50.0, 60.0), "results": {"A": res, "B": res},
                         "guide": [("Burnout", 120.0, 60e3, 3000.0), ("Apogee", 900.0, None, 7e3)]}])
    data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    assert [v["key"] for v in data["vehicles"]] == ["one", "two---co"]
    assert data["vehicles"][1]["guide"] == [["Burnout", 120.0, 60.0, 3.0], ["Apogee", 900.0, None, 7.0]]
    assert "<title>Rocket ascent</title>" in page and 'id="veh"' in page


def test_us76_kinetic_temperature_matches_table():
    from aerosim.atmosphere.standard import kinetic_temperature
    # US Standard Atmosphere 1976, Table I (geometric altitude)
    for z_km, T in [(0, 288.15), (11, 216.774), (50, 270.65), (100, 195.08), (150, 634.39),
                    (200, 854.56), (300, 976.01), (500, 999.24)]:
        assert kinetic_temperature(z_km * 1e3) == pytest.approx(T, abs=0.6)


def test_ascent_closeup_page():
    import json
    import re

    from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
    from aerosim.flight.vehicles import LaunchVehicle, Stage
    from aerosim.plots.ascent_closeup import ascent_closeup_html, ascent_physics

    veh = LaunchVehicle(stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
                        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6,
                         transfer_apogee=300e3, coast_after=100)
    res = OrbitalAscentSim(45.0, 63.0, program=prog).run(veh)
    d = ascent_physics(res, veh, t_end=400)
    assert d["thrust"][0] < d["thrust_vac"][0]                       # back pressure at sea level
    assert d["thrust"][d["h"] > 80e3][0] == pytest.approx(d["thrust_vac"][d["h"] > 80e3][0], rel=1e-3)
    k = int(np.argmax(d["q"]))
    assert d["drag"][k] > 0 and 5e3 < d["h"][k] < 20e3               # max-Q low in the atmosphere
    for v in d["losses"].values():
        assert v[0] == 0 and np.all(np.diff(v) >= -1e-9)
    page = ascent_closeup_html(res, veh, title="Close <up>", t_end=400)
    assert "<title>Close &lt;up&gt;</title>" in page
    imap = json.loads(re.search(r'type="importmap">(.*?)</script>', page).group(1))["imports"]
    assert all(u.startswith("data:text/javascript;base64,") for u in imap.values())  # offline
    data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    names = [e[0] for e in data["events"]]
    assert {"Liftoff", "Mach 1", "Max-Q", "Into the stratosphere", "Kármán line"} <= set(names)
    assert len(data["f"]["t"]) == len(data["f"]["loss_drag"]) and data["layers"][0]["name"] == "Troposphere"


def test_ascent_dashboard_and_report(tmp_path):
    import json
    import re

    from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
    from aerosim.flight.vehicles import LaunchVehicle, Stage
    from aerosim.plots.ascent_dashboard import ascent_dashboard_html
    from aerosim.plots.ascent_data import ascent_dataset
    from aerosim.plots.ascent_report import report_figures, report_tex, tex

    veh = LaunchVehicle(stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
                        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6, transfer_apogee=300e3)
    res = OrbitalAscentSim(45.0, 63.0, program=prog).run(veh)
    data = ascent_dataset({"Model A": (res, veh), "Model B": (res, veh)}, (45.0, 63.0), target="300 km")
    m = data["models"][0]
    s = m["summary"]
    assert s["t_mach1"] < s["t_max_q"] < s["t_insertion"] and s["max_q"] > 0
    assert s["dv_ideal"] > 7000 and all(v > 0 for v in s["losses"].values())
    assert m["enu"]["e"][0] == pytest.approx(0, abs=1e-6) and m["enu"]["n"][0] == pytest.approx(0, abs=1e-6)
    rng = np.hypot(m["enu"]["e"][-1], m["enu"]["n"][-1])           # azimuthal equidistant: true ground range
    assert rng == pytest.approx(np.interp(m["enu"]["t"][-1], m["series"]["t"], m["series"]["dr"]), rel=0.02)
    assert [o["key"] for o in m["orbits"]][-1] == "final" and m["events"][0]["label"] == "Lift-off"

    page = ascent_dashboard_html({**data, "title": "Ascent <A>"}, report_href="r.pdf")
    assert "<title>Ascent &lt;A&gt;</title>" in page and 'href="r.pdf"' in page
    assert "http" not in page.split('id="data"')[0]                   # offline: no network references
    js = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    assert len(js["models"]) == 2 and js["coast"]
    for cid in re.findall(r'plot3?d?\("(f-[^"]+)"', page):             # every plotted canvas exists
        assert f'id="{cid}"' in page

    src = report_tex(data, title="A & B", models_text={"Model A": "is a test."})
    assert r"\title" not in src and r"A \& B" in src and src.count(r"\begin{figure}") == 7
    assert src.count("{") == src.count("}")
    assert tex("50 % of x_1") == r"50 \% of x\_1"
    if not __import__("shutil").which("lualatex"):
        pytest.skip("no LaTeX for the PGF figures")
    names = report_figures(data, tmp_path)
    assert all((tmp_path / f"{n}.pgf").exists() for n in names)
    assert all(f"figures/{n}.pgf" in src for n in names)


def test_site_pages():
    import json
    import re

    from aerosim.plots import site
    t = np.linspace(0, 10, 50)
    pos = np.c_[t * 10, np.zeros_like(t), 1000 - t * 90]
    model = {"name": "B <1>", "t": t, "pos": pos, "channels": {"alt": pos[:, 2], "thr": np.r_[np.full(49, 0.5), np.nan]},
             "events": [("Touchdown", 10.0)], "throttle": np.full(50, 0.5), "thrust_dir": np.tile([0, 0, 1.0], (50, 1))}
    page = site.scene_replay_html([model], "Landing & test", cards=[{"k": "alt", "label": "altitude", "d": 0}],
                                  charts=[{"k": "alt", "label": "Altitude"}], body="booster",
                                  vehicle={"length": 47, "diameter": 3.66}, current="landing.html")
    data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    assert data["models"][0]["ch"]["thr"][-1] is None                 # NaN written as a gap, valid JSON
    assert "<title>Landing &amp; test</title>" in page and 'aria-current=page' in page
    assert "http" not in page.split('id="data"')[0]                   # offline: no network references
    hub = site.hub_html([{"href": "landing.html", "title": "Landing", "text": "x", "stats": {"miss, m": "0.0"}}])
    assert 'href="landing.html"' in hub


def test_theme_switch():
    import json
    import re

    from aerosim.plots import site
    from aerosim.plots.style import THEMES, rc
    from aerosim.plots.web import nav, theme_tag
    page = site.hub_html([{"href": "a.html", "title": "A", "text": "x", "stats": {}}])
    head = re.search(r'<style id="aerosim-themes">(.*?)</style><script>(.*?)</script>', page, re.DOTALL)
    assert head and ":root[data-theme=carbon]" in head.group(1) and "--font:" in head.group(1)
    assert "font/woff;base64," in head.group(1)                      # Latin Modern embedded: offline
    themes = json.loads(re.search(r"var T=(\{.*?\}),F=", head.group(2)).group(1))
    assert set(themes) == {"dark", "carbon", "paper"} and themes["carbon"]["academic"] and themes["carbon"]["page"] == "#000000"
    assert themes["paper"]["key_inside"] and themes["paper"]["series"][0] == "#000000" and themes["paper"]["weights"]
    assert ":root[data-font=technical]" in head.group(1) and "aerosim-font" in head.group(2)
    assert "__" not in re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL).replace("__init__", "")
    assert 'class="theme"' in nav([("A", "a.html")], "a.html")
    assert json.loads(re.search(r'id="theme">(.*?)</script>', theme_tag("dark")).group(1))["name"] == "dark"
    r = rc("carbon")
    assert r["axes.facecolor"] == THEMES["carbon"]["page"] and r["xtick.direction"] == "in" and not r["axes.grid"]


def test_engine_model_geometry():
    pytest.importorskip("skfmm")
    from aerosim.plots.engine_models import nozzle_fields
    from aerosim.propulsion.solid.grains import make_grain
    b = make_grain({"type": "bates", "diameter": 0.05, "length": 0.09, "core_diameter": 0.02})
    hole = b.port_outline(0.005)["holes"][0]
    assert np.hypot(*hole.T) == pytest.approx(0.015)               # core radius + web
    assert b.port_outline(b.web_max)["holes"] == []                 # burned out
    f = make_grain({"type": "finocyl", "diameter": 0.05, "length": 0.09, "core_diameter": 0.016,
                    "fin_length": 0.008, "fin_width": 0.003, "num_fins": 6})
    loops = f.port_outline(0.0)["holes"]
    area = sum(0.5 * abs(np.dot(l[:, 0], np.roll(l[:, 1], 1)) - np.dot(l[:, 1], np.roll(l[:, 0], 1))) for l in loops)
    assert area == pytest.approx(f.port_area(0.0), rel=0.03)         # outline encloses the port area
    x = np.linspace(-0.1, 0.3, 41)
    r = 0.05 + np.where(x < 0, -x * 0.5, x * 0.6)                   # throat at x = 0
    fl = nozzle_fields(x, r, 3500.0, 10e6, 1.2)
    i = int(np.argmin(r))
    assert fl["M"][i] == pytest.approx(1.0) and fl["M"][0] < 1 < fl["M"][-1]
    assert fl["T"][0] > fl["T"][i] > fl["T"][-1] and fl["p"][0] > fl["p"][-1]


def test_theme_head_script_parses():
    import re
    import shutil
    import subprocess

    from aerosim.plots.web import theme_head
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = re.search(r"<script>(.*?)</script>", theme_head("dark"), re.DOTALL).group(1)
    r = subprocess.run([node, "--check", "-"], check=False, input=js, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
