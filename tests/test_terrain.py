"""Terrain: DEM decoding and lookup, route planning, the terrain-following profile and a
JSBSim terrain-following flight on synthetic terrain, and the real Kartverket and Copernicus
grids kept in data/terrain (no downloads)."""

import math
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("tifffile")

from aerosim.terrain import TerrainGrid
from aerosim.terrain.dem import read_cog

DATA = Path(__file__).resolve().parents[1] / "data" / "terrain"


def _float_predictor_tiff(path, a, tile=16, pixel_is_point=True):
    """Minimal tiled GeoTIFF, Deflate + floating-point predictor, written by hand (tifffile
    needs imagecodecs to write that predictor)."""
    H, W = a.shape
    pad = np.zeros((-(-H // tile) * tile, -(-W // tile) * tile), np.float32)
    pad[:H, :W] = a
    tiles = []
    for r in range(0, pad.shape[0], tile):
        for c in range(0, pad.shape[1], tile):
            t = pad[r:r + tile, c:c + tile]
            b = t.astype(">f4").view(np.uint8).reshape(tile, tile, 4).transpose(0, 2, 1).reshape(tile, -1)
            d = np.diff(b, axis=1, prepend=np.zeros((tile, 1), np.uint8)).astype(np.uint8)
            tiles.append(zlib.compress(d.tobytes()))
    n = len(tiles)
    tags = [(256, 3, [W]), (257, 3, [H]), (258, 3, [32]), (259, 3, [8]), (262, 3, [1]),
            (277, 3, [1]), (317, 3, [3]), (322, 3, [tile]), (323, 3, [tile]),
            (324, 4, [0] * n), (325, 4, [len(t) for t in tiles]), (339, 3, [3]),
            (33550, 12, [0.5, 0.25, 0.0]), (33922, 12, [0, 0, 0, 10.0, 20.0, 0]),
            (34735, 3, [1, 1, 0, 1, 1025, 0, 1, 2 if pixel_is_point else 1])]
    fmt = {3: "H", 4: "I", 12: "d"}
    ifd_off = 8
    ifd_size = 2 + 12 * len(tags) + 4
    extra, extra_off = b"", ifd_off + ifd_size
    entries = []
    data_off = extra_off + sum(len(v) * struct.calcsize(fmt[ty]) for _, ty, v in tags
                               if len(v) * struct.calcsize(fmt[ty]) > 4)
    offs, o = [], data_off
    for t in tiles:
        offs.append(o)
        o += len(t)
    tags[9] = (324, 4, offs)
    for code, ty, vals in tags:
        raw = struct.pack("<" + fmt[ty] * len(vals), *vals)
        if len(raw) <= 4:
            entries.append(struct.pack("<HHI", code, ty, len(vals)) + raw.ljust(4, b"\0"))
        else:
            entries.append(struct.pack("<HHII", code, ty, len(vals), extra_off + len(extra)))
            extra += raw
    blob = (b"II*\0" + struct.pack("<I", ifd_off) + struct.pack("<H", len(tags)) + b"".join(entries)
            + struct.pack("<I", 0) + extra + b"".join(tiles))
    Path(path).write_bytes(blob)


def test_read_cog_float_predictor(tmp_path):
    a = (np.random.default_rng(1).random((40, 50)) * 1500 - 20).astype(np.float32)
    _float_predictor_tiff(tmp_path / "p.tif", a)
    b, geo = read_cog(tmp_path / "p.tif")
    assert np.array_equal(a, b)
    # PixelIsPoint: the tie point (10, 20) is the centre of pixel (0, 0)
    assert geo == pytest.approx((9.75, 20.125, 0.5, -0.25))
    _float_predictor_tiff(tmp_path / "a.tif", a, pixel_is_point=False)
    assert read_cog(tmp_path / "a.tif")[1] == pytest.approx((10.0, 20.0, 0.5, -0.25))


def _rgb_tiff(path, a, tile=16):
    """Tiled RGB uint8 GeoTIFF, Deflate + horizontal predictor (as Sentinel-2 TCI COGs)."""
    H, W, _ = a.shape
    pad = np.zeros((-(-H // tile) * tile, -(-W // tile) * tile, 3), np.uint8)
    pad[:H, :W] = a
    tiles = []
    for r in range(0, pad.shape[0], tile):
        for c in range(0, pad.shape[1], tile):
            t = pad[r:r + tile, c:c + tile]
            d = np.diff(t, axis=1, prepend=np.zeros((tile, 1, 3), np.uint8)).astype(np.uint8)
            tiles.append(zlib.compress(d.tobytes()))
    tags = [(256, 3, [W]), (257, 3, [H]), (258, 3, [8, 8, 8]), (259, 3, [8]), (262, 3, [2]),
            (277, 3, [3]), (284, 3, [1]), (317, 3, [2]), (322, 3, [tile]), (323, 3, [tile]),
            (324, 4, [0] * len(tiles)), (325, 4, [len(t) for t in tiles]), (339, 3, [1, 1, 1]),
            (33550, 12, [10.0, 10.0, 0.0]), (33922, 12, [0, 0, 0, 300000.0, 6600000.0, 0]),
            (34735, 3, [1, 1, 0, 1, 1025, 0, 1, 1])]
    _write_tiff(path, tags, tiles)


def _write_tiff(path, tags, tiles):
    fmt = {3: "H", 4: "I", 12: "d"}
    size = lambda ty, v: len(v) * struct.calcsize(fmt[ty])
    extra_off = 8 + 2 + 12 * len(tags) + 4
    data_off = extra_off + sum(size(ty, v) for _, ty, v in tags if size(ty, v) > 4)
    offs = list(np.cumsum([0] + [len(t) for t in tiles[:-1]]) + data_off)
    tags = [(c, ty, [int(o) for o in offs] if c == 324 else v) for c, ty, v in tags]
    extra, entries = b"", []
    for code, ty, vals in tags:
        raw = struct.pack("<" + fmt[ty] * len(vals), *vals)
        if len(raw) <= 4:
            entries.append(struct.pack("<HHI", code, ty, len(vals)) + raw.ljust(4, b"\0"))
        else:
            entries.append(struct.pack("<HHII", code, ty, len(vals), extra_off + len(extra)))
            extra += raw
    Path(path).write_bytes(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(tags)) + b"".join(entries)
                          + struct.pack("<I", 0) + extra + b"".join(tiles))


def test_read_tiff_window_rgb_horizontal_predictor(tmp_path):
    from aerosim.terrain.dem import read_tiff_window
    a = np.random.default_rng(3).integers(0, 256, (37, 45, 3), dtype=np.uint8)
    _rgb_tiff(tmp_path / "rgb.tif", a)
    full, geo = read_tiff_window(tmp_path / "rgb.tif")
    assert np.array_equal(full, a) and geo == pytest.approx((300000.0, 6600000.0, 10.0, -10.0))
    win, g = read_tiff_window(tmp_path / "rgb.tif", rows=(5, 30), cols=(17, 40))   # across tiles
    assert np.array_equal(win, a[5:30, 17:40]) and g == pytest.approx((300170.0, 6599950.0, 10.0, -10.0))


def test_image_layer_round_trip_and_ground_script(tmp_path):
    pytest.importorskip("PIL")
    from aerosim.plots.terrain_site import ground_script
    from aerosim.terrain.imagery import ImageLayer
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:, :30] = (0, 100, 0)
    rgb[:, 30:] = (0, 100, 200)
    p = ImageLayer("lc", "Land cover", "credit", rgb, {"legend": [["Trees", "#006400"]]}).save(tmp_path / "lc.png")
    back = ImageLayer.load(p)
    assert np.array_equal(back.rgb, rgb) and back.meta["legend"] == [["Trees", "#006400"]]
    js = ground_script("r-lc", p)
    assert js.startswith('(window.AEROSIM_GROUND = window.AEROSIM_GROUND || {})["r-lc"] = "data:image/png;base64,iVBOR')


def _grid(fn, size=(20e3, 12e3), spacing=50.0, lat0=60.0, lon0=10.0):
    g = TerrainGrid._empty(lat0, lon0, size, spacing, "synthetic")
    X, Y = np.meshgrid(g.x, g.y)
    g.z = fn(X, Y).astype(np.float32)
    return g


def test_grid_lookup_and_projection():
    g = _grid(lambda x, y: 0.01 * x - 0.02 * y + 300)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-9e3, 9e3, 50), rng.uniform(-5e3, 5e3, 50)
    assert np.allclose(g.elevation(x, y), 0.01 * x - 0.02 * y + 300, atol=1e-3)   # bilinear is exact
    assert g.elevation_at(1234.5, -987.6) == pytest.approx(0.01 * 1234.5 + 0.02 * 987.6 + 300, abs=1e-3)
    lat, lon = g.to_latlon(x, y)
    assert np.allclose(g.to_xy(lat, lon), (x, y), atol=1e-6)
    assert g.to_latlon(0.0, 0.0) == pytest.approx((60.0, 10.0))
    # 1 km north is 1 km on the ground (WGS84 meridian arc at 60 N: 111.41 km per degree)
    assert (g.to_latlon(0.0, 1000.0)[0] - 60.0) * 111_412 == pytest.approx(1000.0, rel=1e-3)


def test_meridian_convergence_matches_pyproj():
    import pyproj
    g = _grid(lambda x, y: 0 * x, lat0=73.3, lon0=55.3)
    proj = pyproj.Proj(g.crs)
    for lat, lon in ((73.3, 53.7), (73.5, 57.0), (73.0, 55.3)):
        c = proj.get_factors(lon, lat).meridian_convergence
        assert math.degrees(g.convergence(lat, lon)) == pytest.approx(-c, abs=1e-4)


def _valley(x, y):
    """A winding valley (centre line y = 2 km sin(2 pi x / 16 km)) 600 m deep, floor at 0."""
    yc = 2000 * np.sin(2 * np.pi * x / 16e3)
    return 600 * (1 - np.exp(-((y - yc) / 700) ** 2))


def test_valley_route_follows_channel_within_turn_radius():
    pytest.importorskip("cvxpy")
    pytest.importorskip("skimage")
    from aerosim.terrain.route import valley_route
    g = _grid(_valley)
    wp = [g.to_latlon(-9e3, 2000 * math.sin(2 * math.pi * -9e3 / 16e3)),
          g.to_latlon(9e3, 2000 * math.sin(2 * math.pi * 9e3 / 16e3))]
    tr = valley_route(g, wp, min_turn_radius=1500)
    yc = 2000 * np.sin(2 * np.pi * tr.x / 16e3)
    assert np.abs(tr.y - yc).max() < 250                 # stays in the valley bottom
    assert tr.meta["max_curvature"] <= 1 / 1500 * 1.02
    assert np.allclose(np.diff(tr.s), 50, rtol=0.05)     # evenly resampled


def test_profile_meets_clearance_and_flight_limits():
    pytest.importorskip("cvxpy")
    from aerosim.core.units import G0
    from aerosim.terrain.route import direct_route, terrain_profile
    ridge = lambda x, y: 700 * np.exp(-(x / 1500) ** 2) + 0 * y
    g = _grid(ridge)
    tr = direct_route(g, [g.to_latlon(-9e3, 0), g.to_latlon(9e3, 0)], min_turn_radius=1500)
    V, clear = 170.0, 100.0
    pr = terrain_profile(g, tr, speed=V, clearance=clear, climb_deg=20, dive_deg=12, n_pull=3, n_push=0.3)
    h, s = pr.altitude, pr.s
    ds = np.diff(s)
    assert np.all(h >= pr.floor - 1e-3) and np.all(pr.floor >= pr.terrain + clear - 1e-3)
    slope = np.diff(h) / ds
    assert slope.max() <= math.tan(math.radians(20)) + 1e-6 and slope.min() >= -math.tan(math.radians(12)) - 1e-6
    curv = np.diff(h, 2) / ds[0] ** 2
    assert curv.max() <= 2 * G0 / V**2 + 1e-7 and curv.min() >= -0.7 * G0 / V**2 - 1e-7
    assert h.min() == pytest.approx(clear, abs=1.0)                # flat ground: the clearance itself
    assert h.max() == pytest.approx(800.0, abs=5.0)                # ridge top + clearance
    i0 = np.argmax(h > clear + 1)                                  # the climb starts well before
    assert s[i0] < s[np.argmax(pr.terrain)] - 2000                 # the ridge (20 deg, 3 g limits)


def test_terrain_following_flight_over_a_ridge():
    pytest.importorskip("jsbsim")
    pytest.importorskip("cvxpy")
    from aerosim.core import EngineDeck
    from aerosim.terrain.flight import fly_route
    from aerosim.terrain.route import direct_route, terrain_profile
    g = _grid(lambda x, y: 500 * np.exp(-((x - 2000) / 1800) ** 2) + 0 * y, size=(26e3, 8e3))
    tr = direct_route(g, [g.to_latlon(-11e3, 0), g.to_latlon(11e3, 0)], min_turn_radius=1500)
    pr = terrain_profile(g, tr, speed=170, clearance=100)
    alts, machs, pw = [0.0, 3000.0], [0.0, 0.5, 0.9], [0.0, 1.0]
    F = np.zeros((2, 3, 2))
    F[:, :, 1] = [[110e3, 100e3, 95e3], [80e3, 75e3, 72e3]]           # F100-class static thrust
    deck = EngineDeck(np.array(alts), np.array(machs), np.array(pw), F, F * 2.2e-5, name="test")
    r = fly_route(g, tr, pr, deck, mach=0.5)
    m = r.meta
    assert m["status"] == "end of route"
    assert m["min_agl"] > 85.0                                    # 100 m set clearance
    assert np.abs(m["cross_track"]).max() < 15.0
    assert np.sqrt(np.mean((r.position[:, 2] - m["h_ref"]) ** 2)) < 10.0
    assert r.position[:, 2].max() > 550                           # it did climb the ridge
    assert abs(m["mach"][-1] - 0.5) < 0.03


def test_packed_grid_round_trip(tmp_path):
    g = _grid(lambda x, y: 1500 * np.sin(x / 3e3) * np.cos(y / 2e3) + 0.01234 * x)
    h = TerrainGrid.load(g.save(tmp_path / "g.npz", packed=True))
    assert np.abs(h.z - g.z).max() <= 0.05 + 1e-4
    assert np.allclose(h.x, g.x) and np.allclose(h.y, g.y) and (h.lat0, h.lon0) == (g.lat0, g.lon0)


def test_repo_grids_and_sources_agree_over_lysefjorden():
    """The grids kept in data/terrain: Kartverket lidar vs Copernicus radar, above the tree line."""
    kv = TerrainGrid.load(DATA / "lysefjorden_kartverket_20.npz")
    cop = TerrainGrid.load(DATA / "lysefjorden_copernicus_20.npz")
    nz = TerrainGrid.load(DATA / "matochkin-shar_copernicus_30.npz")
    assert kv.z.shape == cop.z.shape == (1501, 2401) and nz.z.shape == (1534, 3668)
    assert 1290 < kv.z.max() < 1310 and 1260 < nz.z.max() < 1280
    rock = ~kv.water & ~cop.water & (kv.z > 600)                 # above the tree line
    d = cop.z[rock] - kv.z[rock]
    assert abs(np.median(d)) < 1.0 and np.sqrt(np.mean(d ** 2)) < 6.0


def test_terrain_page():
    pytest.importorskip("matplotlib")
    from aerosim.core import FlightResult
    from aerosim.plots.terrain_site import terrain_hub_html, terrain_replay_html
    from aerosim.terrain.route import Profile, Track
    g = _grid(_valley, size=(4e3, 3e3), spacing=100)
    n = 50
    t = np.linspace(0, 20, n)
    x = np.linspace(-1500, 1500, n)
    meta = {k: np.zeros(n) for k in ("terrain", "agl", "s", "cross_track", "h_ref", "mach", "airspeed",
                                     "nz", "roll_deg", "pitch_deg", "heading_deg")}
    meta |= {"route_length": 3000.0, "min_agl": 100.0, "fuel_used": 10.0, "status": "end of route"}
    f = FlightResult(t, np.c_[x, 0 * x, 0 * x + 100], np.zeros((n, 3)), np.full(n, 1e4), meta=meta)
    tr = Track(x, 0 * x)
    pr = Profile(tr.s, 0 * x, 0 * x + 100, 0 * x + 100, 0 * x, 0 * x)
    page = terrain_replay_html(g, [{"name": "A", "flight": f, "track": tr, "profile": pr}], title="Test",
                               labels=[{"name": "P", "x": 0, "y": 0, "z": 10, "kind": "pk"}])
    assert "importmap" in page and '"nx":41' in page and "Flight summary" in page
    assert "three/addons/loaders/GLTFLoader.js" in page and '"glb":"Z2xURg' in page   # base64 "glTF"
    assert "iedalton" in page and "CC BY 4.0" in page                                  # model credit
    g2 = terrain_replay_html(g, [{"name": "A", "flight": f, "track": tr, "profile": pr}], title="Test",
                             grounds=[{"id": "t-s2", "label": "Sentinel-2", "credit": "c", "src": "terrain-img/t-s2.js"}])
    assert 'id="ground"' in g2 and '"src":"terrain-img/t-s2.js"' in g2
    hub = terrain_hub_html([{"href": "a.html", "title": "A", "text": "t", "img": "data:,"}],
                           [{"label": "x", "value": "y"}], pages=[("Overview", "index.html")], method="")
    assert 'href="a.html"' in hub
