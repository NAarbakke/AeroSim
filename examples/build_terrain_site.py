"""Build the terrain-flight site: the JSBSim F-16 at low level over real terrain, offline HTML.

    python examples/build_terrain_site.py              # -> docs/site/terrain*.html (about 8 minutes)
    python examples/build_terrain_site.py --download   # fetch the terrain again first

Two regions, two flights each (a route along the low ground and a straight line over the
high ground), all terrain-following at 100 m set clearance:

- Lysefjorden, Norway: Kartverket's 1 m laser DTM (resampled to 20 m), from Høgsfjorden
  through the 42 km fjord to Lysebotn and up the headwall to the valley above.
- Matochkin Shar, Novaya Zemlya: Copernicus GLO-30 (30 m), Barents Sea to Kara Sea
  through the strait between the two islands.

The terrain grids are in the repo (``data/terrain``, 10 MB, see its README), so the build needs
no network. ``--download`` fetches them again from Kartverket and Copernicus (about 200 MB of
source tiles, cached in ``outputs/terrain``) and rewrites ``data/terrain``. Over Lysefjorden the
Copernicus model is there too, as an independent check on the Kartverket heights.

Ground images (``data/imagery``, see its README) are draped over the terrain and chosen in the
viewer's Ground menu: Sentinel-2 true colour in two seasons, Kartverket's N50 map (Norway) and
ESA WorldCover land cover. Each is written next to the pages as terrain-img/<region>-<layer>.js
and loaded only when picked.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "site"                            # pages join the main results site
SUMMARY = ROOT / "outputs" / "site" / "terrain.json"    # stats for the main hub tile
DATA = ROOT / "data" / "terrain"
IMAGERY = ROOT / "data" / "imagery"
DOWNLOAD = False

import aerosim.propulsion  # noqa: F401  (registers the engine backends)
from aerosim.core import registry
from aerosim.plots import site
from aerosim.plots import terrain_site as ts
from aerosim.plots.web import write_page
from aerosim.terrain import ATTRIBUTION, TerrainGrid
from aerosim.terrain.dem import COPERNICUS_LIABILITY
from aerosim.terrain.flight import fly_route
from aerosim.terrain.route import direct_route, terrain_profile, valley_route

T0 = time.time()
CLEARANCE = 100.0
PAGES = site.PAGES
SUBNAV = [("Terrain flights", "terrain.html"), ("Lysefjorden", "terrain-lysefjorden.html"),
          ("Matochkin Shar", "terrain-matochkin-shar.html")]

REGIONS = {
    "lysefjorden": {
        "title": "Lysefjorden, Norway", "source": "kartverket", "ramp": "norway",
        "center": (58.99, 6.36), "size": (48e3, 30e3), "spacing": 20.0, "mesh": 2,
        "mach": 0.5, "turn_radius": 1400.0,
        # Høgsfjorden -> Lysebotn quay -> the valley above the headwall
        "valley": [(58.866, 6.12), (59.0552, 6.6388), (59.0729, 6.761)],
        "direct": [(58.866, 6.12), (59.0729, 6.761)],
        "labels": [("Forsand", 58.9033, 6.10478, ""), ("Lysebotn", 59.0554, 6.64853, ""),
                   ("Preikestolen", 58.98666, 6.18748, "pk"), ("Kjerag", 59.02237, 6.58764, "pk"),
                   ("Lysefjorden", 58.985, 6.30, "sea"), ("Høgsfjorden", 58.885, 6.09, "sea")],
        "subtitle": "Kartverket 1 m laser terrain model  ·  Mach 0.5  ·  100 m set clearance",
        "grounds": [
            ("s2-summer", "Sentinel-2, 17 Aug 2025", {"source": "sentinel2", "date": "2025-08-17", "px": 10}),
            ("s2-snow", "Sentinel-2, 7 Apr 2025 (snow)", {"source": "sentinel2", "date": "2025-04-07", "px": 10, "gamma": 1.0}),
            ("n50", "Kartverket N50 topographic map", {"source": "n50", "px": 6}),
            ("landcover", "ESA WorldCover land cover", {"source": "worldcover", "px": 10}),
        ],
    },
    "matochkin-shar": {
        "title": "Matochkin Shar, Novaya Zemlya", "source": "copernicus", "ramp": "arctic",
        "center": (73.33, 55.3), "size": (110e3, 46e3), "spacing": 30.0, "mesh": 2,
        "mach": 0.6, "turn_radius": 1900.0,
        # Barents Sea -> Kara Sea: 50 km west and 52 km east of the grid centre, 4 km south
        "valley": [(73.2883, 53.7427), (73.2879, 56.9196)],
        "direct": [(73.2883, 53.7427), (73.2879, 56.9196)],
        "labels": [("Matochkin Shar", 73.3886, 55.2156, "sea"), ("Barents Sea", 73.42, 53.85, "sea"),
                   ("Kara Sea", 73.2348, 56.8215, "sea"), ("Severny Island", 73.47, 55.95, ""),
                   ("Yuzhny Island", 73.17, 54.95, "")],
        "subtitle": "Copernicus GLO-30 elevation model  ·  Mach 0.6  ·  100 m set clearance",
        "grounds": [
            ("s2-summer", "Sentinel-2, 11 Aug 2024", {"source": "sentinel2", "date": "2024-08-11", "px": 15}),
            ("s2-snow", "Sentinel-2, 3 Jun 2022 (snow, ice)", {"source": "sentinel2", "date": "2022-06-03", "px": 15, "gamma": 1.0}),
            ("landcover", "ESA WorldCover land cover", {"source": "worldcover", "px": 15}),
        ],
    },
}


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


def deck():
    cfg = yaml.safe_load((ROOT / "configs" / "aircraft" / "turbojet.yaml").read_text())
    p = dict(cfg["propulsion"])
    eng = registry.create("airbreathing", p.pop("backend"), **p)
    return eng.deck([0, 1000, 2000, 3500, 6000], [0.0, 0.3, 0.45, 0.6, 0.75, 0.9],
                    cfg["propulsion_deck"]["powers"])


def grid(name: str, source: str, center, size, spacing) -> TerrainGrid:
    """The packed grid from ``data/terrain``, or (``--download`` or missing) fetched and saved there."""
    path = DATA / f"{name}_{source}_{spacing:g}.npz"
    if path.exists() and not DOWNLOAD:
        return TerrainGrid.load(path)
    log(f"downloading {path.name}")
    build = TerrainGrid.kartverket if source == "kartverket" else TerrainGrid.copernicus
    g = build(*center, size=size, spacing=spacing)
    g.save(path, packed=True)
    return TerrainGrid.load(path)             # what the repo holds (heights to 0.1 m)


def grid_for(key: str, r: dict) -> TerrainGrid:
    return grid(key, r["source"], r["center"], r["size"], r["spacing"])


def speed_of_sound(h: float = 0.0) -> float:
    from aerosim.atmosphere.standard import isa
    p, _, rho = isa(h)
    return math.sqrt(1.4 * p / rho)


def fly_region(key: str, r: dict, engine) -> tuple[TerrainGrid, list[dict]]:
    g = grid_for(key, r)
    log(f"{key}: grid {g.z.shape}, {g.z.min():.0f} .. {g.z.max():.0f} m")
    V = r["mach"] * speed_of_sound()
    models = []
    for name, route in (("Valley route", valley_route), ("Direct line", direct_route)):
        tr = route(g, r["valley" if route is valley_route else "direct"], min_turn_radius=r["turn_radius"], name=name)
        pr = terrain_profile(g, tr, speed=V, clearance=CLEARANCE)
        f = fly_route(g, tr, pr, engine, mach=r["mach"])
        me = f.meta
        log(f"{key} {name}: {tr.length / 1e3:.1f} km, {f.time[-1]:.0f} s, {me['status']}, "
            f"min AGL {me['min_agl']:.1f} m, max xtrack {np.abs(me['cross_track']).max():.0f} m")
        models.append({"name": name, "flight": f, "track": tr, "profile": pr})
    return g, models


def grounds_for(key: str, r: dict, g: TerrainGrid) -> list[dict]:
    """The region's draped images: built into data/imagery if missing (or ``--download``), then
    written next to the pages as terrain-img/<region>-<layer>.js for on-demand loading."""
    from aerosim.terrain import imagery as im
    out = []
    (OUT / "terrain-img").mkdir(exist_ok=True)
    for layer, label, spec in r["grounds"]:
        ext = ".png" if spec["source"] == "worldcover" else ".jpg"
        path = IMAGERY / key / f"{layer}{ext}"
        if DOWNLOAD or not path.exists():
            log(f"building {key}/{layer}")
            if spec["source"] == "sentinel2":
                L = im.sentinel2(g, spec["date"], px=spec["px"], key=layer, label=label, gamma=spec.get("gamma", 0.8))
            elif spec["source"] == "n50":
                L = im.kartverket_n50(g, px=spec["px"], key=layer, label=label)
            else:
                L = im.worldcover(g, px=spec["px"], key=layer, label=label)
            L.save(path, **({"quality": 75} if spec["source"] == "n50" else {}))
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        gid = f"{key}-{layer}"
        write_page(OUT / "terrain-img" / f"{gid}.js", ts.ground_script(gid, path))
        out.append({"id": gid, "label": label, "credit": side["credit"], "src": f"terrain-img/{gid}.js",
                    **({"legend": side["legend"]} if "legend" in side else {})})
    return out


def labels(g: TerrainGrid, r: dict) -> list[dict]:
    out = []
    for name, la, lo, kind in r["labels"]:
        x, y = g.to_xy(la, lo)
        out.append({"name": name, "x": round(float(x), 1), "y": round(float(y), 1),
                    "z": round(max(g.elevation(x, y), 0.0), 1), "kind": kind})
    j, i = np.unravel_index(np.argmax(g.z), g.z.shape)
    out.append({"name": f"{g.z[j, i]:,.0f} m", "x": float(g.x[i]), "y": float(g.y[j]),
                "z": float(g.z[j, i]), "kind": "pk"})
    return out


def validation(g: TerrainGrid) -> list[dict]:
    """Kartverket vs Copernicus over the Lysefjorden grid (independent lidar and radar models).

    Published summit heights were tried too and dropped: their coordinates are not precise
    enough to pick one grid cell (Kjerag's published point is 390 m from the highest ground)."""
    c = grid("lysefjorden", "copernicus", (g.lat0, g.lon0), (g.x[-1] - g.x[0], g.y[-1] - g.y[0]), g.spacing)
    d = c.z - g.z
    land = ~g.water & ~c.water
    rows = []
    for label, m in (("all land", land), ("land above 600 m (bare rock)", land & (g.z > 600))):
        dd = d[m]
        rows.append({"label": f"Copernicus − Kartverket, {label}",
                     "value": f"median {np.median(dd):+.1f} m, rms {np.sqrt(np.mean(dd ** 2)):.1f} m, "
                              f"90 % within {np.percentile(np.abs(dd), 90):.1f} m ({dd.size / 1e6:.1f} M points)"})
    log("validation: " + "; ".join(r["value"] for r in rows))
    return rows


METHOD = """<ol>
<li><b>Terrain.</b> Kartverket's terrain model is fetched from Geonorge's coverage service
(EPSG:25833) at 20 m; Copernicus GLO-30 tiles come from the public AWS bucket. Both are resampled
bilinearly onto a local transverse Mercator grid centred on the region, so x and y are true
ground distances. Each tile's pixel convention (area or point) is read from the file.</li>
<li><b>Ground track.</b> The valley route is a least-cost path over the height grid. Low ground
is cheap and ground near walls higher than 120 m costs extra, so the path keeps to the middle of
the fjord or strait. A second-order cone programme then smooths it to the closest curve with
a turn radius no tighter than 63° of bank allows at the planned speed (1.4 km at Mach 0.5,
1.9 km at Mach 0.6). The direct line joins the same end points straight across the high ground.</li>
<li><b>Altitude reference.</b> The lowest profile the aircraft can fly is a linear programme in
the altitude at every 50 m. It stays at least 100 m above the highest terrain within 150 m
either side of the track. Climb and dive are limited to 20° and 12°, and the curvature of the
profile to a 3 g pull-up and a 0.3 g push-over. It begins each climb early enough to clear the
ridge at the pull-up limit.</li>
<li><b>Flight.</b> JSBSim's F-16 is flown on AeroSim's turbojet deck, the same deck as on the
main site. Every step the real terrain height under the aircraft goes to JSBSim, and the
aircraft is matched to the nearest point of the route. Guidance computes one force normal to
the velocity, from curvature feed-forward and cross-track feedback on the horizontal and
profile tracking on the vertical. That force sets the bank angle and load factor, which the
F-16's flight controls then fly.</li>
</ol>"""


CREDITS = (f'<p>{ATTRIBUTION["kartverket"]}.</p><p>{ATTRIBUTION["copernicus"]}. {COPERNICUS_LIABILITY}</p>'
           '<p>Ground images: Sentinel-2 true colour, contains modified Copernicus Sentinel data '
           '(2022, 2024, 2025); N50 Raster © Kartverket (CC BY 4.0); © ESA WorldCover project 2021 / '
           'Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium '
           '(CC BY 4.0).</p>'
           f'<p>{ts.AIRCRAFT_CREDIT}.</p><p>The resampled grids and images are in the repository under '
           'data/terrain and data/imagery, with their licences.</p>')


def main(argv=None):
    global DOWNLOAD
    DOWNLOAD = "--download" in (argv or [])
    OUT.mkdir(parents=True, exist_ok=True)
    engine = deck()
    hub, lows = [], []
    vrows = []
    for key, r in REGIONS.items():
        g, models = fly_region(key, r, engine)
        if key == "lysefjorden":
            vrows = validation(g)
        mesh = g.downsample(r["mesh"])
        notes = (f'<h3>Data</h3><p class="note">{ATTRIBUTION[r["source"]]}. Grid {g.spacing:g} m for '
                 f'the flight, {mesh.spacing:g} m for the 3D mesh, local transverse Mercator centred at '
                 f'{r["center"][0]:.2f}° N {r["center"][1]:.2f}° E.</p>'
                 '<p class="note">Altitudes are above mean sea level as each model gives them; JSBSim '
                 'gets the same heights, so height above ground is consistent.</p>')
        if r["source"] == "copernicus":
            notes += f'<p class="note">{COPERNICUS_LIABILITY}</p>'
        page = ts.terrain_replay_html(mesh, models, title=f"F-16 low level: {r['title']}",
                                      subtitle=r["subtitle"], labels=labels(g, r),
                                      attribution=ATTRIBUTION[r["source"]], ramp=r["ramp"],
                                      clearance=CLEARANCE, pages=PAGES, current="terrain.html",
                                      notes=notes, subnav=SUBNAV, here=f"terrain-{key}.html",
                                      grounds=grounds_for(key, r, g))
        write_page(OUT / f"terrain-{key}.html", page)
        log(f"wrote terrain-{key}.html ({len(page) / 1e6:.2f} MB)")
        st = [ts.flight_stats(m) for m in models]
        lows += [float(s["lowest above ground, m"]) for s in st]
        hub.append({"href": f"terrain-{key}.html", "title": r["title"], "img": ts.hillshade_png(g, r["ramp"], 640),
                    "text": f"{r['subtitle']}<br>Valley route {st[0]['route, km']} km, lowest "
                            f"{st[0]['lowest above ground, m']} m above ground; direct line "
                            f"{st[1]['route, km']} km, lowest {st[1]['lowest above ground, m']} m."})
    write_page(OUT / "terrain.html", ts.terrain_hub_html(hub, vrows, pages=PAGES, method=METHOD,
                                                         credits=CREDITS, current="terrain.html"))
    log("wrote terrain.html")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({"regions": str(len(hub)), "flights": str(len(lows)),
                                   "lowest above ground, m": f"{min(lows):.1f}"}), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
