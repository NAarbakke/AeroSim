"""Ground imagery for a TerrainGrid: satellite photos, maps and land cover, resampled onto the
grid's own transverse Mercator so they drape straight onto the terrain mesh.

    layer = sentinel2(grid, "2025-08-17", label="Sentinel-2, summer")   # true-colour photo
    layer = kartverket_n50(grid)                                         # Norway topographic map
    layer = worldcover(grid)                                             # ESA land cover map
    layer.save("data/imagery/lysefjorden/s2-summer.jpg");  ImageLayer.load(path)

Every layer covers exactly the grid's extent, north up: pixel (i, j) is centred at
x = x[0] + (i + 0.5) (x[-1] - x[0]) / width, y = y[-1] - (j + 0.5) (y[-1] - y[0]) / height,
so a texture coordinate u = (x - x[0]) / (x[-1] - x[0]) (v likewise, from the south) maps it.

Sources (all free to download and redistribute with their credit line):

- **Sentinel-2 L2A true colour** (``TCI``, 10 m), Copernicus Sentinel data, served as COGs by
  Element 84 Earth Search on AWS. One acquisition date per layer, mosaicked across MGRS tiles.
  Credit: "Contains modified Copernicus Sentinel data <year>".
- **Kartverket N50 Raster** (1:50 000 topographic map, 2.5 m, Norway), GeoTIFF cells of
  20 x 20 km from Geonorge's download API (no account needed). CC BY 4.0, © Kartverket.
- **ESA WorldCover 2021 v200** (10 m land cover, 11 classes, global), COGs on AWS, coloured
  with the product's own legend. CC BY 4.0, © ESA WorldCover project 2021.

Not included: EOX Sentinel-2 cloudless (CC BY-NC-SA, bulk delivery only on request), Norge i
bilder aerial photos (permission needed to republish), Google / Bing / Esri imagery (terms
forbid offline copies), Landsat (public domain, but 30 m and the same kind of picture as
Sentinel-2 at 10 m).
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosim.core import require
from aerosim.terrain.dem import CACHE, TerrainGrid, _download, _geo, read_tiff_window

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
GEONORGE_ORDER = "https://nedlasting.geonorge.no/api/order"
N50_UTM32 = "bbe517ba-3b3a-4c96-a78e-104f6c4804d1"     # N50 Raster (UTM32), cells of 20 km
WORLDCOVER_CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
                      50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
                      80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
                      100: "Moss and lichen"}
CREDITS = {
    "sentinel2": "Contains modified Copernicus Sentinel data {year}",
    "n50": "N50 Raster © Kartverket (CC BY 4.0)",
    "worldcover": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data "
                  "(2021) processed by ESA WorldCover consortium (CC BY 4.0)",
}


@dataclass
class ImageLayer:
    key: str
    label: str
    credit: str
    rgb: np.ndarray                      # (height, width, 3) uint8, north up, grid extent
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path, quality: int = 86, subsampling: int = 2) -> Path:
        """JPEG (photos: 4:2:0; maps: pass ``subsampling=0`` to keep thin coloured lines sharp)
        or PNG (``.png``: 32-colour palette, for land cover), plus a ``.json`` sidecar."""
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(self.rgb)
        if path.suffix.lower() == ".png":
            im.quantize(colors=32, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
        else:
            im.save(path, quality=quality, optimize=True, progressive=True, subsampling=subsampling)
        side = {"key": self.key, "label": self.label, "credit": self.credit, "file": path.name,
                "width": self.rgb.shape[1], "height": self.rgb.shape[0], **self.meta}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ImageLayer:
        from PIL import Image
        path = Path(path)
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        rgb = np.asarray(Image.open(path).convert("RGB"))
        meta = {k: v for k, v in side.items() if k not in ("key", "label", "credit", "file", "width", "height")}
        return cls(side["key"], side["label"], side["credit"], rgb, meta)


# ------------------------------------------------------------------ target raster
def _target(grid: TerrainGrid, px: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre x (west -> east) and y (north -> south) for a layer of ~px metres."""
    X0, X1, Y0, Y1 = grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]
    w, h = max(round((X1 - X0) / px), 2), max(round((Y1 - Y0) / px), 2)
    return X0 + (np.arange(w) + 0.5) * (X1 - X0) / w, Y1 - (np.arange(h) + 0.5) * (Y1 - Y0) / h


def _chunks(grid: TerrainGrid, px: float, rows: int = 384):
    """(row slice, lon, lat) over the target raster, ``rows`` at a time (bounded memory)."""
    xs, ys = _target(grid, px)
    inv = grid._tr(True)
    for r0 in range(0, ys.size, rows):
        X, Y = np.meshgrid(xs, ys[r0:r0 + rows])
        lon, lat = inv.transform(X, Y)
        yield slice(r0, r0 + X.shape[0]), lon, lat


def _shape(grid, px):
    xs, ys = _target(grid, px)
    return ys.size, xs.size


def _lonlat_bbox(grid: TerrainGrid, margin: float = 0.01) -> list[float]:
    xs = np.r_[grid.x, grid.x, np.full(grid.y.size, grid.x[0]), np.full(grid.y.size, grid.x[-1])]
    ys = np.r_[np.full(grid.x.size, grid.y[0]), np.full(grid.x.size, grid.y[-1]), grid.y, grid.y]
    lat, lon = grid.to_latlon(xs, ys)
    return [float(lon.min() - margin), float(lat.min() - margin), float(lon.max() + margin), float(lat.max() + margin)]


def _sample(img: np.ndarray, geo, E: np.ndarray, N: np.ndarray, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Values of ``img`` (h, w[, c]) at map coordinates (E, N); mask of points inside it."""
    from scipy.ndimage import map_coordinates
    x0, y0, dx, dy = geo
    r = (N - y0) / dy - 0.5
    c = (E - x0) / dx - 0.5
    inside = (r >= -0.5) & (r <= img.shape[0] - 0.5) & (c >= -0.5) & (c <= img.shape[1] - 0.5)
    chans = [img] if img.ndim == 2 else [img[..., k] for k in range(img.shape[2])]
    out = np.stack([map_coordinates(ch, [r, c], order=order, mode="nearest", prefilter=False)
                    for ch in chans], axis=-1)
    return out, inside


def _coarse_window(grid, px, to_map, geo, shape, step: int = 16):
    """Window of a raster covering the whole target, from every ``step``-th target pixel."""
    xs, ys = _target(grid, px)
    X, Y = np.meshgrid(np.r_[xs[::step], xs[-1]], np.r_[ys[::step], ys[-1]])
    lon, lat = grid._tr(True).transform(X, Y)
    E, N = to_map(lon, lat)
    return _window(geo, shape, E, N, pad=int(step * px / abs(geo[2])) + 4)


def _window(geo, shape, E, N, pad: int = 2):
    """Pixel window of a raster (geo, shape) that contains all of (E, N), or None."""
    x0, y0, dx, dy = geo
    c = (E - x0) / dx
    r = (N - y0) / dy
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    if not ok.any():
        return None
    return ((max(int(r[ok].min()) - pad, 0), min(int(r[ok].max()) + pad + 1, shape[0])),
            (max(int(c[ok].min()) - pad, 0), min(int(c[ok].max()) + pad + 1, shape[1])))


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                  "User-Agent": "AeroSim imagery loader"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


# ------------------------------------------------------------------ Sentinel-2
def sentinel2_items(grid: TerrainGrid, date: str) -> list[dict]:
    """Earth Search items of one acquisition day over the grid, the best copy of each MGRS tile
    (lowest no-data share; Earth Search sometimes lists two processings)."""
    body = {"collections": ["sentinel-2-l2a"], "bbox": _lonlat_bbox(grid),
            "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z", "limit": 50}
    best = {}
    for f in _post(EARTH_SEARCH, body)["features"]:
        p = f["properties"]
        tile = p.get("grid:code") or p.get("s2:mgrs_tile")
        key = (tile, p.get("s2:nodata_pixel_percentage", 0))
        if tile not in best or key[1] < best[tile][0]:
            best[tile] = (key[1], f)
    return [f for _, f in sorted(best.values(), key=lambda v: v[0])]


def sentinel2(grid: TerrainGrid, date: str, *, px: float = 10.0, label: str = "", key: str = "s2",
              gamma: float = 0.8, gain: float = 1.0, cache: Path = CACHE / "sentinel2") -> ImageLayer:
    """True-colour mosaic of the Sentinel-2 L2A ``TCI`` images taken on ``date`` (YYYY-MM-DD).

    TCI maps surface reflectance 0..~0.3 to 0..255 and looks dark over land; ``gamma`` < 1 and
    ``gain`` lift the mid-tones (the same curve everywhere, so colours stay comparable)."""
    pyproj = require("pyproj", "terrain")
    tifffile = require("tifffile", "terrain")
    items = sentinel2_items(grid, date)
    if not items:
        raise ValueError(f"no Sentinel-2 L2A images on {date} over this grid")
    H, W = _shape(grid, px)
    out = np.zeros((H, W, 3), np.uint8)
    filled = np.zeros((H, W), bool)
    used = []
    for f in items:
        href = f["assets"]["visual"]["href"]
        p = f["properties"]
        epsg = int(str(p.get("proj:epsg") or p.get("proj:code", "")).split(":")[-1])
        path = _download(href, cache / f"{f['id']}_TCI.tif")
        tf = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        with tifffile.TiffFile(path) as t:
            p0 = t.pages[0]
            shape = (p0.imagelength, p0.imagewidth)
            geo = _geo(p0)
            page, f_ = 0, 1                       # coarsest overview still finer than px / 2
            for k in range(1, len(t.pages)):
                if geo[2] * shape[1] / t.pages[k].imagewidth <= px / 2:
                    page, f_ = k, shape[1] / t.pages[k].imagewidth
        n_used = 0
        win = _coarse_window(grid, px, tf.transform, (geo[0], geo[1], geo[2] * f_, geo[3] * f_),
                             (round(shape[0] / f_), round(shape[1] / f_)))
        if win is None:
            continue
        img, g = read_tiff_window(path, *win, page=page)       # decoded once per scene
        for rows, lon, lat in _chunks(grid, px):
            E, N = tf.transform(lon, lat)
            v, inside = _sample(img, g, E, N)
            ok = inside & (v.sum(-1) > 0) & ~filled[rows]            # 0, 0, 0 is no-data
            out[rows][ok] = v[ok].astype(np.uint8)
            filled[rows] |= ok
            n_used += int(ok.sum())
        if n_used:
            used.append(f["id"])
        if filled.all():
            break
    lin = np.clip((out.astype(np.float32) / 255.0) * gain, 0, 1) ** gamma
    rgb = np.where(filled[..., None], (lin * 255 + 0.5).astype(np.uint8), 0)
    year = date[:4]
    return ImageLayer(key, label or f"Sentinel-2, {date}", CREDITS["sentinel2"].format(year=year), rgb,
                      {"source": "sentinel2", "date": date, "scenes": used, "pixel_m": px,
                       "coverage": float(filled.mean()), "gamma": gamma, "gain": gain})


# ------------------------------------------------------------------ Kartverket N50 Raster
class _HTTPFile(io.RawIOBase):
    """Seekable read-only HTTP file (range requests), so zipfile can read one member."""

    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120)
        self.url, self.pos = r.geturl(), 0
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _n50_order(codes: list[str]) -> list[dict]:
    body = {"email": "", "orderLines": [{"metadataUuid": N50_UTM32,
            "areas": [{"code": c, "type": "celle", "name": c} for c in codes],
            "projections": [{"code": "25832"}], "formats": [{"name": "TIFF"}]}]}
    return _post(GEONORGE_ORDER, body)["files"]


def n50_cells(e0: float, n0: float, e1: float, n1: float, cache: Path = CACHE / "n50") -> dict[str, tuple]:
    """N50 Raster (UTM32) cells overlapping an EPSG:25832 box: {code: (west, north)}.

    Cells are 20 x 20 km, numbered west to east along rows from the south, only where there is
    land, so the numbers have no formula: each cell's corner is read from the ``.tfw`` in its zip
    (a few kB by range request) and the table is cached."""
    path = cache / "cells_utm32.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    code = 2001
    while True:
        need_more = not table or max(v[1] for v in table.values()) <= n1 + 20000
        if not need_more:
            break
        batch = [str(c) for c in range(code, code + 40) if str(c) not in table]
        code += 40
        if code > 2800:
            break
        if not batch:
            continue
        for f in _n50_order(batch):
            for k in range(4):                    # the download service is sometimes slow
                try:
                    z = zipfile.ZipFile(io.BufferedReader(_HTTPFile(f["downloadUrl"]), 1 << 16))
                    tfw = next(n for n in z.namelist() if n.lower().endswith(".tfw"))
                    v = z.read(tfw).decode().split()
                    break
                except (OSError, zipfile.BadZipFile):
                    if k == 3:
                        raise
                    time.sleep(5 * (k + 1))
            table[f["area"]] = (round(float(v[4]) - float(v[0]) / 2), round(float(v[5]) - float(v[3]) / 2))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table), encoding="utf-8")
    return {c: tuple(v) for c, v in table.items()
            if v[0] < e1 and v[0] + 20000 > e0 and v[1] > n0 and v[1] - 20000 < n1}


def kartverket_n50(grid: TerrainGrid, *, px: float = 6.0, key: str = "n50",
                   label: str = "Topographic map (Kartverket N50)", cache: Path = CACHE / "n50") -> ImageLayer:
    """Kartverket's 1:50 000 map (N50 Raster, 2.5 m) for grids in Norway (UTM zone 32 cells)."""
    pyproj = require("pyproj", "terrain")
    from PIL import Image
    tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    lo0, la0, lo1, la1 = _lonlat_bbox(grid)
    E, N = tf.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    cells = n50_cells(min(E), min(N), max(E), max(N), cache)
    H, W = _shape(grid, px)
    out = np.zeros((H, W, 3), np.uint8)
    filled = np.zeros((H, W), bool)
    files = {f["area"]: f for f in _n50_order(sorted(cells))}
    Image.MAX_IMAGE_PIXELS = None
    for c, (west, north) in sorted(cells.items()):
        zpath = _download(files[c]["downloadUrl"], cache / f"n50_utm32_{c}.zip", magic=(b"PK",))
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".tif"))
            img = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
        geo = (west, north, 20000 / img.shape[1], -20000 / img.shape[0])
        for rows, lon, lat in _chunks(grid, px):
            e, n = tf.transform(lon, lat)
            v, inside = _sample(img, geo, e, n)
            ok = inside & ~filled[rows]
            out[rows][ok] = v[ok].astype(np.uint8)
            filled[rows] |= ok
        del img
    return ImageLayer(key, label, CREDITS["n50"], out,
                      {"source": "kartverket-n50", "cells": sorted(cells), "pixel_m": px,
                       "coverage": float(filled.mean())})


# ------------------------------------------------------------------ ESA WorldCover
def worldcover(grid: TerrainGrid, *, px: float = 10.0, key: str = "landcover",
               label: str = "Land cover (ESA WorldCover 2021)", cache: Path = CACHE / "worldcover") -> ImageLayer:
    """ESA WorldCover 2021 classes in the product's own colours (its GeoTIFF palette)."""
    tifffile = require("tifffile", "terrain")
    lo0, la0, lo1, la1 = _lonlat_bbox(grid)
    H, W = _shape(grid, px)
    out = np.zeros((H, W, 3), np.uint8)
    filled = np.zeros((H, W), bool)
    tiles, seen = [], set()
    for la in range(math.floor(la0 / 3) * 3, math.floor(la1 / 3) * 3 + 1, 3):
        for lo in range(math.floor(lo0 / 3) * 3, math.floor(lo1 / 3) * 3 + 1, 3):
            name = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            path = _download(WORLDCOVER.format(tile=name), cache / f"WorldCover_2021_v200_{name}.tif")
            tiles.append(name)
            with tifffile.TiffFile(path) as t:
                p0 = t.pages[0]
                cmap = np.asarray(p0.colormap).T                   # (256, 3), 16-bit
                shape = (p0.imagelength, p0.imagewidth)
                geo = _geo(p0)
            pal = (cmap // 257).astype(np.uint8)
            win = _coarse_window(grid, px, lambda lo, la: (lo, la), geo, shape)
            if win is None:
                continue
            cls, g = read_tiff_window(path, *win)                  # decoded once per tile
            for rows, lon, lat in _chunks(grid, px):
                v, inside = _sample(cls, g, lon, lat, order=0)
                ok = inside & ~filled[rows]
                cv = v[..., 0][ok].astype(int)
                out[rows][ok] = pal[cv]
                seen.update(np.unique(cv).tolist())
                filled[rows] |= ok
    return ImageLayer(key, label, CREDITS["worldcover"], out,
                      {"source": "esa-worldcover-2021-v200", "tiles": tiles, "pixel_m": px,
                       "coverage": float(filled.mean()),
                       "legend": [[WORLDCOVER_CLASSES[c], "#" + bytes(pal[c]).hex()]
                                  for c in sorted(seen) if c in WORLDCOVER_CLASSES]})
