"""Digital elevation models on a local metric grid.

    g = TerrainGrid.kartverket(59.0, 6.35, size=(48e3, 26e3), spacing=20)   # Norway, 1 m DTM
    g = TerrainGrid.copernicus(73.35, 55.3, size=(110e3, 40e3), spacing=30) # anywhere, 30 m DSM
    g.elevation(x, y)            # bilinear height (m), x east / y north of the centre
    g.to_xy(lat, lon), g.to_latlon(x, y)

Every grid lives on its own transverse Mercator projection centred on (lat0, lon0) (WGS84,
scale 1), so x/y are true ground distances to within 1e-4 over the 100 km regions used here.

Sources (both free, cached under ``outputs/terrain``):

- **Kartverket** "Nasjonal høydemodell" digital terrain model (bare earth, from airborne
  laser scanning, 1 m grid, heights above NN2000), served by Geonorge's WCS for
  EPSG:25833 (ETRS89 / UTM 33N). Requested in chunks at ``spacing`` and resampled here.
  © Kartverket, CC BY 4.0.
- **Copernicus DEM GLO-30** (TanDEM-X based surface model, 1 arc-second, heights above
  EGM2008), public COG tiles on AWS (``copernicus-dem-30m``). Over Novaya Zemlya the surface
  model is the terrain (no forest; glacier surface included). Produced using Copernicus
  WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
  COPERNICUS by the European Union and ESA; all rights reserved. Free to use, modify and
  redistribute under the Copernicus WorldDEM-30 licence, with that notice and its liability
  sentence (``COPERNICUS_LIABILITY``).

The tiles are Deflate-compressed with TIFF's floating-point predictor, which tifffile can only
decode with the optional ``imagecodecs`` package, so ``read_cog`` undoes it here (zlib,
then the byte-wise difference, then the big-endian byte planes).

Sea: both models put the open sea at about 0 m; ``water`` marks hydro-flattened cells (sea and
lakes) as those whose 3x3 neighbourhood varies by less than 5 cm.
"""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosim.core import require

CACHE = Path(__file__).resolve().parents[2] / "outputs" / "terrain"
PACK_STEP = 0.1         # m, height step of packed grids (``TerrainGrid.save(packed=True)``)
KARTVERKET_WCS = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
KARTVERKET_COVERAGE = "nhm_dtm_topo_25833"
COPERNICUS_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
ATTRIBUTION = {
    "kartverket": "Terrain: Nasjonal høydemodell DTM © Kartverket (CC BY 4.0)",
    "copernicus": ("Terrain: Copernicus DEM GLO-30, produced using Copernicus WorldDEM-30 "
                   "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
                   "provided under COPERNICUS by the European Union and ESA; all rights reserved"),
}
# required with any redistribution of Copernicus heights (licence, Article 6(c))
COPERNICUS_LIABILITY = ("The organisations in charge of the Copernicus programme by law or by "
                        "delegation do not incur any liability for any use of the Copernicus WorldDEM-30.")


def _download(url: str, path: Path, tries: int = 4) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AeroSim terrain loader"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if not data.startswith((b"II*\x00", b"MM\x00*")):
                raise ValueError(f"not a TIFF: {data[:200]!r}")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return path
        except Exception as e:
            if k == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise
            time.sleep(2 * (k + 1))
    return path


def read_cog(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a single-band float32 GeoTIFF (tiled or striped, any compression tifffile handles
    plus Deflate + floating-point predictor). Returns (array, (x0, y0, dx, dy)) where (x0, y0)
    is the outer corner of pixel (0, 0) and dy is negative for north-up rasters."""
    tifffile = require("tifffile", "terrain")
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        scale = p.tags["ModelPixelScaleTag"].value
        tie = p.tags["ModelTiepointTag"].value
        geo = (tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1], scale[0], -scale[1])
        keys = p.tags["GeoKeyDirectoryTag"].value
        keys = {keys[i]: keys[i + 3] for i in range(4, len(keys), 4)}
        if keys.get(1025) == 2:                   # PixelIsPoint: the tie point is a pixel centre
            geo = (geo[0] - geo[2] / 2, geo[1] - geo[3] / 2, geo[2], geo[3])
        if int(p.predictor) != 3:
            return p.asarray().astype(np.float32), geo
        H, W = p.shape
        tw, tl = (p.tilewidth, p.tilelength) if p.is_tiled else (W, p.rowsperstrip)
        out = np.zeros((H, W), np.float32)
        nx = -(-W // tw)
        fh = t.filehandle
        for i, (off, n) in enumerate(zip(p.dataoffsets, p.databytecounts)):
            fh.seek(off)
            b = np.frombuffer(zlib.decompress(fh.read(n)), np.uint8)
            rows = b.size // (tw * 4)
            b = np.cumsum(b.reshape(rows, tw * 4), axis=1, dtype=np.uint8)   # undo differencing
            v = b.reshape(rows, 4, tw).transpose(0, 2, 1)[:, :, ::-1].copy().view("<f4")[..., 0]
            r, c = divmod(i, nx)
            hh, ww = min(rows, H - r * tl), min(tw, W - c * tw)
            out[r * tl:r * tl + hh, c * tw:c * tw + ww] = v[:hh, :ww]
        return out, geo


def _bilinear(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    return map_coordinates(a, [rows, cols], order=1, mode="nearest").astype(np.float32)


@dataclass
class TerrainGrid:
    """Heights z[j, i] at x[i] (east), y[j] (north) in m on a local transverse Mercator."""

    lat0: float
    lon0: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source: str = ""
    meta: dict = field(default_factory=dict)

    # -- projection -----------------------------------------------------------------
    @property
    def crs(self) -> str:
        return (f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} +k=1 +x_0=0 +y_0=0 "
                "+ellps=WGS84 +units=m +no_defs")

    def _tr(self, inverse: bool):
        pyproj = require("pyproj", "terrain")
        key = "_inv" if inverse else "_fwd"
        if key not in self.__dict__:
            a, b = ("EPSG:4326", self.crs) if not inverse else (self.crs, "EPSG:4326")
            self.__dict__[key] = pyproj.Transformer.from_crs(a, b, always_xy=True)
        return self.__dict__[key]

    def to_xy(self, lat, lon):
        return self._tr(False).transform(lon, lat)

    def to_latlon(self, x, y):
        lon, lat = self._tr(True).transform(x, y)
        return lat, lon

    def convergence(self, lat, lon):
        """Grid azimuth of true north, rad (grid azimuth = true azimuth + this): the meridian
        convergence of the transverse Mercator, about 1.5 deg at 1.6 deg of longitude off the
        centre at 73 N. Matches pyproj's meridian_convergence (opposite sign convention)."""
        return -np.arctan(np.tan(np.radians(np.asarray(lon) - self.lon0)) * np.sin(np.radians(lat)))

    # -- lookup ---------------------------------------------------------------------
    @property
    def spacing(self) -> float:
        return float(self.x[1] - self.x[0])

    def elevation(self, x, y):
        """Bilinear height at (x, y); clamps to the edge outside the grid."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        fi = np.clip((x - self.x[0]) / self.spacing, 0, self.x.size - 1.000001)
        fj = np.clip((y - self.y[0]) / self.spacing, 0, self.y.size - 1.000001)
        i, j = fi.astype(int), fj.astype(int)
        u, v = fi - i, fj - j
        z = self.z
        out = (z[j, i] * (1 - u) * (1 - v) + z[j, i + 1] * u * (1 - v)
               + z[j + 1, i] * (1 - u) * v + z[j + 1, i + 1] * u * v)
        return out if out.ndim else float(out)

    def elevation_at(self, x: float, y: float) -> float:
        """Scalar fast path of ``elevation`` (called every simulator step)."""
        d = self.x[1] - self.x[0]
        fi = min(max((x - self.x[0]) / d, 0.0), self.x.size - 1.000001)
        fj = min(max((y - self.y[0]) / d, 0.0), self.y.size - 1.000001)
        i, j = int(fi), int(fj)
        u, v = fi - i, fj - j
        z = self.z
        return float((z[j, i] * (1 - u) + z[j, i + 1] * u) * (1 - v)
                     + (z[j + 1, i] * (1 - u) + z[j + 1, i + 1] * u) * v)

    def contains(self, x, y, margin: float = 0.0) -> bool:
        return (self.x[0] + margin <= x <= self.x[-1] - margin
                and self.y[0] + margin <= y <= self.y[-1] - margin)

    @property
    def water(self) -> np.ndarray:
        """Hydro-flattened cells (sea, lakes): 3x3 range below 5 cm."""
        from scipy.ndimage import maximum_filter, minimum_filter
        z = self.z
        return (maximum_filter(z, 3) - minimum_filter(z, 3)) < 0.05

    def downsample(self, k: int) -> TerrainGrid:
        """Every k-th sample after a k x k block mean (for display meshes)."""
        from scipy.ndimage import uniform_filter
        zf = uniform_filter(self.z, k, mode="nearest") if k > 1 else self.z
        return TerrainGrid(self.lat0, self.lon0, self.x[::k].copy(), self.y[::k].copy(),
                           zf[::k, ::k].copy(), self.source, dict(self.meta))

    # -- persistence ----------------------------------------------------------------
    def save(self, path: str | Path, packed: bool = False) -> Path:
        """``packed``: heights in steps of PACK_STEP (0.1 m), differenced along rows and
        LZMA-compressed; about a quarter of the float32 file, for keeping grids in the repo.
        The 0.05 m rounding is far below either source's accuracy (metres)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not packed:
            np.savez_compressed(path, lat0=self.lat0, lon0=self.lon0, x=self.x, y=self.y,
                                z=self.z, source=self.source)
            return path
        import lzma
        q = np.round(self.z / PACK_STEP).astype(np.int32)
        if q.min() < -32768 or q.max() > 32767:
            raise ValueError("heights outside the packed range (+-3276 m)")
        d = np.diff(q, axis=1, prepend=0).astype("<i2")
        np.savez(path, lat0=self.lat0, lon0=self.lon0, x0=self.x[0], y0=self.y[0],
                 spacing=self.spacing, shape=np.array(self.z.shape), step=PACK_STEP,
                 zq=np.frombuffer(lzma.compress(d.tobytes(), preset=9), np.uint8), source=self.source)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TerrainGrid:
        d = np.load(path)
        if "zq" not in d:
            return cls(float(d["lat0"]), float(d["lon0"]), d["x"], d["y"], d["z"], str(d["source"]))
        import lzma
        ny, nx = (int(v) for v in d["shape"])
        q = np.frombuffer(lzma.decompress(d["zq"].tobytes()), "<i2").reshape(ny, nx)
        z = (np.cumsum(q, axis=1, dtype=np.int32) * float(d["step"])).astype(np.float32)
        h = float(d["spacing"])
        return cls(float(d["lat0"]), float(d["lon0"]), float(d["x0"]) + h * np.arange(nx),
                   float(d["y0"]) + h * np.arange(ny), z, str(d["source"]))

    # -- builders -------------------------------------------------------------------
    @classmethod
    def _empty(cls, lat0, lon0, size, spacing, source):
        nx, ny = round(size[0] / spacing) + 1, round(size[1] / spacing) + 1
        x = (np.arange(nx) - (nx - 1) / 2) * spacing
        y = (np.arange(ny) - (ny - 1) / 2) * spacing
        return cls(lat0, lon0, x, y, np.zeros((ny, nx), np.float32), source)

    def _lonlat_mesh(self):
        X, Y = np.meshgrid(self.x, self.y)
        lon, lat = self._tr(True).transform(X, Y)
        return lon, lat

    @classmethod
    def kartverket(cls, lat0: float, lon0: float, size: tuple[float, float],
                   spacing: float = 20.0, cache: Path = CACHE, chunk: int = 1500) -> TerrainGrid:
        """Kartverket 1 m DTM resampled by the WCS to ``spacing`` (in UTM 33), then bilinearly
        onto the local grid. ``size`` = (east-west, north-south) extent in m."""
        pyproj = require("pyproj", "terrain")
        g = cls._empty(lat0, lon0, size, spacing, "kartverket")
        lon, lat = g._lonlat_mesh()
        E, N = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True).transform(lon, lat)
        pad = 4 * spacing
        e0 = math.floor((E.min() - pad) / spacing) * spacing
        n1 = math.ceil((N.max() + pad) / spacing) * spacing
        W = math.ceil((E.max() + pad - e0) / spacing)
        H = math.ceil((n1 - (N.min() - pad)) / spacing)
        src = np.zeros((H, W), np.float32)
        for r0 in range(0, H, chunk):
            for c0 in range(0, W, chunk):
                h, w = min(chunk, H - r0), min(chunk, W - c0)
                bb = (e0 + c0 * spacing, n1 - (r0 + h) * spacing, e0 + (c0 + w) * spacing, n1 - r0 * spacing)
                name = f"kv_{spacing:g}_{bb[0]:.0f}_{bb[1]:.0f}_{bb[2]:.0f}_{bb[3]:.0f}.tif"
                url = (f"{KARTVERKET_WCS}?service=WCS&version=1.0.0&request=GetCoverage"
                       f"&coverage={KARTVERKET_COVERAGE}&format=GeoTIFF&crs=EPSG:25833"
                       f"&bbox={bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}&width={w}&height={h}")
                a, geo = read_cog(_download(url, cache / "kartverket" / name))
                if a.shape != (h, w) or abs(geo[0] - bb[0]) > 1e-3 or abs(geo[1] - bb[3]) > 1e-3:
                    raise ValueError(f"unexpected WCS tile {name}: {a.shape} at {geo}")
                src[r0:r0 + h, c0:c0 + w] = a
        bad = ~np.isfinite(src) | (src < -500) | (src > 3000)    # no-data outside Norway
        src[bad] = 0.0
        # WCS pixel (r, c) covers [e0 + c dx, e0 + (c+1) dx]; its centre is at +0.5
        g.z = _bilinear(src, (n1 - N) / spacing - 0.5, (E - e0) / spacing - 0.5)
        g.meta = {"nodata_fraction": float(bad.mean()), "wcs_spacing": spacing}
        return g

    @classmethod
    def copernicus(cls, lat0: float, lon0: float, size: tuple[float, float],
                   spacing: float = 30.0, cache: Path = CACHE) -> TerrainGrid:
        """Copernicus GLO-30 tiles covering the grid, mosaicked and resampled bilinearly.
        Missing tiles (open ocean) are sea level."""
        g = cls._empty(lat0, lon0, size, spacing, "copernicus")
        lon, lat = g._lonlat_mesh()
        z = np.zeros(lon.shape, np.float32)
        missing = []
        for la in range(math.floor(lat.min()), math.floor(lat.max()) + 1):
            for lo in range(math.floor(lon.min()), math.floor(lon.max()) + 1):
                ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
                n = f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
                try:
                    a, (x0, y0, dx, dy) = read_cog(_download(COPERNICUS_URL.format(n=n),
                                                              cache / "copernicus" / f"{n}.tif"))
                except urllib.error.HTTPError as e:   # 403 / 404 from S3: no tile (ocean)
                    if e.code in (403, 404):
                        missing.append(n)
                        continue
                    raise
                m = (lat >= la) & (lat < la + 1) & (lon >= lo) & (lon < lo + 1)
                z[m] = _bilinear(a, (lat[m] - y0) / dy - 0.5, (lon[m] - x0) / dx - 0.5)
        g.z = z
        g.meta = {"missing_tiles": missing}
        return g
