"""Solid grain geometries: burning area, port area and propellant volume vs web burned.

Every grain regresses uniformly, normal to its burning surface, by the same web
``w`` on the core and on each uninhibited end face:

    Ab(w) = core_perimeter(w) * L(w) + ends * (A_case - A_port(w)),   L(w) = L0 - ends * w

* Closed form: ``Bates``, ``RodAndTube``, ``EndBurner``.
* Any other core shape (``MapGrain`` subclasses) is drawn on a square map of the
  cross-section. A fast-marching distance transform (scikit-fmm) gives each
  propellant cell's distance from the initial port, so the surface after a web
  ``w`` is the level set ``dist == w``; its length (scikit-image marching squares)
  is the core perimeter and the cells with ``dist < w`` are the port. Both are
  tabulated once per grain and interpolated. Needs ``pip install -e '.[grains]'``.

Grain geometry is defined in a cross-section centred on the motor axis, x to the
right, all lengths in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from aerosim.core import require

BURNING_ENDS = {"neither": 2, "top": 1, "bottom": 1, "both": 0}


@dataclass
class Grain:
    diameter: float                # m, outer (case-bonded) diameter
    length: float                  # m
    inhibited_ends: str = "Neither"   # Neither | Top | Bottom | Both

    @property
    def ends(self) -> int:
        """Number of burning end faces."""
        return BURNING_ENDS[self.inhibited_ends.lower()]

    @property
    def case_area(self) -> float:
        return math.pi / 4 * self.diameter**2

    @property
    def web_max(self) -> float:
        """Web at which the grain is fully consumed (radially or axially)."""
        axial = self.length / self.ends if self.ends else math.inf
        return min(self.radial_web, axial)

    def length_at(self, w: float) -> float:
        return max(self.length - self.ends * w, 0.0)

    def burn_area(self, w: float) -> float:
        if w >= self.web_max:
            return 0.0
        return (self.core_perimeter(w) * self.length_at(w)
                + self.ends * (self.case_area - self.port_area(w)))

    def volume(self, w: float) -> float:
        if w >= self.web_max:
            return 0.0
        return (self.case_area - self.port_area(w)) * self.length_at(w)

    # implemented per shape
    radial_web: float = field(init=False, default=0.0)

    def core_perimeter(self, w: float) -> float:
        raise NotImplementedError

    def port_area(self, w: float) -> float:
        raise NotImplementedError


@dataclass
class Bates(Grain):
    """Cylindrical core."""

    core_diameter: float = 0.0

    def __post_init__(self) -> None:
        self.radial_web = (self.diameter - self.core_diameter) / 2

    def core_perimeter(self, w):
        return math.pi * (self.core_diameter + 2 * w)

    def port_area(self, w):
        return math.pi / 4 * min(self.core_diameter + 2 * w, self.diameter) ** 2


@dataclass
class RodAndTube(Grain):
    """Tube with a cylindrical core plus a free-standing rod on the axis; the annular
    port between them burns outward into the tube and inward into the rod."""

    core_diameter: float = 0.0     # tube inner diameter
    rod_diameter: float = 0.0

    def __post_init__(self) -> None:
        self._tube_web = (self.diameter - self.core_diameter) / 2
        self._rod_web = self.rod_diameter / 2
        self.radial_web = max(self._tube_web, self._rod_web)

    def core_perimeter(self, w):
        p = math.pi * (self.core_diameter + 2 * w) if w < self._tube_web else 0.0
        return p + (math.pi * (self.rod_diameter - 2 * w) if w < self._rod_web else 0.0)

    def port_area(self, w):
        tube = math.pi / 4 * (self.diameter**2 - (self.core_diameter + 2 * w) ** 2)
        rod = math.pi / 4 * max(self.rod_diameter - 2 * w, 0.0) ** 2
        return self.case_area - max(tube, 0.0) - rod


@dataclass
class EndBurner(Grain):
    """Cigarette burner: no core, one burning face regardless of inhibitors."""

    def __post_init__(self) -> None:
        self.inhibited_ends = "Top"
        self.radial_web = math.inf

    def core_perimeter(self, w):
        return 0.0

    def port_area(self, w):
        return 0.0


_MAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}


@dataclass
class MapGrain(Grain):
    """Arbitrary core shape regressed on a ``map_dim`` x ``map_dim`` grid."""

    map_dim: int = field(default=501, kw_only=True)
    levels: int = field(default=80, kw_only=True)

    def __post_init__(self) -> None:
        self._tables = None
        self.radial_web = math.inf   # set once the map is built

    def port_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True where the *initial* port (no propellant) is."""
        raise NotImplementedError

    def _key(self) -> tuple:
        skip = {"length", "inhibited_ends", "levels"}
        return (type(self).__name__,) + tuple(
            (k, repr(v)) for k, v in sorted(self.__dict__.items())
            if not k.startswith("_") and k not in skip and k != "radial_web")

    def build(self) -> MapGrain:
        """Run the fast-marching regression and tabulate perimeter / port area."""
        if self._tables is not None:
            return self
        key = self._key()
        if key not in _MAP_CACHE:
            _MAP_CACHE[key] = self._compute()
        self._tables = _MAP_CACHE[key]
        self.radial_web = float(self._tables[3])
        return self

    def _compute(self):
        skfmm = require("skfmm", "grains")
        measure = require("skimage.measure", "grains")
        R, n = self.diameter / 2, self.map_dim
        ax = np.linspace(-R, R, n)
        dx = ax[1] - ax[0]
        x, y = np.meshgrid(ax, ax)
        inside = x**2 + y**2 <= R**2
        port = self.port_mask(x, y) & inside
        if not port.any():
            raise ValueError(f"{type(self).__name__}: core does not fit on the map")
        # smooth the rasterised edge (1.5 cells) so the zero level set is not a staircase;
        # this keeps the initial perimeter within ~0.2 % of the true outline
        phi = np.ma.MaskedArray(0.5 - gaussian_filter(port.astype(float), 1.5), mask=~inside)
        dist = np.ma.filled(skfmm.distance(phi, dx=float(dx)), -1.0)
        web_max = float(dist[inside].max())
        cell = self.case_area / inside.sum()      # per-cell area, exact total
        webs = np.linspace(0.0, web_max, self.levels)
        perim, area = np.zeros_like(webs), np.zeros_like(webs)
        for i, w in enumerate(webs[:-1]):          # last level: burned out, perimeter 0
            area[i] = np.count_nonzero(inside & (dist < w)) * cell
            for c in measure.find_contours(dist, w, mask=inside):
                perim[i] += np.sum(np.hypot(*np.diff(c, axis=0).T)) * dx
        area[-1] = self.case_area
        return webs, perim, area, web_max

    def core_perimeter(self, w):
        webs, perim, _, _ = self.build()._tables
        return float(np.interp(w, webs, perim, right=0.0))

    def port_area(self, w):
        webs, _, area, _ = self.build()._tables
        return float(np.interp(w, webs, area, right=self.case_area))

    @property
    def web_max(self) -> float:
        self.build()
        return super().web_max


def _rotations(x, y, count: int):
    """Yield (x', y') in frames rotated to each of ``count`` equally spaced spokes."""
    for k in range(count):
        a = 2 * math.pi * k / count
        yield x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)


@dataclass
class Finocyl(MapGrain):
    """Round core with ``num_fins`` rectangular slots extending ``fin_length`` past it."""

    core_diameter: float = 0.0
    fin_length: float = 0.0
    fin_width: float = 0.0
    num_fins: int = 0

    def port_mask(self, x, y):
        rc = self.core_diameter / 2
        m = x**2 + y**2 <= rc**2
        for xr, yr in _rotations(x, y, self.num_fins):
            m |= (np.abs(yr) <= self.fin_width / 2) & (xr >= 0) & (xr <= rc + self.fin_length)
        return m


@dataclass
class Star(MapGrain):
    """``num_points`` triangles, base ``point_width`` on the axis, tip at ``point_length``."""

    num_points: int = 0
    point_length: float = 0.0
    point_width: float = 0.0

    def port_mask(self, x, y):
        m = np.zeros_like(x, dtype=bool)
        for xr, yr in _rotations(x, y, self.num_points):
            m |= (xr >= 0) & (xr <= self.point_length) & (
                np.abs(yr) <= self.point_width / 2 * (1 - xr / self.point_length))
        return m


@dataclass
class XCore(MapGrain):
    """Two perpendicular slots of half-length ``slot_length`` and width ``slot_width``."""

    slot_length: float = 0.0
    slot_width: float = 0.0

    def port_mask(self, x, y):
        h = self.slot_width / 2
        return (((np.abs(x) <= self.slot_length) & (np.abs(y) <= h))
                | ((np.abs(y) <= self.slot_length) & (np.abs(x) <= h)))


@dataclass
class CGrain(MapGrain):
    """Slot of width ``slot_width`` from ``slot_offset`` off the axis out to the case."""

    slot_width: float = 0.0
    slot_offset: float = 0.0

    def port_mask(self, x, y):
        return (np.abs(y) <= self.slot_width / 2) & (x >= self.slot_offset)


@dataclass
class DGrain(MapGrain):
    """Circle with the segment beyond ``slot_offset`` from the axis removed."""

    slot_offset: float = 0.0

    def port_mask(self, x, y):
        return x >= self.slot_offset


@dataclass
class MoonBurner(MapGrain):
    """Round core shifted ``core_offset`` off the axis."""

    core_diameter: float = 0.0
    core_offset: float = 0.0

    def port_mask(self, x, y):
        return (x - self.core_offset) ** 2 + y**2 <= (self.core_diameter / 2) ** 2


@dataclass
class CustomGrain(MapGrain):
    """Port outline(s) as closed polygons of (x, y) points in metres."""

    polygons: list = field(default_factory=list)

    def port_mask(self, x, y):
        measure = require("skimage.measure", "grains")
        pts = np.column_stack([x.ravel(), y.ravel()])
        m = np.zeros(pts.shape[0], dtype=bool)
        for poly in self.polygons:
            m |= measure.points_in_poly(pts, np.asarray(poly, float))
        return m.reshape(x.shape)


GRAIN_TYPES: dict[str, type[Grain]] = {
    "bates": Bates, "rod_and_tube": RodAndTube, "end_burner": EndBurner,
    "finocyl": Finocyl, "star": Star, "x_core": XCore, "c_grain": CGrain,
    "d_grain": DGrain, "moon_burner": MoonBurner, "custom": CustomGrain,
}


def make_grain(spec: Grain | dict) -> Grain:
    """``{"type": "finocyl", "diameter": ..., ...}`` -> Grain (configs use this)."""
    if isinstance(spec, Grain):
        return spec
    spec = dict(spec)
    kind = spec.pop("type").lower().replace(" ", "_").replace("-", "_")
    if kind not in GRAIN_TYPES:
        raise ValueError(f"unknown grain type {kind!r}; choose from {sorted(GRAIN_TYPES)}")
    return GRAIN_TYPES[kind](**spec)
