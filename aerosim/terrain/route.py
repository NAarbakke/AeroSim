"""Low-level routes over a TerrainGrid: where to fly (ground track) and how high (profile).

    track = valley_route(grid, [(lat, lon), ...], min_turn_radius=1100)   # follows low ground
    track = direct_route(grid, [(lat, lon), ...], min_turn_radius=1100)   # straight legs
    prof = terrain_profile(grid, track, speed=170, clearance=100)        # altitude reference

Ground track
    ``valley_route`` is a least-cost path on the grid (skimage MCP, 8-connected, geometric):
    a cell costs 1 + (height / ``height_scale``)^2 + ``wall_weight`` exp(-d / ``wall_distance``),
    d = distance to ground higher than ``wall_height``. The first term keeps the route on low
    ground (fjord, strait, valley floor); the second pushes it to the middle of the channel,
    away from the walls. ``direct_route`` joins the waypoints with straight lines.

    Either raw path is then smoothed into something the aircraft can fly: points p_i every
    ``step`` m minimise sum |p_i - raw_i|^2 subject to |p_{i+1} - 2 p_i + p_{i-1}| <= step^2 / R_min
    (a second-order cone programme, cvxpy). For evenly spaced points the second difference over
    step^2 is the curvature, so the constraint is the turn radius; the fidelity term keeps
    the track as close to the planned one as that allows. The result is resampled by arc length.

Altitude profile
    Terrain is sampled under the track and across a corridor ``corridor`` m either side; the
    floor is its maximum plus ``clearance``. The reference altitude h_i is the lowest
    profile above the floor that the aircraft can fly at ``speed``: a linear programme

        minimise sum h_i   subject to   h_i >= floor_i,
        -tan(dive) <= (h_{i+1} - h_i) / ds <= tan(climb),
        -(1 - n_push) g / V^2 <= (h_{i+1} - 2 h_i + h_{i-1}) / ds^2 <= (n_pull - 1) g / V^2,

    i.e. clearance, climb and dive angle, pull-up and push-over load factor. This is the
    classical "ideal" terrain-following profile: the aircraft starts climbing before a ridge
    early enough to clear it at the pull-up limit, and does not dive into a valley faster
    than it can push over and recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from aerosim.core import require
from aerosim.core.units import G0
from aerosim.terrain.dem import TerrainGrid


@dataclass
class Track:
    """Ground track: x, y (m, grid frame) every ``step`` m of arc length s."""

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    raw: np.ndarray | None = None          # (n, 2) planned path before smoothing
    meta: dict = field(default_factory=dict)

    @property
    def s(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def course(self) -> np.ndarray:
        """Course over ground, rad clockwise from north."""
        c = np.arctan2(np.gradient(self.x), np.gradient(self.y))
        return np.unwrap(c)

    @property
    def curvature(self) -> np.ndarray:
        """Signed curvature d(course)/ds, 1/m (positive = turning right)."""
        return np.gradient(self.course, self.s)

    def point(self, s: float) -> tuple[float, float, float, float]:
        """(x, y, course, curvature) at arc length s (clamped)."""
        S = self.s
        return (float(np.interp(s, S, self.x)), float(np.interp(s, S, self.y)),
                float(np.interp(s, S, self.course)), float(np.interp(s, S, self.curvature)))


@dataclass
class Profile:
    """Terrain-following reference along a track: everything indexed by arc length s."""

    s: np.ndarray
    terrain: np.ndarray          # under the track centre line
    floor: np.ndarray            # corridor maximum + clearance
    altitude: np.ndarray         # reference altitude h(s)
    gamma: np.ndarray            # reference flight-path angle, rad
    dgamma_ds: np.ndarray        # its rate of change along the track, rad/m
    meta: dict = field(default_factory=dict)

    def at(self, s: float) -> tuple[float, float, float]:
        S = self.s
        return (float(np.interp(s, S, self.altitude)), float(np.interp(s, S, self.gamma)),
                float(np.interp(s, S, self.dgamma_ds)))


# ------------------------------------------------------------------ ground tracks
def _cost(grid: TerrainGrid, height_scale: float, wall_height: float, wall_distance: float,
          wall_weight: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    z = np.maximum(grid.z, 0.0)
    d = distance_transform_edt(z < wall_height) * grid.spacing
    return 1.0 + (z / height_scale) ** 2 + wall_weight * np.exp(-d / wall_distance)


def _cells(grid: TerrainGrid, x: float, y: float) -> tuple[int, int]:
    i = round((x - grid.x[0]) / grid.spacing)
    j = round((y - grid.y[0]) / grid.spacing)
    return (min(max(j, 0), grid.y.size - 1), min(max(i, 0), grid.x.size - 1))


def _xy(grid, waypoints):
    return [tuple(float(v) for v in grid.to_xy(la, lo)) for la, lo in waypoints]


def valley_route(grid: TerrainGrid, waypoints: list[tuple[float, float]], *,
                 min_turn_radius: float, step: float = 50.0, height_scale: float = 60.0,
                 wall_height: float = 120.0, wall_distance: float = 250.0,
                 wall_weight: float = 4.0, name: str = "Valley route") -> Track:
    """Least-cost path through ``waypoints`` [(lat, lon), ...], smoothed to the turn radius."""
    graph = require("skimage.graph", "grains")
    cost = _cost(grid, height_scale, wall_height, wall_distance, wall_weight)
    pts = _xy(grid, waypoints)
    path = []
    for a, b in pairwise(pts):
        idx, _ = graph.route_through_array(cost, _cells(grid, *a), _cells(grid, *b),
                                           fully_connected=True, geometric=True)
        idx = np.asarray(idx)
        seg = np.c_[grid.x[idx[:, 1]], grid.y[idx[:, 0]]]
        path.append(seg if not path else seg[1:])
    raw = np.vstack(path)
    return _smooth(raw, min_turn_radius, step, name)


def direct_route(grid: TerrainGrid, waypoints: list[tuple[float, float]], *,
                 min_turn_radius: float, step: float = 50.0, name: str = "Direct route") -> Track:
    """Straight legs between ``waypoints``, corners rounded to the turn radius."""
    pts = np.asarray(_xy(grid, waypoints))
    raw = [pts[0]]
    for a, b in pairwise(pts):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        raw.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return _smooth(np.asarray(raw), min_turn_radius, step, name)


def _resample(p: np.ndarray, step: float) -> np.ndarray:
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(round(s[-1] / step), 2)
    t = np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def _circle_curvature(p: np.ndarray) -> np.ndarray:
    """Curvature of the circle through each three consecutive points (ends copied)."""
    a, b = p[1:-1] - p[:-2], p[2:] - p[1:-1]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    k = 2 * np.abs(cross) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                                       * np.linalg.norm(a + b, axis=1), 1e-9)
    return np.r_[k[0], k, k[-1]]


def _smooth(raw: np.ndarray, min_turn_radius: float, step: float, name: str,
            smooth_length: float = 600.0) -> Track:
    """Closest smooth curve to ``raw`` whose curvature never exceeds 1 / min_turn_radius.

    SOCP in the points; the spacing is held near ``step`` along the previous iterate's
    tangent, and the curvature bound is tightened until the resampled curve meets it."""
    cp = require("cvxpy", "optim")
    from scipy.ndimage import gaussian_filter1d
    q = _resample(raw, step / 4)                  # grid staircase -> its centre line
    q = _resample(gaussian_filter1d(q, 4, axis=0, mode="nearest"), step)
    n = len(q)
    p = cp.Variable((n, 2))
    kmax = cp.Parameter(nonneg=True)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    lam = (smooth_length / step) ** 4             # removes wiggles shorter than ~smooth_length
    obj = cp.Minimize(cp.sum_squares(p - q) + lam * cp.sum_squares(d2))
    base = [cp.norm(d2, 2, axis=1) <= kmax, p[0] == q[0], p[-1] == q[-1]]
    ref, target, kv = q, 1.0 / min_turn_radius, step * step / min_turn_radius
    out = q
    for _ in range(8):
        tang = ref[1:] - ref[:-1]
        tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        seg = cp.sum(cp.multiply(p[1:] - p[:-1], tang), axis=1)
        kmax.value = kv
        prob = cp.Problem(obj, base + [seg >= 0.7 * step])
        prob.solve()
        if p.value is None:
            raise RuntimeError(f"route smoothing failed for {name!r} ({prob.status})")
        out = _resample(p.value, step)
        worst = float(_circle_curvature(out).max())
        if worst <= target * 1.01:
            break
        ref = p.value
        kv *= 0.97 * target / worst
    dev = np.min(np.linalg.norm(raw[None, :, :] - out[::10, None, :], axis=2), axis=1)
    return Track(out[:, 0], out[:, 1], name, raw,
                 meta={"min_turn_radius": min_turn_radius, "max_offset_from_plan": float(dev.max()),
                       "max_curvature": float(_circle_curvature(out).max())})


# ------------------------------------------------------------------ altitude profile
def corridor_max(grid: TerrainGrid, track: Track, half_width: float, n_across: int = 7) -> np.ndarray:
    """Highest terrain across the corridor at each track point (and one point either side
    along it, so a peak between samples is not missed)."""
    c = track.course
    nx, ny = np.cos(c), -np.sin(c)                    # right-hand normal (course from north)
    off = np.linspace(-half_width, half_width, n_across)
    z = np.max([grid.elevation(track.x + o * nx, track.y + o * ny) for o in off], axis=0)
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(z, 3, mode="nearest")


def terrain_profile(grid: TerrainGrid, track: Track, *, speed: float, clearance: float = 100.0,
                    corridor: float = 150.0, climb_deg: float = 20.0, dive_deg: float = 12.0,
                    n_pull: float = 3.0, n_push: float = 0.3, start_altitude: float | None = None,
                    min_altitude: float = 0.0) -> Profile:
    """Lowest flyable altitude reference above terrain + clearance (linear programme)."""
    cp = require("cvxpy", "optim")
    s = track.s
    ds = float(np.mean(np.diff(s)))
    terrain = grid.elevation(track.x, track.y)
    floor = np.maximum(corridor_max(grid, track, corridor), min_altitude) + clearance
    n = s.size
    h = cp.Variable(n)
    d1 = (h[1:] - h[:-1]) / ds
    d2 = (h[2:] - 2 * h[1:-1] + h[:-2]) / ds**2
    V2 = speed * speed
    cons = [h >= floor, d1 <= np.tan(np.radians(climb_deg)), d1 >= -np.tan(np.radians(dive_deg)),
            d2 <= (n_pull - 1) * G0 / V2, d2 >= -(1 - n_push) * G0 / V2]
    if start_altitude is not None:
        cons.append(h[0] == max(start_altitude, floor[0]))
    prob = cp.Problem(cp.Minimize(cp.sum(h)), cons)
    prob.solve()
    if h.value is None:
        raise RuntimeError(f"no terrain-following profile ({prob.status})")
    alt = np.asarray(h.value, float)
    # the optimum is bang-bang in curvature (pull limit <-> push limit); the feed-forward
    # derivatives are taken from a copy smoothed over ~100 m so the g demand ramps
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(alt, max(100.0 / ds, 1.0), mode="nearest")
    gamma = np.arctan(np.gradient(sm, s))
    return Profile(s, terrain, floor, alt, gamma, np.gradient(gamma, s),
                   meta={"speed": speed, "clearance": clearance, "corridor": corridor,
                         "climb_deg": climb_deg, "dive_deg": dive_deg, "n_pull": n_pull,
                         "n_push": n_push, "mean_height_above_terrain": float(np.mean(alt - terrain))})
