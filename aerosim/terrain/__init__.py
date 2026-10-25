"""Real terrain for low-level flight: DEMs on a local grid, route planning, terrain following.

dem      TerrainGrid: Kartverket (Norway, 1 m DTM via WCS) and Copernicus GLO-30 (global)
route    least-cost valley routes, curvature-limited smoothing, terrain-following altitude
         profile (a linear programme over clearance, climb / dive angle and pull-up / push-over g)
"""

from aerosim.terrain.dem import ATTRIBUTION, TerrainGrid

__all__ = ["ATTRIBUTION", "TerrainGrid"]
