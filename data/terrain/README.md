# Terrain grids

These are the height grids that `examples/build_terrain_site.py` flies over. They are kept
in the repo so the site builds and the tests run with no network. Load one with
`aerosim.terrain.TerrainGrid.load(path)`.

| File | Region | Source | Grid |
|---|---|---|---|
| `lysefjorden_kartverket_20.npz` | Lysefjorden, Norway, 48 × 30 km centred at 58.99° N 6.36° E | Kartverket DTM | 20 m, 2401 × 1501 |
| `lysefjorden_copernicus_20.npz` | the same grid (used only to check the Kartverket model) | Copernicus GLO-30 | 20 m, 2401 × 1501 |
| `matochkin-shar_copernicus_30.npz` | Matochkin Shar, Novaya Zemlya, 110 × 46 km centred at 73.33° N 55.30° E | Copernicus GLO-30 | 30 m, 3668 × 1534 |

Each grid is on its own transverse Mercator projection (WGS84, scale 1) centred on the
region, so x (east) and y (north) are metres from the centre. Heights are metres above sea
level as each source gives them: NN2000 for Kartverket, EGM2008 for Copernicus.

**Format.** Heights are stored in steps of 0.1 m and differenced along rows, then
LZMA-compressed inside an `.npz` (`TerrainGrid.save(path, packed=True)`). Every height is
within 0.05 m of the source resampling, far below either model's accuracy, and the files are
about a quarter of the size of float32.

**How they were made.** Retrieved on 25 October 2026 by `TerrainGrid.kartverket(...)` and
`TerrainGrid.copernicus(...)` (see `aerosim/terrain/dem.py`):
- Kartverket's 1 m terrain model, requested at 20 m in EPSG:25833 from Geonorge's WCS
  (`wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833`, coverage `nhm_dtm_topo_25833`).
- Copernicus GLO-30 tiles from the public `copernicus-dem-30m` bucket on AWS.

Both were resampled bilinearly onto the local grid. To fetch them again, run
`python examples/build_terrain_site.py --download`.

## Licences and attribution

- **Kartverket:** Nasjonal høydemodell, digital terrengmodell © Kartverket, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed: resampled to 20 m and
  reprojected.
- **Copernicus:** produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus
  Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all
  rights reserved. The organisations in charge of the Copernicus programme by law or by
  delegation do not incur any liability for any use of the Copernicus WorldDEM-30.
  Copernicus DEM GLO-30 is free to reproduce, distribute and modify under the
  [Copernicus WorldDEM-30 licence](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf).
  Anyone redistributing these files must keep this notice and liability sentence and pass the
  same obligations on (licence, Article 6). Changed: resampled and reprojected. This project is
  not endorsed by the Copernicus programme.
