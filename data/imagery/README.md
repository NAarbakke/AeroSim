# Ground images for the terrain flights

These are the images draped over the terrain in the terrain-flight pages (`docs/site/terrain-*.html`,
**Ground** menu). Each one covers exactly its region's terrain grid in `data/terrain` (local
transverse Mercator, north up, see `aerosim/terrain/imagery.py`). A `.json` file beside each
image records its source, dates, pixel size and coverage. `examples/build_terrain_site.py`
writes them next to the pages; `--download` rebuilds them from the sources.

| Region | Layer | Source | Date | Pixel |
|---|---|---|---|---|
| lysefjorden | `s2-summer.jpg` | Sentinel-2 L2A true colour, tile 32VLL | 17 Aug 2025 | 10 m |
| lysefjorden | `s2-snow.jpg` | Sentinel-2 L2A true colour, tile 32VLL | 7 Apr 2025 | 10 m |
| lysefjorden | `n50.jpg` | Kartverket N50 Raster (1:50 000 map), cells 2076-2078, 2096-2098 | 2026 edition | 6 m |
| lysefjorden | `landcover.png` | ESA WorldCover 2021 v200 | 2021 | 10 m |
| matochkin-shar | `s2-summer.jpg` | Sentinel-2 L2A true colour, tiles 39XWB, 40XDG | 11 Aug 2024 | 15 m |
| matochkin-shar | `s2-snow.jpg` | Sentinel-2 L2A true colour, tiles 39XWB, 40XDG | 3 Jun 2022 | 15 m |
| matochkin-shar | `landcover.png` | ESA WorldCover 2021 v200 | 2021 | 15 m |

**Processing.**
- **Sentinel-2:** the `TCI` true-colour images from Element 84 Earth Search on AWS, mosaicked
  across tiles from the same day. Summer images get a mild brightening (gamma 0.8); the snow
  images are left as they are so the snow doesn't blow out.
- **N50:** 20 km cells downloaded as GeoTIFFs through Geonorge's download API.
- **WorldCover:** coloured with the product's own legend.

All of them are resampled bilinearly onto the grid; land cover uses nearest-neighbour so the
classes stay exact.

## Licences and credit

- **Sentinel-2:** Contains modified Copernicus Sentinel data 2022, 2024 and 2025. Copernicus
  Sentinel data are free to use, modify and redistribute with that credit (Legal Notice on
  the use of Copernicus Sentinel Data and Service Information).
- **Kartverket N50 Raster:** © Kartverket, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Changed: resampled to 6 m and reprojected.
- **ESA WorldCover:** © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data
  (2021) processed by ESA WorldCover consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Changed: coloured, resampled and reprojected.

**Not included, and why:**
- **EOX Sentinel-2 cloudless:** CC BY-NC-SA (non-commercial only), and bulk or offline
  delivery is by request to EOX.
- **Norge i bilder aerial photos:** republishing needs permission from the rights holders
  (Geovekst).
- **Google, Bing and Esri imagery:** their terms forbid offline copies.
- **Landsat:** public domain, but at 30 m it's the same kind of picture as Sentinel-2 at 10 m.
