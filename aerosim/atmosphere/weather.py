"""Live / archived weather -> ProfileAtmosphere.

Sources:
- ``gfs`` / ``hrrr`` / ``ifs`` via Herbie (NOAA NODD cloud, ECMWF open data)
- for ERA5 reanalysis, add an adapter using ``cdsapi`` following the same pattern.

Every source returns a ProfileAtmosphere, so flight code never sees GRIB/xarray.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from aerosim.atmosphere.profile import ProfileAtmosphere
from aerosim.core import registry, require

_G0 = 9.80665


@registry.register("atmosphere", "forecast")
def forecast_profile(lat: float, lon: float, time: str | datetime, model: str = "gfs",
                     fxx: int = 0, product: str | None = None) -> ProfileAtmosphere:
    """Download a vertical sounding at (lat, lon) from an NWP model run.

    Pulls temperature, geopotential height and U/V wind on isobaric levels.
    """
    herbie = require("herbie", "weather")
    product = product or {"gfs": "pgrb2.0p25", "hrrr": "prs", "ifs": "oper"}.get(model)
    H = herbie.Herbie(time, model=model, product=product, fxx=fxx)

    # One subset regex per variable; isobaric levels only.
    ds_t = H.xarray(r":TMP:\d+ mb:")
    ds_z = H.xarray(r":HGT:\d+ mb:")
    ds_u = H.xarray(r":UGRD:\d+ mb:")
    ds_v = H.xarray(r":VGRD:\d+ mb:")

    def column(ds, var):
        lon_ = lon % 360 if float(ds.longitude.max()) > 180 else lon
        # TODO: HRRR uses a Lambert grid (2-D lat/lon); use herbie's .pick_points there.
        col = ds[var].sel(latitude=lat, longitude=lon_, method="nearest")
        return col.isobaricInhPa.values * 100.0, np.asarray(col.values, float)

    p, t = column(ds_t, "t")
    _, gh = column(ds_z, "gh")
    _, u = column(ds_u, "u")
    _, v = column(ds_v, "v")
    return ProfileAtmosphere(gh, p, t, u, v, source=f"{model} {time} f{fxx:03d}")
