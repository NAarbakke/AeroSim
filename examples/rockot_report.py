"""LaTeX report of the Rockot / Breeze-KM ascent: both models, all figures and tables generated
from the flight data (the same data as the site's "Rockot data" dashboard).

    python examples/rockot_report.py               # -> docs/reports/rockot/Rockot_Ascent_Report.pdf
    python examples/rockot_report.py --no-compile  # write the .tex and .pgf figures only
    python examples/rockot_report.py --out DIR

Needs a LaTeX installation with lualatex and latexmk (TeX Live or MiKTeX) and the [chem] extra
for the first-principles model. Takes about half a minute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

MODELS_TEXT = {
    "First principles": (r"is built without reference to the User's Guide trajectory: engine performance from "
                         r"NASA CEA at the published chamber pressures and mixture ratios, propellant masses from "
                         r"engine mass flow and published burn times, drag from a component build-up, and steering "
                         r"optimised to leave the most Breeze-KM propellant in the target orbit"),
    "Calibrated to the guide": (r"fits the stage~1 and~2 propellant masses and the pitch programme to the "
                                r"reference trajectory and lift-off mass published in the Rockot User's Guide"),
}


def dataset(models: dict, f: dict) -> dict:
    """Report / dashboard data of flown models ({name: (AscentResult, vehicle)}, first-principles
    first); ``f`` is the mission's flight config (launch site, programme)."""
    from aerosim.plots.ascent_data import ascent_dataset
    data = ascent_dataset(models, (f["latitude"], f["longitude"]), target="a 650 km circular orbit at 86.6°",
                          title="Rockot / Breeze-KM Ascent: Flight Data")
    data["launch_site"] = "Plesetsk"
    data["fairing_heat_flux"] = float(f["program"].get("fairing_heat_flux", 1135.0))
    return data


def flight_data() -> dict:
    """Fly both Rockot models and gather their data."""
    import rockot_ascent as ra
    import rockot_first_principles as fp
    _c, veh_cal, f = ra.load()
    _c, veh_fp, f_fp = fp.load()
    return dataset({"First principles": (fp.fly(veh_fp, f_fp), veh_fp),
                    "Calibrated to the guide": (ra.fly(veh_cal, f), veh_cal)}, f)


def main(argv=None):
    from aerosim.plots.ascent_report import build_report
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "reports" / "rockot"))
    ap.add_argument("--no-compile", action="store_true", help="write the LaTeX sources only")
    args = ap.parse_args(argv)
    data = flight_data()
    s = data["models"][0]["summary"]
    path = build_report(
        data, args.out, name="Rockot_Ascent_Report", compile=not args.no_compile,
        title="Ascent of Rockot / Breeze-KM from Plesetsk",
        subtitle="Flight data of two vehicle models on the 650 km reference mission",
        author="AeroSim",
        abstract=(rf"Two models of the Rockot launch vehicle with the Breeze-KM upper stage are flown from Plesetsk "
                  rf"to a 650\,km circular orbit at 86.6$^\circ$ with a three-degree-of-freedom ascent simulation. "
                  rf"The first-principles model reaches a \num{{{s['perigee']:.0f}}}\,$\times$\,\num{{{s['apogee']:.0f}}}\,km "
                  rf"orbit after a maximum dynamic pressure of \num{{{s['max_q']:.1f}}}\,kPa, and loses "
                  rf"\num{{{s['losses']['gravity']:.0f}}}\,m/s to gravity and \num{{{s['losses']['drag']:.0f}}}\,m/s to drag. "
                  r"Trajectory, atmospheric loads, propulsion, velocity losses and the orbit are reported for both models."),
        models_text=MODELS_TEXT)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
