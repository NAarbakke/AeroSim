"""3D replay of the Rockot / Breeze-KM ascent: calibrated and first-principles models together.

    python examples/rockot_replay.py      # -> docs/briefings/rockot/Rockot_Ascent_Replay.html

Open the page in a browser: drag to orbit the globe, scroll to zoom, space to play / pause.
"""

from __future__ import annotations

from pathlib import Path

import rockot_ascent as ra
import rockot_first_principles as fp

from aerosim.plots.replay import write_replay

OUT = Path(__file__).resolve().parents[1] / "docs" / "briefings" / "rockot" / "Rockot_Ascent_Replay.html"


def main(out: Path = OUT):
    _c, veh_cal, f = ra.load()
    _c, veh_fp, f_fp = fp.load()
    results = {"Calibrated to the guide": ra.fly(veh_cal, f),
               "First principles": fp.fly(veh_fp, f_fp)}
    write_replay(results, out, site=(f["latitude"], f["longitude"]),
                 title="Rockot / Breeze-KM ascent from Plesetsk",
                 subtitle="650 km circular, 86.6°  ·  AeroSim orbital_ascent")
    print("wrote", out)


if __name__ == "__main__":
    main()
