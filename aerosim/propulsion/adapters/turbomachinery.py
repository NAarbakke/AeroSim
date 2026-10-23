"""Component-level turbomachinery analysis (meanline), feeding the cycle model.

Flow of information:
    blade geometry --(TurboFlow meanline)--> efficiency / mass-flow / PR map
    map --(ComponentMap)--> pyCycle Compressor/Turbine element or Turbojet efficiencies

TurboFlow (github.com/turbo-sim/turboflow) currently targets axial turbines.
Compressors need another meanline code or literature maps until that gap is filled;
the ComponentMap contract stays the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerosim.core import require


@dataclass
class ComponentMap:
    """Generic map on (corrected speed, beta/pressure-ratio line)."""

    kind: str                      # "turbine" | "compressor"
    corrected_speed: np.ndarray    # (n_speed,) fraction of design
    pressure_ratio: np.ndarray     # (n_speed, n_line)
    corrected_flow: np.ndarray     # (n_speed, n_line) kg/s
    efficiency: np.ndarray         # (n_speed, n_line) isentropic

    def design_efficiency(self) -> float:
        i = int(np.argmin(abs(self.corrected_speed - 1.0)))
        return float(self.efficiency[i].max())


def turbine_map_from_turboflow(config_file: str, speeds=(0.7, 0.8, 0.9, 1.0, 1.05),
                               pressure_ratios=None) -> ComponentMap:
    """Run TurboFlow performance analysis over a speed/PR grid and package as a ComponentMap.

    TODO: load the TurboFlow YAML config, call its performance-map routine for each
    (speed, PR) pair, collect mass flow and total-to-static efficiency.
    """
    require("turboflow", "airbreathing")
    raise NotImplementedError("TODO: TurboFlow performance map -> ComponentMap")
