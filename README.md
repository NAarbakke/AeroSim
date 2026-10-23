# AeroSim

Unified propulsion, atmosphere and flight simulation: solid and liquid rockets,
airbreathing engines with turbomachinery, and live weather, in one modular Python tool.

A mission is a YAML file that picks one backend per layer (engine model, atmosphere,
flight simulator). AeroSim wires them together and reports apogee, speed, impulse, etc.

```bash
pip install -e .                      # core + built-in models
pip install -e '.[chem,fluids,gas]'   # NASA CEA, CoolProp, Cantera
pip install -e '.[control,optim]'     # python-control, OpenMDAO + Dymos
pip install -e '.[rocket,weather]'    # add RocketPy + Herbie
pip install -e '.[all,dev]'           # everything on PyPI (except turboflow, see below)
aerosim run configs/examples/rocket/solid/knsb_isa.yaml
aerosim backends
```

The distribution is called `AeroSim`; the import name and command are lowercase
(`import aerosim`, `aerosim run ...`), as is conventional for Python packages.

### Development environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e '.[all,dev]'
.venv/Scripts/python.exe -m pytest
```

Notes:

- **Python 3.12** is recommended: RocketPy's dependency `simplekml` may fail to build on 3.13.
- **TurboFlow** is a separate `turboflow` extra, not part of `all`: it pins `numpy<2`, which
  conflicts with NASA `cea` (`numpy>=2`). Install it in its own environment if you need it.
- **openMotor** is GPL-3.0 and not on PyPI, so it is never installed automatically
  (see `docs/licensing.md`). Clone it next to this repo, build its one Cython module
  (needs a C compiler, e.g. MSVC Build Tools) and point the venv at it:
  ```bash
  git clone https://github.com/reilleya/openMotor ../openMotor
  uv pip install --python .venv/Scripts/python.exe scikit-fmm scikit-image cython setuptools
  cd ../openMotor && ../AeroSim/.venv/Scripts/python.exe setup.py build_ext --inplace && cd -
  echo "$(cd ../openMotor && pwd -W)" > .venv/Lib/site-packages/openmotor.pth
  ```
  If `setup.py` fails on missing PyQt tooling, build only the extension
  `mathlib._find_perimeter_cy` from `mathlib/_find_perimeter_cy.pyx` with `cythonize`.
- **Herbie on Windows** prints emoji on import; set `PYTHONUTF8=1` if the console
  encoding raises `UnicodeEncodeError`.

## Layout

```
aerosim/
├── core/             contracts every layer speaks (no deps on other layers)
│   ├── types.py          ThrustCurve, EngineDeck, AtmosphereState, FlightResult
│   ├── interfaces.py     Protocols: CombustionModel, RocketEngineModel,
│   │                     AirbreathingEngineModel, AtmosphereModel, FlightSimulator
│   ├── registry.py       plugin registry (lazy: heavy libs import only when used)
│   ├── errors.py         require(): optional-dependency guard with install hint
│   ├── units.py          SI internally; convert only at the edges
│   └── eng.py            RASP .eng thrust-curve read/write
├── thermo/           chemistry & fluids
│   ├── combustion/       NASA CEA
│   │   ├── nasa_cea.py       rocket problem -> Isp, c*, Tc, gamma, station data
│   │   ├── propellants.py    aliases (LOX, RP1, Ethanol...) and blends -> CEA species
│   │   └── solid_propellants.py  KNSB/KNSU/KNDX/KNER/APCP recipes, burn rates, CEA performance
│   ├── gas/              Cantera
│   │   ├── mixture.py        air + Jet-A products, cp(T, far), isentropes, burner balance
│   │   └── kinetics.py       finite-rate (ignition delay) + equilibrium cross-check vs CEA
│   └── fluids.py         CoolProp: real-fluid props, saturation, pump power
├── propulsion/       engines -> ThrustCurve | EngineDeck
│   ├── nozzle.py         isentropic relations (shared)
│   ├── solid/            bates.py (CEA propellants or manual)
│   ├── liquid/           engine.py (CEA-based) · tanks.py (CoolProp, N2O blowdown)
│   ├── airbreathing/     ideal_turbojet.py (constant-cp or Cantera)
│   └── adapters/         unfinished wrappers: openmotor · pycycle_engine ·
│                         turbomachinery (TurboFlow) · feed_system (FullFlow / pyskyfire)
├── atmosphere/       standard.py (US76) · profile.py · weather.py (GFS/HRRR/IFS via Herbie)
├── flight/           vehicles.py
│   ├── rocket/           point_mass.py (3-DOF) · planar.py (attitude, TVC)
│   └── adapters/         unfinished wrappers: rocketpy_sim · jsbsim_sim
├── gnc/              guidance & control
│   ├── blocks.py         PID (anti-windup), Actuator (lag/rate/limits), Sensor (noise/bias)
│   ├── controllers.py    AirbrakeApogeeController, AttitudeHoldTVC (gain-scheduled)
│   └── design.py         python-control: TVC plant, pole placement, LQR, margins, discretize
├── optimization/     mdao.py (OpenMDAO components) · trajectory.py (Dymos optimal ascent)
├── missions/         YAML-driven runner wiring propulsion -> atmosphere -> flight
└── cli.py            aerosim run | backends | atm
configs/examples/
├── rocket/solid/     knsb_isa (base install) · knsb_gfs (live winds) · apcp_cea (CEA propellant)
├── rocket/liquid/    cea_engine
└── aircraft/         turbojet
examples/             solid motor -> trajectory · air brakes · TVC design · Dymos ascent
tests/                pytest, one folder per layer (core, thermo, propulsion, ...)
docs/                 architecture.md · licensing.md
```

`adapters/` folders hold wrappers around external libraries that are interface-only or
partly built. Everything outside them is a working built-in model.

## Status

| Module | Backend | State |
|---|---|---|
| Atmosphere | US76 ISA + power-law wind, tabulated profiles | working |
| Atmosphere | GFS / HRRR / IFS via Herbie | implemented, untested here |
| Solid | BATES 0-D ballistics, manual propellant | working |
| Solid | Propellant library + NASA CEA c*/γ(Pc), burn-rate range warnings | working, KNSB within 0.5 % of Nakka |
| Solid | openMotor (`propulsion/adapters`) | working, N2950 sample within 3 % of its designation |
| Thermo | NASA CEA rocket problem (equilibrium / frozen, blends) | working, validated vs RP-1311 Ex. 8 |
| Thermo | CoolProp fluids (saturation, density, transport) | working, validated vs NIST |
| Thermo | Cantera gas model + ignition delay | working, equilibrium within 0.15 % of CEA |
| Liquid | CEA-based chamber + throttle | working |
| Liquid | Tanks (sizing, loaded mass) + N2O self-pressurising blowdown | working |
| Liquid | FullFlow / pyskyfire feed & regen (`propulsion/adapters`) | interface only |
| Airbreathing | built-in turbojet cycle (constant-cp or Cantera real-gas) | working |
| Airbreathing | pyCycle, TurboFlow maps (`propulsion/adapters`) | interface only |
| Flight | 3-DOF point-mass rocket with wind (open or closed loop) | working |
| Flight | Planar rigid-body rocket: attitude, weathercocking, TVC | working |
| Flight | RocketPy 6-DOF (`flight/adapters`) | motor done, Rocket build TODO |
| Flight | JSBSim (`flight/adapters`) | runner done, deck->XML TODO |
| GNC | PID / actuators / sensors, air-brake apogee control, TVC attitude hold | working |
| GNC | python-control design: pole placement, LQR, margins | working |
| Optimization | Dymos optimal throttle (Goddard), max-q constraint | working, verified by independent replay |
| Optimization | OpenMDAO solid-apogee component | working skeleton |

See `docs/architecture.md` for the design rules and how to add a backend.
