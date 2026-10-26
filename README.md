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
pip install -e '.[grains,plot]'       # non-circular solid grains, matplotlib plots
pip install -e '.[all,dev]'           # everything on PyPI (except turboflow, see below)
pip install tifffile pyproj scikit-image cvxpy jsbsim   # terrain flights (aerosim.terrain)
aerosim run configs/solid/knsb_isa.yaml
aerosim run configs/solid/finocyl_grain_motor.yaml
aerosim backends
```

The distribution is called `AeroSim`; the import name and command are lowercase
(`import aerosim`, `aerosim run ...`), as is conventional for Python packages.

### Development environment

```bash
uv venv --python 3.12 .venv
PY=.venv/bin/python                  # Linux / macOS
# PY=.venv/Scripts/python.exe        # Windows
uv pip install --python $PY -e '.[all,dev]'
uv pip install --python $PY tifffile pyproj scikit-image cvxpy jsbsim
$PY -m pytest
```

The code is plain Python with pathlib paths and runs the same on Linux, macOS and Windows; the
paths below differ only in where the venv keeps its interpreter and site-packages.

Notes:

- **Python 3.12** is recommended: RocketPy's dependency `simplekml` may fail to build on 3.13.
- **TurboFlow** is a separate `turboflow` extra, not part of `all`: it pins `numpy<2`, which
  conflicts with NASA `cea` (`numpy>=2`). Install it in its own environment if you need it.
- **pyCycle**: the PyPI release (om-pycycle 4.4.0) fails on numpy 2.5; install it from its
  master branch, e.g. `uv pip install --python $PY git+https://github.com/OpenMDAO/pyCycle@ee7e161`.
- **openMotor** is GPL-3.0 and not on PyPI, so it is never installed automatically
  (see `docs/licensing.md`). Clone it next to this repo, build its one Cython module
  (needs a C compiler: gcc/clang on Linux and macOS, MSVC Build Tools on Windows) and point
  the venv at it:
  ```bash
  git clone https://github.com/reilleya/openMotor ../openMotor
  uv pip install --python $PY scikit-fmm scikit-image cython setuptools
  (cd ../openMotor && ../AeroSim/$PY setup.py build_ext --inplace)
  # Linux / macOS
  echo "$(cd ../openMotor && pwd)" > "$($PY -c 'import site; print(site.getsitepackages()[0])')/openmotor.pth"
  # Windows (Git Bash)
  # echo "$(cd ../openMotor && pwd -W)" > .venv/Lib/site-packages/openmotor.pth
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
│   ├── solid/            bates.py (fast BATES) · grain_motor.py (any grain stack, time-stepped)
│   │                     grains.py (BATES, finocyl, star, C, D, X, moon, rod-and-tube,
│   │                     end burner, custom outline) · propellant.py · ric.py (openMotor files)
│   ├── hybrid/           motor.py (N2O self-pressurising or regulated feed, a*G^n regression,
│   │                     O/F shift, CEA performance over O/F and Pc)
│   ├── liquid/           engine.py (CEA-based) · design.py (engine design point: CEA + losses ->
│   │                     Isp, mdot, nozzle) · cycle.py (pressure-fed, electric pump, gas
│   │                     generator, staged combustion, expander power balance) · tanks.py
│   ├── airbreathing/     ideal_turbojet.py (constant-cp or Cantera) · ramjet.py (ramjet and
│   │                     scramjet, Cantera equilibrium)
│   └── adapters/         external wrappers: openmotor (cross-check) · pycycle_engine (turbojet,
│                         turbofan) · regen (pyskyfire cooling) · turbomachinery (TurboFlow) ·
│                         feed_system (FullFlow)
├── atmosphere/       standard.py (US76, exponential above 86 km) · profile.py · weather.py
├── flight/           vehicles.py (RocketVehicle, AircraftVehicle, Stage, LaunchVehicle)
│   ├── aero.py           launcher drag build-up from geometry (friction, wave, base)
│   ├── stability.py      CG as tanks drain and stages drop, Barrowman CP, static margin,
│   │                     pitch divergence and steering needed along an ascent
│   ├── orbits.py         Earth model (WGS-84, J2, rotation), orbit elements
│   ├── rocket/           point_mass.py (3-DOF) · planar.py (attitude, TVC) ·
│   │                     orbital_ascent.py (multi-stage launcher to orbit; programme fitting
│   │                     and trajectory optimisation)
│   └── adapters/         rocketpy_sim (6-DOF) · jsbsim_sim (JSBSim airframe on an AeroSim
│                         engine deck, autopilot)
├── gnc/              guidance & control
│   ├── blocks.py         PID (anti-windup), Actuator (lag/rate/limits), Sensor (noise/bias)
│   ├── controllers.py    AirbrakeApogeeController, AttitudeHoldTVC (gain-scheduled)
│   ├── design.py         python-control: TVC plant, pole placement, LQR, margins, discretize
│   └── landing.py        convex powered-descent guidance (cvxpy) + closed-loop landing sim
├── terrain/          real terrain for low-level flight: dem.py (Kartverket 1 m DTM via WCS,
│                     Copernicus GLO-30 tiles, local metric grid) · route.py (least-cost valley
│                     routes, turn-radius smoothing, terrain-following altitude profile) ·
│                     flight.py (JSBSim aircraft flying a route at low level over the DEM) ·
│                     imagery.py (Sentinel-2, Kartverket N50, ESA WorldCover ground images)
├── optimization/     mdao.py (OpenMDAO components) · trajectory.py (Dymos optimal ascent)
├── missions/         YAML-driven runner wiring propulsion -> atmosphere -> flight
├── plots/            matplotlib charts: thrust curves, motor history, grain burnback,
│                     flight profiles, ascent small multiples, ground tracks;
│                     replay.py: animated 3D ascent replay as a standalone HTML page;
│                     site.py + web.py: results site (local-frame 3D replays, engines page);
│                     engine_models.py + engines3d.py: 3D engine cutaways coloured by flow fields;
│                     ascent_closeup.py: Rockot up close through the atmosphere's layers;
│                     terrain_site.py: low-level flights over a terrain mesh (glTF F-16 from
│                     data/f16, CC BY 4.0, see its ATTRIBUTION.md)
└── cli.py            aerosim run | backends | atm
configs/             example YAML mission configs (aerosim run <file>)
├── solid/            knsb_isa (base install) · knsb_gfs (live winds) · apcp_cea (CEA propellant)
│                     finocyl_grain_motor (GrainMotor)
├── hybrid/           n2o_htpb_sounding (N2O/HTPB hybrid sounding rocket)
├── liquid/           cea_engine
├── orbital/          rockot (calibrated to the User's Guide) · rockot_first_principles
│                     (CEA engines, drag from geometry, optimised steering)
└── aircraft/         turbojet (JSBSim F-16 airframe on an AeroSim turbojet deck)
examples/             solid motor -> trajectory · air brakes · TVC design · Dymos ascent ·
                      rockot_ascent (calibrate + fly the Rockot reference mission) ·
                      rockot_first_principles (predict it, then check) · rockot_replay (3D) ·
                      build_site (all results as an offline website, docs/site/) ·
                      build_terrain_site (F-16 low level over Norway and Novaya Zemlya, docs/site/terrain*.html) ·
                      bake_f16_model (the F-16 glTF for those pages)
data/terrain/         height grids for the terrain flights (Kartverket, Copernicus; 10 MB, licences
                      in its README), so the terrain site builds offline
data/imagery/         ground images draped over them (Sentinel-2, N50 map, land cover; 31 MB, licences
                      in its README)
tests/                pytest, one folder per layer (core, thermo, propulsion, ...)
docs/                 architecture.md · licensing.md · references/rockot (User's Guide) ·
                      briefings/rockot (PDF + PPTX briefing, 3D replay page, generators) ·
                      site/ (results website, terrain flights included: open index.html)
```

`adapters/` folders hold wrappers around external libraries. Everything outside them is a
built-in model.

## Status

| Module | Backend | State |
|---|---|---|
| Atmosphere | US76 ISA + power-law wind, tabulated profiles | working |
| Atmosphere | GFS / HRRR / IFS via Herbie | implemented, untested here |
| Solid | BATES 0-D ballistics, manual propellant | working |
| Solid | Propellant library + NASA CEA c*/γ(Pc), burn-rate range warnings | working, KNSB within 0.5 % of Nakka |
| Solid | GrainMotor: 10 grain types, tabbed propellants, erosion/slag, .ric files | working; 13 openMotor regression motors: mass within 1.5 %, Isp 0-8 % high |
| Solid | openMotor (`propulsion/adapters`, cross-check) | working, N2950 sample within 3 % of its designation |
| Thermo | NASA CEA rocket problem (equilibrium / frozen, blends) | working, validated vs RP-1311 Ex. 8 |
| Thermo | CoolProp fluids (saturation, density, transport) | working, validated vs NIST |
| Thermo | Cantera gas model + ignition delay | working, equilibrium within 0.15 % of CEA |
| Liquid | CEA-based chamber + throttle | working |
| Liquid | Tanks (sizing, loaded mass) + N2O self-pressurising blowdown | working |
| Liquid | Engine cycles: pressure-fed, electric pump, gas generator, staged combustion (fuel/ox-rich), expander | working; F-1 turbopump power 13 % low, RD-180 Isp within 0.5 %, RS-25 fuel-pump discharge 11 % low |
| Liquid | Regenerative cooling via pyskyfire (`propulsion/adapters/regen.py`), coupled to the expander cycle | working; RL10-class expander closes, Isp 1.4 % below the RL10A-3-3A |
| Liquid | FullFlow feed networks (`propulsion/adapters`) | interface only |
| Hybrid | N2O self-pressurising / regulated feed, regression law, O/F shift, CEA | working; mass balance exact; regression a, n must come from firings |
| Airbreathing | built-in turbojet cycle (constant-cp or Cantera real-gas) | working |
| Airbreathing | Ramjet / scramjet, Cantera equilibrium burner and nozzle | working; kerosene ramjet Isp 1,400-1,700 s at Mach 2-4 |
| Airbreathing | pyCycle turbojet / two-spool turbofan, design + map-based off-design (`propulsion/adapters`) | working (needs om-pycycle from its master on numpy 2.5); J79 air flow within 1 %, TSFC 11 % low; CFM56-7B27 OPR, air flow and takeoff TSFC inside published ranges |
| Airbreathing | TurboFlow maps (`propulsion/adapters`) | interface only |
| Flight | 3-DOF point-mass rocket with wind (open or closed loop) | working |
| Flight | Planar rigid-body rocket: attitude, weathercocking, TVC | working |
| Flight | RocketPy 6-DOF (`flight/adapters`) | working; no wind: apogee within 0.3 % of point-mass |
| Flight | Multi-stage orbital ascent (rotating oblate Earth, staging, fairing, restarts) | working; Rockot reference mission reproduced, see briefing |
| Flight | Trajectory optimisation (`optimize_program`), drag from geometry | working; first-principles Rockot reaches orbit, burnout speeds within 2.5 % of the guide |
| Flight | JSBSim airframe on an AeroSim engine deck, altitude / Mach / heading autopilot | working; F-16 holds 4,000 m, Mach 0.80, 90 deg |
| Terrain | Kartverket DTM (Norway) and Copernicus GLO-30 (global) on a local grid | working; the two agree over Lysefjorden to a median 0.1 m, rms 3.9 m above the tree line |
| Terrain | Valley routes, terrain-following profile (LP), JSBSim F-16 terrain following | working; four routes at 100 m set clearance, lowest 95.9 m above ground |
| Terrain | Ground images: Sentinel-2 true colour (two seasons), Kartverket N50 map, ESA WorldCover land cover, switchable in the viewer | working |
| GNC | PID / actuators / sensors, air-brake apogee control, TVC attitude hold | working |
| GNC | python-control design: pole placement, LQR, margins | working |
| GNC | Booster landing: convex powered-descent guidance, replanned with drag and wind | working; Falcon 9-class touchdown under 2 m/s, on target |
| Optimization | Dymos optimal throttle (Goddard), max-q constraint | working, verified by independent replay |
| Optimization | OpenMDAO solid-apogee component | working skeleton |
| Plots | `aerosim.plots` (validated colour-blind-safe palette, light/dark) | working |
| Plots | 3D ascent replay (`aerosim.plots.replay`, three.js page, works offline) | working |
| Flight | Pitch stability along the ascent (`aerosim.flight.stability`): CG from a mass layout with draining tanks, Barrowman CP, static margin, time to double, gimbal per degree of AoA | working; layout mass matches the simulator to 1e-14; Rockot −6.1 to −2.6 calibers, doubling every 0.43 s at max-Q |
| Plots | Ascent close-up (`aerosim.plots.ascent_closeup`): three.js Rockot model, sky / Earth scattering shader, atmospheric layers, drag, max-Q, heating, thrust gain and Δv losses | working; US76 thermosphere temperatures match the 1976 tables within 0.6 K |
| Plots | Results site (`aerosim.plots.site`, `examples/build_site.py`): landing, F-16 and hybrid 3D replays, engines page | working |
| Plots | 3D engine models (`aerosim.plots.engine_models`): liquid chambers (pyskyfire contour, isentropic gas fields, wall temperature), regressing solid grains, hybrid port and tank, ramjet / scramjet and gas-turbine cutaways | working |

See `docs/architecture.md` for the design rules and how to add a backend.
