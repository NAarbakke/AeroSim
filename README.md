# aerosim

Unified propulsion, atmosphere and flight simulation: solid and liquid rockets,
airbreathing engines with turbomachinery, and live weather, in one modular Python tool.

```bash
pip install -e .                      # core + built-in models
pip install -e '.[chem,fluids,gas]'   # NASA CEA, CoolProp, Cantera
pip install -e '.[control,optim]'     # python-control, OpenMDAO + Dymos
pip install -e '.[rocket,weather]'    # add RocketPy + Herbie
pip install -e '.[all,dev]'           # everything on PyPI
aerosim run configs/examples/solid_rocket_isa.yaml
aerosim backends
```

## Layout

```
aerosim/
├── core/             contracts every layer speaks (no deps on other layers)
│   ├── types.py          ThrustCurve, EngineDeck, AtmosphereState, FlightResult
│   ├── interfaces.py     Protocols: CombustionModel, RocketEngineModel,
│   │                     AirbreathingEngineModel, AtmosphereModel, FlightSimulator
│   ├── registry.py       plugin registry (lazy: heavy libs import only when used)
│   ├── errors.py         require(): optional-dependency guard with install hint
│   └── units.py          SI internally; convert only at the edges
├── thermo/           chemistry & fluids
│   ├── nasa_cea.py       NASA CEA (default)  -> Isp, c*, Tc, gamma, station data
│   ├── propellants.py    aliases (LOX, RP1, Ethanol...) and blends -> CEA species
│   ├── rocketcea.py      legacy RocketCEA, cross-check only (GPL)
│   ├── gas.py            Cantera: air + Jet-A products, cp(T, far), isentropes, burner balance
│   ├── kinetics.py       Cantera: finite-rate (ignition delay) + equilibrium cross-check vs CEA
│   ├── solid_propellants.py  KNSB/KNSU/KNDX/KNER/APCP recipes, burn rates, CEA performance
│   └── fluids.py         CoolProp: real-fluid props, saturation, pump power
├── propulsion/
│   ├── nozzle.py         isentropic relations (shared)
│   ├── solid/            bates.py (CEA propellants or manual) · openmotor.py (GPL, optional)
│   ├── liquid/           engine.py (CEA-based) · tanks.py (CoolProp, N2O blowdown)
│   │                     · feed_system.py (FullFlow / pyskyfire)
│   └── airbreathing/     ideal_turbojet.py · pycycle_engine.py · turbomachinery.py (TurboFlow)
├── atmosphere/       standard.py (US76) · profile.py · weather.py (GFS/HRRR/IFS via Herbie)
├── flight/           vehicles.py · rocket/{point_mass, planar, rocketpy_sim} · aircraft/jsbsim_sim
├── gnc/              guidance & control
│   ├── blocks.py         PID (anti-windup), Actuator (lag/rate/limits), Sensor (noise/bias)
│   ├── controllers.py    AirbrakeApogeeController, AttitudeHoldTVC (gain-scheduled)
│   └── design.py         python-control: TVC plant, pole placement, LQR, margins, discretize
├── optimization/     mdao.py (OpenMDAO components) · trajectory.py (Dymos optimal ascent)
├── data_io/          RASP .eng read/write (more formats here)
├── missions/         YAML-driven runner wiring propulsion -> atmosphere -> flight
└── cli.py
configs/examples/     solid (KNSB manual / APCP via CEA / GFS), liquid, turbojet aircraft
examples/             solid motor -> trajectory · air brakes · TVC design · Dymos ascent
tests/                pytest, runs on the base install
docs/                 architecture.md · licensing.md
```

## Status

| Module | Backend | State |
|---|---|---|
| Atmosphere | US76 ISA + power-law wind, tabulated profiles | working |
| Atmosphere | GFS / HRRR / IFS via Herbie | implemented, untested here |
| Solid | BATES 0-D ballistics, manual propellant | working |
| Solid | Propellant library + NASA CEA c*/γ(Pc), burn-rate range warnings | working, KNSB within 0.5 % of Nakka |
| Solid | openMotor | adapter, verify channel names |
| Thermo | NASA CEA rocket problem (equilibrium / frozen, blends) | working, validated vs RP-1311 Ex. 8 |
| Thermo | CoolProp fluids (saturation, density, transport) | working, validated vs NIST |
| Thermo | Cantera gas model + ignition delay | working, equilibrium within 0.15 % of CEA |
| Liquid | CEA-based chamber + throttle | working |
| Liquid | Tanks (sizing, loaded mass) + N2O self-pressurising blowdown | working |
| Liquid | FullFlow / pyskyfire feed & regen | interface only |
| Airbreathing | built-in turbojet cycle (constant-cp or Cantera real-gas) | working |
| Airbreathing | pyCycle, TurboFlow maps | interface only |
| Flight | 3-DOF point-mass rocket with wind (open or closed loop) | working |
| Flight | Planar rigid-body rocket: attitude, weathercocking, TVC | working |
| GNC | PID / actuators / sensors, air-brake apogee control, TVC attitude hold | working |
| GNC | python-control design: pole placement, LQR, margins | working |
| Optimization | Dymos optimal throttle (Goddard), max-q constraint | working, verified by independent replay |
| Flight | RocketPy 6-DOF | motor done, Rocket build TODO |
| Flight | JSBSim | runner done, deck->XML TODO |
| Optimization | OpenMDAO solid-apogee component | working skeleton |

See `docs/architecture.md` for the design rules and how to add a backend.
