# Licensing boundaries

Choose AeroSim's own licence deliberately; the table shows what each choice implies.

| Library | Licence | How AeroSim uses it | Effect |
|---|---|---|---|
| RocketPy | MIT | optional extra | none |
| OpenMDAO, Dymos, pyCycle | Apache-2.0 | optional extra | none (keep NOTICE) |
| NASA CEA (`cea`) | NASA open source; export-control notice | `chem` extra, the combustion backend | follow NASA terms |
| openMotor | GPL-3.0 | **not a dependency**; user installs it | distributing a bundle that includes it is GPL |
| JSBSim | LGPL-2.1 | optional extra, unmodified library; its aircraft files are copied to a temp folder and patched at run time, never redistributed | permissive use OK |
| Herbie | MIT | optional extra | none |
| pyskyfire | MIT | `liquid` extra, regen cooling | none |
| cvxpy | Apache-2.0 | `landing` extra, landing guidance | none (keep NOTICE) |
| TurboFlow, FullFlow | check each repo | optional extra | verify before release |
| CoolProp | MIT | `fluids` extra | none |
| Cantera | BSD-3-Clause | `gas` extra | none (keep notice) |

**Recommendation:** licence the core under MIT or Apache-2.0 and keep every
GPL backend behind a runtime import that the user installs themselves, as done
for openMotor. If you ever ship a single bundled binary that includes openMotor,
that bundle must be GPL-3.0. Verify each licence at the pinned version before release.

RocketCEA (GPL-3.0) was removed; NASA CEA covers the same rocket problem.

## Bundled data

- `aerosim/plots/data/coastline_110m.json`: Natural Earth 1:110m coastline, public domain
  (naturalearthdata.com), simplified to two decimals.
- three.js r160 (MIT, (c) three.js authors): `aerosim/plots/data/three/` holds the unmodified
  `three.module.min.js` and `OrbitControls.js` from the npm package, with its `LICENSE`. The 3D
  replay page embeds them so it works offline.
