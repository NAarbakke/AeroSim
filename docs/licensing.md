# Licensing boundaries

Choose aerosim's own licence deliberately; the table shows what each choice implies.

| Library | Licence | How aerosim uses it | Effect |
|---|---|---|---|
| RocketPy | MIT | optional extra | none |
| OpenMDAO, Dymos, pyCycle | Apache-2.0 | optional extra | none (keep NOTICE) |
| NASA CEA (`cea`) | NASA open source; export-control notice | `chem` extra, **default** combustion backend | follow NASA terms |
| RocketCEA | GPL-3.0 | separate `rocketcea` extra, cross-check only | distributing a bundle that includes it is GPL |
| openMotor | GPL-3.0 | **not a dependency**; user installs it | same as above |
| JSBSim | LGPL-2.1 | optional extra, unmodified library | permissive use OK |
| Herbie | MIT | optional extra | none |
| TurboFlow, pyskyfire, FullFlow | check each repo | optional extra | verify before release |
| CoolProp | MIT | `fluids` extra | none |
| Cantera | BSD-3-Clause | `gas` extra | none (keep notice) |

**Recommendation:** licence the core under MIT or Apache-2.0 and keep every
GPL backend behind an optional extra / runtime import, as done here. If you
ever ship a single bundled binary that includes RocketCEA or openMotor, that
bundle must be GPL-3.0. Verify each licence at the pinned version before release.
