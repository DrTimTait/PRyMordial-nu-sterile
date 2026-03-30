# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRyMordial is a Python package for precise Big Bang Nucleosynthesis (BBN) computations within and beyond the Standard Model. It predicts primordial light element abundances (Yp, D/H, He3/H, Li7/H) and related cosmological observables (Neff, etc.). Reference paper: [arXiv:2307.07061](https://arxiv.org/abs/2307.07061).

## Running

```bash
# Quick demo run (small + large network, optionally Julia)
python runPRyM_julia.py
```

Jupyter notebooks `PRyMdemoSM.ipynb` (Standard Model) and `PRyMdemoNP.ipynb` (New Physics) provide interactive usage examples.

## Dependencies

- **Required:** numpy, scipy
- **Recommended:** numba (JIT for integrands in PRyM_thermo.py), numdifftools (numerical derivatives in PRyM_main.py), vegas (Monte Carlo integration for thermal radiative corrections)
- **Optional:** julia (PyJulia) + diffeqpy (for Julia ODE solvers) — requires Julia lang + SciML

## Architecture

The main entry point is `PRyM/PRyM_main.py` which defines `PRyMclass`. A typical call:

```python
import PRyM.PRyM_init as PRyMini
# Set flags before importing PRyM_main
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False

import PRyM.PRyM_main as PRyMmain
res = PRyMmain.PRyMclass().PRyMresults()
```

### Module responsibilities

- **PRyM_init.py** — Global configuration: physical constants (PDG values), cosmological parameters, temperature ranges, all boolean flags, and nuclear rate data loaded from `PRyMrates/`. This is a flat module (no classes) where flags are set as module-level globals *before* importing other modules.
- **PRyM_main.py** — `PRyMclass`: orchestrates the full BBN calculation. Constructor solves the thermodynamic background (temperature evolution), then the neutron-proton freeze-out, then the nuclear network ODE. `PRyMresults()` returns [Neff, Omega_nu h^2, sum_m_nu/Omega_nu h^2, Yp_CMB, Yp_BBN, D/H, He3/H, Li7/H].
- **PRyM_thermo.py** — Thermodynamic functions: energy densities, pressures, and collision terms for photons, electrons, and neutrinos. Loads QED plasma corrections and neutrino scattering/annihilation rates from `PRyMrates/thermo/`. Uses `@njit` from numba when available.
- **PRyM_eval_nTOp.py** — Computes neutron-to-proton weak rates with radiative corrections (Fermi/Coulomb, resummation). Uses `vegas` for thermal corrections when `compute_nTOp_thermal_flag` is set.
- **PRyM_nTOp.py** — Loads or recomputes n<->p weak rates, returning interpolation functions across three temperature eras (HT, MT, LT).
- **PRyM_nuclear_net12.py** — 12-reaction nuclear network (sufficient for Yp and D/H).
- **PRyM_nuclear_net63.py** — 63-reaction extended network (needed for Li7/H). Both define `UpdateNuclearRates` classes with forward/backward rate methods.
- **PRyM_jl_sys.py** — Julia ODE system definitions via PyJulia for optional Julia solver path.

### Key flags in PRyM_init.py

| Flag | Effect |
|------|--------|
| `smallnet_flag` | True = 12-reaction network, False = 63-reaction network |
| `julia_flag` | Use Julia ODE solvers instead of scipy |
| `compute_bckg_flag` | Recompute thermodynamic background (vs. loading pre-stored) |
| `compute_nTOp_flag` | Recompute weak rates from scratch |
| `NP_thermo_flag` / `NP_nTOp_flag` / `NP_nuclear_flag` | Enable new-physics modifications to thermodynamics / weak rates / nuclear rates |
| `numba_flag` | Use numba JIT acceleration |
| `nacreii_flag` | Use NACRE II rates instead of default PRIMAT compilation |

### Data directory: PRyMrates/

- `thermo/` — QED plasma corrections, neutrino scattering/annihilation matrix elements
- `nTOp/` — Pre-computed n<->p weak rate tables and thermal corrections
- `nuclear/key_primat_rates/` and `nuclear/key_nacreii_rates/` — 12 key nuclear reaction rate tables
- `nuclear/other_nucl_rates/` — Additional 51 nuclear rates for the extended network

### Design patterns

- Configuration is done by setting module-level flags on `PRyM_init` *before* importing downstream modules, since data loading happens at import time.
- Nuclear rates support log-normal parameterization via `p_*` weights (defaulting to 0 = median) for MCMC analyses.
- `PRyMclass` constructor accepts optional new-physics energy density/pressure functions (`my_rho_NP`, `my_p_NP`, etc.) for BSM scenarios.
- All rate data uses scipy `interp1d` interpolation throughout.
