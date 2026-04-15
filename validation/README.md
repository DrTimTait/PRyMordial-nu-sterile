# PRyMordial-nu validation scripts

This directory contains scripts that run PRyMordial-nu in reference
configurations and compare against literature values or visualize
physics signatures. Used as regression / reproducibility tests and as
starting points for BSM analyses.

## Scripts

### `literature_comparison.py`
Runs five SM-like configurations (Standard thermal, Boltzmann diagonal,
QKE density matrix; with and without the O(e⁴) two-loop QED correction)
and prints the Neff values side-by-side with Bennett+2021 and
Froustey+2020 reference tables.

- Expected runtime: ~6–8 min (two thermal + three Boltzmann/QKE runs).
- Sample output: see `literature_comparison.out.txt`.
- Key takeaway: QKE mode reproduces Bennett recommended Neff = 3.0440
  to 10⁻⁴ precision.

### `spectral_distortion.py`
Runs one QKE density-matrix BBN and saves a plot of the six per-species
Δf(y)/f_FD distortions at the end of neutrino decoupling.

- Expected runtime: ~2 min.
- Output: `spectral_distortion.png`.
- Shows the classic Dolgov-style hot-tail excess signature, with ν_e
  distortion ≈ 3% and ν_μ/ν_τ ≈ 2.3% at y/T_ν_com = 10.

### `bsm_demos.py`
Three toy BSM scenarios demonstrating the new μ-τ / ν-ν̄ asymmetric
modes (Task 2):

- **A. L_μ − L_τ anomaly**: ν_μ 2.5% hotter than ν_τ, equal total ρ.
  Uses n=4 diagonal. Shows Neff preserved, small D/H shift.
- **B. Lepton asymmetry**: ξ_μ = +0.1, ξ_μ̄ = −0.1. Uses n=6 diagonal.
  Shows annihilation-rate-suppression Neff signature.
- **C. Flavor-specific DM decay**: Gaussian bump in f_ν_τ, f_ν̄_τ at
  y = 5 T_ν. Uses n=6 diagonal. Shows large +Neff from injected
  energy, Yp unchanged (weak rates only see ν_e).

- Expected runtime: ~10 min (5 runs including 2 SM baselines).

## Conventions

All scripts set:
- `smallnet_flag = True` (12-reaction network)
- `compute_bckg_flag` as needed (True when toggling `two_loop_QED_flag`
  between runs so the background ODE picks up the change)
- `compute_nTOp_flag = False` (pre-stored weak rates)

Absolute Neff values differ from Bennett/Froustey at the ~10⁻³ level
because of distinct a(T) and entropy-handoff conventions. Shifts
between configurations (δNeff from each physical ingredient) agree with
the literature at the ~10⁻⁴ level, which is the claimed precision of
the reference calculations.
