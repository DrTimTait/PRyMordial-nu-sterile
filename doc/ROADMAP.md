# PRyMordial-nu: Future work

After Task 1 (O(e⁴) QED) and Task 2 (μ-τ / ν-ν̄ symmetry breaking) landed,
the code was rounded out with literature comparisons, a spectral-distortion
plot, BSM demo scenarios, updated demo notebooks, a pytest regression
suite, numba persistent caching, exact 3-flavor PMNS for n=4/n=6, a
massive-electron n=6 ν-e integral, the `nlo_weak_flag` placeholder, and
a nu-e dispatcher refactor. See the git log for details (commits
`efe4968` through `6ea0bf4`).

Sterile neutrino production (the previous Item 1) was then built out in
three stages on the sister repository
[DrTimTait/PRyMordial-nu-sterile](https://github.com/DrTimTait/PRyMordial-nu-sterile):

- **Stage A** (`231b525`): 4×4 density-matrix infrastructure gated by
  `sterile_flag`. Bit-identical to the 3-flavor path when θ_14=0.
- **Stage B** (`e04b01f`): Dodelson-Widrow production via a quasi-static
  Sigl-Raffelt diagonal-transfer block in `evolve_step`; sterile
  distribution plumbed through `rho_3nu` / `drho_3nu_dTg`.
- **Stage C** (`ea04426`): Shi-Fuller MSW resonance with asymmetric initial
  conditions (`xi_nue_init`, `xi_numu_init`, `xi_nutau_init`) and the
  SF-complete matter potential — per-sector sign flip on V_CC and V_nunu,
  plus the `trace(n_ξ)·I_active` contribution that shifts H_αα−H_ss for
  active-sterile transitions.

All three stages are regression-tested (modes 1, 2, 5, 6) and validated
against expected DW / SF qualitative behavior (Dolgov+2002,
Hannestad+2012).

---

## Full QKE as an independent ODE driver

**Stage D.1 (landed, `36ee0b7`):** Strang-split unitary variant of
`evolve_step`, gated by `qke_full_ode_flag` (default False). Replaces the
two quasi-static diagonal-transfer approximations (Sigl-Raffelt
active-active relaxation and the Stage B Dodelson-Widrow active-sterile
transfer) with an exact per-mode unitary conjugation
ρ → U^(½) ρ (U^(½))†, U = exp(-iH dt), wrapped Strang-symmetrically
around a pure-damping collision step. Implemented in
`DensityMatrixSolver.evolve_step_ode` and `_apply_unitary`; regression-
tested via `test_mode5b_qke_full_ode`.

**Stage D.2 (landed):** dt-convergence diagnostic and sterile validation
of the ODE driver. Key finding from `validation/stage_d_convergence.py`:

| n_B  | Neff (ODE)  | Yp (ODE)   | D/H (ODE) |
|------|-------------|-----------:|-----------|
| 2400 | 3.040568    | 0.248463   | 2.4673    |
| 4800 | 3.040774    | 0.248351   | 2.4664    |
| 9600 | 3.040880    | 0.248305   | 2.4633    |
| ∞ (Richardson) | 3.0409  | 0.24829  | 2.462  |

The ODE-driver Neff converges to ~3.0409, **not** to the
Sigl-Raffelt/evolve_step value of 3.0445. The ~4×10⁻³ gap is therefore
a *genuine systematic bias* in the quasi-static Sigl-Raffelt approximation
— it overestimates active-flavor equilibration when the damping
timescale and oscillation timescale are comparable. Literature
reference values (Bennett+2021: 3.0440 ± 0.0002) use similar quasi-static
treatments, so `evolve_step` matches them better; the ODE driver is
arguably more self-consistent with the literal QKE as stated.

Sterile validation (`validation/sterile_DW_ode_demo.py`): the ODE driver
reproduces Stage B Dodelson-Widrow near-thermalization (ΔNeff ≈ +0.93)
without any explicit DW block in `evolve_step_ode` — the exact unitary
conjugation plus pure-damping collision step handles the active↔sterile
dynamics correctly on its own.

### Open follow-ups (Stage D.3 and beyond)

- **D.3 — ETDRK2 corrector.** Add a predictor-corrector on top of the
  Strang-split unitary for O(dt²) local error. Will NOT close the
  ~4×10⁻³ Neff bias (which is a modeling choice, not splitting error)
  but would tighten the convergence and let the default n_B suffice
  for tighter tolerances.
- **D.4 — Eigenbasis collision step.** Reformulate the damping and
  collision gain in the instantaneous H-eigenbasis, so that H and C
  commute and no splitting is needed at all. Research-grade.
- **D.5 — Magnus / implicit BDF.** For extreme-stiffness robustness
  (e.g., keV-mass sterile regimes where ω_41 dt → 10⁶).

No urgency unless a concrete physics use-case demands tighter than
~10⁻³ accuracy in a regime where the Sigl-Raffelt approximation
breaks down. Estimated effort per item: **~2–4 days**.
