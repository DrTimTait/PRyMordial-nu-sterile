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

**Stage D.3 (landed):** Shi-Fuller validation of the ODE driver, plus
a ν̄ sector convention bug fix. `validation/sterile_SF_ode_demo.py` runs
the 3×3 reference, DW baseline, and SF (ξ_νe = 1e-2, 5e-2) scenarios
through both Strang and ODE drivers.

In the process, Stage D.3 uncovered a convention bug in the original
Stage D.1 `_apply_unitary`: PRyM's `evolve_step` stores
`rho_all[1] = ρ̄*` (complex conjugate of the antineutrino density
matrix), as encoded by its `osc_signs = [+1, -1]` in the off-diagonal
update. The original `_apply_unitary` naively applied U ρ U† to both
sectors, which for sector 1 amounts to a different physical operation.
Fix: for sector 1 the correct unitary transformation is
`(ρ̄)*_new = U_ν̄* · (ρ̄)* · U_ν̄ᵀ`, not `U · ρ · U†`.

Before the fix the ODE DW test produced a spurious ν − ν̄ asymmetry of
magnitude |n_ξe| ~ 24 (should be ~0 for ξ=0). After the fix it drops
to ~10⁻¹, and the Stage B ΔNeff result agrees with the Strang
quasi-static block to within 10⁻³ (0.931 vs 0.930; previously 0.83 vs
0.93).

Under the corrected ODE driver, the SF regime exhibits genuine
differences vs the Strang quasi-static result:

| Scenario                | Strang ΔNeff | ODE ΔNeff | ΔΣρ_ss (ODE−Strang) |
|-------------------------|-------------:|----------:|---------------------:|
| DW (sin²=1e-3, ξ=0)     | +0.930       | +0.931    | -0.02                |
| SF (sin²=1e-3, ξ=1e-2)  | +1.016       | +0.929    | -0.46                |
| SF (sin²=1e-3, ξ=5e-2)  | +0.939       | +0.912    | -0.31                |

The ~0.03–0.09 gap in the SF cases traces to a physics distinction:
Stage B's quasi-static DW formula `Γ_DW = 2|H_αs|²D/(D²+ω²)` has a
1/(D²+ω²) structure that spikes at the MSW resonance (ω → 0),
overestimating the transfer efficiency when the resonance sweeps
through a momentum mode. The ODE driver's exact unitary evolution does
not suffer this overestimate, producing a somewhat smaller ΔNeff and
retaining more of the initial asymmetry. Both are physically
consistent; the ODE driver is more self-consistent with the stated
QKE, while the Strang path matches literature references that use
similar quasi-static treatments.

### Open follow-ups (Stage D.4 and beyond)

- **D.4 — ETDRK2 corrector.** Add a predictor-corrector on top of the
  Strang-split unitary for O(dt²) local error. Will NOT close the
  modeling-choice Neff gap (already shown to be a real bias in D.2)
  but would tighten convergence.
- **D.5 — Eigenbasis collision step.** Reformulate the damping and
  collision gain in the instantaneous H-eigenbasis, so that H and C
  commute and no splitting is needed at all. Research-grade.
- **D.6 — Magnus / implicit BDF.** For extreme-stiffness robustness
  (e.g., keV-mass sterile regimes where ω_41 dt → 10⁶).

No urgency unless a concrete physics use-case demands tighter than
~10⁻³ accuracy in a regime where the Sigl-Raffelt/DW quasi-static
approximation breaks down. Estimated effort per item: **~2–4 days**.
