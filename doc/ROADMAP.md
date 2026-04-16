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

**Stage D.1 (landed):** Strang-split unitary variant of `evolve_step`,
gated by `qke_full_ode_flag` (default False). Replaces the two
quasi-static diagonal-transfer approximations (Sigl-Raffelt active-active
relaxation and the Stage B Dodelson-Widrow active-sterile transfer) with
an exact per-mode unitary conjugation ρ → U^(½) ρ (U^(½))†,
U = exp(-iH dt), wrapped Strang-symmetrically around a pure-damping
collision step. Implemented in `DensityMatrixSolver.evolve_step_ode` and
`DensityMatrixSolver._apply_unitary`, regression-tested via
`test_mode5b_qke_full_ode`.

SM observable shift vs. the quasi-static path: Neff 3.0445 → 3.041
(-4×10⁻³, ~0.13 %). This is a consequence of the numerical regime,
not an error: the quasi-static path assumes the off-diagonal coherence
has reached its steady-state within each timestep, while the exact-
unitary path resolves the build-up explicitly. The current evolve_step
happens to be closer to Bennett+2021; whichever is "more correct"
against a full QKE ODE reference is an open question the Stage D.2/D.3
follow-ups are meant to answer.

### Open follow-ups (Stage D.2 and beyond)

- **D.2 — ETDRK2 corrector.** Upgrade the Strang-split integrator to
  an ETDRK2 scheme (exponential time differencing with predictor +
  φ_2 corrector) for O(dt²) splitting error.
- **D.3 — Eigenbasis collision step.** Reformulate the collision
  operator in the instantaneous H-eigenbasis so that the off-diagonal
  damping and H-commutator are diagonal simultaneously. This is the
  "proper" way to avoid splitting error entirely.
- **Magnus expansion** (order 4) for the unitary block, or an
  **implicit BDF** integrator adapted to the Hermitian ρ evolution
  with Jacobian support. Either would make the ODE driver robust at
  extreme parameters (very small mixing, very stiff damping).

Stage D.1 delivers the infrastructure; D.2+ deliver quantitative
improvements. No urgency unless a concrete physics use-case demands
tighter than ~10⁻³ accuracy in a regime where the quasi-static
approximation breaks down.

Estimated effort for each follow-up: **~2–4 days** on top of D.1.
