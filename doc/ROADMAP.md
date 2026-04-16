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

One item remains on the backlog as genuine open work:

## Full QKE as an independent ODE driver

Current QKE path uses Strang operator splitting — exact unitary oscillation
`exp(-i H dt/2)` interleaved with explicit collision + off-diagonal damping.
A **fully-coupled implicit driver** on `dρ/dt = -i[H, ρ] + C[ρ]` would be
a research-grade architectural contribution.

The main obstacle is stiffness: oscillation timescales (ω ~ Δm²/2E) are
much shorter than collision timescales (Γ ~ G_F² T⁵), especially at high T.
Realistic options:

- **Exponential integrators** that handle the oscillation part exactly
  while integrating the collision part explicitly — conceptually similar
  to current Strang splitting but with smaller splitting error.
- **Implicit BDF methods** adapted to the Hermitian ρ evolution, with
  careful handling of the non-Hermitian `-i H` block.
- **Hybrid schemes** (e.g., Magnus expansion for the unitary part).

No SM observable impact expected — Strang splitting already reproduces
Bennett+2021 to 10⁻⁴. The payoff would be architectural: cleaner
interaction with BSM matter potentials, easier to extend to new osc
sectors (the sterile stages already benefit, but more would be added
at lower cost with a full QKE driver).

Estimated effort: **1–2 weeks** of ODE engineering + re-validation.
