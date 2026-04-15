# PRyMordial-nu: Future work

After Task 1 (O(e⁴) QED) and Task 2 (μ-τ / ν-ν̄ symmetry breaking) landed,
the code was rounded out with literature comparisons, a spectral-distortion
plot, BSM demo scenarios, updated demo notebooks, a pytest regression
suite, numba persistent caching, exact 3-flavor PMNS for n=4/n=6, a
massive-electron n=6 ν-e integral, the `nlo_weak_flag` placeholder, and
a nu-e dispatcher refactor. See the git log for details (commits
`efe4968` through `6ea0bf4`).

Two items remain on the backlog as genuine open work:

## 1. Sterile neutrino production via MSW resonance

Biggest new BSM capability. Would extend the QKE density-matrix path with
active-sterile mixing (θ_14, θ_24, θ_34 + sterile mass m_s) and add the
corresponding Hamiltonian and (optionally) lepton-asymmetry-tracking
infrastructure for:

- **Dodelson-Widrow** (non-resonant) sterile production via frequent
  active-active collisions mixing into a sub-leading sterile mass eigenstate.
- **Shi-Fuller** (resonant) production via MSW level-crossing driven by a
  non-zero primordial lepton asymmetry.

Opens access to a large literature of sterile-ν constraints from BBN, Neff,
warm dark matter, and short-baseline anomalies. Implementation requires:

- Expanding `DensityMatrixSolver`'s 3×3 ρ to 4×4 (or keeping 3×3 with an
  auxiliary sterile equation).
- New Hamiltonian terms with the sterile mass-squared splitting.
- Re-using the `n_species=6` ν/ν̄-distinct collision machinery (Stage 3)
  for Shi-Fuller lepton-asymmetry feedback.
- Validation benchmarks against Dolgov+2002 / Hannestad+2012.

Estimated effort: **1–2 weeks** of careful physics + validation work.

## 2. Full QKE as an independent ODE driver

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
sectors (items like item 1 above).

Estimated effort: **1–2 weeks** of ODE engineering + re-validation.
