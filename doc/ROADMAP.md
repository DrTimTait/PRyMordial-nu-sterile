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

**Stage D.4 (landed):** SF dt-convergence diagnostic. Unlike the 3×3
case (D.2), the 4×4 SF regime does NOT converge cleanly at the default
`n_B=2400`. Running the ξ=5e-2 SF scenario through the ODE driver at
n_B ∈ {2400, 4800, 9600}:

| n_B   | ODE Neff  | ΔNeff to prior |
|------:|----------:|---------------:|
| 2400  | 3.95297   | —              |
| 4800  | 3.96172   | +8.8×10⁻³      |
| 9600  | 3.96562   | +3.9×10⁻³      |

Drift ratio = 0.45, closer to 0.5 (O(dt)) than 0.25 (O(dt²)). The
Strang-split integrator degrades from its usual O(dt²) to O(dt) at the
MSW resonance crossing, because the in-medium mixing angle sweeps
through π/4 on a time-scale shorter than dt near resonance. Richardson
extrapolation gives ODE Neff_∞ ≈ 3.97, so the Stage D.3 reported
values (at default n_B=2400) are ~1.4×10⁻² below the true ODE answer.

Two gaps to distinguish:

- **Numerical gap** (~1.4×10⁻² Neff, ODE at default n_B vs ODE-∞).
  ETDRK2 would restore O(dt²) at the resonance and close this gap at
  default dt. A simpler workaround for users who need precision is to
  set `PRyMini.n_B_override = 9600` or higher for SF runs — the
  diagnostic script in `validation/stage_d_sf_convergence.py` uses
  this knob.
- **Physics gap** (~1×10⁻² Neff, ODE-∞ ≈ 3.97 vs Strang 3.98). The
  quasi-static DW formula spikes at resonance (see D.3 discussion).
  No numerical improvement closes this — it's the modeling choice.

Recommendation for SF ODE users: set `n_B_override` ≥ 9600 in
`PRyMini` before constructing `PRyMclass`. The diagnostic output
(`validation/stage_d_sf_convergence.out.txt`) documents the trend.

**Stage D.5 (documentary finding):** A naive predictor-corrector on
the Hamiltonian was attempted — predictor Strang step with H(ρ_n),
then corrector Strang step starting from ρ_n with H(ρ_mid) where
ρ_mid = ½(ρ_n + ρ_pred). Hypothesis was that this would restore
O(dt²) at the MSW resonance. Result
(see `validation/stage_d5_sf_convergence_attempt.py` /.out.txt):

| n_B  | Corrector Neff | vs. D.4 single-stage |
|-----:|---------------:|---------------------:|
| 2400 | 3.95428        | +1.3×10⁻³ (improving)|
| 4800 | 3.96442        | +2.7×10⁻³ (improving)|
| 9600 | 3.84467        | -12.1×10⁻² (UNSTABLE)|

The corrector helps at moderate n_B but destabilizes at finer dt,
because near the MSW resonance `ω_αs = H_αα − H_ss` sweeps through
zero. The predictor sees ω on one side of zero, the corrector sees ω
on the other side; the unitary halves `exp(-iω·dt/2)` rotate in
opposite senses across the two stages, and cumulative interference
corrupts the state. This is a known failure mode of naive H-averaging
predictor-corrector schemes at turning points; the fix is to handle
the commutator in the H-eigenbasis, where the phase-flip issue
disappears because each eigenmode gets its own phase.

Decision: the naive corrector was reverted; Stage D.5 ships as this
documentary finding. The attempt script is preserved at
`validation/stage_d5_sf_convergence_attempt.py`.

**Stage D.6 (landed, partial):** ETDRK2-in-eigenbasis scaffold, plus
the concrete finding that the Cox-Matthews corrector breaks ν-ν̄
symmetry in this formulation and the target fix is actually D.7.

What D.6 shipped:

- Flag `qke_ode_etdrk2_flag` (default False), gated inside
  `qke_full_ode_flag`. Reset in `tests/test_regression.py::_reset_flags`
  and wired into `PRyM_main.py`'s Phase-B dispatcher.
- New `DensityMatrixSolver` methods (non-invasive — `evolve_step` and
  `_apply_unitary` untouched):
  * `_build_H_list` — per-sector H in eV, shape `(Ny, N, N)`.
  * `_apply_unitary_from_eigs` — in-place unitary conjugation reusing
    precomputed `(lam, V)`; preserves the ν̄ storage convention
    (`U*·ρ·Uᵀ` on sector 1) from D.3.
  * `_assemble_collision_N` — full collision RHS as per-sector
    Hermitian `(Ny, N, N)` in eV natural units, returning `(N_list,
    I_total)`.
  * `_etdrk2_phi_apply` — rotate into H-eigenbasis (sector-aware sign
    flip + `V̄ᵀ A V̄*` for sector 1), apply per-eigenpair
    `dt·φ_k(-iω dt)` element-wise (closed form + Taylor series below
    `|z|<1e-4`), rotate back, Hermitise.
  * `evolve_step_ode_etdrk2` — IMEX step that runs one full-step
    unitary, then adds `dt·φ_1(L dt)·N_off(ρ_n)` in the eigenbasis
    (the Stage-D.6 payload, giving each eigenpair its own φ factor at
    the MSW turning point), then the same flavor-basis diagonal
    exp-Euler `evolve_step_ode` uses, then the standard off-diagonal
    clamp / diagonal clip. **Predictor-only: first-order accurate.**
- Smoke test `test_mode5c_qke_ode_etdrk2` (3×3 SM) — PASSES inside the
  mode-5b tolerance envelope.

What D.6 discovered but did NOT ship:

The Cox-Matthews second-order corrector
`ρ_{n+1} = ρ* + dt·φ_2(L dt)·(N(ρ*) − N(ρ_n))` breaks ν-ν̄ symmetry
in the current L=−i[H,·] formulation. Measured on sterile DW (ξ=0,
sin²(2θ_14)=0.1):

| Variant                              | Neff    | Σρ_ss | n_ξe       |
|--------------------------------------|---------|-------|-----------:|
| Strang (evolve_step_ode)             | 3.97520 | 6.127 | -8.53×10⁻² |
| ETDRK2 full corrector (tried)        | 3.99122 | 6.257 | -1.895     |
| ETDRK2 corrector, k=l eigendiag = 0  | 3.97085 | 6.001 | -1.987     |
| ETDRK2 + eigenvector phase-fix       | 3.98147 | 6.257 | -5.661     |
| **ETD1 predictor only (landed)**     | 3.97638 | 6.379 | -0.123     |

Root cause (proved analytically): for a ν-ν̄-symmetric state
(ρ_phys = ρ̄_phys so stored `ρ̄* = conj(ρ)`), sector 1 enters
`_etdrk2_phi_apply` through `(V*, Vᵀ)` rotations while sector 0 uses
`(V, V†)`. After applying φ_k elementwise, the rotated-back results
are
```
A_0_out = V · F · Ã · V†
A_1_out = V* · conj(F) · Ã · Vᵀ
```
which are **not** complex conjugates of each other for generic complex
V (only for V permutation-real). The predictor avoids the accumulation
because its sole N-contribution `dt·φ_1·N(ρ_n)` is evaluated at the
same ρ_n for both sectors and its symmetry breaking resolves into the
sector propagation exactly. The corrector accumulates `N(ρ*) − N(ρ_n)`
at a post-unitary ρ*, where the V-rotation mismatch integrates into
an O(1) asymmetry over 2400+ Phase-B steps.

The SF (ξ≠0) regime shows the same class of drift even in the
predictor-only path because the initial ν-ν̄ asymmetry seeds the
instability; see
`validation/stage_d6_sf_convergence.out.txt`.

**Deferred to Stage D.7:** full ETDRK2 with `L = −i[H,·] − D_αβ`
(flavor-basis pair damping absorbed into the linear part). L is then
not H-eigenbasis-diagonal, so the per-mode matrix exponential and
φ_k construction must be done on the 9×9 (or 16×16) superoperator
directly — a ~1-week refactor that genuinely closes the SF O(dt²) gap.

Until D.7 lands, the practical recommendation for SF ODE users is
unchanged from D.4: set `n_B_override ≥ 9600` to converge the
single-stage Strang path. The new `qke_ode_etdrk2_flag` is a working
scaffold but should be left False for SF runs — it's first-order and
not an accuracy improvement over Strang at moderate `n_B`.
