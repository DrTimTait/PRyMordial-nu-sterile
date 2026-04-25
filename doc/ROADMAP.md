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

**Stage D.7 (landed, hybrid scope):** full ETDRK2 with off-diagonal
damping absorbed into the linear part `L = −i[H, ·] − D_off ⊙ ·`,
computed per mode via the Al-Mohy & Higham (2011) augmented-matrix
`expm` on the 9×9 (or 16×16) vectorised Liouvillian. No H-eigenbasis
rotation anywhere; `D` is real so the sector-1 damping block is
identical to sector-0 and ν-ν̄ symmetry is algebraic. **The D.6
Cox-Matthews symmetry-breaking instability is eliminated.**

What D.7 shipped:

- `DensityMatrixSolver._build_L_list` — assembles per-sector 9×9/16×16
  `L` in eV and the gain-only off-diagonal `N` with `D_off·ρ` added back
  to cancel `_assemble_collision_N`'s damping subtraction.
- `DensityMatrixSolver._etdrk2_expm_phi` — Al-Mohy & Higham augmented
  matrix `M = [[L·dt, I, 0], [0, 0, I], [0, 0, 0]]`; one `scipy.linalg.expm`
  call yields `e^{L·dt}`, `dt·φ_1(L·dt)`, `dt²·φ_2(L·dt)` as top-row blocks.
  (Third block divided by `dt_nat` to yield `dt·φ_2` for the ETDRK2
  corrector formula.)
- `DensityMatrixSolver.evolve_step_ode_etdrk2` rewritten: full
  predictor + corrector on off-diagonals, diagonal exp-Euler between
  them (identical to `evolve_step_ode` regularisation). Clamp + clip
  blocks carried over from D.6.
- D.6 auxiliary scaffolding `_apply_unitary_from_eigs` and
  `_etdrk2_phi_apply` removed — the superoperator approach supersedes
  them.
- `tests/test_regression.py::test_qke_etdrk2_nu_nubar_symmetry`: fast
  unit test asserting `max |rho[0] − conj(rho[1])| < 1e-10` after 10
  steps with V_CC forced to zero. Includes pre-asserts for vec/mat
  round-trip and kernel-level sector symmetry so an L-side bug can be
  isolated from a pre-existing gain-kernel asymmetry.
- `test_mode5c_qke_ode_etdrk2` Neff tolerance tightened from 3e-3 to
  1e-3 (matches mode-5b Strang exactly now that the D.7 driver is
  second-order).
- `validation/stage_d7_sf_convergence.py` + `.out.txt` — SF sweep at
  n_B ∈ {2400, 4800, 9600}.

Hybrid-scope rationale: damping-in-L is applied ONLY to off-diagonal
entries. Diagonal occupations continue on the existing `phi1_dt`
exp-Euler step. Off-diagonal coherence is what drives O(dt²) at MSW;
PRyMordial's `I_total` is the full collision kernel, not a schematic
`−Γ_α·ρ_αα` relaxation, so the brief's `I_total_α + Γ_α·ρ_αα`
diagonal subtraction would be a large cancellation during the SF
sweep when `ρ_ee` departs from FD. Full-diagonal-in-L with an
`I_total`-derived damping is the documented D.7.1 fallback.

Measured SF convergence (`validation/stage_d7_sf_convergence.out.txt`,
sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2):

| n_B  | Neff     | Σρ_ss | n_ξe   | wall (s) |
|------|---------:|------:|-------:|---------:|
| 2400 | 3.94000  | 5.533 | +3.40  | 1243     |
| 4800 | 3.95633  | 6.043 | +3.06  | 1661     |
| 9600 | 3.96223  | 5.999 | +2.19  | 3368     |

- ΔNeff(2400→4800) = +16.3×10⁻³, ΔNeff(4800→9600) = +5.9×10⁻³.
- Drift ratio = 0.362. **Clear second-order character** (contrast D.6's
  ratio −3.3), but not yet dominated by O(dt²) (target 0.25).
- Richardson-extrapolated Neff_∞ = 3.9642; D.4 Strang Richardson limit
  3.9682. **Agreement to 4×10⁻³** — D.6 had no defined Richardson limit
  (non-convergent). The residual 4×10⁻³ gap to Strang's limit is
  consistent with the hybrid scope's Lie-Trotter diagonal/off-diagonal
  split being formally O(dt).
- n_ξe stabilises at **+2.2 at n_B=9600**, matching the brief's target
  "in the same ballpark as Strang's +2.2". D.6 predictor drifted to
  −34 at n_B=9600.

What D.7 does NOT yet do:

- **Strict O(dt²) convergence at default n_B=2400**. Ratio 0.36 and
  default-n_B Neff 2.8×10⁻² below the Richardson limit indicate a
  residual O(dt) contribution. The most likely source is the
  Lie-Trotter-style split: off-diag ETDRK2 predictor → diag exp-Euler
  → off-diag ETDRK2 corrector is not symmetric, so the operator-split
  error is O(dt). A Strang-symmetric diagonal split (half-diag before
  the predictor + half-diag after the corrector) would make the split
  O(dt²) with minimal code change. **Deferred as D.7.1.**
- Closing the 4×10⁻³ Richardson gap to Strang-Strang requires either
  D.7.1 or the brief's full-diagonal-in-L variant. The latter has the
  cancellation concern noted above and would need an `I_total`-derived
  damping estimator.

Practical recommendation (updated): for SF ODE users, D.7 is now a
**qualitative improvement** over Strang-alone at default n_B — the
n_ξe runaway is eliminated and Richardson extrapolation is meaningful.
For O(dt²) accuracy targets at default n_B, continue using
`n_B_override ≥ 9600` with either driver (Strang or D.7). The D.7
driver is slower (~4 min overhead per Phase-B call from the per-mode
`expm`), so there is no hard reason to prefer it over Strang until
D.7.1 (or full-diagonal-in-L) closes the gap.

Full regression suite status after D.7: modes 1, 2, 5, 5b, 5c (with
tightened 1e-3 Neff), 6, sterile_stage_a_invariant,
sterile_dw_production, sterile_sf_asymmetry_depletion, and the new
test_qke_etdrk2_nu_nubar_symmetry all PASS. Default-path
(qke_ode_etdrk2_flag=False) unchanged — modes 1/2/6 pass with their
existing 1e-5 Neff tolerances, confirming bit-identity on the default
Strang path.

**Stage D.7.1 (landed):** Strang-symmetric diagonal split around the
ETDRK2 off-diagonal predictor/corrector, closing the formal-O(dt)
Lie-Trotter gap that D.7 left open.

D.7's per-step structure was `predictor → full-dt diag exp-Euler →
corrector`. That ordering is Lie-Trotter-style and its operator-split
error is O(dt), which poisoned the off-diag ETDRK2's formal O(dt²) and
landed the SF drift ratio at 0.36 with default-n_B Neff 2.8×10⁻² below
the Richardson limit.

D.7.1 replaces that with

```
  ½-dt diag exp-Euler  [ I_total evaluated at ρ_n ]
  ETDRK2 predictor     [ off-diag, uses L(ρ_n) and N_gain(ρ_n) ]
  ETDRK2 corrector     [ off-diag, uses Phi2(L(ρ_n)) and dN(ρ*, ρ_n) ]
  ½-dt diag exp-Euler  [ I_total evaluated at ρ_after-corrector ]
```

Two implementation details matter:

1. `phi1_dt` supplied by `PRyM_main.py` is computed for the **full** dt:
   `phi_1(Γ·dt)·dt`. Naively halving it would give `½·phi_1(Γ·dt)·dt`
   which differs from the correct half-step regulariser
   `phi_1(Γ·dt/2)·(dt/2)` by up to a factor of 2 in the stiff
   (`Γ·dt ≫ 1`) limit. `evolve_step_ode_etdrk2` now **computes
   `phi_1(Γ·dt/2)·(dt/2)` locally** using the solver's own `C_D` and
   the PRyM_main convention (3 channels: [nue, nuebar, numu_eff], the
   first two sharing the electron-flavor `C_D[0]`).
2. The second half-step uses `I_total` re-evaluated at the state going
   into it (after the corrector), not `I_total(ρ_n)`. Concretely, an
   extra `_assemble_collision_N` call is made between the corrector
   and the second half-diag. This costs one collision-integral
   evaluation per step — well below the per-step `expm` budget — and
   is what promotes the split from frozen-coefficient O(dt) to
   state-consistent O(dt²).

The `phi1_dt` argument to `evolve_step_ode_etdrk2` is kept for
dispatcher signature stability but is not consumed by the D.7.1 body.

Measured SF convergence
(`validation/stage_d7_sf_convergence.out.txt`, identical configuration
to the D.7 sweep above — sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2):

| n_B  | Neff     | Σρ_ss | n_ξe   | wall (s) |
|------|---------:|------:|-------:|---------:|
| 2400 | 3.96184  | 5.894 | +3.10  | 967      |
| 4800 | 3.95865  | 5.810 | +2.79  | 2432     |
| 9600 | 3.96445  | 5.897 | +2.60  | 4124     |

- Default-n_B (2400) Neff jumped from D.7's 3.94000 to D.7.1's **3.96184**,
  collapsing the gap to Strang's Richardson limit 3.96823 from D.7's
  2.8×10⁻² down to **6.4×10⁻³**. This is ~6× off the brief's strict
  "within ~10⁻³" target but is already **better than Strang at its own
  default n_B=2400** (Strang lands at 3.95297, which is 1.5×10⁻² below
  its own Richardson limit); D.7.1 at default n_B is 9×10⁻³ *above*
  Strang-at-default-n_B and closer to the true n_B→∞ limit.
- Richardson-extrapolated Neff (from n_B=4800, 9600) = **3.96638**.
  D.4 Strang Richardson = 3.96823. **Gap 1.85×10⁻³** — 2.2× closer
  to Strang Richardson than D.7's 4×10⁻³ Richardson.
- Drift ratio = +1.82. The Neff sequence oscillates around the
  asymptote: ΔNeff(2400→4800) = −3.2×10⁻³, ΔNeff(4800→9600) = +5.8×10⁻³.
  Amplitude (~6×10⁻³) is small relative to the n_ξe scale, suggesting
  we are close to the asymptotic plateau rather than on a clean
  O(dt²) curve. Possible residual-effect candidates:
  * **Frozen-coefficient L.** The ETDRK2 caches `(Phi0, Phi1, Phi2)` at
    L(ρ_n) and uses them through both predictor and corrector. For
    state-dependent H(ρ) (via V_νν, trace terms), this freezes out an
    O(dt²) correction per step that may be non-negligible at SF MSW.
  * **Diagonal exp-Euler with full-kernel I_total.** The phi_1
    regulariser is exact for `dρ_αα/dt = −Γ_α·ρ_αα + gain_const`, but
    I_total is the full collision kernel. The gain piece varies with
    f_all and is evaluated at a single state per sub-step; this gives
    an O(dt) correction per sub-step → O(dt²) accumulated, but with
    coefficients that interact non-trivially with the Strang split.
  Closing the ratio to exactly 0.25 likely requires either a
  state-updated L cache (expensive: doubles the `expm` cost) or a
  midpoint I_total evaluation in the half-diag steps.

**Attempted D.7.2 (state-space midpoint-L, discarded):** A session
tried building `L_mid` at `ρ_mid = ½·(ρ_n + ρ_*_tentative)` where
`ρ_*_tentative` is a full-dt ETD predictor at `L(ρ_n)`, then using
`L_mid` for the real predictor and corrector. **Made SF accuracy
worse** — default-n_B Neff moved from 3.96184 (D.7.1) to 3.95571
(gap to Strang Richardson 1.3×10⁻² vs D.7.1's 6.4×10⁻³), and `n_ξe`
drifted from +3.10 to +4.01 at n_B=2400.

Root cause of the failure: for stiff-oscillatory L with `|L·dt_nat|
~ 10⁴`, the ETD predictor `e^{L·dt}·ρ_n` rotates off-diagonal
components by huge phases — `ρ_*_tentative` has off-diagonal content
that looks like noise relative to `ρ_n`. The state-space average
`(ρ_n + ρ_*_tentative)/2` has DILUTED off-diagonal amplitude rather
than approximating `ρ(t_n + dt/2)`. `V_νν(ρ_mid)` is therefore
mis-estimated and `L_mid` is worse than `L(ρ_n)` as a representative
linear operator.

The principled fix requires a separate half-dt ETD predictor
(`e^{L·dt/2}·ρ_n + dt/2·φ_1(L·dt/2)·N_n`) to obtain a genuine
time-midpoint estimate. That is another `expm` call per step at
`L·dt_nat/2` — on top of the `expm` at `L·dt_nat`, so ~3× D.7.1
runtime with uncertain payoff. Deferred indefinitely; the residual
6.4×10⁻³ gap at default n_B is below the BBN-observable sensitivity
floor (Yp and D/H shift by &lt;10⁻⁴ from this level of Neff
uncertainty), so tightening further is not physics-motivated.

The D.6 ν-ν̄ asymmetry failure mode remains eliminated:
`test_qke_etdrk2_nu_nubar_symmetry` PASSES under the D.7.1 driver
without modification (Strang splitting of a symmetry-preserving
operator is trivially symmetric).

Full regression suite status after D.7.1: `test_mode5c_qke_ode_etdrk2`
PASSES with the tightened 1e-3 Neff tolerance (as under D.7);
`test_sterile_dw_production` PASSES. The remaining slow tests
(modes 1, 2, 5, 5b, 6, sterile_stage_a_invariant,
sterile_sf_asymmetry_depletion) are unaffected by the D.7.1 change —
they do not enable `qke_ode_etdrk2_flag`, so they route through
`evolve_step_ode` (Strang) which D.7.1 does not touch. Default-path
bit-identity is preserved.

Practical recommendation (updated): D.7.1 is the recommended driver
for SF ODE runs at default n_B. The `expm`-per-mode overhead adds
~30% runtime versus Strang, but Neff now lands within ~10⁻² of the
n_B→∞ limit at n_B=2400, versus needing `n_B_override ≥ 9600` to
match Strang's Richardson limit under the Strang driver alone.

---

## Stage E: literature validation and physics corrections

**Stage E.1 (landed):** pair-specific off-diagonal damping formula
via new `DensityMatrixSolver._compute_D_pair_matrix` helper, selected
by `PRyMini.qke_damping_formula ∈ {"symmetric", "mirizzi", "gariazzo"}`
(default now `"mirizzi"`). The shared helper is called from all four
live damping sites — `_build_L_list`, `_assemble_collision_N`,
`evolve_step_ode`, and `evolve_step` — so the formula is consistent
across every QKE driver (D.7.1 ETDRK2, pre-D.7.1 ODE, and Strang).
The dead `collision_step` method retains its inline symmetric form.

The Mirizzi form (Mirizzi+2012 Eq. 28, coefficients from their
Eq. 29 citing Hannestad) is `D_αβ = 0.5·G_F²·T⁴·E · [(g_α^s − g_β^s)²
+ (g_α^a + g_β^a)²]` with `g^s = √C_D`, `g^a = √C_A`, and the new
`C_A = [0.50, 0.28, 0.28, 0.0]` annihilation coefficients. For
active-sterile pairs this gives `D_μs = 1.25·base` vs the legacy
symmetric `0.5·Γ_μ = 1.11·base` — a 12.6% increase that is the
physically correct Lindblad structure but quantitatively small in
the saturated-damping regime of our benchmarks.

**Validation ladder:**

| Sub-step | Target | Result |
|---|---|---|
| diag_2level_damped.py | L-expm ≡ analytic damped Rabi | ρ_ss = 1.580e-4 both (4 sig figs) ✓ |
| fast tests (mode1/2/6 + ν-ν̄ symmetry) | all green | 4/4 ✓ |
| mode5c + sterile_dw_production | all green | 2/2 ✓ |
| stage_a_invariant + sf_asymmetry_depletion | all green | 2/2 ✓ |

**Literature-comparison outcomes (Hannestad+2012 Fig. 2, Δm²=0.93 eV²):**

| Point | sin²2θ | H+2012 δNeff | Ours (Mirizzi) | Δ/H |
|---|---:|---:|---:|---:|
| A full therm. | 1e-1 | 1.00 | 0.955 | −4.5% ✓ |
| B partial | 2.26e-3 | 0.50 | 0.970 | +94% ✗ |
| C minimal | 1e-4 | 0.04 | 0.858 | +2045% ✗ |

**Gariazzo+2019 Fig. 3 benchmark** (Δm²=1.29, |U_μ4|²=1e-4,
i.e. sin²2θ≈4e-4): their δNeff ≈ 0.09 vs ours 0.923 → +925%.

**Diagnosis.** Mirizzi's correction to the damping *coefficients* has
essentially no effect at small mixing (C goes from 0.85 → 0.858),
confirming that the literature gap is **not** a damping-prefactor
problem. Gariazzo+2019 App. A.17-A.20 (paper verified, Eq. A.16)
uses a quite different coefficient structure (`D_μs/D_μτ = 4.23` vs
Mirizzi's 2.23 vs our legacy 0.50), but the physics says a larger D
in the Sigl-Raffelt weak-damping regime (`|H_diff|·dt ~ 2.6e4`,
`D·dt ~ 11`) gives linearly *more* DW rate, yet Gariazzo's code
reports ~10× *less* thermalization than ours. This is structurally
inconsistent with a damping-coefficient fix, so Stage E.1 stops
at Mirizzi-as-default and the Gariazzo form is left as a stub
(`NotImplementedError`) behind the `"gariazzo"` flag value until
Stage E.2 opens.

**Stage E.2 (partially landed — hypothesis A falsified, hypothesis B
unresolved):** deeper bug hunt for the small-mixing DW over-production.
Three candidates flagged by
[validation/diagnostics/README.md](../validation/diagnostics/README.md);
E.2's first sprint tested the two highest-likelihood.

**E.2(a) — hypothesis A (active-sterile gain asymmetry): FALSIFIED.**
`validation/diagnostics/diag_as_gain.py` patches three variants of
`_offdiag_collision_gain` at Hannestad Point C (sin²2θ_24=1e-4,
Δm²_41=0.93, PMNS on, 4-flavor, Mirizzi damping):
  (i) baseline, (ii) zero all off-diagonal gain, (iii) symmetric
  active-sterile gain (pair q ← pair q).

| Variant | ΔNeff | sum ρ_ss |
|---|---:|---:|
| (i) baseline | +0.8580 | 4.96 |
| (ii) zero all gain | +0.8452 | 5.08 |
| (iii) sym active-sterile gain | +0.8424 | 5.01 |

All three agree within ~2%, and variants (ii) and (iii) move in the
*wrong* direction (ρ_ss slightly rises). The `S_gain_si = 0.0`
active-sterile branch at `PRyM_boltzmann.py:4473` is not load-bearing.

A prior PMNS-off run
(`validation/diagnostics/diag_as_gain_pmns_off.out`) gave all three
variants at ΔNeff = +0.2928 identically — expected, because
active-active coherences have no Hamiltonian driving without PMNS
and `_offdiag_collision_gain` is linear in the coherence input. The
PMNS-off null does NOT rule out A; it's physically unobservable
there. The PMNS-on run is the real test, and it falsifies.

**E.2(b) — hypothesis B (QKE evolution-window mismatch): UNRESOLVED.**
`validation/diagnostics/diag_qke_window.py` varies `T_boltz_start`
at Point C (PMNS on, Mirizzi):

| T_boltz_start | Neff | ΔNeff | sum ρ_ss |
|---:|---:|---:|---:|
| 5 MeV (default) | 3.886 | +0.8453 | 5.08 |
| 30 MeV | 3.090 | +0.0496 | 26.17 |
| 60 MeV | 13.66 | +10.62 | 35.81 |

The 30 MeV run lands at ΔNeff=0.05, suspiciously close to
Hannestad's 0.04 — but its `sum ρ_ss = 26` is internally
inconsistent with that Neff (thermal sterile would give ΔNeff ~ 0.5).
The 60 MeV run gives Neff = 13.66, unphysical for 3+1.
The Phase B Froustey formalism and/or the ETDRK2 step-size policy
appears to break at T ≫ 5 MeV.

A sanity control (`validation/diagnostics/diag_phaseA_only.py`)
pushes `T_start = 60 MeV` but keeps `T_boltz_start = 5 MeV`
(Phase A extended, Phase B default): ΔNeff = +0.8514, matching the
default-window baseline to 0.7%. This confirms the dramatic shifts
above are Phase-B-QKE effects, not Phase-A artefacts.

**B conclusion (sprint 1).** Window extension moves ΔNeff in the
right direction, but the numerics at T > 5 MeV are unreliable, so we
could not commit a fix. A proper test of B requires stabilising
Phase B's Froustey entropy equation and the ETDRK2 step-size policy
at high T — addressed in sprint 2 below.

**E.2 sprint 2 — Phase B stabilisation at extended windows: LANDED.**
Three suspects from `doc/STAGE_E2_SPRINT2_BRIEF.md` were
investigated and two of three mitigations landed as code changes.

**Suspect 1 (n_B T-range-blind): partially confirmed, mitigated.**
`validation/diagnostics/diag_nB_convergence.py` scanned
`n_B ∈ {2000, 5000, 10000}` at T_boltz_start = 30 MeV (Point C):

| n_B | Neff | ΔNeff | sum ρ_ss | runtime |
|---:|---:|---:|---:|---:|
| 2000 | 3.11177 | +0.1019 | 25.829 | 13 min |
| 5000 | 3.04367 | +0.0338 | 26.976 | 34 min |
| 10000 | 3.08632 | +0.0764 | 27.405 | 66 min |

`sum ρ_ss` converges monotonically (Δ: +1.15 → +0.43), asymptote
~27.6. Neff is noisier at the ±0.05 level; step count isn't the
sole driver. Sprint 1's n_B=2000 was clearly too coarse at 30 MeV
(~6% error in sum ρ_ss, and most damaging at the 60 MeV catastrophe
below). Fix landed in `PRyM.PRyM_init.validate_configuration`: when
`qke_full_ode_flag=True` and `n_B_override is None`,
`n_B_override = int(2000 · max(1, decades/3)^4)`, calibrated so that
at T_boltz_start=30 MeV it lands `n_B ~ 5031` (sum ρ_ss stabilised),
at 60 MeV `n_B ~ 6836`, and at the default 5 MeV window yields
exactly 2000 (regression tests bit-identical).

**Suspect 2 (QED extrapolation above 40 MeV): confirmed by
inspection, mitigated.**
`PRyMrates/thermo/QED_P_int.txt` + siblings all cap at T = 40 MeV;
baseline `interp1d`s used `fill_value="extrapolate"`, so at T = 60
MeV the corrections are linearly extrapolated (see
`validation/diagnostics/diag_thermo_ranges.py`). The O(e^4) tables
already clamped to zero via `fill_value=0.0`. Fix landed in
`PRyM.PRyM_thermo`: baseline `PofT`, `dPdT`, `d2PdT2` now
zero-clamp above `_T_QED_TABLE_TOP = 40 MeV` (matching the e⁴
precedent), with a validator warning when
`T_boltz_start > 40 MeV and qke_full_ode_flag=True`. Fractional
effect on `spl` at 60 MeV: +0.18% (safely dominated by
`drho_g_dT + drho_e_dT` in the Phase A denominator). Below 40 MeV
behaviour is bit-identical.

**Suspect 3 (ETDRK2 expm at |L·dt| ≫ 1): ruled out.**
`validation/diagnostics/diag_2level_high_T.py` built the full 4×4 L
at T=60 MeV for the first Phase B step and compared
`_etdrk2_expm_phi`'s Phi0 against `scipy.linalg.expm(L·dt)` directly.
At `||L·dt||_inf = 2.65e4` the Al-Mohy augmented-matrix construction
matches to `3.4e-21` relative, with finite Phi1/Phi2 and no
NaN/Inf. The expm driver is not the bug.

**Post-fix window scan (`diag_qke_window.py`, Point C).**

| T_boltz_start | Neff (sprint 1) | Neff (sprint 2) | ΔNeff (s2) | sum ρ_ss (s2) |
|---:|---:|---:|---:|---:|
| 5 MeV | 3.886 | 3.886 | +0.845 | 5.08 |
| 30 MeV | 3.090 | 3.156 | +0.116 | 26.8 |
| 60 MeV | **13.66** | **3.795** | +0.754 | 35.9 |

Sprint 2's **60 MeV value is physical**; sprint 1's was not. All
three runs are now internally runnable without spurious Neff
blowups. However, ΔNeff across the three windows is
**non-monotone** (0.85 → 0.12 → 0.75) — so hypothesis B is **not a
clean window-mismatch story**. Sprint 1's "suspiciously close to
Hannestad 0.04" ΔNeff=0.0496 at 30 MeV was partly a methodology
artefact (3x3 reference at window=5 MeV produces `Neff=3.04071`
while at window=30 MeV it is `Neff=3.00991` — the matched-window
ΔNeff is 0.15, not 0.05).

**B conclusion (post-sprint 2).** Window extension does alter
ΔNeff substantially, so the 5-MeV default window is not neutral —
but the shift is not monotone, so the dominant small-mixing DW
overproduction bug is not simply a missing high-T window. The
engineering fixes (auto-scaled n_B, zero-above-40-MeV QED clamp)
are worth keeping regardless. Hypothesis B is effectively
**closed**; hypothesis C (y-grid discretisation) tested below.

**E.2 sprint 3 — hypothesis C (y-grid discretisation): FALSIFIED.**
`validation/diagnostics/diag_ny_convergence.py` runs Point C at
`Ny_boltz ∈ {50, 100, 200}` (default 5-MeV window, PMNS on, Mirizzi).
Matched-Ny 3x3 references used for ΔNeff (Ny=50/100 use the Ny=100
reference, Ny=200 uses the Ny=200 reference — the 3x3 Neff shifts by
only 2e-4 across 100↔200, so reference drift is a non-issue).

| Ny | Neff | ΔNeff | sum ρ_ss | runtime |
|---:|---:|---:|---:|---:|
| 50 | 3.90904 | +0.8683 | 2.624 | 6 min |
| 100 | 3.89870 | +0.8580 | 4.963 | 17 min |
| 200 | 3.89523 | +0.8547 | 9.751 | 66 min |

ΔNeff drifts monotonically 0.8683 → 0.8580 → 0.8547. Total drift
Ny=50→200 is +0.014, just above the brief's 1e-2 rule-out threshold
but clearly converging (deceleration ratio 0.3 between successive
steps; Richardson asymptote ≈ 0.852). **Even at Ny→∞, ΔNeff
stays at ~0.85** — nowhere near the Hannestad+2012 target of 0.04.
The y-grid discretisation is NOT the cause of the 10× small-mixing
DW overproduction. Ny=400 not tested (O(Ny³) cost prohibitive); the
trend is decisive without it.

Note: `sum_rho_ss` doubles with each Ny doubling (2.6 → 5.0 → 9.8).
That's a diagnostic-script artefact — `rho[:, 3, :].sum()` is an
un-normalised grid-point sum, so O(Ny)-scaling is expected and not
a physics signal. Neff is computed correctly via the integral
normalisation inside PRyMclass, which is the quantity that matters.

**Open: all three E.2 brief-named structural candidates now
falsified or non-causal.** The 10× small-mixing DW overproduction
is elsewhere.

**E.2 sprint 4 — damping-magnitude sensitivity: ruled out.**
FortEPiaNO was cloned to `References/external/` (gitignored) and
an Explore-agent surveyed its collision-term and QKE structure.
Three differences vs PRyMordial surfaced: (i) FortEPiaNO has no
`V_NC` on active diagonals (matter.f90 comment literally
`!missing: term for NC!`), (ii) damping absolute normalisation
differs by ~2.6× (PRyMordial Mirizzi > FortEPiaNO McKellar for
active-sterile), (iii) FortEPiaNO uses a single-solver LSODA while
PRyMordial does Strang-symmetric ETDRK2 splitting.

To test whether damping magnitude is the bug, a `qke_damping_scale`
diagnostic knob was added to `DensityMatrixSolver._compute_D_pair_matrix`
(default 1.0 = no-op, preserving regression tests bit-identically).
`validation/diagnostics/diag_damping_scale.py` swept scale ∈
{0.3, 1.0, 3.0} at Point C:

| scale | Neff | ΔNeff | sum ρ_ss | runtime |
|---:|---:|---:|---:|---:|
| 0.3 | 3.86804 | +0.8273 | 5.18 | 16 min |
| 1.0 | 3.89870 | +0.8580 | 4.96 | 15 min |
| 3.0 | 3.89937 | +0.8587 | 4.98 | 17 min |

**ΔNeff varies by only 4% across a 10× damping sweep.** The
classical Dodelson-Widrow regime is linear in damping (0.3× ⇒
0.3× ΔNeff); we see essentially no response. PRyMordial is in a
**fully-saturated regime** where ρ_ss reaches near-thermal
regardless of D magnitude. Damping coefficient form (Mirizzi vs
FortEPiaNO McKellar vs Bennett+2020) cannot close the 20× gap.

Initial-conditions comparison (sprint-4, no commit): both codes
start from empty sterile and thermal actives. FortEPiaNO evolves
the *deviation* `ρ_dev = ρ_full/f_eq − δ`, PRyMordial evolves
`ρ_full`. These are mathematically equivalent. Physical ICs match.
No IC-layer bug identified, but two IC-adjacent audits are parked:
(a) Phase A's perfect-thermal assumption at T_boltz_start vs the
active-depletion FortEPiaNO tracks across 60→5 MeV, and (b)
whether any `_build_L_list` / `_assemble_collision_N` term fires
specifically because ρ_ss=0 (empty-sterile edge case).

**E.2 sprint 5 — Hamiltonian potential scope audit: V_nunu bug
identified and fix landed as opt-in.** Three diagnostic knobs added
to `_build_H_list`: `qke_v_nc_scale`, `qke_v_thermal_scale`,
`qke_v_nunu_scale`. `validation/diagnostics/diag_potential_scope.py`
scaled each to 0 at Point C:

| scenario | Neff | ΔNeff | sum ρ_ss |
|---|---:|---:|---:|
| baseline | 3.89870 | +0.8580 | 4.963 |
| V_NC off | 3.89199 | +0.8513 | 5.170 |
| V_thermal off | 3.88995 | +0.8493 | 4.973 |
| **V_nunu off** | **3.01081** | **−0.0299** | **0.006** |

**V_nunu was the entire driver of the saturation.** V_NC and
V_thermal are innocent (ΔNeff moves by ~0.01, noise level).

Physics of the bug: the Pantaleone-Sigl-Raffelt ν-ν self-interaction
goes through Z exchange, which couples only to the SU(2)_L doublet.
Sterile is a singlet with zero NC charge, so
`V_nunu[α, s] = V_nunu[*, s] = V_nunu[s, s] = 0` identically at
tree level. PRyMordial was computing the full 4×4 `(ρ − ρ̄)` matrix
and feeding every entry into H at `_build_H_list` — the non-zero
`V_nunu[α, s]` entries bootstrapped active-sterile coherence past
the tiny vacuum mixing, driving the saturation. The active-only
projection matches FortEPiaNO's implementation (`matter.f90:46-52`
explicitly zeroes sterile rows/cols of their `nuDensities`) and the
Sigl-Raffelt derivation.

Fix: `PRyMini.qke_v_nunu_active_only` flag (default `False`
preserves regression bit-identically). When `True`, the sterile row
and column of `V_nunu_eV` are zeroed before H assembly in all three
`_build_H_list` code paths. `validation/diagnostics/diag_vnunu_active_only.py`
at default 5 MeV window (qke_v_nunu_active_only=True):
ΔNeff = 0.858 → −0.012 (Hannestad target 0.040). **~70×
improvement in small-mixing agreement.**

Full Hannestad A/B/C suite with projection:

| Point | sin²2θ | Hannestad | orig | proj+5 MeV | proj+15 MeV |
|---|---:|---:|---:|---:|---:|
| A | 1e-1 | 1.00 | 0.953 | 0.173 | 0.342 |
| B | 2.26e-3 | 0.50 | 0.963 | 0.133 | 0.155 |
| C | 1e-4 | 0.04 | 0.853 | −0.012 | 0.009 |

The pre-fix numbers were nearly flat across A/B/C (0.95, 0.96, 0.85)
— the signature of V_nunu saturation happening to land near full
thermalisation by coincidence. The fix correctly unsaturates the
production, showing the true oscillation-damping rate. Points B and
C now undershoot, and A undershoots badly — pointing to a secondary
issue: PRyMordial's default Phase B window starts at T=5 MeV,
which is below the MSW resonance at T_MSW ≈ 10 MeV for Δm²=0.93.
Starting above T_MSW recovers adiabatic passage, which would drive
Point A back toward 1.0.

Extended-window attempts:

- proj+15 MeV: runs cleanly, gives partial progress (A: 0.17 → 0.34).
- proj+20 MeV, proj+30 MeV, proj+pointC-only @ 30 MeV: **numerically
  unstable**. Point A crashes with NaN in `scipy.linalg.expm`
  (from ETDRK2 off-diagonal clamp at `PRyM_boltzmann.py:4750`);
  Point C at 30 MeV completes but produces unphysical values
  (Neff=7.22, Yp=0.29, ΔNeff=+4.21).

Root cause of numerical instability: with V_nunu's off-diagonal
active-sterile "ballast" removed by the projection, L's eigenstructure
at high T develops a giant dynamic range — large thermal-damping
eigenvalues coexisting with tiny residual vacuum-mixing eigenvalues.
The condition number tanks `expm` precision, and the existing
off-diagonal magnitude clamp in `evolve_step_ode_etdrk2` isn't
NaN-safe. This is a separable engineering problem from the physics
of the projection fix.

Sprint-5 landing posture: projection lands as **opt-in**
(`qke_v_nunu_active_only=False` default). Users who need
small-mixing DW accuracy can flip it at the default 5 MeV window.
Document the extended-window caveat. `sterile_DW_literature.py`
continues to serve as the regression target at default config.

**Next structural candidates**, ranked post-sprint-5:

1. **Extended-window numerical stability (blocks full projection
   default flip).** The clamp at `PRyM_boltzmann.py:4750` needs a
   NaN-safe branch. Also consider whether the Al-Mohy augmented-matrix
   expm in `_etdrk2_expm_phi` is the right choice at extreme
   dynamic range vs. direct `scipy.linalg.expm`. If this is fixed,
   the projection + T_boltz_start ≈ 30-60 MeV combination should
   recover Hannestad A/B/C within 10-20% and the default can flip.
2. **Strang-split time evolution**. ρ_ss is populated *only*
   through the off-diagonal ETDRK2 commutator; diagonal collisions
   never touch it. Controlled test: drop into the D.7 predictor
   path (no Strang split, single-phase off-diag ETDRK2) at Point C
   and compare. May become irrelevant if candidate 1 solves.
3. **Representation-factor leak**. FortEPiaNO's
   `nuDensMatVecFD(i,j)` for off-diagonals divides by f_eq(y).
   PRyMordial's ρ_full(α,β) carries the f_eq(y) shape in. If
   PRyMordial's L-builder accidentally applies damping to ρ/f_eq
   somewhere, we'd see a y-shape mismatch that could amplify
   coherence. Grep `_build_L_list` for any `f_eq` or
   `np.maximum(f, ...)` multiplier.
4. **Gariazzo damping form** (from original sprint-2 post-fix list):
   `qke_damping_formula = "gariazzo"` still stubbed. Lower priority
   after sprint-4 shows damping form doesn't matter.
5. **Shi-Fuller literature (Saviano+2013)**: cross-check on related
   physics; may expose the same structural bug indirectly.

The new `validation/sterile_DW_gariazzo.py` script is parked for
reuse in post-E.2 acceptance testing. `validation/sterile_DW_literature.py`
is unchanged (Dm²=0.93 is Hannestad-specific) and continues to
serve as the regression target.

**E.2 sprint 6 — extended-window numerical stability: primary
crash resolved, deeper physics drift surfaced.** Sprint 5's
extended-window crash (Point A at w20 raising `cannot convert
float NaN to integer` inside `scipy.linalg.expm`) was targeted via
option (a) from `doc/STAGE_E2_SPRINT6_BRIEF.md`: NaN-safe clamp
fix only, no changes to the Strang-symmetric D.7.1 sequence or
the V_nunu projection physics.

Phase 1 probe (temporary, discarded before commit) instrumented
`evolve_step_ode_etdrk2` with finite-state checks at entry and after
each of the four sub-steps (half-diag-1, predictor, corrector,
half-diag-2). Ran Point A at w20 with projection on. First non-finite
entry: **step 4099, T=5.5 keV, `post-half-diag-2`, single entry at
(sector=0, comp=0, y_idx=0)**. That is: ν-sector, ρ_ee diagonal,
lowest-momentum mode, at the very tail of the Phase B window
(~5× the T_end floor of 5 keV). The origin is in the cold-T
collision integral `I_total_post` — one rare numerical excursion
that half-diag-2 writes into a diagonal. The existing `np.clip` on
line 4762 passes NaN through (numpy semantics), so the next step's
`_build_L_list` inherits NaN, `L` carries NaN, and
`_etdrk2_expm_phi`'s `scipy.linalg.expm` crashes in its
norm-estimate. The sprint-5 RuntimeWarning at the off-diagonal
clamp (line 4750) was a downstream symptom one step later, not the
origin.

Fix (Phase 2, committed): two additive sanitisation passes in
`evolve_step_ode_etdrk2`:

1. **Off-diagonal clamp** (line ~4749): before computing `ab_mag`,
   `rho_ab = np.where(np.isfinite(ab_mag), rho_ab, 0 + 0j)`. Zeros
   non-finite coherences so the clamp's `np.where(ab_mag > max_mag, …)`
   always operates on finite input.
2. **Diagonal clip** (line ~4774): `np.clip(np.nan_to_num(x,
   nan=f_min, posinf=f_max, neginf=f_min), f_min, f_max)`. Sanitises
   the diagonal at the origin (half-diag-2 write) so NaN does not
   poison the next step's `L`.

Both passes are mathematical no-ops on finite state (`np.where` with
a true-everywhere condition returns the input; `nan_to_num` on
finite input is identity). Gates 1-4 confirm bit-identity.

Validation ladder:

| # | Test | Result |
|---|------|--------|
| 1 | `pytest -m "not slow"` (default) | 4/4 bit-identical ✓ |
| 2 | `pytest -k sterile` (default) | 3/3 bit-identical ✓ |
| 3 | `diag_2level_damped.py` | ratio 1.000 ✓ |
| 4 | `diag_vnunu_active_only.py` (proj + 5 MeV) | Point C ΔNeff = −0.0119 (sprint-5: −0.012) bit-identical ✓ |
| 5a | `diag_hannestad_proj_w20.py` | no crash ✓, all Yp ∈ [0.24, 0.26] ✓ |
| 5b | `diag_hannestad_proj_w30.py` | no crash ✓, Yp unphysical at A (0.300) and C (0.293) ✗ |
| 6 | w30 Hannestad A/B/C vs literature | A=1.474, B=0.098, C=4.212 — outside tolerance ✗ |

Extended-window summary (all with projection on):

| Point | Hannestad | w5 | **w20** | w30 | w20 Yp | w30 Yp |
|-------|---:|---:|---:|---:|---:|---:|
| A | 1.000 | 0.173 | 0.113 | 1.474 | 0.24954 | 0.30046 |
| B | 0.500 | 0.133 | 0.045 | 0.098 | 0.24844 | 0.24900 |
| C | 0.040 | −0.012 | −0.032 | 4.212 | 0.24870 | 0.29320 |

sum ρ_ss grows monotonically with window size (w15 sprint-5:
9.7-15.6; w20: 12.3-21.8; **w30: 20.4-29.3**), and the w20→w30 ΔNeff
jump for Point C (−0.032 → +4.212) is non-monotonic. The deeper
physics bug is a window-dependent over-sterilisation, strongly
correlated with the same cold-T `I_total_post` numerical hazard
whose first-NaN the probe caught. When the NaN is sanitised to
`f_min`, nearby steps' integrals shift enough to bias the sterile
production at w30; at w20 the hazard is rarer (first NaN only
occasionally fires) and Yp stays physical. This is a separate
structural bug from the Phase-2 fix — the NaN-safe clamp makes the
run complete robustly, but does not fix the physical over-production.

**Sprint-6 landing posture**: clamp + diagonal sanitisation lands
as a standalone numerical-robustness fix. **Default
`qke_v_nunu_active_only` remains `False`** — the sprint-5 5 MeV
baseline is preserved bit-identical, and the extended-window
recovery of Points A and B (the original motivation for the default
flip) is blocked by the cold-T over-sterilisation, not by crashes.
Users can still opt in at the default 5 MeV window. Two benign
RuntimeWarnings at line 4755 remain in extended-window runs (clamp
sees NaN `max_mag` from `rho_aa*rho_bb` before the current step's
diagonal sanitisation fires; scale=1.0 fallback keeps the off-diag
finite — not a crash path). A future sprint may promote the
diagonal sanitisation earlier in the step sequence to suppress
these warnings entirely.

**Next structural candidates**, ranked post-sprint-6:

1. **Cold-T `I_total_post` non-finite origin.** The Phase-1 probe
   localised the first NaN to sector=0, comp=0, y_idx=0 at T≈5.5 keV
   during half-diag-2. The origin is inside
   `_assemble_collision_N(rho_all, a, Tg)`. Likely culprits:
   Fermi-Dirac exponent overflow at E/T ≫ 1 at the lowest-y edge,
   `f_eq` division-by-zero, or a numerical edge-case in one of the
   `D_pair` / scattering kernels at very cold T. Instrument
   `_assemble_collision_N` to report first non-finite intermediate,
   then fix at origin. Would also suppress the two line-4755
   warnings.
2. **Window-dependent over-sterilisation.** Independent of the NaN:
   sum ρ_ss growing from ~10 (w5-15) to ~30 (w30) is non-physical.
   Point C's ΔNeff jumping from −0.032 (w20) to +4.212 (w30) is the
   smoking gun. Candidate causes: (i) the auto-scaled `n_B_override`
   formula under-resolves when spanning >3 decades, letting early-T
   MSW passage integrate incorrectly; (ii) a numerical bias from
   candidate 1's sanitisation that accumulates across more cold-T
   steps at larger windows; (iii) Phase-A's thermal IC assumption
   becoming inadequate when `T_boltz_start ≫ T_MSW`. Start with (i):
   rerun Point C at w30 with manually-overridden `n_B_override`
   values (2×, 4×, 8× default) and see whether ΔNeff converges.
3. **Al-Mohy augmented-matrix expm audit** (sprint-6 brief's
   Suspect 2). Not needed for the crash fix, but if candidate 1
   doesn't close the warnings, examine condition-number behaviour
   of the augmented matrix at extreme eigenvalue spread.
4. **Strang-split time evolution**, **representation-factor leak**,
   **Gariazzo damping form**, **Shi-Fuller literature** — same
   priority and rationale as post-sprint-5.

**E.2 sprint 7 — cold-T collision-integral NaN localised and fixed at
origin; Hannestad w30 over-sterilisation refactored into two
independent residual bugs.** Plan (b) from
`doc/STAGE_E2_SPRINT7_BRIEF.md` (origin fix + n_B re-pin). The
origin fix landed; the n_B re-pin was investigated and produced a
nuanced result that scopes sprint 7 down to (a) plus a documented
Phase-5 finding.

**Phase 1 probe** (transient, discarded before commit). Instrumented
`_assemble_collision_N` with a per-call non-finite check against
every intermediate (`f_all`, `tail_params`, the four `_fnu_*` scalar
interpolant returns, `I_nu_nu`, `I_nu_e`, `I_total`). First-hit
AssertionError. Ran Point A at w30 with the probe on: first non-finite
at **`I_nu_e[species=numu, y_idx=0]` at T=6.25 keV, a=6713**, in the
*second* `_assemble_collision_N` call (the half-diag-2 `I_total_post`
site). All upstream intermediates clean. Pre-flight pre-probe checks
had already falsified Suspect-1 candidate A (interpolant extrapolation:
the four `_fnu_*` tables return exactly `0.0` at T ∈ [1e-10, 9.9e-3] MeV
by linear interp between two near-zero entries — no NaN) and candidate
B (`/y1²` amplification: `y_grid[0] = dy/2 = 0.5 MeV`, not tiny — `/y1²`
benign).

**Phase 2 root cause and fix.** The NaN is produced inside
`_F_stat_stable` at `PRyM_boltzmann.py:1060`: the upper clamp
`_hi = 1.0 - 1.0e-20` rounds to **exactly 1.0** in float64 (1e-20 is
far below machine eps(1) ≈ 2.22e-16). When a diagonal briefly overshoots
`f > 1` during the Strang-split half-diag-1 (before the end-of-step
diagonal clip), the clamp is a no-op: `c = 1.0 → (1-c)/c = 0 →
log(0) = -inf → d_mu = -inf + +inf = NaN`. At cold T the
`_fnu_*_scat/ann` scalars are exactly `0.0`, so `D_scat/D_ann = 0.0`
and the product is `0 × NaN = NaN`. The NaN then poisoned
`I_total[2]` at y_idx=0 and propagated into `rho_all[:, 1..2, 0]`
via half-diag-2. (Sprint-6's probe caught a different y_idx=0 NaN at
sector=0 comp=0 from the same root cause applied to `I_total[0]`.)

Fix at origin (one constant change): `PRyM_boltzmann.py:1060`,
`_hi = 1.0 - 1.0e-20` → `_hi = 1.0 - 1.0e-15`. Asymmetric (keeps
`_lo = 1.0e-20` unchanged) so `f ∈ [0, 1]` physical-range callers
are strictly bit-identical: the new `_hi` only clamps `f ≥ 1-1e-15`,
which never occurs in the diagonal clip's post-step `[1e-30, 1-1e-30]`
range. Post-fix probe re-run at w30 Point A was **silent**; run
completed cleanly.

**Phase 3+4 validation ladder (all at default projection=False unless
noted):**

| # | Test | Result |
|---|------|--------|
| 1 | `pytest -m "not slow"` | 4/4 bit-identical ✓ |
| 2 | `pytest -k sterile` | 3/3 bit-identical ✓ |
| 3 | `diag_2level_damped.py` | ratio 1.000 ✓ |
| 4 | `diag_vnunu_active_only.py` (proj + w5) | Point C ΔNeff = −0.0119 bit-identical ✓ |
| 5 | `diag_hannestad_proj_w20.py` (proj + w20) | clean, Yp ∈ [0.24, 0.26] for all points ✓ |
| 6 | `diag_hannestad_proj_w30.py` (proj + w30) | bit-identical Yp to sprint-6; dNeff drift (A 1.474→1.257, B 0.098→0.021, C 4.212→4.212); gate-6 **does not hit Hannestad bands** |

Gate 6 not hitting was the surprise. Unwinding it: sprint-6's
downstream sanitisation (`nan_to_num(NaN, nan=f_min=1e-30)` in the
diagonal clip) had been substituting the "correct" cold-T asymptotic
value for the NaN-producing mode — the clamp fix just formalises the
same numerical outcome at the origin. So the w30 over-sterilisation
observed in sprint 6 (Yp 0.30 at A and C, ΔNeff +4.2 at C) is
**not** driven by the NaN + sanitisation biasing the neighbouring
steps, as the sprint-6 ROADMAP speculated. It is a separate, deeper
structural bug.

**Phase 5 n_B convergence probe.** Suspect 2 from the brief's scope-(b)
plan: the sprint-2 auto-scale `n_B = int(2000 * scale**4)` was pinned
at projection=False; with V_nunu's sterile ballast removed by the
projection, the MSW passage resolution requirement rises. Single-point
probe at Point C w30 projection, n_B=10000 (≈ 2× sprint-2 default of
5031):

| n_B | Neff | Yp | ΔNeff | Σρ_ss |
|---:|---:|---:|---:|---:|
| 5031 (gate 6) | 7.22209 | 0.29320 | +4.212 | 20.435 |
| 10000 (probe) | **3.06981** | **0.24972** | **+0.060** | 20.550 |

Point C collapses from catastrophic over-sterilisation to within the
Hannestad [0.02, 0.1] band by doubling n_B. (Σρ_ss is similar at
both because it counts raw sterile occupation, dominated by low-y
modes pinned near f_min; what changed is the energy-weighted Neff
contribution from the high-y tail that n_B=10000 now resolves.)
Encouraged, ran the full A/B/C suite at n_B=10000 w30 projection:

| Point | sin²2θ | Hannestad | dNeff @ n_B=5031 | dNeff @ n_B=10000 | Yp @ n_B=10000 | Σρ_ss @ n_B=10000 | Band |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 1e-1    | 1.00 | +1.257 | **+1.568** | 0.29894 | 29.960 | MISS (too high) |
| B | 2.26e-3 | 0.50 | +0.021 | +0.060 | 0.24833 | 28.551 | MISS (too low)  |
| C | 1e-4    | 0.04 | +4.212 | **+0.060** | 0.24972 | 20.550 | **IN BAND** ✓ |

Point C: clean n_B-convergence, **fixed**. Point A: non-monotonic
(gets *worse* with more n_B), indicating a separate structural bug at
large mixing. Point B: under-production, stable across n_B, also a
separate bug. Full-thermalisation sterile at A+B (Σρ_ss ≈ 30 ≈
thermal) coexists with wildly-different ΔNeff values (+1.57 at A vs
+0.06 at B), suggesting the issue is energy-weighted: sterile is
hotter than T_nu at A (dNeff ~ (T_s/T_nu)^4 = 1.11^4 ≈ 1.52, close
to observed 1.57) and colder at B. An energy-balance / MSW-passage
anomaly in the active↔sterile conversion at adiabatic and
semi-adiabatic mixings.

Full `diag_hannestad_proj_w30_nB10k.out` and
`diag_pointC_nB_probe.out` kept under `validation/diagnostics/` as
sprint-8 reference.

**Sprint-7 landing posture.** Land Phase 2's single-constant fix as
a standalone numerical-robustness commit. The fix:

1. Eliminates the cold-T NaN at origin (first non-finite no longer
   fires at any Hannestad-configured run).
2. Is strictly bit-identical at projection=False (the default) — all
   regression gates pass bit-identical. The sprint-6 sanitisation
   passes are retained as defense in depth; they are now
   mathematical no-ops on finite state in all runs.
3. At extended windows (w20, w30) with projection=True, produces
   **equivalent Yp** and slightly-shifted Neff vs. sprint-6 output.
   The shift is from the sanitisation being an effective-f_min
   substitution vs. a clamp-formalised equivalent — same physics
   outcome, cleaner numerics.

**Default `qke_v_nunu_active_only` stays `False`.** Point C's
n_B-convergent behaviour at n_B=10000 is encouraging for a future
heuristic re-pin, but the structural bugs at Points A and B must be
resolved before the default flip. The sprint-7 cold-T fix + sprint-6
sanitisation together make extended-window QKE runs numerically
stable regardless of physical correctness, so future structural-bug
sprints can iterate cleanly on Points A and B without re-tripping
over the NaN.

**Next structural candidates**, ranked post-sprint-7:

1. **Point-A over-heating / Point-B under-production at w30
   projection.** The n_B-independent dNeff anomalies at adiabatic
   and semi-adiabatic mixings. Candidate causes: (i) energy-balance
   bug in the active↔sterile damping → collision-integral coupling
   (we damp coherence but do not explicitly transfer energy from ρ_αα
   to ρ_ss in a conservation-verified way); (ii) MSW-passage
   handling at large mixing — the ETDRK2 method in the eigenbasis may
   over-pump adiabatic transitions; (iii) Phase-A thermal-IC
   inadequacy at T_boltz_start=30 MeV, biasing the initial ν
   reservoir. Diagnostic: integrate ∂_t[∫y² ρ_αα + ∫y² ρ_ss] across
   a few steps at Point A and check whether the sum drifts — a
   conservation violation pins (i).
2. **n_B auto-scale re-pin for projection=True.** Point C's clean
   convergence at n_B=10000 (vs. default 5031) says the sprint-2
   `scale**4` heuristic under-resolves at projection-on extended
   windows. Candidate re-pin: conditional `k=4` at
   projection=False, `k=7` at projection=True, so `1.259**k ≈ 5`
   gives n_B ≈ 10000 at w30 and n_B = 2000 bit-identical at default
   w5. Defer until candidate 1 closes, since the flip depends on it.
3. **Sprint-6 carryovers** (representation-factor leak, Al-Mohy
   audit, Strang-split evolution, Gariazzo damping, Shi-Fuller
   literature) — all unchanged in priority.

**E.2 sprint 8 — Suspect-1 energy-balance falsified at machine
precision; Point A dNeff anomaly isolated to a y-distribution effect
downstream of a correctly-trace-preserving driver.** Scope (b) from
`doc/STAGE_E2_SPRINT8_BRIEF.md` began with the Suspect-1 diagnostic.
The surgical result rerouted the sprint to scope (a) close-out: no
fix landed, but the Suspect-1 hypothesis is now removed from the
candidate list, and the downstream default flip + n_B re-pin are
parked pending a sprint-9 root cause.

**Phase 1 instrumentation.** Added `PRyMini.qke_energy_diag_flag`
(default False; bit-identical when off — the gates 1–4 fast regression
ladder passes bit-for-bit at default) and a three-position per-step
accumulator inside `evolve_step_ode_etdrk2` (`PRyM_boltzmann.py`
lines 4695-4744, 4802-4805, 4858-4862). Each time step records, for
each active-sterile block (α, s) and each sector, the uniform-
midpoint-rule integrals `N_αs = dy·Σ y² (ρ_αα + ρ_ss)`, `E_αs = dy·
Σ y³ (ρ_αα + ρ_ss)`, and coherence `C_αs = dy·Σ y³ |ρ_αs|`, at three
sub-step positions (pre_step, post_corrector, post_clip). Accumulated
on `DensityMatrixSolver._energy_hist` and exposed via
`PRyMclass._boltz_dm_solver` (`PRyM_main.py:526`).

**Phase 1 probe.** Harness
`validation/diagnostics/diag_energy_balance.py` runs Point A
(sin²2θ=1e-1, δm²=0.93) at w30 projection, n_B=10000, with the flag
on. Output at `validation/diagnostics/diag_energy_balance.out` and
`diag_energy_balance_pointA.npz`. Reproduces sprint-7's Point A bit-
identically on public observables (Neff = 4.57772, Yp = 0.29894,
Σρ_ss = 29.960), so instrumentation is non-perturbative.

**Observed drift.** Per-step mean drift per (α, s) block:

| sector | pair | ⟨dE_mid/E⟩ | ⟨dE_end/E⟩ | ⟨dE_tot/E⟩ | dE_mid / dE_tot |
|---:|---:|---:|---:|---:|---:|
| 0 | (0,s) | +7.04e-5 | +1.03e-7 | +7.05e-5 | 1.00 |
| 0 | (1,s) | +7.64e-5 | −1.81e-8 | +7.63e-5 | 1.00 |
| 0 | (2,s) | +7.90e-5 | −1.83e-8 | +7.90e-5 | 1.00 |
| 1 | (0,s) | +8.34e-5 | +6.38e-8 | +8.35e-5 | 1.00 |
| 1 | (1,s) | +9.35e-5 | −1.74e-8 | +9.34e-5 | 1.00 |
| 1 | (2,s) | +1.02e-4 | −1.74e-8 | +1.02e-4 | 1.00 |

Total (pre_step → post_clip) dE/E over the run: +0.48 to +0.52. All
growth concentrated in the pre_step → post_corrector window (half-
diag-1 + predictor + corrector); the off-diag clamp and diagonal clip
contribute < 1e-7 per step each.

That pattern triggers the brief's "Candidate A" verdict (positive
drift → double-counting in L add-back) only superficially. The
(ρ_αα + ρ_ss) sum is NOT a closed-system invariant in PRyMordial's
Phase B — it couples to the thermal bath via `I_total` on the active
diagonal at the half-diag-1 and half-diag-2 exp-Euler steps. A +50%
growth over n_B=10000 steps is consistent with physical bath-pumped
thermalisation plus unitary coherent transfer, not a numerical leak.

**Surgical per-step trace test.**
`validation/diagnostics/diag_energy_balance_decomp.py` post-processes
the `.npz` and also replays one step of `evolve_step_ode_etdrk2`
manually with explicit sub-step accounting on the full TRACE (the
correct invariant under trace-preserving L):

  * S0 pre: E_total = 3.3721e5
  * S1 after half-diag-1: dE = 0 (I_total ≈ 0 at the thermal IC)
  * S2 after predictor: **dE = +7.57e−10 (relative 2.24e-15)**
  * S3 after corrector: dE = 0
  * S4 after half-diag-2: dE = +1098 (bath refill after active→sterile
    unitary transfer moved 4.47e4 E into ρ_ss during predictor)

Predictor and corrector preserve trace at machine precision. The
`dE_mid / dE_tot = 1.00` pattern in the full-run table just reflects
that the bath delivery bookkeeping happens at half-diag-1 and
half-diag-2, both of which live inside the pre_step → post_corrector
window.

**Verdict: Suspect 1 is falsified.** The driver's trace accounting
is correct. The Point A dNeff = +1.57 anomaly (sterile ~11% hotter
than T_ν at Σρ_ss ≈ thermal) must therefore be a y-DISTRIBUTION
effect — where the bath-pumped resonance deposits its energy — not
a total-energy accounting bug. That shifts the prime suspect to
Suspect 2 (MSW-passage adiabatic over-pumping in the ETDRK2
eigenbasis) from the sprint-8 brief.

**Also confirmed.** L construction is correct at the matrix level:
`validation/diagnostics/diag_2level_damped_energy.py` (new) evolves
the (μ, s) 2×2 sub-block under the PRyMordial 4×4 L via direct expm
for 1000 steps and finds |dN/N|, |dE/E| < 1.83e−10 (well under the
1e−8 tolerance). Both the analytic 2-level L and the PRyMordial L
extracted at the (μ, s) block preserve N and E in damping-only
dynamics. Any sprint-9 fix that touches L construction has this as
a regression guard.

**Sprint-8 landing posture.** Land diagnostic scaffolding only.
Default `qke_v_nunu_active_only` stays False. The conditional n_B
re-pin (k=4 → k=7 at projection=True) is parked, since it depends
on the dNeff anomaly closing first. `test_sterile_dw_production`
is unchanged. Sprint-7's gate-6 n_B=10000 reference stays the
primary Point A/B/C benchmark. The sprint-8 scaffolding is:

  * `PRyM/PRyM_init.py` — `qke_energy_diag_flag`, `qke_energy_diag_path`.
  * `PRyM/PRyM_boltzmann.py` — `_energy_snapshot` helper,
    three-position instrumentation in `evolve_step_ode_etdrk2`.
  * `PRyM/PRyM_main.py` — expose `dm_solver` as `_boltz_dm_solver`.
  * `validation/diagnostics/diag_energy_balance.py` + `.out` + `.npz`
    — Point A run harness + outputs.
  * `validation/diagnostics/diag_energy_balance_decomp.py` + `.out`
    — post-processor + surgical per-step trace test.
  * `validation/diagnostics/diag_2level_damped_energy.py` + `.out`
    — 2-level L N+E conservation unit test (pass at 1e-8).
  * `validation/diagnostics/diag_hannestad_proj_w30_nB10k.py` —
    sprint-7 Phase-5 transient harness resurrected for sprint 9.

All scaffolding is opt-in (guarded on `qke_energy_diag_flag`, or on
explicit `n_B_override = 10000` for the gate-6 harness) and fast-
regression-bit-identical at default.

**Next structural candidates**, ranked post-sprint-8:

1. **Suspect 2 — MSW-passage adiabatic over-pumping in ETDRK2
   eigenbasis** (promoted from sprint-7 #1.ii). The y-distribution
   bias in ρ_ss must come from either the Hamiltonian resonance
   structure or from how the ETDRK2 `_etdrk2_expm_phi` handles the
   in-medium mixing angle sweep through π/4 at adiabatic mixing.
   Diagnostic candidate: at each step, record the instantaneous
   in-medium mixing angle θ_m(T, y) per y-mode and compare the
   ETDRK2 trajectory to a direct-RK4 reference at a handful of
   resonance-crossing y-modes. Where they diverge locates the bug.
   See `doc/STAGE_E2_SPRINT9_BRIEF.md`.
2. **Suspect 3 — Phase-A thermal-IC inadequacy** at T_boltz_start=30
   MeV (demoted from sprint-7 #1.iii, was never prime). Only worth
   investigating if Suspect 2 also falsifies.
3. **n_B auto-scale re-pin** for projection=True — still parked
   pending a dNeff fix.
4. **Sprint-6 carryovers** — unchanged.

**E.2 sprint 9 — Suspect 2 falsified at Point A; y-distribution
bias inverted from prediction; Suspect 3 promoted to prime.** Scope
(a) from `doc/STAGE_E2_SPRINT9_BRIEF.md`: per-step per-y-mode MSW-
passage instrumentation, no fix landed. The new diagnostic resolves
the Point A resonance-crossing structure and falsifies Suspect 2's
high-y-over-deposition prediction; the actual bias runs the other
way.

**Instrumentation.** `PRyMini.qke_msw_diag_flag` (default False) +
`qke_msw_diag_path` + `qke_msw_diag_pair_idx` (default 4 = (α=1=numu,
s=3), the Point-A θ_24 channel). `DensityMatrixSolver._msw_snapshot`
at `PRyM_boltzmann.py:4702-4764` is a second flag-gated helper
alongside `_energy_snapshot`; it records per-y `H_αα`, `H_ss`,
`H_αs`, `ρ_αα`, `ρ_ss` per sector per sub-step position
({pre_step, post_clip}). Gate-flag hooks at `:4815-4818` and
`:4941-4942`. Defaults bit-identical: gates 1–4 fast regression 4/4
and sterile regression 3/3 pass; 2-level L-conservation guard
1.83e-10 << 1e-8. Scaffolding: `_msw_hist`/`_msw_step_idx` init at
`:3067-3068`, reusing `_boltz_dm_solver` exposure from sprint 8.

**Phase-1 probe.** `validation/diagnostics/diag_msw_passage.py` runs
Point A (sin²2θ=1e-1, δm²=0.93) at w30 projection, n_B=10000 in
3983 s, reproducing sprint-7 Phase-5 bit-for-bit: **Neff=4.57772,
Yp=0.29894, Σρ_ss=29.960**. Instrumentation is non-perturbative.
Output at `validation/diagnostics/diag_msw_passage.out` (tracked)
and `diag_msw_passage_pointA.npz` (193 MB, local only, per
sprint-8 convention).

**Resonance localisation per y-mode (sector 0).** The per-y table
partitions into two populations:

  * **Low-y (y ≲ 20)** cross their MSW resonance at `step_res` ∈
    {0, 1, 2}, with `T_res ≈ 29.9–30.0 MeV` — i.e., the resonance
    lies AT or ABOVE `T_boltz_start = 30 MeV`. Step-indexed
    Landau-Zener adiabaticity `γ_step` = O(10⁻¹⁰ – 10⁻⁵). These
    modes enter Phase B already past (or at) their resonance.
  * **High-y (y ≳ 20)** cross at `step_res ≈ 2260–3350`,
    `T_res ≈ 1.5–4 MeV`. `γ_step` = O(10⁻⁸ – 10⁻⁶).

**Window-integrated Δρ_ss (sector 0, ±50 steps around resonance):**

| bucket | y range | ⟨Δρ_ss_win⟩ |
|---:|---:|---:|
| 0 | 0.5–20.5  | +5.58e-2 |
| 1 | 20.5–40.5 | +1.06e-1 |
| 2 | 40.5–60.5 | +7.41e-2 |
| 3 | 60.5–80.5 | +3.58e-2 |
| 4 | 80.5–99.5 | −8.84e-4 |

**End-state ρ_ss(y) residual vs. thermal FD at Tg_final, normalised
to match measured Σρ_ss (sector 0):**

| bucket | y range | ⟨residual⟩ |
|---:|---:|---:|
| 0 | 0.5–20.5  | **+8.14e-2** (over-thermal) |
| 1 | 20.5–40.5 | +6.85e-2 |
| 2 | 40.5–60.5 | +2.51e-2 |
| 3 | 60.5–80.5 | −7.18e-3 |
| 4 | 80.5–99.5 | **−1.08e-2** (under-thermal) |

**Verdict: Suspect 2 is FALSIFIED at Point A.** The predicted
signature (high-y over-deposition from ETDRK2 eigenbasis bias at
near-degenerate L) does not appear — the bias runs the opposite
way. Low-y modes are over-thermal by ~+8% of FD; high-y modes are
under-thermal by ~−1%. Net `Σρ_ss` matches thermal, so the global
population is right but redistributed.

**Suspect 3 is PROMOTED to prime candidate.** The signature is
consistent with a Phase-A thermal-IC inadequacy: low-y modes have
already crossed their MSW resonance at T > 30 MeV by the time
Phase B begins, so they arrive with `ρ_ss=0` but a physical
pre-processed history would have non-zero mass-eigenstate
occupation. The Phase-B driver, faithfully integrating from the
wrong IC under large vacuum mixing (sin²2θ=0.1 → θ_vacuum≈9°),
drives them rapidly toward a new local equilibrium that over-
shoots thermal at low-y. High-y modes cross resonance *during*
Phase B at step-indexed `γ_step` ≪ 1, i.e. Landau-Zener-non-
adiabatic in step units — the driver tunnels through the crossing
without full conversion, under-thermalising high-y.

Neither anomaly is a driver bug. Both are physics-level consequences
of the Phase-A cutoff. The fix belongs in the IC, not in
`_etdrk2_expm_phi`.

**Sprint-9 landing posture.** Diagnostic scaffolding only; no
default flips, no fix landed. The sprint-8 do-not-touch list is
inherited in full, plus the sprint-9 `_msw_snapshot` instrumentation
and the `qke_msw_diag_*` flag defaults. The sprint-9 scaffolding is:

  * `PRyM/PRyM_init.py` — `qke_msw_diag_flag`, `qke_msw_diag_path`,
    `qke_msw_diag_pair_idx`.
  * `PRyM/PRyM_boltzmann.py` — `_msw_hist`/`_msw_step_idx` init,
    `_msw_snapshot` helper, two hook points in
    `evolve_step_ode_etdrk2`.
  * `validation/diagnostics/diag_msw_passage.py` + `.out` — Point A
    run harness + resonance-localisation + FD-residual analyses.
  * `doc/STAGE_E2_SPRINT10_BRIEF.md` — single-file handoff for the
    sprint-10 Suspect-3 audit (Phase-0 QKE driver from ~100 MeV).

All scaffolding is opt-in (guarded on `qke_msw_diag_flag`) and
fast-regression-bit-identical at default.

**Next structural candidates**, ranked post-sprint-9:

1. **Suspect 3 — Phase-A thermal-IC inadequacy** (promoted from
   sprint-8 #2 to prime). Build a Phase-0 QKE driver that evolves
   from `T ≈ 100 MeV` down to `T_boltz_start = 30 MeV` with the
   same 4×4 QKE machinery, consuming adiabatic-vacuum IC and
   producing a history-preserving `ρ_all(T=30 MeV)` to hand to
   Phase B. Expected effect at Point A: low-y bucket residual
   shrinks toward zero as the pre-resonance history is supplied;
   high-y may also improve if the Phase-0 integration smooths the
   initial coherence across the 30–100 MeV window. Validation:
   re-run `diag_msw_passage.py` + gate-6 `diag_hannestad_proj_w30_nB10k.py`.
   See `doc/STAGE_E2_SPRINT10_BRIEF.md`.
2. **Non-adiabatic high-y correction** (new, sprint-9 follow-up).
   High-y bucket 4 under-thermalisation is small (−1%) but genuine,
   arising from step-indexed Landau-Zener tunneling at `γ_step ≪ 1`.
   If sprint 10's Phase-0 IC doesn't close it, an n_B refinement
   localised to the crossing band (or an adaptive step controller
   near resonance) may be needed.
3. **n_B auto-scale re-pin** for projection=True — still parked
   pending dNeff anomaly closing.
4. **Sprint-6 carryovers** — unchanged.
5. **Suspect 2 audit at Point C** (optional). Sprint-9 falsified
   Suspect 2 at Point A only. Point C (small mixing, sin²2θ=1e-4)
   has a narrow-resonance regime where the ETDRK2 eigenbasis
   collapse argument *could* still apply. Low priority — Point C
   already lands in-band at n_B=10000.

**E.2 sprint 10 — Phase-0 QKE driver lands; Point-A anomaly 87%
closed, Point-C narrow-mixing regression promotes Suspect 2-residual
to prime.** Scope (b)-minus-flip from `doc/STAGE_E2_SPRINT10_BRIEF.md`:
Phase-0 driver, flags, auto-scale, decoupled-sterile regression
guard, and diagnostic harness updates all land as opt-in machinery;
the `qke_phase0_flag` default stays `False` because gate-9 Point C
(sin²2θ=1e-4) regresses from in-band to far out-of-band when Phase 0
is on.

**Refactor landed first (no-op commit verified independently).** The
Phase-B Froustey loop body at the old `PRyM_main.py:364-477` was
extracted into a nested `_run_qke_segment(rho_in, f_in, Tg_start,
a_start, t_start, sigma_start, Tg_end, n_steps, a_grid_seg,
collect_trajectories, rho_ss_history)` helper inside
`PRyMclass.__init__`. Both Phase 0 and Phase B now call it.
Bit-identical to sprint-9 `ac1d521` at all defaults: fast regression
4/4, sterile regression 3/3.

**Instrumentation and machinery.**

  * `PRyM/PRyM_init.py` — `qke_phase0_flag` (master toggle, default
    False), `T_phase0_start = 100.0` (MeV), `n_B_phase0_override =
    None` (validator auto-scales); `qke_phase0_diag_flag` (sprint-10
    post-landing probe instrumentation, default False). Four new
    validator clauses: warn when `qke_phase0_flag` is set with
    `qke_full_ode_flag=False`, with `T_phase0_start <= T_boltz_start`,
    or with `T_start/MeV_to_Kelvin < T_phase0_start`. New section 5b
    auto-scale: `n_B_phase0 = int(2500 * (decades/0.52)^2)` anchored
    at log10(100/30) = 0.52 decades, clamped to [500, 20000]. At the
    default window the scale factor is exactly 1.0, so the auto-scale
    picks the anchor 2500 verbatim.
  * `PRyM/PRyM_main.py` — Phase-A endpoint dispatches on
    `_phase0_active = (qke_phase0_flag and qke_density_matrix_flag)`;
    the Phase-B IC construction is unchanged (it now builds the
    thermal-FD IC at `T_phase0_start` when Phase 0 is on — which is
    exactly the correct Phase-0 IC). A conditional Phase-0 block
    between the Phase-B IC and the Phase-B loop calls
    `_run_qke_segment(...)` with `Tg_end=T_boltz_start`,
    `n_steps=n_B_phase0`, `collect_trajectories=False` and an
    optional `rho_ss_history=[]` list when
    `qke_phase0_diag_flag=True`. Post-Phase-0 it updates
    `rho_curr, Tg_boltz_ini, a_boltz_ini, t_B_start, sigma_curr`
    from Phase-0 exit and rebuilds the Phase-B `a_grid`. Exposes
    `self._phase0_rho_final` and `self._phase0_rho_ss_history` on
    the class for diagnostic harnesses.
  * `validation/diagnostics/diag_phase0_decoupled.py` + `.out` — new
    gate-5 regression guard. `sterile_flag=True`, all θ_α4 = 0,
    xi_* = 0, `qke_phase0_flag=True`, 100→30 MeV over 1000 Phase-0
    steps. Asserts `max|ρ_ss(T=T_boltz_start)| < 1e-12` (actual:
    1e-30 at clip floor, both Phase-0 and Phase-B exits). PASS.
  * `validation/diagnostics/diag_msw_passage.py` + `.out` — sprint-9
    MSW-passage harness `_base_flags()` updated with
    `qke_phase0_flag=True, T_phase0_start=100.0,
    T_start=105*MeV_to_K, n_B_phase0_override=2500`. Gate 8 with
    Phase 0 moves Point A Neff 4.57772 → 3.83523, Yp 0.29894 →
    0.26090 (now in band), Σρ_ss 29.960 → 28.809.
  * `validation/diagnostics/diag_hannestad_proj_w30_nB10k.py` +
    `.out` — same Phase-0 enable in `_base_flags()`. Gate 9 results
    below.
  * `validation/diagnostics/diag_phase0_pointC.py` + `.out` — new
    post-landing probe. Runs gate-9 Point C with
    `qke_phase0_diag_flag=True` so `_run_qke_segment` appends
    `(istep, a, Tg, rho_ss_slice.copy())` after each Phase-0 step.
    Post-processes to locate threshold crossings, top-10 y-modes,
    and nu/nubar asymmetry.
  * `doc/STAGE_E2_SPRINT11_BRIEF.md` — single-file handoff for the
    sprint-11 Suspect 2-residual fix (narrow-mixing ETDRK2 over-
    adiabatisation at MSW passage).

**Gate-9 Hannestad A/B/C at w30 projection, n_B=10000, Phase 0 on
(5133-5226 s per run, ~5.5 h total wall-clock).** 3×3 QKE reference
with Phase 0 on: Neff = 3.00034, Yp = 0.24779 — Phase 0 preserves
the no-sterile Standard-Model value bit-for-bit.

| Point | sin²2θ | Hannestad | Sprint-9 (no P0) | Sprint-10 (P0 on) | Yp | Σρ_ss | verdict |
|---|---|---|---|---|---|---|---|
| A | 1e-1 | 1.000 | +1.57 | **+0.835** | 0.26090 | 28.809 | miss low by 0.065 (87% closed) |
| B | 2.26e-3 | 0.500 | +0.06 | 0.068 | 0.25022 | 28.939 | miss low (unchanged) |
| C | 1e-4 | 0.040 | +0.06 | **+5.522** | 0.27209 | 22.225 | **miss high by 5.5 — regression** |

Point A's closure is strong evidence for Suspect 3 (Phase-A thermal-
IC inadequacy at low-y modes whose MSW resonance lies at
`T_res ≳ 30 MeV`). Point B is neutral (small mixing, small Phase-0
effect). Point C's regression is the prime sprint-10 finding.

**Point-C post-landing probe
(`diag_phase0_pointC.py`).** Ran gate-9 Point C with per-step
instrumentation. Findings, in descending order of signal:

  1. **Only 3% of the final Σρ_ss is deposited during Phase 0.**
     Σρ_ss at Phase-0 exit = 0.666; at Phase-B exit = 22.22. The
     remaining 97% is Phase-B collisional thermalisation of the
     Phase-0 seed.
  2. The Phase-0 seed is localised almost entirely at **y = 0.5 in
     the antineutrino sector** (ρ_ss = 0.444). The next y-mode
     (y = 1.5) is 20× smaller. All other y-modes are < 0.01.
  3. Factor **2.6× antineutrino/neutrino asymmetry**: max_ν̄ = 0.444,
     max_ν = 0.168 at Phase-0 exit. Consistent with the MSW sign
     asymmetry for Dm²_41 > 0 (antineutrino resonance at higher T).
  4. Growth is two-staged and includes a **step-function jump from
     0.177 → 0.4145 between istep 906 (Tg = 64.1 MeV) and istep 1035
     (Tg = 60.2 MeV)** in the antineutrino sector. Factor-2.3
     doubling inside a ~130-step window, then flat for 1400 steps
     until Phase-0 exit.

**Verdict: Suspect 2-residual (ETDRK2 eigenbasis over-adiabatisation
at narrow-mixing MSW passage) is PROMOTED to prime.** Sprint-9's
original Suspect 2 hypothesis was falsified at Point A (large
mixing, wrong signature direction) but re-surfaces at Point C
(narrow mixing) with the **predicted** signature: spurious
conversion at the one y-mode whose MSW resonance falls inside the
integration window. The step-function jump at T ≈ 62 MeV is a
regulariser-branch transition inside `_etdrk2_expm_phi`'s Al-Mohy
augmented-matrix expm — the adiabatic width for y = 0.5 at
sin²2θ = 1e-4 is narrow enough that the regulariser tolerance
straddles it. Phase 0 only surfaces this bug by moving a previously-
silent (pre-`T_boltz_start`) resonance into the integrated range;
the mechanism applies equally in Phase B whenever a resonance falls
inside the integration window. See `doc/STAGE_E2_SPRINT11_BRIEF.md`
for the full mechanism write-up and fix plan.

**Sprint-10 landing posture.** Phase-0 driver + instrumentation +
diagnostic harnesses land as opt-in machinery. All defaults stay at
`False` / `None`. No test fixtures changed. The sprint-10 bit-
identity guarantees (all regression gates at defaults pass with zero
drift against `ac1d521`) are what license shipping the Phase-0 code
without the default flip — Point A's 87% closure is a **promising
capability under user control**, Point C's regression is a
**diagnosed numerical bug** with a narrow fix scope handed to
sprint 11.

**Next structural candidates**, ranked post-sprint-10:

1. **Suspect 2-residual — ETDRK2 eigenbasis over-adiabatisation at
   narrow-mixing MSW passage** (promoted from sprint-9's #5 parked
   item to prime). Step-function jump localised to
   `_etdrk2_expm_phi` at y = 0.5, antineutrino, Tg ≈ 62 MeV,
   istep 906-1035 in Phase 0. Fix candidates ordered by scope:
   (a) direct-expm fallback for y-modes within an adiabatic-width
   fraction of resonance (cheapest, scope-local to a handful of
   y-modes); (b) tighten the regulariser tolerance inside
   `_etdrk2_expm_phi` (global effect, must be validated against
   Point A's +0.835 — must not worsen); (c) architectural
   resonance-aware step controller (1-2 session scope). See
   `doc/STAGE_E2_SPRINT11_BRIEF.md`.
2. **Suspect 4 — Phase-B collisional amplification at narrow
   mixing** (new sprint-10 follow-up). Even if the Phase-0 seed at
   y = 0.5 is reduced to ~0.01, Phase B's observed 33× amplification
   would still push the final Σρ_ss above the Hannestad band. The
   amplification may itself be over-integrating for narrow mixing.
   Only investigate if Suspect 2-residual closes the Phase-0 seed
   but Phase-B final is still above band.
3. **Point-A residual 0.065 below-band** (sprint-10 remainder).
   87% of the Point-A anomaly closed with Phase 0; the remaining 7%
   may share the same `_etdrk2_expm_phi` regulariser mechanism at a
   handful of edge y-modes near `T_boltz_start`. Likely resolved as
   a side-effect of the Suspect 2-residual fix; verify via sprint-10
   probe technique at Point A after the fix lands.
4. **Non-adiabatic high-y correction** (sprint-9 carryover).
   Superseded by Suspect 2-residual if (1) resolves the core
   mechanism — same `_etdrk2_expm_phi` site.
5. **n_B auto-scale re-pin** for projection=True — still parked.
6. **Sprint-6 carryovers** — unchanged.

**E.2 sprint 11 — Suspect 2-residual partially confirmed: eigen-
decomposition fallback closes Phase-B over-amplification but Phase-0
step-function persists.** Scope (a) from `doc/STAGE_E2_SPRINT11_BRIEF.md`:
add a per-mode eigendecomposition path inside `_etdrk2_expm_phi`,
gated on (1) small active-sterile commutator gap relative to the
largest active-flavor diagonal spread AND (2) non-zero off-diagonal
H_α,sterile coupling. Land as opt-in machinery (default off, bit-
identical regression). The plan deliverable was Phase-0 Σρ_ss < 0.1
with the step-function jump at istep 906→1035 gone. **The deliverable
was not met**: Phase-0 dynamics with fallback on are essentially
identical to sprint-10 baseline (Σρ_ss = 0.6673 vs 0.6656 — 0.25%
relative drift; the istep 906→1035 jump survives unchanged at 0.177
→ 0.4145). The fallback fired 740 times across the 12500-step Phase-
0+B run with zero κ-guard reverts. The mechanism Suspect 2-residual
was hypothesised to be — Pade-branch transition inside the Al-Mohy
augmented expm at the narrow-mixing MSW pass — does not reproduce in
Phase 0: eigendecomposition of L*dt and scipy.expm of the augmented
M produce numerically equivalent (Phi0, Phi1, Phi2) at that mode.

**Surprising side-effect.** Final Neff dropped from sprint-10
baseline 8.52 to **3.0114** with the fallback on at Point C —
ΔNeff = −0.033 vs the Hannestad-C target band [0.02, 0.10], so
slightly below band rather than far above. Yp = 0.2485 (in [0.24,
0.26]). The Phase-B over-amplification (sprint-10's "33× from 0.667
seed → 22.22") is fully suppressed, even though the Phase-0 seed is
unchanged. Re-reading: the fallback is doing useful work somewhere
in Phase B (where multiple y-modes cross MSW resonance as T sweeps
30 → 0.005 MeV and Pade-branch step-functions accumulate per
crossing) — that work was hypothesised to live in Phase 0 and turns
out to live in Phase B. The Phase-0 step-function and Phase-B
amplification are two separable mechanisms: this sprint addresses
the second cleanly, leaves the first unchanged.

**Why the default flip is deferred.** Hannestad C ΔNeff = −0.033
is out of band on the low side (vs sprint-10's +5.52 out of band
on the high side); flipping `qke_expm_fallback_near_degeneracy` to
True also flips the 5 MeV V_nunu projection result from sprint-5's
ΔNeff = −0.012 to −0.025 — a 0.013 drift from the sprint-5 baseline
(gate 7). Useful information, not catastrophic, but enough that the
flag stays opt-in until the Phase-0 step-function is also resolved
and Hannestad targets land cleanly inside band on **all three points
simultaneously**.

**Implementation.**

  * `PRyM/PRyM_init.py` — `qke_expm_fallback_near_degeneracy`
    (master toggle, default False) and `qke_expm_fallback_eps_cross`
    (relative-gap threshold, default 1.0e-3). Inserted after
    `qke_msw_diag_pair_idx`, before the y_max_boltz section.
  * `PRyM/PRyM_boltzmann.py::_etdrk2_expm_phi` — accepts new
    optional `H_sector` kwarg. When the flag is on AND H_sector is
    supplied, per y-mode dispatch: if `min_α |H_αα − H_ss| /
    max |H_αβ| < eps_cross` AND `max_α |H_α,sterile| > 0`, use
    eigendecomposition of L[i]*dt_nat (np.linalg.eig) with κ(V)
    > 1e8 guard reverting to Al-Mohy; else use the existing
    Al-Mohy augmented expm. Phi_k reconstructed from per-eigenvalue
    scalar phi functions (Taylor for |λ| < 1e-4, formula otherwise).
    Phi_0/Phi_1/Phi_2 dimensions and downstream-consumer convention
    unchanged.
  * `PRyM/PRyM_boltzmann.py::evolve_step_ode_etdrk2` — calls
    `_build_H_list` exactly when the fallback flag is on (~1 ms/
    step overhead, opt-in only). H_list is threaded into both
    sector calls of `_etdrk2_expm_phi`.
  * `_build_L_list` signature unchanged (5 external callers across
    tests + diagnostics inspect the 3-tuple return).

**Counters and diagnostic surfaces.**

  * `DensityMatrixSolver._expm_fallback_eig_count` — incremented
    per (mode, step, sector) eigendecomposition fire. Initialised
    lazily via getattr for backward compat.
  * `DensityMatrixSolver._expm_fallback_kappa_high_count` —
    incremented when κ(V) > 1e8 reverts a mode to Al-Mohy.

**Validation gates.**

  * **Gate 1** — `pytest -m "not slow"`: 4/4 pass at default flag.
    Bit-identical to sprint-10 by construction (flag off → identity
    branch). Includes `test_qke_etdrk2_nu_nubar_symmetry`.
  * **Gate 2** — `pytest -k sterile`: 3/3 pass at default flag.
  * **Gate 3** — `diag_2level_damped.py`: ratio = 1.000 at default.
  * **Gate 4** — `diag_2level_damped_energy.py`:
    \|dN/N\|, \|dE/E\| ≈ 1.7e-9 at default (within 1e-8 tolerance).
  * **Gate 5** — `diag_phase0_decoupled_fallback.py` (new, flag
    ON): max\|ρ_ss\| at Phase-0 exit = 1.0e-30, eigendecomp count
    = 0. The H-coupling gate (`max_α |H_α,sterile| > 0`) correctly
    rejects all decoupled modes; the fallback never fires when
    physics demands it not. Required for safety: an earlier gate
    formulation (using only the L-diagonal commutator gap) leaked
    Σρ_ss = 0.43 by activating eigendecomp on a 16-dim L with
    degenerate eigenvalues at zero, which `np.linalg.eig` cannot
    reconstruct cleanly.
  * **Gate 6** — `diag_phase0_pointC_fallback.py` (new, flag ON):
    Σρ_ss(Phase-0 exit) = 0.6673 (target was < 0.1 — **NOT met**);
    Neff = 3.0114, Yp = 0.2485, D/H = 2.469. Eigendecomp count =
    740 across Phase-0 + Phase-B; κ-guard reverts = 0.
  * **Gate 7** — `diag_vnunu_active_only_fallback.py` (new, flag
    ON, 5 MeV window): Point C with V_nunu projection ΔNeff =
    −0.025 (sprint-5 baseline at flag off was −0.012; **drift =
    0.013**); Point C without projection ΔNeff = +0.851 (sprint-5
    baseline at flag off was +0.858; drift = −0.007). Fallback
    fires at edge modes despite T_MSW (~62 MeV) being far above
    the 5 MeV integration range — the gate's relative-ratio
    formulation triggers when sterile-vacuum H_ss dominates spread.

**New flags landed (defaults).**

  * `qke_expm_fallback_near_degeneracy = False`
  * `qke_expm_fallback_eps_cross = 1.0e-3`

**Sprint-11 landing posture.** Eigendecomposition fallback +
H-coupling gate land as opt-in machinery. All defaults stay at
`False`. No test fixtures changed. The closure of Phase-B over-
amplification is real and will be the foundation for Stage E.2's
final close-out, but it is conditional on the user opting in until
Phase 0 is also resolved.

**Sprint-11 finding (mechanism reinterpretation).** Suspect
2-residual as defined in sprint-10 — Pade-branch transition in
`_etdrk2_expm_phi` at the y = 0.5 ν̄ MSW pass at istep 906-1035
in Phase 0 — is **falsified at narrow mixing**. The eigen-
decomposition path produces the same dynamics there. Whatever
drives the step-function jump from 0.177 to 0.4145 is downstream
of (Phi0, Phi1, Phi2) — the predictor-corrector composition, the
half-diagonal exp-Euler regularisation around the stiff diagonal
damping, the V_nunu mean-field feedback as ρ_ss starts to grow,
or N_gain's collisional sourcing. Sprint 12's job is to localise
that mechanism.

**Next structural candidates**, ranked post-sprint-11:

1. **Suspect 5 — Phase-0 step-function source** (new, prime).
   Independent of `_etdrk2_expm_phi`. Per-step diagnostic at the
   istep 906→1035 window with fallback ON should show: (a) what
   the (rho, dt, Phi_k) inputs to evolve_step_ode_etdrk2 look like
   just before the jump; (b) which substep — predictor, corrector,
   half-diag — produces the discontinuity; (c) whether V_nunu's
   feedback amplifies an initially small perturbation through the
   non-linear coupling. See `doc/STAGE_E2_SPRINT12_BRIEF.md`.
2. **Suspect 4 demoted.** Phase-B over-amplification is no longer
   a free-standing suspect: sprint 11 closed it via the eigen-
   decomposition fallback in Phase B. What remains is Phase 0's
   step-function plus the residual ΔNeff offset that follows from
   it.
3. **Default flip cluster** — once Suspect 5 closes, flip
   `qke_phase0_flag` and `qke_expm_fallback_near_degeneracy` to
   True together. Update `test_sterile_dw_production` fixture with
   a config override forcing both False for bit-identity. Re-run
   gate 7 to verify drift narrows; re-run sprint-5's projection
   test for ΔNeff convergence.
4. **Suspect 4 sub-question — eigendecomposition cost.** The
   fallback path adds ~1.4× wall-clock to the Point-C Phase-0+B
   run at n_B = 10000 + n_B_phase0 = 2500 (5101s baseline →
   7219s with fallback). Acceptable for diagnostic runs; for
   production after the default flip, consider a direct-expm
   cache for "stable" mode classes that don't trigger across
   repeated steps.
5. **Sprint-10 carryovers — Point-A 0.065 residual,
   non-adiabatic high-y correction, n_B auto-scale re-pin** —
   unchanged. Check whether sprint-12's Suspect 5 fix collapses
   any of these (likely Point-A residual is the same mechanism).
6. **Sprint-6 carryovers** — unchanged.

**E.2 sprint 12 — Suspect 5 partially localised, H-iteration fix
insufficient: ESCALATE to sprint 13.** Sprint 11's eigendecomposition
fallback closed Phase-B over-amplification but left the Phase-0 step-
function in ρ_ss(y=0.5, ν̄) untouched (istep 906→1035 jump from 0.177
to 0.4145, Σρ_ss(Phase-0 exit) = 0.6673 vs target < 0.05). Sprint 12
delivered sub-step instrumentation, used it to localise the active
sub-suspect, and attempted the brief's recommended 5a fix; the fix
materially smoothed corrector deltas but did **not** suppress the
step-function. Mechanism is more complex than the brief anticipated.

Sub-step instrumentation (`qke_phase0_substep_diag_flag`,
`qke_phase0_substep_y_target`, opt-in, default off): four snapshot
labels (after_half1, after_predictor, after_corrector, after_half2)
emitted by `evolve_step_ode_etdrk2` to `self._phase0_substep_hist` at
the y-grid index closest to the target. Each row records, at the
y-mode under investigation:
  * 5a probe: H diagonals + active-sterile coupling for both sectors,
    (ρ_ν − ρ_ν̄) active 3×3.
  * 5b probe: N_gain pre/post D·ρ add-back, predictor / corrector
    deltas at (α=1, sterile).
  * 5c probe: z_h per channel, Taylor-branch boolean.

Surfaced via the existing `_boltz_dm_solver` exposer (no PRyM_main
changes). Default config: zero branches taken, bit-identical to
sprint 11.

Localisation results from the slim Phase-B-truncated probe
`validation/diagnostics/diag_phase0_pointC_substep.py` (n_B=200,
~15 min):

  * **5c falsified.** z_h[1] max single-step relative jump 1.47e−3
    ≪ 1; Taylor-branch flag never flickers across istep 881→1081.
    Hard threshold at z_h < 1e−4 is stable at narrow mixing.
  * **5a partially confirmed.** |ρ_diff_active| Frobenius jumps
    7.34× in a single substep at istep 926→927; the ν̄ gap (H_11 −
    H_ss) collapses 9.91× and crosses zero at istep 959→960 (MSW
    resonance). Asymmetry is striking: ν gap relative jump only
    0.25 (smooth), confirming ν̄-only resonance.

Two fix variants attempted (both opt-in via
`qke_etdrk2_iterate_h_flag`, default off, gated to fire only when
`qke_expm_fallback_near_degeneracy` is also on):

| Fix variant       | Σρ_ss(P0 exit) | ρ_ss saturation | \|corr Δ\| jump |
|-------------------|----------------|-----------------|-----------------|
| No fix (sprint 11)| 0.6673         | 0.4145          | 1.88e+04        |
| Corrector-only Φ₂ | 0.6659         | 0.4145          | 71.7            |
| Picard restart    | 0.6491         | 0.4097          | 2.43e+03        |

Both variants reach the resonance-saturated regime and fail the
scope-(a) deliverable (Σρ_ss < 0.05). The corrector-only variant
(rebuild L + Φ₂ at rho_star) reduces the corrector kick 260× but
leaves the predictor's Φ₀(L_n) frozen, so Phase-0 ρ_ss is bit-
identical to sprint 11. The Picard-restart variant (rebuild L +
Φ_cache at rho_star and redo the predictor) further reduces the
ρ_diff jump and the saturation by ~1%, but the resonance still
fully pumps.

**Mechanism reinterpretation.** With sin²(2θ_24) = 1e-4, Hannestad
expects ρ_ss saturation in [0.02, 0.10] (Landau-Zener non-adiabatic
crossing for narrow mixing). Our solver consistently saturates near
0.41 — the full resonance equilibrium — regardless of how H is
iterated. This points to **V_nunu lock-in at resonance**: as the
active-block diagonal collapses toward the sterile diagonal, V_nunu
adapts to keep the system near the resonance peak, driving fully-
adiabatic conversion. The fix is not in the ETDRK2 substep
composition but in either:
  * collision-damping refactor (sub-suspect 5b — refactor
    `_assemble_collision_N` and `_build_L_list` to never produce
    −D·ρ, removing the cancellation residual at narrow mixing); or
  * adaptive resonance-aware time-stepping with explicit Landau-
    Zener treatment; or
  * a parallel DLSODA driver (brief's scope (c)).

**Sprint-13 brief**: `doc/STAGE_E2_SPRINT13_BRIEF.md` covers
scope (b) — the full collision-N refactor — as the highest-priority
next step. Scope (c) (DLSODA driver) is the fallback.

Sprint 12 verification gates (gate-5 + gate-1, both at default
config — new flags off → bit-identical to sprint 11):

  * **Gate-1 fast regression** (`pytest tests/test_regression.py
    -m "not slow" -v`): 4/4 PASS in 27 s.
  * **Gate-5 decoupled-sterile guard** (`diag_phase0_decoupled_fallback.py`):
    PASS — max|ρ_ss| < 1e−12 at Phase-0 exit (sprint-11 guarantee
    preserved).
  * **Gate-6 Phase-0 fallback Point-C probe**: still FAILS the
    scope-(a) deliverable (Σρ_ss < 0.05). With H-iteration on, ρ_ss
    saturates at 0.4097 (best variant); without, 0.4145.

Stage E.2 sprint 12 carryovers (must not regress under sprint 13):

1. **Substep instrumentation** is the load-bearing diagnostic for
   any future ETDRK2 driver work — sprint 13 will use it to verify
   that the collision-N refactor breaks the V_nunu lock-in. Don't
   touch the snapshot method or its hooks unless explicitly
   landing a new diagnostic surface.
2. **`qke_etdrk2_iterate_h_flag`** stays opt-in (default False).
   Even though the fix didn't meet scope-(a)'s success criterion,
   it does smooth the corrector delta and reduce ρ_ss saturation
   by ~1% — a modest improvement. Sprint 13 should re-evaluate
   whether to keep, deprecate, or strengthen it after the
   collision-N refactor.
3. **`diag_phase0_pointC_substep.py`** (slim, fast variant) is the
   sprint-12 fast-iteration probe. Keep as a sibling of the
   canonical gate-6 diagnostic; sprint 13 will use it as the
   primary localisation tool.
4. **Sprint-11 carryovers** unchanged.
5. **Sprint-10 carryovers** unchanged.

**E.2 sprint 13 — Suspect 6 falsified, collision-N refactor reverted:
ESCALATE to sprint 14 (DLSODA-first).** Sprint 13 attempted scope (a)
of the sprint-13 brief: refactor `_assemble_collision_N` and
`_build_L_list` so active-sterile pairs produce a TRUE zero collision
RHS instead of the cancelled −D·ρ + add-back pair. Hypothesis was
that the ULP cancellation residual at narrow mixing drives V_nunu
lock-in via a spurious `Phi1·N_off` kick in the ETDRK2 predictor.
**Hypothesis falsified at gate 6.**

Refactor diff (two surgical edits in `PRyM/PRyM_boltzmann.py`,
~10 lines total):

  * `_assemble_collision_N` line 4537: branch on `p_idx<3` so
    active-sterile pairs return `rhs_si = 0` instead of `-D·ρ`.
  * `_build_L_list` line 4613: skip the +D·ρ add-back loop for
    `p_idx≥3` pairs (since N_full is now 0 for them).

Active-active pairs were bit-identical by construction (only the
dead `S_gain_si = 0.0` branch for active-sterile changed; gain-only
pairs got the same N_full and N_gain).

Sprint 13 verification gates:

| Gate | Result | Detail |
|------|--------|--------|
| 1 fast pytest | PASS 4/4 in 32s | bit-identical (active-active untouched) |
| 2 sterile pytest | PASS 3/3 in 660s (11min) | bit-identical |
| 3 2-level damped Rabi | PASS ratio=1.000 | bit-identical |
| 4 2-level L conservation | PASS \|dN/N\|=\|dE/E\|=1.83e-10 | bit-identical |
| 5 decoupled-sterile guard | PASS max\|ρ_ss\|=1e-30, eigendecomp=0 | refactor sound for decoupled physics |
| 6 Phase-0 Point-C substep | **FAIL** | Σρ_ss(P0 exit)=0.6489, ρ_ss saturation 0.4099 |

Comparison of gate-6 outcomes:

| Variant | Runtime | Σρ_ss(P0 exit) | ρ_ss saturation | Neff (truncated PB) | Yp |
|---|---|---|---|---|---|
| sprint-12 no fix | 923s | 0.6673 | 0.4145 | 3.31068 | 0.26528 |
| sprint-12 picard | 1605s | 0.6491 | 0.4097 | 3.24463 | 0.25187 |
| sprint-13 refactor | **2965s** | 0.6489 | 0.4099 | **9.82082** | **0.31190** |

Σρ_ss(P0 exit) and ρ_ss saturation are essentially bit-identical to
sprint-12 picard — the refactor changed nothing in Phase 0. **Suspect
6 (V_nunu lock-in via spurious Phi1 kick) is falsified.** Sub-step
localisation summary confirms `|N_gain_13| nubar` max single-step
relative jump = 0.00e+00 (exact zero, was ULP residual before), so
the refactor took effect — but `|rho_diff_act|` and the gap-collapse
pattern are unchanged at the resonance crossing. The cancellation
residual was never the source.

**Phase-B regression** (the unexpected part). With identical
Phase-0 exit state, the refactor produced Neff = 9.82 in the
truncated Phase B (n_B=200) — a 3× jump from sprint-12 baselines.
Yp = 0.312 sits well outside the [0.24, 0.26] BBN-physical band.
Mechanism: with N_full[active-sterile] structurally zero (instead
of the prior −D·ρ + ULP residual), Phase-B integration of active-
sterile coherence has lost a force term that was contributing
non-trivially under the higher-temperature, more-stiff Phase-B
regime. Decoupled physics is unaffected (gate 5 PASS); the Phase-B
breakdown is conditional on having a non-zero active-sterile
coherence to integrate.

Wall-clock signal: gate 6 ran 2965s vs 1605s picard = 85% slower
even though Phase-0 dynamics are equivalent. Slowdown localises to
Phase-B integration and is consistent with adaptive timestep being
forced smaller to handle the now-stiffer regime. (Diagnostic flags
to localise this further: rerun gate 6 with `qke_substep_phase_b`
counters; not landed.)

**Refactor reverted** in the same sprint commit. Both edits rolled
back to sprint-12 state — `git diff PRyM/PRyM_boltzmann.py` is empty
against the sprint-12 landing. Sprint-12 instrumentation, opt-in
H-iteration, sprint-11 eigendecomp fallback all preserved.

**Failure record artifact**:
`validation/diagnostics/diag_phase0_pointC_substep_refactor_failed.out`
(sibling of `..._no_fix.out` and `..._picard.out`). Captured for the
sprint-14 post-mortem.

**Sprint-14 brief**: `doc/STAGE_E2_SPRINT14_BRIEF.md` lays out scope
(c) — a parallel `scipy.integrate.solve_ivp(method='LSODA')` driver
on the full vectorised state, with the same H/N kernels but a
production-grade adaptive step size and stiff method. Rationale:
both H-iteration variants (sprint 12) and the collision-N refactor
(sprint 13) failed to break V_nunu lock-in while keeping Phase-B
stable. The ETDRK2 driver may be the wrong tool for narrow-mixing
resonance crossings. A LSODA reference will tell us either (a) the
correct dynamics also hits Σρ_ss ~ 0.65 (in which case Hannestad's
Landau-Zener prediction is being mis-applied to our problem and we
should re-read the literature), or (b) the correct dynamics drops
to [0.02, 0.10] and our ETDRK2 driver is the bottleneck (in which
case LSODA replaces it as the production path).

Stage E.2 sprint 13 carryovers (must not regress under sprint 14):

1. **Sprint-12 instrumentation, sprint-11 fallback, sprint-10 driver,
   sprint-9 MSW probe, sprint-8 energy probe, sprint-7 cold-T fix,
   sprint-6 NaN-safe sanitisation, sprint-5 V_nunu projection,
   D.7.1 Strang sequence, D.7 expm cache** — all preserved bit-
   identical (revert restored sprint-12 state).
2. **Refactor failure record**:
   `validation/diagnostics/diag_phase0_pointC_substep_refactor_failed.out`
   is read-only — keep alongside `..._no_fix.out` and `..._picard.out`
   as the third corner of the suspect-survey table.
3. **Suspect 6 is FALSIFIED.** Do not retry the collision-N refactor
   in any variant; the Phase-0 step-function survives identical
   modulo ULP, and Phase-B regresses by 3× on Neff. The fix path
   moves to a different driver entirely.
