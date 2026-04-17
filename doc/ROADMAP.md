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
  midpoint I_total evaluation in the half-diag steps. Deferred as a
  potential D.7.2 if tighter SF accuracy is needed.

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
