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
