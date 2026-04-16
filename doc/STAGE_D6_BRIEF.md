# Stage D.6 brief: ETDRK2 with eigenbasis collision

This is a single-file handoff for a fresh context window picking up the
Stage D ODE-driver work on `PRyMordial-nu-sterile`. Read this first.
By the end of it you should know: what's been built, why the obvious
next step failed, what D.6 actually needs to do, and what not to
touch.

## One-paragraph orientation

PRyMordial-nu is a BBN code. We have extended its 3×3 QKE
density-matrix solver to 3+1 (active + sterile) on the
`DrTimTait/PRyMordial-nu-sterile` repo, and then built an alternate
full-ODE driver behind `qke_full_ode_flag` that replaces two
quasi-static diagonal-transfer approximations with an exact
per-mode unitary. Stage D.4 showed that driver converges cleanly in
3×3 but exhibits O(dt) behaviour at the Shi-Fuller MSW resonance. A
naive predictor-corrector was tried in D.5 and reverted because it
destabilizes at sharp turning points (explained below). D.6 is the
proper fix: move the collision operator into the instantaneous
H-eigenbasis, where each eigenmode carries its own phase through the
resonance sign-flip and the turning-point pathology disappears.

## What to read, in order

Budget ~15 minutes for this before writing anything.

1. `CLAUDE.md` — project overview.
2. `doc/ROADMAP.md` — entire Stage D section. Ends with the Stage D.6
   scope.
3. `git log --oneline -12` — commit headlines.
4. `git show 905010f` — full Stage D.5 commit message with the
   turning-point analysis.
5. `validation/stage_d5_sf_convergence_attempt.py` docstring — hands-on
   explanation of what breaks in the naive corrector.
6. `validation/stage_d_sf_convergence.py` + `validation/stage_d_sf_convergence.out.txt`
   — the single-stage SF convergence baseline to beat.
7. `PRyM/PRyM_boltzmann.py`:
   * `DensityMatrixSolver.__init__` (circa line 3000) — state layout,
     flavor-pair indexing (`_all_pair_flavors`, `_all_pair_write`,
     `_active_offdiag_idx`).
   * `DensityMatrixSolver._to_mat` / `_to_vec` (module-level
     `_rho_vec_batch_to_matrix` / `_matrix_batch_to_rho_vec`) —
     flavor-basis matrix↔vector packing.
   * `DensityMatrixSolver._apply_unitary` — sector-aware unitary
     (see "ν̄ convention" below).
   * `DensityMatrixSolver.evolve_step_ode` — the Strang-split ODE
     driver being upgraded.
   * `DensityMatrixSolver.evolve_step` — **do not touch** (Strang /
     quasi-static path; existing references depend on it).
   * `_offdiag_collision_gain` / `_offdiag_collision_gain_massive`
     (~line 1600) — Numba-accelerated active-active gain kernels.
8. `PRyM/PRyM_main.py` around line 370–410 — Phase B dispatcher where
   the ODE driver is invoked.
9. `tests/test_regression.py` — the list of green tests that must
   stay green.

## What Stage D.1–D.5 actually shipped

Commit → headline → what it delivered, in chronological order:

| Commit | Stage | Delivered |
|--------|-------|-----------|
| 231b525 | A | 4×4 density-matrix infrastructure, `sterile_flag` |
| e04b01f | B | Dodelson-Widrow production via quasi-static block |
| ea04426 | C | Shi-Fuller MSW + `xi_*_init` + per-sector V_sign |
| baf76cb | cleanup | pytest gates for sterile, demo notebook |
| 36ee0b7 | D.1 | `evolve_step_ode` + `_apply_unitary`, Strang-split |
| 5130926 | D.2 | ODE dt-convergence (3×3 OK); `n_B_override` knob |
| 3983297 | D.3 | SF ODE validation + **ν̄ convention bugfix** |
| e5fd08f | D.4 | SF dt-convergence finding: O(dt) at resonance |
| 905010f | D.5 | naive predictor-corrector fails at turning point |

## Conventions and gotchas

**ν̄ storage convention.** `rho_all[0]` stores ρ directly; `rho_all[1]`
stores ρ̄\* (complex conjugate of the antineutrino density matrix).
This is baked into `evolve_step`'s `osc_signs = [+1, -1]` for the
off-diagonal update. Consequence for the ODE driver:

```
Sector 0 (ρ):   ρ_new = U · ρ · U†                          (standard)
Sector 1 (ρ̄*): (ρ̄*)_new = U_ν̄* · (ρ̄*) · U_ν̄ᵀ             (NOT U · ρ · U†)
```

`_apply_unitary` already handles this correctly (D.3 bugfix). Any D.6
code that touches the eigenbasis transform must handle it too — ask
"am I treating sector 1 the same as sector 0?" and if yes, it is
probably wrong.

**Collision-operator parts.** What currently fills `evolve_step_ode`'s
middle step:

1. **Diagonal Euler** using `I_total[0..2]`: `nue`, `nuebar`,
   `numu_equiv`. The `phi1_dt` argument already encodes an
   exponential-Euler regularization for diagonal-collision stiffness
   as a (3, Ny) array from `PRyM_main.py`.
2. **Off-diagonal damping** `D_αβ = ½(Γ_α + Γ_β)` applied as
   `exp(-D·dt)` on each of the (at most 6) pair Re/Im slots.
3. **Off-diagonal collision gain** — `_offdiag_collision_gain[_massive]`
   returns `(6, Ny)` for the 3 active-active pairs; active-sterile
   pairs get zero gain.
4. **No Sigl-Raffelt block, no DW block** — those are deliberately
   absent from `evolve_step_ode` because the unitary handles the
   H-commutator exactly.

**Flavor-pair layout.** Read once in `_all_pair_flavors` and
`_all_pair_write`:

```
3-flavor: diag [ρ_ee, ρ_μμ, ρ_ττ]; pairs (0,1), (0,2), (1,2)
          stored at indices (3,4), (5,6), (7,8)
4-flavor: diag [ρ_ee, ρ_μμ, ρ_ττ, ρ_ss]; pairs interleaved:
          (0,1)→4,5  (0,2)→6,7  (0,3)→8,9  (1,2)→10,11
          (1,3)→12,13  (2,3)→14,15
```

Active-active pairs are the first three of `_all_pair_flavors` in both
3- and 4-flavor layouts; active-sterile pairs only exist in the
4-flavor layout.

## What D.6 needs to do

The root problem: near the MSW resonance, `ω_αs = H_αα − H_ss` sweeps
through zero. Per-pair Strang sees the phase `exp(-iω·dt/2)` rotate
by O(1) in different directions on either side of zero, and the
splitting error between unitary and collision becomes O(dt) because
the collision is applied after a phase that has "swung through"
rather than averaged.

Stage D.5 tried averaging H over ρ_n and ρ_pred; the average ω is
still close to zero at resonance so the predictor and corrector land
on opposite sides and the two unitary halves rotate in opposite
senses. Doesn't work.

The honest fix: **work in the instantaneous H-eigenbasis.** In that
basis the commutator `-i[H, ·]` is diagonal per pair of eigenvalues
— each eigenmode has its own phase, and applying the collision
element-wise with an eigenbasis-aware weighting respects the actual
per-mode dynamics. Specifically, ETDRK2:

```
Predictor:  ρ* = e^{L dt} ρ_n + dt · φ_1(L dt) · N(ρ_n)
Corrector:  ρ_{n+1} = ρ* + dt · φ_2(L dt) · (N(ρ*) − N(ρ_n))
```

where `L[ρ] = -i[H, ρ]` and `N(ρ)` is the collision operator. In the
H-eigenbasis (eigenvalues λ_k, eigenvector matrix V):

```
(L[A])̃_{kl} = -i(λ_k − λ_l) A_{kl}                 (acts element-wise)
e^{L dt} [A] = diag(e^{-iλ_k dt}) · Ã · diag(e^{+iλ_l dt})        ← already in _apply_unitary
dt·φ_1(L dt) [A] on element (k,l) = (e^{-i(λ_k-λ_l)dt} - 1) / (-i(λ_k-λ_l))·A_{kl}
dt²·φ_2(L dt) [A] on element (k,l) = same with the 2nd-order φ_2
```

For `k = l` (diagonals), `ω = 0` so φ_1 = dt, φ_2 = dt²/2; for
`k ≠ l` with `|ω dt|` large, φ_1 and φ_2 are bounded oscillatory
factors. This is precisely what goes wrong in a flavor-basis Strang
split at resonance: the appropriate φ factor depends on `ω` for each
eigenpair individually, and averaging in the flavor basis is the
wrong thing.

## Concrete implementation sketch

You will likely want to:

1. **Add a flag `qke_ode_etdrk2_flag`** (default False) in
   `PRyM_init.py`, gated inside `qke_full_ode_flag`. Reset it in
   `_reset_flags` in `tests/test_regression.py`.
2. **Factor the collision operator into a single callable**
   `N(ρ) → (dρ/dt)_collision` that returns a per-sector-per-mode
   Hermitian matrix (shape `(2, Ny, N, N)` in flavor basis). This
   involves assembling:
   - Diagonals from `I_total` (already computed).
   - Active-active off-diagonals from `_offdiag_collision_gain`.
   - Damping contribution `-D_αβ ρ_αβ` on off-diagonals.
   - Zero on active-sterile off-diagonals (already correct).
3. **Build two eigen bases per step**: H(ρ_n) for the predictor, then
   H(ρ*) for the corrector. Reuse the `eigh` already needed by
   `_apply_unitary`.
4. **Transform N into eigenbasis**, apply φ_k element-wise with the
   right per-pair phase, transform back. This is the part Stage D.5
   skipped and that the naive H-averaging can never do correctly.
5. **Watch for the ν̄ convention** (`U*·ρ·Uᵀ` instead of `U·ρ·U†`)
   everywhere you rotate into or out of the eigenbasis.

## Target metrics

A working D.6 should:

- **Pass all existing slow regression tests** (modes 1, 2, 5, 5b, 6,
  sterile_stage_a_invariant, sterile_dw_production,
  sterile_sf_asymmetry_depletion).
- **Match Stage D.1's frozen `test_mode5b_qke_full_ode` values** when
  `qke_ode_etdrk2_flag = False` (ETDRK2 is a new code path; default
  stays Strang-split).
- **Beat the Stage D.4 SF convergence baseline**: at default
  `n_B = 2400`, SF ODE Neff should land within 10⁻³ of the
  Richardson-extrapolated ODE limit (≈ 3.967) that D.4 measured.
  Currently the default-n_B result is 3.95297 (1.4×10⁻² below the
  limit).
- **Be stable under dt refinement**, i.e. the SF convergence
  diagnostic rerun with `qke_ode_etdrk2_flag = True` at
  `n_B ∈ {2400, 4800, 9600}` should show monotonic O(dt²) convergence
  and NOT the D.5 divergence at n_B=9600.

## Do-not-touch list

- `evolve_step` — the Strang/quasi-static path. All pre-Stage-D tests
  and the Stage B/C sterile tests depend on its exact reference
  values. If you need to touch shared helpers (`_offdiag_collision_gain`,
  `_to_mat`), add new methods rather than modifying the old ones.
- `_apply_unitary` — already contains the ν̄-convention fix from D.3.
  If the new code path needs a variant (e.g. half-step vs full-step
  unitary), write a new helper rather than swapping behaviour under a
  flag.
- Default values of any existing flag. Default behaviour of the code
  must remain unchanged — every new capability is opt-in.

## Session realism

The ROADMAP scoped D.6 as 4–7 days of work. In one context window
plan to achieve **one** of these, not all three:

- Full clean plan + scaffolded code (flag, refactor, stub tests)
  with a working 3×3 smoke test. Defer SF validation and debugging
  to a follow-up session.
- Complete implementation including eigenbasis φ-factor assembly and
  the `N(ρ)` refactor, without thorough validation. Mode 5b pytest
  green; SF convergence deferred.
- Debugging an existing partial D.6 implementation handed off from a
  prior session.

Pick one of those at the start and say so explicitly in the opening
message to the session. Trying all three in one context will thin
out quality on each.

## Minimum viable opening message for the new session

> Read `doc/STAGE_D6_BRIEF.md` end-to-end before writing any code.
> Then use `EnterPlanMode` to propose a concrete Stage D.6
> implementation plan. My target for this session is {ONE of the
> three in "Session realism"}. Do not touch `evolve_step` or change
> any existing default behaviour.
