# Stage E.2 sprint 14 brief: DLSODA-first reference driver (scope c)

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 13 (commit `<sprint-13 hash>`; will be set by the
sprint-13 landing). Read this first. By the end you should know
why the collision-N refactor failed even though Phase-0 dynamics
were preserved, why both H-iteration (sprint 12) and the collision-N
refactor (sprint 13) left V_nunu lock-in intact, and why the path
forward is a parallel LSODA driver — not another ETDRK2 surgical
edit.

## One-paragraph orientation

Sprint 12 attempted to break V_nunu lock-in with ETDRK2 H-iteration
(corrector-only Φ₂ at L(ρ⋆); Picard restart). Both smoothed
corrector residuals 100×–7× but only nudged Σρ_ss(P0 exit) from
0.6673 → 0.6491 (1%); ρ_ss saturation barely moved (0.4145 →
0.4097). Sprint 13 attempted to remove the spurious `Phi1·N_off`
kick that Suspect 6 implicated — refactor `_assemble_collision_N`
and `_build_L_list` so active-sterile pairs produce a TRUE zero
collision RHS (no −D·ρ + add-back cancellation). Result: Σρ_ss(P0
exit) stayed at 0.6489 — bit-equivalent to sprint-12 picard, **no
improvement**. Worse, the refactor produced a Phase-B Neff blow-up
(Neff = 9.82 vs 3.31 baseline) and 85% wall-clock slowdown,
indicating the cancellation residual was carrying load in the
stiff Phase-B regime. Suspect 6 is falsified; the refactor was
reverted in-sprint. Two converging signals (H-iteration + collision-
N refactor both failed to break lock-in) point to the ETDRK2 driver
itself being structurally incompatible with narrow-mixing resonance
crossings — not just a coefficient bug. Sprint 14's job: build a
parallel `scipy.integrate.solve_ivp(method='LSODA')` driver as a
reference, run Hannestad A/B/C through it, and either close Stage
E.2 (LSODA hits target → ship LSODA as production) or escalate to
literature re-read (LSODA also saturates → re-read Hannestad's
non-adiabatic prediction; we may be solving the right equations and
the target is wrong).

## What to read, in order

Budget ~60 min before writing any driver code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — the sprint-13 landing record at the
   bottom has the gate-by-gate validation, the gate-6 falsification
   table, the Phase-B Neff regression, and the Suspect 6
   falsification narrative. Skim sprint-12 for the H-iteration
   variants and sprint-11 for the eigendecomposition fallback.
3. **`doc/STAGE_E2_SPRINT13_BRIEF.md`** — the sprint that just
   ran. Suspect 6 is falsified; its hypothesis was exactly what
   we just tried.
4. **`doc/STAGE_E2_SPRINT12_BRIEF.md`** — substep instrumentation
   and H-iteration design. Same outcome (lock-in not broken).
5. **`validation/diagnostics/diag_phase0_pointC_substep_no_fix.out`**
   — sprint-12 baseline.
6. **`validation/diagnostics/diag_phase0_pointC_substep_picard.out`**
   — sprint-12 H-iteration variant.
7. **`validation/diagnostics/diag_phase0_pointC_substep_refactor_failed.out`**
   — sprint-13 collision-N refactor (Phase-B regression artifact).
   Compare line-by-line against picard: Σρ_ss(P0 exit) is bit-
   equivalent (0.6489 vs 0.6491), but Neff jumps 3×.
8. **`PRyM/PRyM_boltzmann.py`**:
   * `evolve_step_ode_etdrk2` — the current driver (lines
     ~5005-5170 after sprint-12 edits). Sprint 14 leaves this
     untouched; LSODA is a *parallel* path, not a replacement.
   * `_build_H_list`, `_build_L_list`, `_assemble_collision_N` —
     the kernels that LSODA's RHS must reuse. They produce H, L,
     N in eV units; LSODA needs a single RHS function `f(t, y)`
     that returns `dy/dt` in 1/eV (or whichever unit aligns with
     `solve_ivp`'s default).
   * `_apply_unitary` and the diagonal exp-Euler step — *not*
     used by LSODA; LSODA solves the unsplit ODE.
9. **`PRyM/PRyM_init.py`** — flags: a new `qke_lsoda_driver_flag`
   should land here, defaulting False (parallel path, opt-in).

## What's probably broken (the new framing)

The two-converging-signals argument:

  * **Sprint 12 (H-iteration)**: rebuilding H at ρ⋆ and re-doing
    the predictor at L(ρ⋆) reduces corrector residuals ~100×, but
    Phase-0 ρ_ss saturation stays at 0.4097. The unitary part of
    Phi0 = e^{L·dt} is already pumping the resonance to full
    adiabatic conversion, regardless of what coefficient ETDRK2
    uses for the corrector.
  * **Sprint 13 (collision-N refactor)**: removing the −D·ρ +
    add-back cancellation residual leaves Σρ_ss bit-equivalent to
    sprint-12 picard. Worse, Phase-B regresses 3× on Neff. The
    cancellation residual was carrying load in the stiff Phase-B
    regime — removing it broke a force balance we hadn't named.

Both suspects (5b, 6) shared the framing "the ETDRK2 split is fine,
just one term is mis-coefficient'd." Both are now falsified. The
remaining options:

### Suspect 7 — Strang split is incorrect for narrow-mixing resonance

The current driver uses a half-step unitary, then a collision step
(diagonal exp-Euler + off-diagonal ETDRK2 with L = -i[H,·] - D),
then a second half-step unitary. At narrow mixing, V_nunu non-
linearly couples ρ_diff (the ν-ν̄ asymmetry) to H's active-sterile
gap. The resonance crossing happens within a single sub-step at
istep ~960 (sprint-12 5a probe). Strang split assumes the unitary
and collision pieces commute well enough over dt; in this regime
they don't.

### Suspect 8 — Hannestad's prediction is being mis-applied

Hannestad's [0.02, 0.10] band for ρ_ss saturation at sin²(2θ)=1e-4
assumes a specific Landau-Zener treatment. Our QKE solves the full
3+1 density matrix evolution under V_nunu lock-in. These may not be
comparable apples-to-apples — particularly if Hannestad's
non-adiabatic crossing assumes a non-self-consistent V_nunu, or a
different lepton-asymmetry boundary condition.

LSODA-first cuts both ways: it's a definitive test of whether ETDRK2
is the bottleneck (Suspect 7 confirmed if LSODA gives [0.02, 0.10])
or whether the target itself is wrong (Suspect 8 confirmed if LSODA
also gives 0.6).

## Refactor plan (scope c)

### Phase 1 — Build the parallel LSODA driver (~6 hours)

Add `evolve_full_ode_lsoda(self, T_start, T_end, n_steps, ...)` to
`PRyM/PRyM_boltzmann.py:DensityMatrixSolver` that:

1. Flatten the full state `rho_all[2, n_components, Ny]` to a 1-D
   real vector of length `2 * 2 * n_components * Ny` (real + imag
   parts; both ν and ν̄ sectors).
2. Define `f(t, y_flat)` that:
   * Reshape `y_flat` → `rho_all`.
   * Call `_build_H_list(rho_all, a(t), Tg(t))` → H_list per sector.
   * Call `_build_L_list(rho_all, a(t), Tg(t))` → L_list, N_gain,
     I_total.
   * Compute `dρ/dt` per sector as `-i [H, ρ] + N_collision(ρ)`
     where N_collision is the gain-only term plus the diagonal
     I_total contributions, all in eV.
   * Convert eV → 1/s via `_eV_to_secm1`, flatten, return.
3. Call `scipy.integrate.solve_ivp(f, [T_start, T_end], y0_flat,
   method='LSODA', rtol=1e-6, atol=1e-10, dense_output=False)` with
   appropriate tolerance settings.
4. Reshape solution at terminal time → `rho_all_final`.

The driver must reuse `_build_H_list`, `_build_L_list`, and
`_compute_D_pair_matrix` exactly so that physics is identical
across the two paths. The only difference is the time integrator.

### Phase 2 — Wire into `PRyMclass` behind a flag (~1 hour)

In `PRyM/PRyM_init.py`:

```python
qke_lsoda_driver_flag = False  # opt-in parallel reference driver
```

In `PRyM/PRyM_boltzmann.py:DensityMatrixSolver.evolve_phase_0` (or
whichever wrapper Phase-0 / Phase-B currently dispatch through),
gate on `PRyMini.qke_lsoda_driver_flag` to use the LSODA path
instead of `evolve_step_ode_etdrk2` for both phases.

### Phase 3 — Cross-check on Hannestad A/B/C (~3 hours)

Re-use the existing diagnostic harness:
`validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`. Run
twice — once with `qke_lsoda_driver_flag=False` (current ETDRK2
path), once with True (new LSODA path). Compare:

  * Σρ_ss(P0 exit), ρ_ss saturation
  * Final Neff, ΔNeff vs SM, Yp
  * Wall-clock

**Decision tree**:

| LSODA result | Interpretation | Action |
|---|---|---|
| Σρ_ss(P0 exit) ∈ [0.02, 0.10], ΔNeff matches Hannestad | ETDRK2 is the bottleneck (Suspect 7) | Ship LSODA as production. Default flip `qke_lsoda_driver_flag=True`; keep ETDRK2 path opt-in for performance benchmarking. |
| Σρ_ss(P0 exit) ≈ 0.65, ΔNeff matches ETDRK2 | Hannestad target is wrong for our problem (Suspect 8) | Re-read Hannestad+2008 carefully. Check whether their "non-adiabatic" assumption holds when V_nunu is solved self-consistently. May need to redefine the BBN bands. |
| LSODA crashes or diverges | RHS function bug; debug | n/a |

### Phase 4 — Validation (post-LSODA decision)

If Suspect 7 wins (LSODA closes Stage E.2):

  1. `pytest tests/test_regression.py -m "not slow"` — 4/4 bit-
     identical at default (LSODA flag off).
  2. `pytest -k sterile` — 3/3 at default. Then re-run with LSODA
     flag on and adjust fixture tolerances if needed.
  3. Hannestad A/B/C through LSODA — A ∈ [0.9, 1.1], B ∈
     [0.3, 0.7], C ∈ [0.02, 0.10]; Yp ∈ [0.24, 0.26] for all three.
  4. Default flip: `qke_lsoda_driver_flag = True`.
  5. Update `test_sterile_dw_production` fixture and any other
     sterile tests with `qke_lsoda_driver_flag = False` overrides
     for bit-identity to the sprint-11 frozen oracle.

If Suspect 8 wins (LSODA also saturates):

  1. Stage E.2 closure deferred to a literature-review sprint.
  2. Document the discrepancy with explicit numerical evidence
     (LSODA + ETDRK2 agree on Σρ_ss = 0.65; Hannestad predicts
     [0.02, 0.10]).
  3. Consider whether to redefine the Hannestad band as the new
     baseline target or whether to dig into self-consistent
     V_nunu treatment in literature.

## Stage inheritance: what NOT to touch

Sprint-13 reverted the collision-N refactor. Sprint-14 inherits the
sprint-12 state of `PRyM_boltzmann.py`. Specifically preserve:

  * `_assemble_collision_N` and `_build_L_list` exactly as in
    sprint 12 (the −D·ρ / +D·ρ cancellation pair stays; suspect 6
    is falsified, the refactor regressed Phase B).
  * `evolve_step_ode_etdrk2` and all sprint-12 H-iteration machinery.
  * Sprint-11 eigendecomposition fallback.
  * Sprint-10 Phase-0 driver and instrumentation.
  * Sprint-9 MSW probe.
  * Sprint-8 energy probe.
  * Sprint-7 `_F_stat_stable` clamp.
  * Sprint-6 NaN-safe sanitisation.
  * Sprint-5 V_nunu projection.
  * D.7.1 Strang-symmetric composition.

Sprint-14 adds `evolve_full_ode_lsoda` as a *parallel* method —
it does not replace the existing path. ETDRK2 stays as the
default until LSODA proves it can carry production.

## Diagnostic plan (post-LSODA)

If LSODA closes Suspect 7:

  * Run the sub-step probe at Phase-0 Point C through the LSODA
    driver (instrumentation needs minor adapter; LSODA doesn't do
    "sub-steps" in the ETDRK2 sense, so capture the full
    trajectory and project onto an istep=2460 grid post-hoc).
  * Compare LSODA's ρ_ss(istep) curve against ETDRK2's at the
    istep ~926→960 transition. The shape difference is the
    structural artifact we couldn't see before.

If LSODA confirms Suspect 8:

  * Re-derive the Hannestad Landau-Zener prediction from
    Hannestad's exact assumptions, allowing self-consistent
    V_nunu. May require literature search beyond Hannestad+2008.

## Validation targets (post-implementation, scope (c))

| Gate | Command | Target |
|---|---|---|
| 1 fast tests | `pytest -m "not slow" -v` | 4/4 bit-identical at default (LSODA off) |
| 2 sterile tests | `pytest -k sterile -v` | 3/3 at default |
| 3 LSODA sanity (no sterile) | LSODA-mode `_run_qke_with_rho` of `cfg_3x3` | Neff matches ETDRK2 to <1e-4, Yp <1e-5 |
| 4 LSODA sanity (decoupled sterile) | LSODA-mode `cfg_4x4_decoupled` | Σρ_ss < 1e-12 (parallels gate 5 of sprint-13) |
| 5 LSODA Hannestad | `diag_hannestad_proj_w30_nB10k.py` with LSODA flag | A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10]; Yp ∈ [0.24, 0.26] |
| 6 LSODA wall-clock | Same as gate 5 | <2× ETDRK2 baseline (1605s picard); >5× is a red flag |

If all 6 gates hit: flip `qke_lsoda_driver_flag` default to True,
update `test_sterile_dw_production` fixture, close Stage E.2.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT14_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete
> sprint-14 plan. Target for this session is to land the LSODA
> driver and run gate 5 (Hannestad). Suspect 6 (collision-N
> cancellation residual) is falsified at the gate-6 level; do not
> retry that refactor in any variant.
>
> The refactor was reverted in-sprint; sprint-14 inherits
> sprint-12 state of `PRyM_boltzmann.py`. The new code surface
> is `evolve_full_ode_lsoda`, a *parallel* method that reuses
> `_build_H_list`, `_build_L_list`, and `_compute_D_pair_matrix`.
> Do not touch `evolve_step_ode_etdrk2` or any of its sprint-12
> instrumentation.

## Commit chain for context

  * `<sprint-13 hash>` — **Sprint 13** (collision-N refactor
    landed and reverted; Suspect 6 falsified; Phase-B Neff
    regression captured). Sprint 14 builds directly on top.
  * `e699c00` — Sprint 12 (sub-step instrumentation;
    H-iteration variants).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver).
  * `ac1d521` — Sprint 9 (MSW diagnostic).
  * `bb0ea32` — Sprint 8 (Suspect 1 falsified).

## Post-LSODA: downstream opportunities

If sprint 14 closes Suspect 7 and lands the default flip:

  1. **Gariazzo benchmark** (`validation/sterile_DW_gariazzo.py`),
     ΔNeff ∈ [0.05, 0.2].
  2. **Shi-Fuller literature** (Saviano+2013).
  3. **ETDRK2 deprecation path**. If LSODA wins production, the
     ETDRK2 driver becomes a benchmarking tool only. Decide
     whether to keep maintaining it or freeze it.
  4. **Performance** — LSODA's adaptive step may be 1-3× ETDRK2
     wall-clock; profile and tune `rtol`/`atol`.

If sprint 14 confirms Suspect 8:

  1. **Literature re-read sprint** — re-derive Hannestad's
     prediction under self-consistent V_nunu assumptions.
  2. **BBN-band redefinition** — agree on a new pass criterion
     for Stage E.2 closure that reflects the actual physics.
