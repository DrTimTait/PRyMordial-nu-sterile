# Stage E.2 sprint 13 brief: collision-N refactor (scope b carryover from sprint 12)

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 12 (commit `<sprint-12 hash>`; will be set by the
sprint-12 landing). Read this first. By the end you should know why
sprint 12's H-iteration fix only nibbled at the Phase-0 step-function
in ρ_ss(y=0.5, ν̄), what the V_nunu lock-in mechanism is, and the path
to closing Stage E.2 with all three Hannestad points in band via a
collision-RHS refactor.

## One-paragraph orientation

Sprint 11 closed Phase-B over-amplification at narrow mixing via an
opt-in eigendecomposition fallback in `_etdrk2_expm_phi`
(`qke_expm_fallback_near_degeneracy`). Sprint 12 added sub-step
instrumentation, used it to localise the active sub-suspect for the
unchanged Phase-0 step-function (sub-suspect 5a confirmed: V_nunu
non-linear feedback drives a sub-step ν̄ gap collapse and MSW resonance
crossing at istep ~960; 5c falsified; 5b inconclusive at the
diagnostic level — exact cancellation in `_build_L_list`'s D·ρ
add-back means the residual is at machine precision), and attempted
the brief's recommended H-iteration fix in two variants (corrector-
only Φ₂ at L(rho_star), then Picard restart of the full predictor at
L(rho_star)). Both fixes materially smoothed corrector deltas (260×
reduction with corrector-only) and modestly reduced ρ_ss saturation
(0.4145 → 0.4097, ~1%) — but **neither suppressed the step-function**.
Σρ_ss(Phase-0 exit) only dropped from 0.6673 to 0.6491. The system
still saturates near the full resonance equilibrium of 0.5 instead of
the expected Landau-Zener non-adiabatic value in [0.02, 0.10]. The
mechanism is V_nunu *lock-in*: as the active diagonal of H collapses
toward H_ss, V_nunu adapts to keep the system AT resonance, forcing
fully-adiabatic conversion regardless of where in the Strang split we
build H. The fix lives outside `evolve_step_ode_etdrk2` —
specifically in the collision RHS that should suppress coherence
faster than the resonance can pump it. Sprint 13's job: refactor
`_assemble_collision_N` and `_build_L_list` to never produce −D·ρ on
active-sterile pairs (move all damping into L), then verify the
resonance lock-in breaks.

## What to read, in order

Budget ~60 min before writing any refactor code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — the sprint-12 landing record at the bottom
   has the gate-by-gate validation, the V_nunu lock-in
   reinterpretation, the three carryover artifacts, and the full
   table of fix-variant outcomes. Also skim sprint-11 (Suspect-2
   residual partially confirmed; eigendecomp fallback) and
   sprint-10 (Phase-0 driver landed).
3. **`git show <sprint-12 hash>`** — the sprint-12 landing commit.
4. **`validation/diagnostics/diag_phase0_pointC_substep_no_fix.out`**
   — sprint-12 baseline (no H-iteration). 5a/5b/5c tables in the
   istep 881→1081 window. Reference for what "broken" looks like.
5. **`validation/diagnostics/diag_phase0_pointC_substep_picard.out`**
   — sprint-12 Picard-restart variant. Same window, reduced ρ_diff
   jump but same saturation. Reference for what "5a fix"
   accomplishes (and doesn't).
6. **`PRyM/PRyM_boltzmann.py::_assemble_collision_N`** (search for
   `def _assemble_collision_N`, lines ~4397-4536). The damping
   piece for active-sterile pairs lives at line 4528:
   `S_gain_si = 0.0`, so `rhs_si = -D_pairs[p_idx] * rho_ab_stored`
   — pure damping, no gain. This is the −D·ρ that gets cancelled
   by the add-back in `_build_L_list:4596-4601`.
7. **`PRyM/PRyM_boltzmann.py::_build_L_list`** (lines 4540-4603).
   Particularly the cancellation block 4591-4601: `N_gain[s][:,
   alpha, beta] += D_off_eV[alpha, beta] * rho_mat[:, alpha, beta]`.
   For active-sterile pairs this produces N_gain ≈ 0 (within ULP).
   The damping, meanwhile, lives in L via `np.diag(D_diag_vec)` at
   line 4586/4589 — exponentiated by `_etdrk2_expm_phi` so the
   off-diagonal coherence ρ_α,sterile decays with rate D in the Φ₀
   propagator.
8. **`PRyM/PRyM_boltzmann.py::evolve_step_ode_etdrk2`** (lines
   ~5005-5170 after sprint 12's edits). Sprint-12 landed the
   Picard-restart H-iteration here behind `qke_etdrk2_iterate_h_flag`.
   Don't touch unless you're refactoring the ETDRK2 driver itself.

## What's probably broken (one new suspect)

### Suspect 6 — V_nunu lock-in driven by collision RHS structure (PRIME)

The sprint-12 sub-step probe shows that even with H rebuilt at
rho_star (Picard restart), the resonance still fully pumps ρ_ss to
~0.41. The H-iteration fixes the *coefficient* of ETDRK2 but not the
*driving term*. The driving term is the off-diagonal evolution
through Φ₀ = e^{L·dt}, where L includes the [H,·] commutator.

At narrow mixing (sin²(2θ) = 1e-4), in-medium H is dominated by V_nunu
shifts as ρ_diff grows. The collision damping rate D (from
`_compute_D_pair_matrix`) lives in the diagonal of L (line 4586),
exponentiated alongside the unitary [H,·] commutator. The off-
diagonal coherence ρ_α,sterile decays in Φ₀ at rate D — **but** the
unitary part of Φ₀ also pumps ρ_α,sterile through the resonance.

The hypothesis: the cancellation between gain (which is zero for
active-sterile) and damping (which goes into L) leaves the collision
RHS with a structural artifact. When sprint 12's `_build_L_list`
add-back fires, N_gain[active, sterile] = N_full[active, sterile] +
D·ρ = -D·ρ + D·ρ = 0 (within ULP). But the L-side damping `-D` on
the diagonal of L acts on the FULL vec(ρ), including diagonal
components. In the row-major vec layout, `vec(ρ)[α*N + α] = ρ_αα`,
so the damping diagonal `D_diag_vec[α*N + α] = D_off[α, α] = 0` (no
self-damping). But for the `(α, sterile)` entries
`vec(ρ)[α*N + 3]`, `D_diag_vec[α*N + 3] = D_off[α, 3]` damps the
coherence in Φ₀.

**This is correct in isolation**, but the predictor's `Phi1 · N_off`
term still adds a *coherent* drive that's not balanced against the
damping in N_off itself (because we set N_off[active, sterile] = 0
via the cancellation). The result: every time-step injects a small
coherent kick from N_gain (or what's left after cancellation, which
is non-trivial when ρ is changing fast), and that kick survives in
Φ₁·N_off until eaten by Φ₀'s damping in the next step. The damping
is correct but the *kick* is wrong.

### Diagnostic plan

Re-use sprint 12's substep probe with the same Point C config. Add
two new captured fields to `_phase0_substep_snapshot`:

  * `Phi0_diagonal_at_(1,3)`: the (1, sterile) entry of Phi0 at y_idx,
    both sectors. Tells you how much of the diagonal damping `D` makes
    it through `e^{L·dt}` per step.
  * `(Phi1 · N_off_n)[1, 3]` raw vs after enforcement of N_off[1,
    sterile] = 0: see how big the residual kick is.

If the residual kick is non-negligible at the istep ~927 transition,
that's the smoking gun for Suspect 6.

### Refactor plan

The cleanest fix: **eliminate the −D·ρ piece from
`_assemble_collision_N` for active-sterile pairs**. Move all damping
into L (already done) and have `_assemble_collision_N` return only
the gain piece (which is zero for active-sterile, by construction).
Then `_build_L_list`'s add-back becomes a no-op for active-sterile
pairs, and the cancellation at the eV ↔ 1/s unit boundary disappears.

#### Code surface for the refactor

  * `_assemble_collision_N`: change the active-sterile pair branch
    (line 4528) from `rhs_si = -D_pairs[p_idx] * rho_ab_stored` to
    `rhs_si = 0.0`. Active-sterile pairs have zero collision RHS.
  * `_build_L_list`: remove the conditional add-back for active-
    sterile pairs in the loop at lines 4596-4601 (or skip them via
    `if (alpha, beta) in self._active_sterile_pairs: continue`).
    Active-active pairs still need the add-back.
  * `_compute_D_pair_matrix`: unchanged — D_off still computed
    correctly for the L diagonal.
  * `evolve_step_ode`: same `_build_L_list` change applies; mirror.

This is bit-identical to the current code modulo ULP, except for
active-sterile pairs at narrow mixing where the cancellation residual
was non-trivial. With the residual gone, the predictor's Phi1·N_off
drive on (active, sterile) entries is exactly zero, and the only
evolution there is unitary [H,·] + L-side damping. The hypothesis is
that this breaks V_nunu lock-in: without the spurious Phi1 kick, the
resonance can't be artificially reinforced.

## Stage inheritance: what NOT to touch

Sprint-12 additions (all retained):

  * `qke_phase0_substep_diag_flag` / `qke_phase0_substep_y_target`
    defaults (False / 0.5).
  * `_phase0_substep_snapshot` method, four hook callsites in
    `evolve_step_ode_etdrk2`, the `_phase0_substep_hist` /
    `_phase0_substep_y_idx` solver fields.
  * `qke_etdrk2_iterate_h_flag` default False — preserve the opt-in
    H-iteration even though it didn't fix the step-function. Sprint
    13 should re-evaluate after the collision-N refactor lands.
  * `validation/diagnostics/diag_phase0_pointC_substep.py` (slim
    fast probe), and the additive substep extension to
    `validation/diagnostics/diag_phase0_pointC_fallback.py`.
  * `validation/diagnostics/diag_phase0_pointC_substep_no_fix.out`
    and `..._picard.out` baselines. Sprint 13 should produce a new
    `..._refactor.out` for comparison.

Sprint-11 / -10 / earlier carryovers unchanged (see ROADMAP).

## Scope options

  * **(a) Scope-(b) refactor only.** Land the
    `_assemble_collision_N` / `_build_L_list` refactor described
    above. Verify with sub-step probe. If Σρ_ss(Phase-0 exit) drops
    below 0.05 and Hannestad gate-6 sits in band for A/B/C, **close
    Stage E.2**: flip `qke_phase0_flag` and
    `qke_expm_fallback_near_degeneracy` defaults to True, update
    `test_sterile_dw_production` fixture with overrides forcing
    both False for bit-identity. ~10-14 hours.
  * **(b) (a) + DLSODA cross-check.** On top of (a), build a
    parallel `scipy.integrate.solve_ivp(method='LSODA')` driver on
    the full vectorised state. Run Hannestad A/B/C through both
    drivers; if they agree, ship the Strang-split as production.
    ~2 sessions.
  * **(c) DLSODA-first.** Skip the refactor and write the LSODA
    driver as the production path. The collision-N refactor may
    still be needed for the LSODA path's RHS function, so this is
    not a shortcut. ~2-3 sessions.

State the choice in the opening message.

## Validation targets (post-refactor, scope (a))

  1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
     4/4 must pass bit-identical at default config.
  2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
     3/3 at default.
  3. **2-level damped Rabi** —
     `python validation/diagnostics/diag_2level_damped.py`. Ratio
     1.000 at default.
  4. **2-level L conservation** —
     `python validation/diagnostics/diag_2level_damped_energy.py`.
     |dN/N|, |dE/E| within 1e-8 at default.
  5. **Phase-0 decoupled-sterile guard** —
     `python validation/diagnostics/diag_phase0_decoupled_fallback.py`.
     `max|ρ_ss| < 1e-12` (sprint-11 gate, sprint-12 verified).
  6. **Sub-step probe** —
     `python validation/diagnostics/diag_phase0_pointC_substep.py`.
     **Target: Σρ_ss(Phase-0 exit) < 0.05; ρ_ss saturation at the
     resonance < 0.1.**
  7. **Sprint-11 fallback Point-C probe** —
     `python validation/diagnostics/diag_phase0_pointC_fallback.py`.
     **Target: same as (6) at full Phase-B precision (n_B=10000).**
  8. **5 MeV V_nunu projection** —
     `python validation/diagnostics/diag_vnunu_active_only_fallback.py`.
     Point C with projection ΔNeff within 0.005 of sprint-5
     baseline ΔNeff = −0.012.
  9. **Hannestad gate** —
     `python validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`.
     **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10].**
     Yp ∈ [0.24, 0.26] for all three.
  10. If gate 9 hits: flip `qke_phase0_flag` AND
      `qke_expm_fallback_near_degeneracy` defaults to True; update
      `test_sterile_dw_production` with config overrides forcing
      both False for bit-identity. Re-run gates 1-5.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT13_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete
> sprint-13 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-12 sub-step instrumentation or the opt-in
> H-iteration fix; the sprint-11 eigendecomposition fallback or
> H-coupling gate; the sprint-10 Phase-0 driver or instrumentation;
> the sprint-9 MSW instrumentation; the sprint-8 energy
> instrumentation or 2-level energy regression guard; the sprint-7
> `_F_stat_stable` clamp; the sprint-6 NaN-safe sanitisation; the
> sprint-5 V_nunu projection; or the D.7.1 Strang-symmetric
> composition. The primary suspect is the active-sterile collision
> RHS structure: `_assemble_collision_N` produces −D·ρ for active-
> sterile pairs which `_build_L_list` then cancels via add-back,
> leaving a residual that drives V_nunu lock-in at the istep
> 906→1035 window of Phase-0 Point C.

## Commit chain for context

  * `<sprint-12 hash>` — **Sprint 12** (sub-step instrumentation;
    Suspect 5 partially localised; H-iteration fix landed but
    insufficient; V_nunu lock-in identified). Sprint 13 builds
    directly on top.
  * `<sprint-11 hash>` — Sprint 11 (eigendecomp fallback; Phase-B
    over-amplification closed).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver; Point-C narrow-mixing
    regression introduced).
  * `ac1d521` — Sprint 9 (MSW diagnostic; Suspect 2 falsified at
    Point A; Suspect 3 promoted).
  * `bb0ea32` — Sprint 8 (Suspect 1 falsified).
  * `0c28d8d` — Sprint 7 (cold-T NaN origin fix).
  * `968c936` — Sprint 6 (NaN-safe sanitisation).
  * `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
    dependency** — projection exposed the MSW anomaly.
  * `ca589b0` — Sprint 2 (Phase B stabilisation; n_B auto-scale).
  * `e2f41f6` — D.7.1 (Strang-symmetric sequence).
  * `4d8ab2c` — D.7 (per-mode expm via Al-Mohy augmented matrix).

## Post-refactor: downstream opportunities

If sprint 13 closes Suspect 6 and lands the default flip:

  1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
     (|U_μ4|² = 1e-4, Δm² = 1.29). ΔNeff ∈ [0.05, 0.2].
  2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
     resonance-driven physics at non-zero lepton asymmetry.
  3. **Representation-factor audit** — parked from sprint 4.
  4. **Performance** — eigendecomposition fallback cost (~1.4× wall
     on Point-C n_B=10000). Cache "stable" mode classes.
  5. **Unified telemetry facility** folding sprint-8/9/10/11/12
     per-step instrumentation into one reusable surface.
