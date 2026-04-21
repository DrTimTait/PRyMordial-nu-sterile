# Stage E.2 sprint 6 brief: extended-window numerical stability

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 5 (commit `a2a975c`). Read this first. By the end you
should know why PRyMordial crashes when the V_nunu active-only
projection is turned on together with T_boltz_start ≳ 20 MeV, what
to land to make it stable, and why flipping the projection default
to `True` is conditional on this sprint's success.

## One-paragraph orientation

Sprint 5 identified the dominant structural bug in PRyMordial's
small-mixing Dodelson-Widrow prediction: V_nunu was being applied as
a full 4×4 `(ρ − ρ̄)` matrix including active-sterile entries that
shouldn't exist (sterile has no NC charge), and these spurious
entries were bootstrapping active-sterile coherence past the tiny
vacuum mixing. The fix landed behind the opt-in flag
`PRyMini.qke_v_nunu_active_only` (default `False`, preserves all
regression tests bit-identically). At the default 5 MeV Phase B
window, the fix drops Hannestad Point C from ΔNeff=+0.858 to
−0.012 vs the Hannestad target of 0.040 — a ~70× improvement in
small-mixing agreement. Hannestad A and B are under their targets
(A=0.17 vs 1.0, B=0.13 vs 0.5) because the default 5 MeV window
starts *below* the MSW resonance at T_MSW ≈ 10 MeV for
Δm²=0.93 eV², missing adiabatic passage. Extending the window to
30-60 MeV would recover A and B — but the V_nunu projection
combined with an extended window triggers a numerical instability
in `evolve_step_ode_etdrk2`: Point A crashes with NaN from
`scipy.linalg.expm`, and Point C at 30 MeV completes with
unphysical values (Neff=7.22, Yp=0.29). Sprint 6's job: find the
stability fix, confirm the full Hannestad A/B/C suite lands within
10-20% of literature, flip the projection default to `True`.

## What to read, in order

Budget ~45 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 5". The full
   sprint 5 landing table is there, including the extended-window
   failure matrix. Also review sprint 4 (damping-magnitude ruled out)
   and sprint 2 (n_B auto-scale + QED-table zero-clamp that made
   extended windows numerically feasible in the first place).
3. `git show a2a975c` — the sprint 5 landing commit. Read the
   commit message end to end; it's the densest summary of the V_nunu
   physics and the stability failure matrix.
4. **`validation/diagnostics/diag_hannestad_proj_w20.out`** — the
   NaN crash log at T_boltz_start=20 MeV, Point A.
   **`validation/diagnostics/diag_pointC_proj_w30.out`** — Point C
   completes at 30 MeV but produces Neff=7.22, Yp=0.29, D/H=3.89.
   **`validation/diagnostics/diag_hannestad_proj_w15.out`** — the
   largest window that runs cleanly. Gives A=0.34, B=0.16, C=0.01.
5. `PRyM/PRyM_boltzmann.py` lines **4596-4760**: the D.7.1 driver
   `evolve_step_ode_etdrk2`. Particularly **line 4750** — the
   off-diagonal magnitude clamp (`ab_mag > max_mag`) that becomes
   a silent no-op when `ab_mag` is already NaN.
6. `PRyM/PRyM_boltzmann.py` lines **4550-4595**: `_etdrk2_expm_phi`,
   the Al-Mohy-Higham augmented-matrix exponential. The crash is
   *inside* `scipy.linalg.expm` called here. The Al-Mohy
   construction embeds `L*dt` in a 3N²×3N² matrix; at very large
   dynamic range between eigenvalues, the norm estimation in
   scipy's `_expm` internals hits NaN.
7. `PRyM/PRyM_boltzmann.py` lines **4300-4369**: `_build_H_list`.
   The sprint-5 projection is at lines 4337-4340 (zeroing sterile
   row/col of V_nunu_eV when `qke_v_nunu_active_only=True`).
   The projection is **physics-correct**; don't revert it — the
   instability is a numerical consequence, not a symptom of wrong
   physics.
8. **`References/external/fortepiano_public/sources/equations.f90`**
   lines ~598 onwards — FortEPiaNO's evolution driver. They use
   DLSODA (stiff ODE from ODEPACK) on the full system, no
   operator splitting, no per-mode expm. If the right sprint-6
   answer turns out to be "switch to a different solver path", this
   is the reference.

## What's probably broken (three suspects, ordered by likelihood)

### Suspect 1 — off-diagonal magnitude clamp not NaN-safe (HIGH)

At `PRyM_boltzmann.py:4750` the clamp compares `ab_mag > max_mag`.
If `ab_mag` already contains NaN from a prior step's numerical
instability, the comparison produces NaN (treated as False), the
clamp silently does nothing, and NaN propagates to the next
`_etdrk2_expm_phi` call which crashes inside scipy's `expm`.

**Diagnostic**: insert `assert not np.any(np.isnan(...))` at the
start of `evolve_step_ode_etdrk2` to catch the first NaN. Or simpler:
instrument to print the step index and T at which NaN first appears
at Point A with projection + 20 MeV window. This tells you whether
NaN enters from the off-diag clamp loop or from the ETDRK2 corrector.

**Fix candidate**: add a NaN-sanitising pass before the clamp —
replace `np.isnan(ab_mag)` entries with `max_mag` or zero — so the
clamp always operates on finite values. This is a small edit and
preserves the clamp's intended semantics.

**Likelihood**: **high**. This is the most direct interpretation of
the warnings observed at sprint 5 (`RuntimeWarning: invalid value
encountered in greater`). A couple of lines of code, no physics.

### Suspect 2 — Al-Mohy augmented expm loses precision at extreme dynamic range (MEDIUM)

With the V_nunu projection on, L's eigenvalues at high T span from
~1e-7 eV (residual vacuum mixing) to ~1e-4 eV (thermal damping) —
a dynamic range of 10³–10⁴ on eigenvalue magnitudes. After
embedding in the 3N²×3N² augmented block at
`_etdrk2_expm_phi:4582-4594`, the matrix norm estimate inside
scipy's `_expm` can hit NaN at `_expm:666` (the
`np.log2(eta_5 / theta_13)` step that crashed in sprint 5).

**Diagnostic**: swap `_etdrk2_expm_phi` for a direct
`expm(L*dt) = Φ₀` + finite-difference Φ₁, Φ₂ (`phi_1(z) ≈
(e^z − 1)/z`, `phi_2(z) ≈ (φ_1(z) − 1)/z`) at high condition
number. Compare against baseline at the default 5 MeV window
(should be bit-identical there since eigenvalue spread is smaller).

**Fix candidate**: guard the augmented-matrix construction with a
condition-number check; fall back to the eigen-decomposition path
(or direct expm + finite-difference φ's) when the norm estimate
would fail. Stage D.7.1 was validated against the 2-level damped
Rabi test — that test should still pass after the guard.

**Likelihood**: **medium**. Sprint 2's `diag_2level_high_T.py`
showed Φ₀ agreement at `||L·dt||_inf = 2.65e4` to 3.4e-21 relative,
ruling out this mechanism in the saturation regime. But with
projection on, the eigenvalue *spread* is different (not just the
largest magnitude), which is what scipy's norm estimator reacts to.

### Suspect 3 — Strang-split step size at high T under-resolves oscillation (LOW)

The half-diagonal exp-Euler steps might be using a step regulariser
(`phi_1`-based) that, when combined with very fast off-diag
oscillations, leaves a residual that the next half-step doesn't
cancel — accumulating to instability. Sprint 2 made n_B auto-scale
per window, but the heuristic `n_B = 2000·(decades/3)⁴` might be
undersized for projection + 20 MeV onwards.

**Diagnostic**: at T_boltz_start=20 MeV, projection on, manually
override `n_B_override` to {2·default, 4·default, 8·default} on
Point A. If the crash survives all three, Suspect 3 is ruled out
and the bug is in Suspect 1 or 2.

**Likelihood**: **low**. The `n_B` auto-scale was validated by
sprint 2's clean 5/30/60 MeV scan (baseline, no projection). If it
worked there, step size shouldn't be the differentiator.

## Stage inheritance: what NOT to touch

- `DensityMatrixSolver._compute_D_pair_matrix` (Stage E.1).
- `PRyMini.qke_damping_formula = "mirizzi"` default.
- `_apply_unitary` (Stage D.3 ν-bar convention).
- `_build_H_list` physics (sprint 5 projection is correct, just
  numerically fragile under extension — don't revert).
- `_etdrk2_expm_phi` mathematical form (Al-Mohy is correct; any
  reformulation needs the 2-level Rabi test to still pass exactly).
- `evolve_step_ode_etdrk2` sequence (half-diag + ETDRK2 corrector
  + half-diag — Stage D.7.1 baseline). A NaN-safe clamp is an
  additive fix that doesn't change the sequence.
- Default value of `qke_v_nunu_active_only` (still `False` when
  this sprint starts; the flip to `True` is the sprint's *output*,
  not its input).
- Any other flag default. If you add new flags (e.g. a
  condition-number threshold), default them to current behaviour.

## Scope options

- **(a) NaN-safe clamp only** (Suspect 1 fix): add a sanitisation
  pass so the off-diagonal clamp operates on finite values. Re-run
  diag_hannestad_proj_w20.py / _w30.py. If crashes disappear and
  A/B/C approach Hannestad targets, commit + flip the default.
  If crashes disappear but A is still far from 1.0, escalate to (b).
  ~3 hours total including diagnostic reruns.
- **(b) NaN-safe clamp + expm path audit** (Suspects 1+2): land
  (a), then instrument `_etdrk2_expm_phi` to detect the
  norm-estimate NaN path, add a guard that falls back to direct
  `expm` + finite-difference φ's when the augmented matrix would
  fail. Re-run extended-window suite. ~6 hours.
- **(c) Full DLSODA path** (radical): write a parallel driver that
  calls `scipy.integrate.solve_ivp` with `method='BDF'` or
  `'LSODA'` on the full vectorised (ρ, Tg, a) state, matching
  FortEPiaNO's approach. Compare against the Strang-split driver
  in the default 5 MeV regime for correctness, then use it
  specifically for extended windows. ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix)

Same ladder as sprints 2 / 4 / 5:

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must pass bit-identical at default config
   (`qke_v_nunu_active_only=False`).
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 must pass at default config.
3. **2-level damped Rabi** —
   `python validation/diagnostics/diag_2level_damped.py`. Must
   still give ratio 1.000 between PRyMordial 4x4 and analytic
   damped 2-level expm.
4. **Sprint-5 regressions** — re-run
   `validation/diagnostics/diag_vnunu_active_only.py` (projection
   at 5 MeV: expect Point C ΔNeff ≈ −0.012, bit-identical to
   sprint-5 numbers).
5. **Extended-window test** — re-run
   `validation/diagnostics/diag_hannestad_proj_w20.py`,
   `diag_hannestad_proj_w30.py`. No crashes. Point C Yp must be in
   [0.24, 0.26] (no unphysical values).
6. **Hannestad literature at extended window** — run
   `validation/diagnostics/diag_hannestad_proj_w30.py` (or create a
   w60 variant). **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7],
   C ∈ [0.02, 0.1]**. Hitting all three closes Stage E.2.
7. If targets hit: flip `qke_v_nunu_active_only` default to `True`
   in `PRyM_init.py`. Re-run 1–5 and verify `sterile_DW_literature.py`
   now gives Hannestad agreement. Update
   `validation/sterile_DW_literature.out.txt` accordingly.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT6_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-6 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-5 V_nunu projection physics or the D.7.1
> driver's Strang-symmetric sequence; the bug is numerical
> robustness, not physics scope.

## Commit chain for context

- `a2a975c` — **Sprint 5** (V_nunu projection opt-in). Sprint 6 is
  built directly on top of this — the starting state.
- `66ba8ec` — Sprint 4 (damping magnitude ruled out via FortEPiaNO
  survey; introduced diagnostic scale knobs).
- `f403442` — Sprint 3 (y-grid discretisation falsified).
- `ca589b0` — Sprint 2 (Phase B stabilisation: auto-scaled n_B,
  QED-table zero-clamp). **Critical dependency** — enables the
  extended-window runs sprint 6 needs to validate.

## Post-fix: downstream opportunities

If sprint 6 closes the extended-window stability problem and lands
the default flip:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). Target: ΔNeff ∈ [0.05, 0.2] per
   Gariazzo+2019. With the V_nunu fix + extended window, this
   should now agree with FortEPiaNO.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   different physics focus; sprint-5 fix should carry over.
3. **Representation-factor audit** — still parked from sprint 4.
   Check whether `_build_L_list` implicitly applies damping to
   `ρ/f_eq` anywhere. May be moot after sprint 6.
4. **Phase A thermal approximation audit** — still parked. Whether
   assuming exact thermal equilibrium at T_boltz_start biases the
   subsequent DW production. Likely negligible vs the V_nunu effect,
   but worth closing out.
5. **asymmetric-gain-at-ρ_ss=0 audit** — still parked. Low priority
   after sprint 5 identified the real driver.
