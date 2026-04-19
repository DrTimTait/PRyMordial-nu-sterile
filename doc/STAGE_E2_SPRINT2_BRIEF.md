# Stage E.2 sprint 2 brief: Phase B numerics at extended windows

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 1 (commit `5d56a43` — the validator commit; sprint 1's
diagnostics landed at `3892c22`). Read this first. By the end you
should know why the QKE window-extension result is contaminated, what
to fix to get a clean signal, and what NOT to touch.

## One-paragraph orientation

Sprint 1 tested two of three E.2 structural hypotheses. Hypothesis A
(active-sterile gain asymmetry) was **falsified**: PMNS-on variants of
`_offdiag_collision_gain` move ΔNeff by ~2% at Hannestad Point C.
Hypothesis B (QKE evolution-window mismatch) is **unresolved but
suggestive**: pushing `T_boltz_start` from 5 MeV to 30 MeV drops ΔNeff
from 0.845 to **0.0496** — within 25% of Hannestad's 0.04 — but
`sum ρ_ss = 26.17` at that run is internally inconsistent with
ΔNeff=0.05, and the 60 MeV run gives Neff=13.66 (unphysical for 3+1).
A Phase-A-only control (extend `T_start` to 60 MeV, keep
`T_boltz_start = 5 MeV`) gives ΔNeff=0.8514, matching the
default-window baseline — confirming the dramatic shifts come from
extending Phase B, not Phase A. Sprint 2's job: figure out why Phase B
breaks at T ≫ 5 MeV, stabilise it, and re-run the window scan to see
whether hypothesis B is real.

## What to read, in order

Budget ~45 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 (partially landed".
   The sprint-1 summary table and the three hypothesis outcomes are
   there. Note the "Open: hypothesis C" line — C is still untested.
3. **`doc/STAGE_E2_BRIEF.md`** — parent E.2 brief. Read the
   hypothesis-B section (diagnostic sketch, likelihood, target). The
   rest of the brief is superseded by sprint-1 results.
4. **`doc/FLAG_AUDIT.md`** — awareness only. Sprint 1 landed a
   minimal `validate_configuration()`; the audit doc lists larger
   follow-ups (auto-scale `n_B` with `T_boltz_start` is directly
   relevant here).
5. **`validation/diagnostics/diag_qke_window.py` + `.out`** — the
   window-scan script and its raw results. The 30 MeV and 60 MeV
   anomalies live here.
6. **`validation/diagnostics/diag_phaseA_only.py` + `.out`** — the
   sanity control proving Phase A extension is a no-op.
7. `PRyM/PRyM_main.py` lines **225–345** — the Phase B setup
   (`T_boltz_start` initialisation, `n_B = max(2000, 2·n_sampling)`,
   log-uniform `a_grid`, Froustey entropy equation). **This is the
   prime suspect.**
8. `PRyM/PRyM_boltzmann.py` — `evolve_step_ode_etdrk2`
   (line 4596 onwards) and `_build_L_list` (line 4483). Cheap read;
   you don't need to re-derive D.7.1, just understand the step
   structure.
9. **`References/1204.5861_hannestad2012.pdf`** (if present) —
   Section 3 describes their integration from 60 MeV to 1 MeV. Their
   step-size policy is worth comparing.

## What's probably broken (three suspects, ordered by likelihood)

### Suspect 1 — `n_B` is T-range-blind (HIGH)

`n_B = max(2000, 2·n_sampling)` is fixed regardless of the range
`T_boltz_start → T_boltz_end`. At the default 5 → 0.005 MeV the
2000-step log-uniform-in-`a` grid has ~660 steps per decade. Pushing
`T_boltz_start` to 60 MeV stretches the range over 4.08 decades in T
but keeps `n_B` fixed — so steps at high T are ~4× coarser. The
collision rate scales as T^5 (ν-ν) or T^4·E (ν-e damping), so at
T=60 MeV it is ~(60/5)^5 = **7776× larger** than at 5 MeV. Step
size dt and damping rate D multiply as D·dt, and once D·dt ≫ 1 the
ETDRK2 per-mode expm is faithful but the linearised Neff / f
integration is not.

**Diagnostic**: n_B convergence scan at fixed T_boltz_start = 30 MeV.
`n_B ∈ {2000, 5000, 10000, 20000}`. Record ΔNeff, sum ρ_ss, and the
ratio `sum ρ_ss / (ΔNeff * something-thermal)`. If ΔNeff converges
monotonically and sum ρ_ss becomes consistent, this is the culprit.

### Suspect 2 — Phase B's Froustey entropy equation is out of range (MEDIUM)

Phase B evolves `Tg` via the plasma entropy equation
`d(spl·a³)/dt = -(Q/Tg)·a³` (PRyM_main.py:~305–315). The tabulated
`PRyMthermo.spl(Tg)` is the plasma entropy per unit a^-3, which
includes the relativistic photon + e+/e− + corrections. At T=60 MeV,
e+/e− are fully relativistic (me/T ~ 0.009), which the tables should
handle — but the QED corrections in `P_QED`, `dP/dT`, `d²P/dT²` may
not have been tabulated above some cutoff. If `spl(60 MeV)` returns
extrapolated garbage, the initial `a_boltz_ini = (spl_ref/spl(Tg))^(1/3)`
is wrong, and everything downstream inherits that.

**Diagnostic**: print `PRyMthermo.spl(T)` and `PRyMthermo.rho_e(T)`
at T ∈ {5, 10, 30, 60, 100} MeV. Compare to analytic expectations
(`rho_e ∝ T^4` at `T ≫ m_e`). A clean way: query the underlying
tables/interpolators for their T-range.

### Suspect 3 — ETDRK2 expm regime at D·dt ≫ 1 (LOW)

`_etdrk2_expm_phi` (PRyM_boltzmann.py:~4550) computes
`Phi0 = e^{L·dt}`, `Phi1 = dt·phi_1(L·dt)`, `Phi2 = dt·phi_2(L·dt)`
via an augmented 3N²×3N² matrix exponential. When `|L·dt|` is
modest the Higham-Al-Mohy formula is exact; at extreme `|L·dt|` the
`e^{L·dt}` block goes to zero correctly but `phi_1`, `phi_2` saturate,
and the N multiplier (which is the collision RHS) gets scaled by an
O(1/|L|) factor. That's the mathematically correct ETDRK2 behaviour,
so suspect 3 is low. But worth confirming once suspect 1 is ruled out.

**Diagnostic**: verify the 2-level damped Rabi test
(`diag_2level_damped.py`) still matches analytic at an extreme
high-T single step (manually force T=60 MeV, dt from the 2000-step
grid). If it still matches, ETDRK2 is innocent.

## Stage inheritance: what NOT to touch

- `DensityMatrixSolver._compute_D_pair_matrix` (Stage E.1 helper).
- `PRyMini.qke_damping_formula = "mirizzi"` default.
- `C_D` and `C_A` values.
- `_build_H_list` (V_NC and V_thermal are physics-complete).
- `_apply_unitary` (Stage D.3 ν-bar convention).
- `_etdrk2_expm_phi` body (Stage D.7 Al-Mohy augmented expm).
- `evolve_step_ode_etdrk2` body (Stage D.7.1 Strang-symmetric split).
- `PRyM_init.validate_configuration` scope (sprint 1's validator; if
  you want auto-scale `n_B`, add it to that validator rather than
  sprinkling T-range logic through Phase B).
- Default value of any flag (if you introduce a new one, default it
  to the current behaviour).

## Scope options

- **(a) n_B convergence scan, one window (30 MeV)**: add an
  `n_B_override` sweep on top of `diag_qke_window.py`'s 30 MeV config.
  4 BBN solves (~60 min). If ΔNeff converges and sum ρ_ss becomes
  consistent, you have your answer. ~3 h total including the analysis.
- **(b) Full stabilisation sprint**: (a) + query `spl` / `rho_e`
  interpolant ranges (cheap, 1-hour diagnostic), + a 2-level expm
  sanity check at high T. If any of the three suspects fires, land
  a fix (auto-scale `n_B` in the validator, or extend the
  interpolation tables, or add a step-size guard). ~6 h.
- **(c) Phase B re-architecture for high-T robustness**: bypass
  the plasma entropy equation at T ≫ m_e by falling back to the
  analytic thermal Tg and using Phase B only for the density-matrix
  evolution. Large scope, large diff. Only if (a) and (b) fail.
  ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix)

Same ladder as E.1 / sprint-1:

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must still pass. The validator runs in all of them now.
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 must pass.
3. **2-level damped Rabi** — `python validation/diagnostics/diag_2level_damped.py`.
   Still exact to 4 sig figs. If your fix touches the step-size
   policy, verify it at both the default and any new extreme.
4. **Window scan re-run** — re-run `diag_qke_window.py` with
   whatever stabilisation landed. Target: internally consistent
   ΔNeff vs sum ρ_ss across 5 / 30 / 60 MeV, and monotone ΔNeff
   decrease as the window widens (if hypothesis B is real).
5. **Hannestad literature** — `validation/sterile_DW_literature.py`
   at the new window if default was changed. If the default was NOT
   changed, this target is informational only.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT2_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-2 plan. My target for this session is {one of a / b / c}.
> Do not touch `_compute_D_pair_matrix` or the Stage D.7.1 driver
> body; stage-1 sprint landed them and they are the committed
> baseline.

## Commit chain for context

- `5d56a43` — **Flag audit + validator** (sprint 1 closeout).
- `3892c22` — **Stage E.2 sprint 1**: hypothesis A falsified, B
  unresolved. Three diagnostic scripts under `validation/diagnostics/`.
- `baeec1c` — doc: Add STAGE_E2_BRIEF.md (parent brief).
- `18b1fa7` — Stage E.1: Mirizzi pair-specific damping (default).

## Post-fix: downstream opportunities

If sprint 2 closes the hypothesis-B question (either confirms a
real physical effect and lands a window-default change, or
definitively falsifies):

1. **Hypothesis C (y-grid discretisation)** — the third E.2
   candidate. Untouched so far; parent E.2 brief has the diagnostic
   sketch. Cheap to rule in/out.
2. **Gariazzo damping form** — still stubbed with
   `NotImplementedError` behind `qke_damping_formula = "gariazzo"`.
   Worth implementing once the structural DW picture stabilises.
3. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   different physics focus; a clean window-extension fix should
   also help there.
4. **ROADMAP update** with the E.2 sprint-2 landing and any new
   default changes.
