# Stage E.2 sprint 10 brief: Suspect-3 Phase-A thermal-IC audit

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 9 (commit **`<sprint-9 hash>`**; will be set by the
sprint-9 landing). Read this first. By the end you should know why
the Point A dNeff anomaly likely survives every fix applied inside
`evolve_step_ode_etdrk2` — it lives in the Phase-A → Phase-B
hand-off, not the driver — and the path to closing Stage E.2 with a
Phase-0 QKE driver that supplies a history-preserving IC to Phase B
at T = 30 MeV.

## One-paragraph orientation

Sprint 8 falsified Suspect 1 (active-sterile energy-balance driver
bug) at machine precision. Sprint 9 built a per-step per-y-mode MSW-
passage diagnostic and falsified Suspect 2 (ETDRK2 eigenbasis over-
pumping) at Point A — but with an unexpected, crisp signal. Rather
than high-y over-deposition (which is what an eigenbasis regulariser
bias would produce), Point A shows **low-y over-thermalisation**
(+8.1% vs. FD at y ≲ 20) and **mild high-y under-thermalisation**
(−1.1% at y ≳ 80). The per-y resonance table explains it: low-y
modes have their MSW resonance at `T_res ≳ 30 MeV` (step_res ∈ {0,
1, 2} of Phase B), meaning they arrive at Phase B already past the
crossing. Phase B's IC sets ρ_ss=0 regardless, so the driver,
under large vacuum mixing (sin²2θ=0.1 → θ_vacuum≈9°), rapidly over-
drives them toward a new local equilibrium over the 10000-step run.
High-y modes cross resonance during Phase B at step-indexed
`γ_step` ≪ 1 — Landau-Zener-non-adiabatic in step units — and
under-convert. The driver is not the problem: both anomalies are
physics-level consequences of starting Phase B at T = 30 MeV without
supplying the pre-resonance adiabatic history. **Suspect 3 is
promoted to prime.** Sprint 10's job: build a Phase-0 QKE driver
that evolves the 4×4 system from ~100 MeV down to 30 MeV with the
same machinery, produces a history-preserving `ρ_all(T=30 MeV)`,
and closes the Point A/B/C anomaly.

## What to read, in order

Budget ~60 min before writing any code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 9". The
   sprint-9 landing record has the full per-y resonance table, the
   bucketed Δρ_ss and FD-residual summaries, and the ranked next-
   candidate list with Suspect 3 promoted. Also skim sprint 8 (the
   trace-test falsification of Suspect 1), sprint 7 (n_B auto-scale
   + Phase-5 n_B=10000 baseline), and sprint 5 (V_nunu projection).
3. **`git show <sprint-9 hash>`** — the sprint-9 landing commit.
   Covers the MSW instrumentation, the diagnostic harness, the
   ROADMAP falsification of Suspect 2, and this brief.
4. **`validation/diagnostics/diag_msw_passage.out`** — per-y
   resonance-localisation tables, bucketed Δρ_ss, end-state
   FD-residual, verdict line. Read the sector-0 and sector-1 tables
   side-by-side: sector-1 anti-neutrino crossings happen *during*
   Phase B even at low y (`step_res` ≈ 1138 at y=0.5), which is
   a second clue that the Phase-A IC pays asymmetric attention to
   sectors.
5. **`PRyM/PRyM_boltzmann.py` lines 2940-3068** — `DensityMatrixSolver`
   constructor. Where the Phase-B initial condition `ρ_all` is set
   up. Currently: `ρ_αα(y) = f_FD(y; T=T_boltz_start)` for active
   flavors, `ρ_ss(y) = 0`, off-diagonals = 0. Sprint 10's Phase-0
   driver output replaces this IC.
6. **`PRyM/PRyM_main.py` Phase A/B handoff** — search for
   `T_boltz_start` and the `dm_solver` construction. Where the
   Phase-B entry point lives; Phase 0 wires in before it.
7. **`PRyM/PRyM_boltzmann.py` lines 4321-4390** — `_build_H_list`.
   Phase 0 reuses this for the Hamiltonian; no modification expected.
8. **`PRyM/PRyM_boltzmann.py` lines 4533-4598** — `_build_L_list`.
   Reused by Phase 0; do NOT touch.
9. **Sabti et al. BBN reference (`Sabti-BBN.pdf`)** — Section on
   how they handle the HNL-decay initial distribution at their
   equivalent of our Phase-A boundary. Their construction is the
   template for a history-preserving IC under general vacuum mixing.
10. **`validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`** —
    gate-6 harness (resurrected by sprint 8, untouched by sprint 9).
    Use this for the sprint-10 fix validation.

## What's probably broken (one suspect, one follow-up)

### Suspect 3 — Phase-A thermal-IC inadequacy (PRIME)

At Point A, the MSW resonance for y ≲ 20 crosses at `T_res` in
{~29.9 MeV, ~30 MeV}. Phase B starts at `T_boltz_start = 30 MeV`.
Modes with `T_res ≥ 30 MeV` have already passed the crossing in
the physical universe by the time we begin numerical evolution.
PRyMordial's Phase-B IC throws that away: ρ_ss=0 everywhere, and
active diagonals set to `f_FD(y; T=30 MeV)`. The driver then drives
those modes toward a new equilibrium under large-mixing dynamics,
overshooting.

**Primary fix.** Build a Phase-0 driver that integrates the 4×4 QKE
from `T_phase0_start ≈ 100 MeV` down to `T_boltz_start = 30 MeV`
with adiabatic-vacuum IC (ρ_αα = f_FD at 100 MeV, ρ_ss=0, off-
diagonals=0 — this IC is correct at T=100 MeV where the medium
suppression is strong enough that θ_m ≈ 0 for all relevant y).
Phase 0 produces `ρ_all(T=30 MeV)` which replaces the current
Phase-B IC. Reuses `evolve_step_ode_etdrk2`, `_build_L_list`,
`_build_H_list`, `_assemble_collision_N` with no modification.

**Scope of integration.** n_B for Phase 0 needs enough resolution to
capture the T ≈ 30 MeV MSW crossings that sprint 9 saw at low y.
Start with n_B_phase0 = 2000–3000 across 100 → 30 MeV (a factor-3.33
scale-factor increase); this gives ~10× the sprint-9 step density at
the crossings that matter. Use the existing sprint-2 auto-scale
logic with a phase-0 branch (or a dedicated override).

**Expected effect.**
- Low-y bucket residual at Point A should collapse from +8% toward
  zero (the pre-resonance history is now present in the IC).
- High-y bucket may also improve: Phase-0 collision integrals will
  thermalise the ν-ν coherence into a less-special IC that couples
  more cleanly to the Phase-B resonance passage.
- Point B (sin²2θ=2.26e-3) under-production likely closes by the
  same mechanism at reduced magnitude.
- Point C (sin²2θ=1e-4) already in-band; change should be
  bit-identical or near (small mixing = small Phase-0 effect).

**Likelihood**: **high**. The sprint-9 diagnostic is a direct
measurement of `step_res = {0, 1, 2}` for low-y modes, which is the
textbook signature of Phase-A IC inadequacy.

**Validation gates.** Rerun `diag_msw_passage.py` with Phase 0 on:
bucket-0 residual should drop below 2%. Rerun gate-6
`diag_hannestad_proj_w30_nB10k.py`: Points A/B/C all in Hannestad
band closes Stage E.2. Fast regression (4/4) and sterile regression
(3/3) must stay bit-identical at default (Phase 0 off).

### Suspect 2-residual — non-adiabatic high-y tunneling (FOLLOW-UP)

If Phase 0 closes Point A's low-y over-thermalisation but leaves a
residual high-y under-thermalisation, the mechanism is step-indexed
Landau-Zener non-adiabaticity: sprint-9 measured `γ_step` = O(10⁻⁸
– 10⁻⁶) at the high-y crossings, far below 1. A fix option:
adaptive step controller near resonance, or a refined
n_B localised to the `T ≈ 1.5 MeV` crossing band.

Only investigate if Suspect 3 leaves high-y residuals > 1%.

## Stage inheritance: what NOT to touch

Sprint-9 additions to the "do not touch" list (existing items from
sprint-9 brief retained):

- **Sprint-9 MSW instrumentation** at `PRyM_boltzmann.py:4702-4764`
  (`_msw_snapshot` helper), `:3067-3068` (`_msw_hist`/`_msw_step_idx`
  init), `:4815-4818`/`:4941-4942` (hook points). Flag-gated off at
  default; do not remove or inline.
- **`PRyMini.qke_msw_diag_flag` / `qke_msw_diag_path` /
  `qke_msw_diag_pair_idx`** defaults (all opt-in; stay at the
  sprint-9 defaults).

From sprint 9's inherited list (all retained):

- Sprint-8 `_energy_snapshot` + `qke_energy_diag_flag` instrumentation
  and 2-level L-conservation regression guard
  (`diag_2level_damped_energy.py`).
- `PRyMclass._boltz_dm_solver` expose at `PRyM_main.py:530-534`.
- `_F_stat_stable` upper clamp at `PRyM_boltzmann.py:1048` (sprint 7).
- NaN-safe sanitisation at `PRyM_boltzmann.py:4831` (isfinite check)
  and `:4857` (nan_to_num before clip) (sprint 6).
- V_nunu active-only projection at `PRyM_boltzmann.py:4351-4354`
  (sprint 5).
- `_build_L_list`, `_build_H_list`, `_build_PMNS` internals — Phase 0
  is a new *caller* of these, not an editor.
- `_etdrk2_expm_phi` — sprint 9 falsified the high-y bias hypothesis;
  Phase 0 reuses it unchanged.
- `PRyMini.qke_damping_formula = "mirizzi"` default.
- `_apply_unitary` (Stage D.3).
- D.7.1 Strang-symmetric sequence in `evolve_step_ode_etdrk2` — reused
  by Phase 0 unchanged.
- Default of `qke_v_nunu_active_only` (still `False`).
- Any other flag default.

## Scope options

- **(a) Phase-0 driver + Point A diagnostic re-run**: build the
  Phase-0 driver (new `_phase0_integrate` method on `PRyMclass` or
  on `DensityMatrixSolver`, gated on a new `qke_phase0_flag`, default
  False), expose its output as the Phase-B IC when on. Re-run
  `diag_msw_passage.py` to measure the effect on the low-y bucket
  residual. No gate-6 run, no default flip. ~6-10 hours.
- **(b) (a) + gate-6 re-run + conditional default flip**: on top of
  (a), re-run `diag_hannestad_proj_w30_nB10k.py`. If Points A/B/C
  all land in Hannestad band, flip `qke_phase0_flag` default to
  True and re-run gates 1–6. Close Stage E.2. ~14-20 hours.
- **(c) (a) falsifies Suspect 3**: escalate to Suspect 2-residual
  (sprint-10 follow-up) or Suspect 4 (to-be-identified). ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix, for scope (b))

Same ladder as sprint 9 with the new Phase-0 regression guard at
gate 5:

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must pass bit-identical at default (Phase 0 off).
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 must pass at default config (Phase 0 off).
3. **2-level damped Rabi** —
   `python validation/diagnostics/diag_2level_damped.py`. Ratio 1.000.
4. **2-level L conservation** —
   `python validation/diagnostics/diag_2level_damped_energy.py`. Both
   ratios within 1e-8.
5. **Phase-0 regression guard** (sprint-10 addition): build a unit
   test that exercises the Phase-0 driver on a decoupled toy case
   (theta_14 = theta_24 = theta_34 = 0, sterile_flag=True) and
   verifies that `ρ_ss(T=30 MeV)` stays at zero to < 1e-12 across
   1000 Phase-0 steps. Guards against leakage through Phase 0 when
   there's nothing to mix.
6. **Sprint-5 5 MeV baseline** —
   `python validation/diagnostics/diag_vnunu_active_only.py`. Point C
   ΔNeff ≈ −0.012 bit-identical (Phase 0 off default path).
7. **Sprint-6 w20 clean completion** —
   `python validation/diagnostics/diag_hannestad_proj_w20.py`. No
   crash; all Yp ∈ [0.24, 0.26].
8. **Sprint-9 MSW diagnostic with Phase 0 on** —
   `python validation/diagnostics/diag_msw_passage.py` (add a
   `PRyMini.qke_phase0_flag = True` to its `_base_flags`).
   Target: bucket-0 residual drops from +8.1% toward ≤ 2%;
   bucket-4 residual stays near 0 (−1% allowable).
9. **Hannestad literature at w30 n_B=10000 with Phase 0 on** —
   `python validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`
   (add Phase-0 enable to its `_base_flags`). **Target: A ∈ [0.9,
   1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.1]**. Yp for all three ∈
   [0.24, 0.26]. All three in-band closes Stage E.2.
10. If gate 9 hits: flip `qke_phase0_flag` default to `True` in
    `PRyM_init.py`. Re-run gates 1–6. Sterile regression (gate 2)
    will shift — `test_sterile_dw_production` will need either a
    config override or a tolerance update per precedent from sprint 8.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT10_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-10 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-9 MSW instrumentation, the sprint-8 energy
> instrumentation or 2-level energy regression guard, the sprint-7
> `_F_stat_stable` clamp, the sprint-6 NaN-safe sanitisation, the
> sprint-5 V_nunu projection, or the D.7.1 Strang-symmetric
> composition. The primary suspect is Phase-A thermal-IC inadequacy;
> sprint 9 falsified the ETDRK2 eigenbasis over-pumping candidate
> and localised the bias to low-y over-thermalisation of modes whose
> MSW resonance lies at `T_res ≳ T_boltz_start = 30 MeV`.

## Commit chain for context

- `<sprint-9 hash>` — **Sprint 9** (Suspect 2 falsified at Point A;
  Suspect 3 promoted to prime; MSW instrumentation landed).
- `<sprint-8 hash>` — Sprint 8 (Suspect 1 falsified at machine
  precision; energy-balance instrumentation landed).
- `0c28d8d` — Sprint 7 (cold-T NaN origin fix).
- `968c936` — Sprint 6 (NaN-safe sanitisation).
- `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
  dependency** — projection is what exposes the low-y bias.
- `ca589b0` — Sprint 2 (Phase B stabilisation; n_B auto-scale). The
  auto-scale template to mirror for n_B_phase0.
- `e2f41f6` — D.7.1 (Strang-symmetric sequence; base of the integrator
  that Phase 0 reuses).
- `4d8ab2c` — D.7 (per-mode expm via Al-Mohy augmented matrix).

## Post-fix: downstream opportunities

If sprint 10 closes Suspect 3 and lands the default flip:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). Target: ΔNeff ∈ [0.05, 0.2] per
   Gariazzo+2019. Should come for free.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   resonance-driven physics at non-zero lepton asymmetry.
3. **Representation-factor audit** — still parked from sprint 4.
4. **Non-adiabatic high-y correction** (sprint-9 follow-up) — if any
   residual in high-y bucket remains after Phase 0.
5. **Fold sprint-8 and sprint-9 instrumentation into a unified per-
   step telemetry facility** that future stages reuse.
