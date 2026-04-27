# Stage E.2 sprint 19 brief: active-sector pathology at production n_B and four-point Hannestad scan

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 18. Read this first. By the end you should know
why sprint 18 declared **partial Stage E.2 closure** (sterile-sector
δNeff_ss = 0.054 IN HTT 2012 band [0.02, 0.10] at production n_B
for the `qke_phase0_flag=False, qke_damping_formula='symmetric'`
configuration) and why the **active sector exhibits a numerical
pathology (Neff=417, Yp=0.36) at the same configuration and same
n_B**, and why sprint 19's job is to (a) localise and cure the
active-sector pathology and (b) extend the verification to all
four Hannestad benchmark points.

## One-paragraph orientation

Sprint 18 falsified the V_nunu axis as the divergence point:
flipping `qke_v_nunu_active_only` only made δNeff_ss WORSE while
restoring near-SM Neff/Yp at reduced n_B, and the joint
configuration with HTT-matching damping showed the two flag axes
do not compose. The breakthrough came from the Phase-0/Phase-B
handoff diagnostic: disabling Phase 0 entirely (Phase B runs
directly from T=100 MeV down) brought δNeff_ss from 0.331 (Run C)
to **0.0304** at reduced n_B — squarely in the HTT 2012 band. The
sprint-12-localised "Tg ≈ 60-64 MeV resonance jump" was a Phase-0/
Phase-B handoff numerical artefact, NOT a physical resonance
(consistent with HTT 2012 Appendix A: no resonance for L=0 NH).
The production-n_B confirmation run kept δNeff_ss in band (0.054,
robust to n_B doubling — sterile cure is solid), but produced
Neff=417 and Yp=0.36 — the active sector explodes at production
n_B for the no-Phase-0 path. Sprint 19's task is to localise the
active-sector instability (likely an ETDRK2 step-size pathology
in the extended Phase-B segment at small dt) and verify the cure
across all four Hannestad benchmark points.

## What to read, in order

Budget ~45 min before writing any code or starting an experiment.

1. **`CLAUDE.md`** — project overview.
2. **`doc/STAGE_E2_SPRINT17_LITERATURE.md`** — HTT 2012 reference
   table, units audit (the cornerstone of the sprint-17 + 18
   reframing).
3. **`doc/STAGE_E2_SPRINT18_BRIEF.md`** — sprint 18's framing and
   the joint axis bracket motivation.
4. **`doc/ROADMAP.md`** — sprint-18 landing record at the bottom
   has the breakthrough finding (Phase-0 disable cures sterile
   sector) and the active-sector pathology. Skim sprints 14-17.
5. **`validation/diagnostics/diag_sprint18_no_phase0_production.py`**
   and `.out` — the production-n_B confirmation harness; this is
   the run that shows Neff=417 and Yp=0.36 at n_B=12000 while
   δNeff_ss=0.054 stays in band.
6. **`validation/diagnostics/diag_sprint18_phase0_handoff.py`**
   and `.out` — the reduced-n_B (n_B=3500) reference giving
   Neff=3.91, Yp=0.249, δNeff_ss=0.030.
7. **`PRyM/PRyM_main.py`**:
   * Phase-A → Phase-B segment chain, lines 213-265. With
     `qke_phase0_flag=False` and `T_boltz_start=100 MeV`,
     Phase A runs from `Tstart_MeV` (~105 MeV) to T_boltz_start;
     Phase B runs from T_boltz_start down to T_boltz_end=0.005.
   * `_run_qke_segment` body (line 376+) — the Froustey loop body
     that ETDRK2 calls; outer-step grid `a_grid_seg` is logspaced,
     so dt at the high-T end (T~100 MeV) is much smaller than at
     the low-T end. Sprint-19 should instrument the per-step
     Neff / Yp / sigma_curr to find where the active sector
     starts diverging.
   * The Tg=100 MeV → T_boltz_end=0.005 MeV span is 9.9 decades
     in scale factor; n_B=12000 gives ~1200 steps/decade. Sprint
     16 production used n_B=10000 over a smaller span (8.7
     decades), giving ~1150 steps/decade. The step density is
     comparable; the pathology may not be a step-density issue.
8. **`PRyM/PRyM_thermo.py`** — the QED tables zero-clamp above
   T=40 MeV (the warning emitted at T_boltz_start=100 MeV). The
   sprint-18 reduced-n_B run gave benign Neff/Yp; the production
   run with the same QED clamp gave catastrophic Neff. So the QED
   clamp alone isn't sufficient explanation for the ×100 Neff
   blowup.

## What's broken (the framing inherited from sprint 18)

* **Active sector pathology at production n_B for the no-Phase-0
  path**:
  - Reduced n_B=3500: Neff=3.909, Yp=0.249 (near-SM, healthy)
  - Production n_B=12000: Neff=417.19, Yp=0.36 (catastrophic)
  - Same flags, just larger n_B. Sterile sector is robust
    (δNeff_ss=0.030 → 0.054, modest +12% on Σρ_ss(raw)) — only
    the active sector diverges.
* The pathology is NOT in the active-flavor density matrix
  diagonals at end-of-Phase-B (the harness's δNeff_ss
  computation uses `c._boltz_rho_final[:, 0, :]` as the active
  ν_e, and the ratio m3_ss/m3_e is sensible at 0.054 — meaning
  ρ_νe at end of Phase B is approximately one fully-thermal
  species worth of energy density). The pathology is in the
  Neff calculation downstream, possibly via the Phase A → B
  entropy/sigma bookkeeping or the `c.PRyMresults()` Neff
  formula's normalisation.
* Yp = 0.362 vs SM Yp = 0.247 → ΔYp = +0.115 (47% above SM).
  Standard relation dYp/dNeff ≈ 0.013 → ΔNeff_implied ≈ 8.8
  from Yp. But reported Neff is 417 — vastly more than 8.8.
  The Neff calculation is breaking, not just over-estimating.
* HTT 2012's Fig. 2 spans **all four Hannestad benchmark points**
  (Point A: sin²2θ=0.1 / curve red; Point B: 2.26e-3 / orange;
  Point C: 1e-4 / blue; plus the global-fit point at sin²2θ=0.089,
  δm²=0.9 eV²). Sprint 18 only verified Point C. Stage E.2 closure
  requires reproducing the band for all four points.

## Recommended sequence

Sprint 19 should be a two-phase sprint: **active-sector pathology
debug** first (single-localised problem; ~1-2 days), **then** the
four-point Hannestad scan (~2-3 days, runs).

### Phase 1 — Active-sector pathology localisation (~6 hours, instrumented run)

Build `validation/diagnostics/diag_sprint19_active_sector_probe.py`
extending `diag_sprint18_no_phase0_production.py` with per-step
instrumentation:

* Save `(t, Tg, a, sigma_curr, Σ_y y³ ρ_νe·dy)` at every outer
  step into a numpy array.
* Run at n_B=12000 (production, where pathology is present) and
  at n_B=3500 (reduced, where pathology is absent). Both with
  Phase-0 disabled.
* Diff the two trajectories: identify the istep at which
  Σ_y y³ ρ_νe (active energy density) diverges between the two
  runs.

Three candidate localisations:

1. **Early-Phase-B explosion (T ~ 100→50 MeV)**: ETDRK2 step-size
   pathology in the high-T regime. At T=100 MeV the collision
   rate is ~G_F²·T⁵ ~ 10⁵ s⁻¹ at production n_B's dt ~ 1e-7 s,
   the dimensionless damping z ~ 0.01 — well in the φ_1
   computation regime. Should be benign. But cumulative drift
   over n_B=12000 outer steps could amplify a small per-step
   error.
2. **Mid-Phase-B amplification (T ~ 30→10 MeV)**: the sprint-12
   "resonance" region (Tg=60-64 MeV) sits inside this window.
   With Phase 0 disabled the QKE driver sees this region in a
   single segment without the Phase-0/B boundary discontinuity,
   but the ETDRK2 corrector might still over-pump in this
   region at high n_B.
3. **End-of-Phase-B numerical drift (T ~ 1→0.005 MeV)**:
   accumulated entropy/sigma errors in the Froustey integration
   over n_B=12000 small steps could violate energy conservation.

The instrumentation should determine which window is responsible.

### Phase 2 — Active-sector cure (~6 hours, code change)

Depending on Phase 1 verdict:

* **If early Phase B**: switch the early-Phase-B segment (T > 30
  MeV) to a different driver (re-enable Phase 0 with a tighter
  ETDRK2 substep rule, or use a single-pass ETDRK4 driver for
  the high-T segment).
* **If mid Phase B**: the sprint-12 "resonance" region needs a
  different corrector; ETDRK4 is already opt-in (sprint 16).
  Re-run the sprint-18 closure config with `qke_etdrk4_flag=True`
  to see if ETDRK4 cures the active sector while preserving
  the sterile cure.
* **If end-of-Phase-B drift**: investigate the entropy/sigma
  bookkeeping in the Froustey loop; possibly tighten the
  Phase-A initial-condition transfer at T_boltz_start=100 MeV.

### Phase 3 — Four-point Hannestad benchmark scan (~3 hours, three new runs)

Once active-sector pathology is cured, write
`validation/diagnostics/diag_sprint19_hannestad_scan.py`:

* Point A: sin²2θ=0.1, δm²=0.93 eV² (HTT 2012 expected δNeff ≈ 1.0)
* Point B: sin²2θ=2.26e-3, δm²=0.93 eV² (expected δNeff ≈ 0.5)
* Point C: sin²2θ=1e-4, δm²=0.93 eV² (expected δNeff ≈ 0.03,
  already verified in sprint 18)
* Global-fit point: sin²2θ=0.089, δm²=0.9 eV² (HTT 2012 §4
  reports δNeff = 0.55 in NH, ~0 in IH; we run NH)

All under the Stage-E.2-closure config: `qke_phase0_flag=False`,
`qke_damping_formula='symmetric'`, `qke_v_nunu_active_only=True`,
production n_B (likely n_B=12000 if the cure preserves the
no-Phase-0 path; otherwise whatever the cured config requires).
Each ~35 min wall-clock; total ~2.5 hours.

### Phase 4 — Verification gates

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default |
| 2 | `pytest -k sterile -v` | 3/3 at default |
| 3 | Sprint-19 active-sector probe | Neff and Yp converge to near-SM values across n_B (active-sector pathology resolved) |
| 4 | Hannestad four-point scan | All four points in band: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10], global ∈ [0.4, 0.7] |

### Decision tree at gate 4

| Outcome | Verdict | Action |
|---|---|---|
| All four points in HTT-comparable bands | **Stage E.2 fully closed** | Document the closure config (qke_phase0_flag=False + qke_damping_formula='symmetric') in PRyM_init.py; propose default flips for Hannestad-style benchmarks; write Stage F brief (BSM extension follow-ups). |
| Three of four in band, one out | **Substantial closure** with one outlier | Investigate the outlying point; possibly accept as Stage-E.2-imperfect with documentation; proceed to Stage F. |
| Two or fewer in band | Stage E.2 closure incomplete | Sprint-20 investigation: Hannestad benchmark points at strong mixing (Point A, sin²2θ=0.1) are nominally NOT in the L=0-NH non-resonant regime and may genuinely have different physics (resonance possible, Stage C asymmetry effects) than our setup. Re-read HTT 2012 §3.2 for the Point A regime. |

## Stage inheritance: what NOT to touch

* `evolve_step_ode_etdrk2`, `evolve_step_ode_etdrk4`,
  `evolve_step_lsoda` — production drivers, untouched. Sprint 19
  may add a NEW driver flag if Phase 1 localises the pathology
  to a specific segment, but does not modify existing drivers.
* `_etdrk_expm_phi_4`, `_build_H_list`, `_build_L_list`,
  `_assemble_collision_N`, `_compute_D_pair_matrix` — kernels.
* `qke_phase0_flag` default stays True until Stage F default-flip.
  Sprint 19 work uses False as an experiment-only configuration
  unless a cure is found that makes False default-safe.
* `tests/test_regression.py` — must pass 6/6 fast bit-identical
  at default.
* The sprint-17 + 18 axis bracket data (Runs A, B, C, joint,
  no-Phase-0 reduced and production) is the canonical reference
  for any future bracket experiments. Do NOT regenerate or
  overwrite the `.out` files.

## Why not declare Stage E.2 closed and move to Stage F?

Sprint 18's δNeff_ss = 0.054 in band IS a real result, and the
sterile-sector cure is real. But the active-sector pathology at
production n_B is unprecedented in the project's prior n_B
convergence behavior (sprint 16 established that active-sector
Neff and Yp are stable across n_B for the Phase-0-ON path), and
declaring Stage E.2 closed without resolving Neff=417 / Yp=0.36
would leave a known-broken regime in the closure config. Stage F
should not start until the closure configuration is uniformly
healthy across all four Hannestad benchmark points at production
n_B.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT19_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete
> sprint-19 plan. Target for this session is to localise the
> active-sector pathology at production n_B for the no-Phase-0
> path (Phase 1) and propose a cure (Phase 2 design only,
> implementation can wait for sprint 19 part 2).

## Commit chain for context

  * **Sprint 18** (this) — joint axis bracket falsifies V_nunu
    composition; Phase-0/B handoff diagnostic delivers
    sterile-sector cure (δNeff_ss in HTT 2012 band); production
    n_B confirms sterile robustness but reveals active-sector
    pathology; PARTIAL Stage E.2 closure.
  * **Sprint 17** (1e01f26) — HTT 2012 reproduction; unit-mismatch
    identified; three-run axis bracket inconclusive on its own.
  * **Sprint 16** — Krogstad ETDRK4 driver; Suspect 7 falsified.
  * `b8dd47a` — Sprint 15 (analytic Jacobian + segment-only LSODA).
  * `c6149dd` — Sprint 14 (LSODA driver; gate 5 wall-clock infeasible).
  * `db511c5` — Sprint 13 (collision-N refactor falsified Suspect 6).
  * `e699c00` — Sprint 12 (sub-step instrumentation; localised
    Tg ≈ 60-64 MeV jump that sprint 18 traced to Phase-0/B
    handoff numerical artefact).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver; the segment now
    falsified as the source of the sterile-sector divergence).
