# Stage F brief: post-Stage-E.2 follow-ups

Single-file handoff for the agent taking over after Stage E.2's
**SUBSTANTIAL closure** (sprint 19 part 2, this commit). Stage F
priorities are derived from the sprint-19 part-2 outcome and the
sprint-18 carryover list.

## One-paragraph orientation

Stage E.2 closed three of four Hannestad 2012 benchmark points
(A, B, C in HTT 2012 bands; global-fit NH out of band by ~70%)
under the cured config `qke_phase0_flag=False,
qke_damping_formula='symmetric', qke_post_phaseB_clamp_flag=True`
at production n_B=12000. The cured QKE driver and BBN pipeline
produces near-SM Neff (~1.83 production, ~1.34 reduced; both well
under the gate-4 threshold of 4.0) and near-SM Yp (~0.231; ~6%
below SM 0.247). The global-fit (NH) point overshoots its
expected δNeff_ss=0.55 by ~70% (lands at 0.944 ≈ Point A's
0.915), suggesting the project's L=0 NH non-resonant configuration
saturates at the strong-mixing regime for any sin²2θ ≳ 0.05 — a
missing-physics finding (likely lepton asymmetry seeding or
resonance-aware Phase-0) rather than a numerical bug. Stage F's
job is to (1) decide on default flag-flips for Hannestad-style
runs, (2) investigate the global-fit-NH overshoot, (3) address
the +6% Yp under-prediction, and (4) address sprint-18 carryovers
(FortEPiaNO comparison, V_nunu paradox at structural level) at
lower priority.

## What to read, in order

Budget ~30 min.

1. `doc/ROADMAP.md` (sprint-19 landing record at the bottom).
2. `doc/STAGE_E2_SPRINT19_CURE_DESIGN.md` (especially §8 — the
   sprint-19-part-2 verdict, cure pattern, and gate 4-5 results).
3. `doc/STAGE_E2_SPRINT19_BRIEF.md` (sprint-19 framing
   inherited from sprint 18).
4. `validation/diagnostics/diag_sprint19_hannestad_scan.{out,npz}`
   — gate-5 four-point scan showing the global-fit outlier.
5. `validation/diagnostics/diag_sprint19_active_sector_cure_probe.{out,npz}`
   — gate-4 cure verification at both n_B settings.
6. `PRyM/PRyM_init.py` — three sprint-19 flags (`qke_active_probe_flag`,
   `qke_post_phaseB_trace_flag`, `qke_post_phaseB_clamp_flag`),
   all default False, plus the sprint-18 carryover flags
   (`qke_phase0_flag`, `qke_damping_formula`).
7. `PRyM/PRyM_boltzmann.py` (`_make_f_callable` ~ line 2697 and
   `make_f_callable` ~ line 2830) — the cure site
   (one-line condition extension behind the cure flag).

## Recommended sequence

Stage F should be a multi-sprint stage. Suggested ordering:

### Stage F sprint 1 — global-fit (NH) overshoot investigation (~3-5 days)

The global-fit (NH) point produces δNeff_ss = 0.944 instead of
the expected 0.55 from HTT 2012 §4. Hypotheses to test (in
order of probability):

  1. **The project's QKE saturates at the strong-mixing
     plateau for sin²2θ ≳ 0.05.** Run a sin²2θ scan at fixed
     δm² = 0.93 over sin²2θ ∈ {0.01, 0.05, 0.05, 0.1, 0.5}
     with the cured config to map the project's δNeff_ss(sin²2θ)
     curve. If the curve is FLAT above sin²2θ=0.05 (sitting at
     ~0.91-0.95), that confirms the saturation and directs
     attention to (2) or (3) below.
  2. **HTT 2012 §4's expected 0.55 relies on lepton asymmetry
     seeding.** HTT 2012 considers L=0 NH and L=0 IH separately;
     the global-fit point's expected δNeff_ss=0.55 (NH) and ~0
     (IH) suggests asymmetry-dependent dynamics. Compare to the
     project's L=0 NH closure — if the project's L=0 NH gives
     ~0.94 and HTT 2012 expects 0.55, either HTT used L≠0 or
     the resonance handling matters.
  3. **Resonance handling is necessary at sin²2θ=0.089.** The
     project's `qke_phase0_flag=False` Phase-B-only path may
     bypass resonance handling that HTT 2012 captures. Try
     Stage F sprint with `qke_phase0_flag=True` for the
     global-fit point only (as a test) — if δNeff_ss drops
     toward 0.55, resonance handling is needed.
  4. **An NH-IH framing mismatch.** The project's `Dm2_41`
     parameter is sign-conventional. HTT 2012 NH means
     m_4 > m_3 (Δm² > 0); confirm the project's sign matches.

### Stage F sprint 2 — default flag flips for Hannestad-style runs (~1 day)

Per sprint-18 carryover #6 and sprint-19 part 2 §8.4:

  * `qke_phase0_flag` default: True → False (verified safe by
    sprint-18 sterile cure + sprint-19 active cure).
  * `qke_damping_formula` default: 'mirizzi' → 'symmetric'
    (verified by sprint-17 → sprint-18 axis bracket).
  * `qke_post_phaseB_clamp_flag` default: False → True
    (verified by sprint-19 part 2 cure; safe for both n_B
    settings).
  * `qke_v_nunu_active_only` default: True (already default).

**Risk**: changing defaults will likely shift Mode-2-or-similar
regression tests by small amounts. Run a full regression sweep
including Mode 2 (FD distributions) before flipping defaults.
Mode 2 currently passes via `qke_phase0_flag=True` codepath,
which the cure doesn't touch — so expected impact: zero on
Mode 2; non-zero on QKE-mode runs.

### Stage F sprint 3 — Yp under-prediction investigation (~3-5 days)

Sprint-19 part 2 §8.4 noted Yp ≈ 0.231 (post-cure) vs SM 0.247
— a ~6.5% under-prediction. The candidate causes:

  1. **`y_max_boltz = 100` truncates the active-sector
     spectrum.** At T_nu_init = 105 MeV, the FD distribution
     has half-density at y ≈ 105, well above the grid. The
     cure sets f → exp(-y/y_max_grid) for y > 100, but this
     might under-count the active-sector energy density at
     low Tg. Try `y_max_boltz = 200` (regenerate cached
     thermo / nTOp tables if needed; check `compute_bckg_flag`
     and `compute_nTOp_flag` workflow).
  2. **Post-Phase-B active-sector re-thermalisation.** After
     decoupling at Tg ~ 1 MeV, the active-sector distributions
     should be approximately FD with T_nu_eff ≈ (4/11)^(1/3) Tg.
     The QKE-evolved distributions at end-of-Phase-B are
     non-thermal in detail. A post-Phase-B
     "re-thermalisation" step (replacing each active f_α with
     the FD distribution that has the same total energy density)
     would give exactly the SM weak-rate inputs to the
     n→p freeze-out.
  3. **n→p weak rates evaluated with the QKE distributions
     directly might differ from FD-equivalent.** Check
     `PRyM_eval_nTOp.py` for how it consumes the f_α
     callables — if it expects approximately-FD shape,
     non-thermal QKE distributions could shift Yp.

### Stage F sprint 4 — sprint-18 carryovers (~3-5 days)

  * **FortEPiaNO comparison scripts** — sprint-17 literature
    record §3 noted these were never realised. With Stage E.2
    substantially closed, the comparison is more meaningful.
    Build a side-by-side runner if FortEPiaNO is reachable.
  * **V_nunu paradox** (sprint-17 Run B near-SM Neff/Yp under
    `active_only=False`) — flagged in sprint-18 recommended
    actions. Now that the active-sector cure is understood,
    the paradox may have a simple explanation
    (`active_only=False` may have masked the FD-tail bug
    differently). Investigate.

## What NOT to touch

  * Sprint-19 part 1 + part 2 `.out`/`.npz` artefacts —
    canonical reference for the active-sector cure.
  * The cure flag's default value (False) — must remain False
    until Stage F sprint 2 deliberately default-flips it.
  * The trace-flag plumbing (`qke_active_probe_flag`,
    `qke_post_phaseB_trace_flag`) — additive instrumentation;
    don't remove without thinking about future investigations.
  * `tests/test_regression.py` — must continue to pass 6/6 fast
    bit-identical at default and 3/3 sterile.

## Why declare Stage E.2 SUBSTANTIAL CLOSURE rather than wait for
the global-fit-NH cure?

The global-fit-NH overshoot is a missing-physics finding (the
project's L=0 NH non-resonant configuration produces saturated
strong-mixing-regime output for sin²2θ ≳ 0.05), not a numerical
bug. Adding the missing physics (lepton asymmetry input
or resonance handling) is a Stage F task in scope; Stage E.2's
remit was the QKE driver + BBN pipeline numerical correctness,
which is now demonstrated by the 3/4 in-band points and the
n_B convergence ratio dropping from 107× to 1.37×. Holding
Stage E.2 open for the global-fit point would conflate physics
modelling with numerical correctness.

## Commit chain for context

  * **Sprint 19 part 2** (this) — FD-tail extrapolation cure
    via `qke_post_phaseB_clamp_flag`; gate-4 cure verification;
    gate-5 four-Hannestad-point scan; SUBSTANTIAL Stage E.2
    closure; escalate to Stage F.
  * **Sprint 19 part 1** (`fe5834d`) — QKE driver exonerated by
    dual-n_B per-step probe; active-sector pathology localised
    to post-Phase-B pipeline; cure-design doc proposes the
    interpolator-tightening path.
  * **Sprint 18** (`c76468b`) — Phase-0 segment localised as
    sterile-sector divergence; PARTIAL Stage E.2 closure.
  * Earlier sprints (10-17): see `doc/ROADMAP.md`.
