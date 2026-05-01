# Stage F sprint 1b brief: lepton asymmetry / resonance physics for the unified missing-physics question

Single-file handoff for a fresh context window taking over after
**sprint 3h-c**, which closed the Yp sign-flip investigation and
unified two previously-separate Stage F open issues into a single
missing-physics question. By the end of reading this you should
know why **the Global-NH dNss overshoot and the narrow-mixing C
Yp deficit are the same question**, why both originate in
PRyMordial-nu's L=0 NH non-resonant configuration, and what the
two upstream physics extensions are (lepton asymmetry seeding
and resonance-aware Phase-0) that should resolve them.

## One-paragraph orientation

Sprint 19 part 2 shipped the post-Phase-B clamp cure that closed
**3/4 Hannestad benchmark points** (A, B, C in HTT 2012 bands)
under `qke_phase0_flag=False, qke_damping_formula='symmetric',
qke_post_phaseB_clamp_flag=True`; the global-fit (NH) point
overshot expected δNeff_ss=0.55 by ~70%, landing at 0.944 (≈
strong-mixing Point A's 0.915). Stage F sprint 3 closed the
**Yp sign+magnitude question**: sprint 3h-b' OR-firing pair-
symmetric clamp (`qke_phaseB_clamp_pair_symmetric=True,
rule="min"`) preserves the 3/4 coverage and **delivers right-
sign sterile-induced Yp at saturation** (A: +0.0038, NH: +0.0026
— both inside Saviano 2013 [+0.001, +0.012]) with active sector
universally healthy. Sprint 3h-c then **falsified the n→p code-
slip hypothesis** by inspection: the `(1 − f_νebar)` consumer
is dispatch-correct across all four backends. The remaining
literature-mismatch — narrow-mixing Yp deficit (C: −0.0152) and
Global-NH dNss overshoot (≈0.94 vs expected 0.55) — is now
understood as the **same missing-physics question**: PRyMordial-
nu's L=0 NH non-resonant QKE configuration produces what its
chosen physics actually predicts; HTT 2012 / Saviano 2013
reference results presume different physics (L≠0 and/or
resonance handling).

Sprint 1b's job is to add one or both upstream extensions to
the QKE configuration, then re-run the four-Hannestad scan
under sprint 3h-b' OR + L≠0 (or + resonance Phase-0) and check
whether C's Yp deficit closes to positive AND Global-NH
dNss drops into [0.4, 0.7].

## What to read, in order

Budget ~60 min before writing any code or starting an experiment.

1. **`CLAUDE.md`** — project overview.
2. **`doc/STAGE_F_BRIEF.md`** — Stage F priorities; sprint 1
   was identified there as the global-fit-NH overshoot; this
   sprint 1b is the same thread, now unified with the Yp deficit.
3. **`doc/STAGE_F_SPRINT3HC_FINDINGS.md`** (just landed) — the
   inspection verdict that unified the two issues.
4. **`doc/STAGE_E2_SPRINT19_CURE_DESIGN.md` §8.5** — the gate-5
   Hannestad scan record showing per_flavor's 3/4 + global-fit
   overshoot.
5. **`doc/STAGE_E2_SPRINT17_LITERATURE.md`** — the HTT 2012
   reference table, units audit, and asymmetry/resonance notes.
   Pay attention to HTT 2012 Appendix A on resonance crossings
   and the L=0 vs L≠0 distinction.
6. **`validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.{py,out,npz}`**
   — the live cure's four-point scan (sprint 3h commit `ff3f81f`).
7. **`PRyM/PRyM_init.py`** — flags relevant to this sprint:
   * `xi_nue_init`, `xi_numu_init`, `xi_nutau_init` (currently 0,
     all three) — these set the initial chemical potentials
     entering the QKE. **L≠0 seeding goes here.**
   * `qke_phase0_flag` (currently False, the closure default) —
     re-enabling Phase 0 brings back the resonance-crossing
     handling that was disabled in sprint 18 to cure sterile-
     sector divergence.
   * `qke_phase0_diag_flag` and `qke_active_probe_flag` —
     existing instrumentation flags useful for sprint 1b.
8. **`PRyM/PRyM_main.py`** Phase-0 segment driver (line ~376;
   the `_run_qke_segment` Froustey loop body). Reading this is
   essential before re-enabling Phase 0 — sprint 12 localised a
   "Tg ≈ 60-64 MeV resonance jump" here that sprint 18 framed
   as a numerical artefact for L=0 NH; for L≠0 or non-trivial
   IH, the same site may carry genuine resonance physics.
9. **`PRyM/PRyM_boltzmann.py`** lines 2697–2860 — the post-Phase-B
   clamp / pair-symmetric firing logic. Sprint 1b should NOT
   change this; the cure is the live default once flags are flipped.

## What's known (the framing inherited from sprint 3h)

* **Live cure flags shipped (default off, opt-in):**
   * `qke_post_phaseB_clamp_flag=True`
   * `qke_phaseB_clamp_anchor="T_nu_init"`
   * `qke_phaseB_clamp_mode="per_flavor"`
   * `qke_phaseB_clamp_pair_symmetric=True`
   * `qke_phaseB_clamp_pair_symmetric_rule="min"`
* **Live cure result (sprint 3h-b' four-Hannestad scan):**

  | Point      | Neff   | Yp     | dYp_st     | dNss   | Verdict                |
  |------------|--------|--------|------------|--------|------------------------|
  | A (strong) | 2.393  | 0.2509 | +0.00378   | 0.9341 | PASS                   |
  | B (mid)    | 2.942  | 0.2462 | -0.00102   | 0.6781 | PASS                   |
  | C (narrow) | 3.345  | 0.2319 | **-0.01523** | 0.0276 | PASS (but wrong-sign Yp) |
  | Global-NH  | 2.758  | 0.2497 | +0.00256   | **0.9457** | FAIL (dNss above [0.4,0.7]) |

* **Sprint 3h-c verdict:** n→p Pauli-blocking dispatch is
  correct across all four backends. C's Yp deficit reflects the
  genuine physics of L=0 NH non-resonant QKE evolution: at
  narrow mixing, gate-5's polyfit-flat ν̄_e tail accidentally
  provided Pauli-blocking protection on n decay (artificially
  bumping Yp upward); 3h-b' OR removes this artifact, exposing
  the underlying physics deficit.
* **Unified missing-physics question:** the Global-NH overshoot
  (saturation plateau extending to sin²2θ ≥ 0.05) and the
  narrow-mixing Yp deficit (sign-opposite to Saviano 2013) are
  the same question. HTT 2012 §4 expected δNeff_ss=0.55 (NH)
  and ~0 (IH); Saviano 2013 expects positive Yp shift +0.001
  to +0.012 across mixing. Both rely on physics that
  PRyMordial-nu currently lacks at the QKE configuration:
  lepton asymmetry seeding (`xi_nue_init`/etc != 0) and/or
  resonance-aware Phase-0.

## Sprint 1b plan

Two upstream extensions to test, in increasing implementation cost:

### Phase 1 — L≠0 lepton asymmetry seeding test (~3-5 days)

The cheapest test. The flags are already there
(`xi_nue_init`, `xi_numu_init`, `xi_nutau_init`). HTT 2012 and
Saviano 2013 specifications for L≠0 NH typically use total
lepton number L = ξ_νe + ξ_νμ + ξ_ντ around 1e-3 to 1e-2 (the
range that matters for sterile production at sub-eV masses).

Phase 1 plan:

1. **Reproduce the gate-5 Hannestad scan at L=0 NH (live cure)**
   to confirm the new commits don't introduce drift relative to
   sprint 3h-b' OR's reference. Should be 3/4 PASS, identical to
   sprint 3h-b' .out/.npz.
2. **Add a pre-Phase-A seed of `xi_nue_init = 1e-3`** (or a
   total-L equivalent across flavors) and re-run the four-point
   scan. Predict: at narrow mixing C the seeded ν_e excess
   should bias channel 4 (p+ν̄_e→n+e⁺) absorption asymmetrically,
   producing a positive Yp shift. At strong mixing (A, NH) the
   QKE saturates regardless, so the seed should mostly cancel
   between ν and ν̄ — minimal effect on dNss.
3. **Sweep `xi_nue_init ∈ {1e-3, 5e-3, 1e-2}`** at Point C and
   Global-NH and fit dYp_sterile vs xi_nue_init. The literature-
   matching value (where dYp at C lands inside Saviano [+0.001,
   +0.012]) is a candidate cure parameter.
4. **Cross-check Global-NH dNss under the same xi_nue_init
   sweep.** Expectation: at sin²2θ=0.089 + L≠0 the asymmetry
   reservoir biases sterile production; the saturation plateau
   may shift down so dNss lands in [0.4, 0.7].

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_init.py` | (top-level) | No change — the flags exist. |
| `validation/diagnostics/diag_stage_f1b_l_nonzero_scan.py` | new | Cloned from sprint 3h-b' harness; sweeps `xi_nue_init` and re-runs the four Hannestad points + a Yp-signature point at C. |

Cost: ~5h sequential per scan × ~3 scans = ~15h wall-clock
(running each scan fresh; if `xi_nue_init` doesn't enter Phase
B's QKE evolution path the cost may collapse — verify before
committing to multiple full scans).

### Phase 2 — Resonance-aware Phase-0 segment (~5-10 days)

If Phase 1 doesn't cure the C deficit, the secondary path is
re-enabling Phase 0 with resonance handling. Sprint 18 disabled
Phase 0 entirely to cure the sterile-sector divergence at
production n_B; the trade-off was loss of resonance-crossing
physics that HTT 2012 captures.

Phase 2 plan:

1. **Re-enable `qke_phase0_flag=True`** and verify that the
   sprint-18 sterile-sector divergence (δNeff_ss out of HTT
   band) is REPRODUCED at production n_B without the cure flag.
   Confirms the sprint-18 closure of Phase 0 was not regression-
   masked by the post-Phase-B cure.
2. **Re-enable Phase 0 with the live cure on** and the
   pair-symmetric OR rule and re-run the four-Hannestad scan.
   Question: does the cure flag suppress the Phase-0 sterile-
   sector divergence, AND does the resonance crossing produce
   the HTT-band δNeff_ss at Global-NH?
3. **If yes:** the live cure + Phase 0 is the closure
   configuration for Stage F. Run a benchmark sweep (sterile
   mass × mixing scan) to characterise the literature
   agreement.
4. **If no:** Phase 0 needs structural work (likely the
   ETDRK2 step-size at the resonance crossing) — handoff to
   a Phase-0-driver sprint.

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_main.py` | `_run_qke_segment` | Possibly tighten step-size near the resonance crossing region; gate behind a flag. |
| `validation/diagnostics/diag_stage_f1b_phase0_resonance_scan.py` | new | Phase-0-on harness re-running the four points. |

Cost: ~5h sequential per Hannestad scan × ~2 scans = ~10h
wall-clock for the basic test; more if Phase-0 driver work
becomes necessary.

## Test strategy

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default (no flag flips beyond the live cure). |
| 2 | `pytest -k sterile -v` | 3/3 at default. |
| 3 | live-cure-only Hannestad reproduction | matches sprint 3h-b' .npz to within 0.5% per-point in Neff/Yp/dNss. |
| 4 | L≠0 sweep at Point C | dYp_sterile lands inside Saviano [+0.001, +0.012] at some `xi_nue_init`; identifies the candidate cure parameter. |
| 5 | L≠0 four-Hannestad scan | 3/4 or 4/4 PASS. Global-NH dNss lands in [0.4, 0.7] OR is documented as residual. |
| 6 | (if Phase 2) Phase-0-on scan with cure | sterile-sector divergence not reproduced; Global-NH and C close as expected. |

## Out of scope for sprint 1b (defer to subsequent sprints)

* **Default flag flips for the Hannestad-style closure config.**
  Once a working configuration is identified, flipping defaults
  is its own sprint with full regression sweep.
* **FortEPiaNO comparison** (sprint 4 carryover from sprint 18).
  The closure-config + xi_nue_init scan should provide a
  comparison point against FortEPiaNO's L≠0 reference — but
  the apples-to-apples FortEPiaNO comparison (Phase-B inner-
  iteration matching) is its own sprint.
* **V_nunu paradox** (sprint-18 carryover). Already characterised
  as not Stage E.2-blocking.
* **QED-table extension above 40 MeV** (sprint 3i). Every run
  emits the warning; the fractional Neff effect is O(α/π) ~
  0.2%. Worth a sprint at some point but not blocking
  sprint 1b's literature-match question.
* **Sprint 3h-c re-investigation.** The n→p dispatch is verified
  correct; do NOT spend cycles re-auditing it.

## Suspects table snapshot (post sprint 3h-c)

| Suspect | Status |
|---|---|
| 6 (V_nunu axis as divergence point) | FALSIFIED (sprint 18) |
| 7 (Phase-0 ETDRK2 step pathology at L=0 NH) | FALSIFIED (sprint 16 + 18 closure) |
| 8 (active-sector pathology at production n_B) | CHARACTERISED + CURED (sprint 19 + 3h post-Phase-B clamp) |
| 9 (n→p code slip producing sign-flipped Yp) | **FALSIFIED (sprint 3h-c)** |
| 10 (L=0 NH non-resonant configuration) | **NEW; the unified missing-physics question; this sprint's target** |

## Commit chain expected for sprint 1b landing

* Sprint 1b part 1 — L=0 reproduction commit (live-cure baseline,
  no code change, just .npz re-record).
* Sprint 1b part 2 — `xi_nue_init` scan harness + .out/.npz +
  ROADMAP entry.
* Sprint 1b part 3 (if needed) — Phase-0-on scan harness +
  .out/.npz + ROADMAP entry.
* Sprint 1b handoff — single-file brief for the next sprint
  (default-flip sprint OR Phase-0-driver sprint, depending on
  Phase 1/2 outcome).
