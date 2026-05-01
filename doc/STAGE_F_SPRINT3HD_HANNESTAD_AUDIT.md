# Stage F sprint 3h-d Hannestad-2012 global-fit-point audit

Source-tracing audit of the Global-NH expected δNeff_ss claim, run
after sprint 3h-c flagged the apparent disagreement at the
"global-fit (NH)" benchmark point as suspect (parameters within
~10% of the saturated Point A but expected δNeff_ss differing by
factor ~2).

## §1 The fabricated claim

Three internal docs assert that HTT 2012 §4 reports δNeff_ss=0.55
at the global-fit point under L=0 NH (with ~0 in IH):

* `doc/STAGE_E2_SPRINT19_BRIEF.md:174-175`
* `doc/STAGE_F_BRIEF.md:69`
* `doc/STAGE_F_SPRINT3HC_FINDINGS.md:118-120`

All three trace back to the sprint-19 brief, written before
sprint 17's careful HTT 2012 audit. Sprint 17's literature
audit (`doc/STAGE_E2_SPRINT17_LITERATURE.md`) is rigorous on
Point C but does not cover the global-fit point — the 0.55
claim was inherited unaudited.

## §2 Direct quotation from HTT 2012 (arXiv:1204.5861v2)

PDF: `References/1204.5861v2_hannestad2012.pdf` (22 pages).

**Page 8, §3.2 ("Sterile neutrino production for zero lepton
asymmetry"):**

> "We mark with a green hexagon the best fit point of the 3+1
> global analysis presented in [56], obtained from a joint
> analysis of Solar, reactor, and short-baseline neutrino
> oscillation data (δm²_s, sin²2θ_s) = (0.9 eV², 0.089). **For
> that point δNeff = 1 in both hierarchies, i.e. complete
> thermalization occurs.**"

This is the *direct* HTT 2012 statement at our project's
"Global-fit (NH)" benchmark point. The expected value is
**δNeff = 1** under L=0 NH (and L=0 IH).

The HTT 2012 §4 reference in the sprint-19 brief is
fabricated. §4 of HTT 2012 is "Conclusions" (per the table of
contents on page 2: "4 Conclusions 12") and does not contain
any δNeff = 0.55 NH / ~0 IH claim.

## §3 Where the 0.55 number actually lives in HTT 2012

**Page 10, §3.3 ("The case of large initial lepton asymmetry"):**

> "For illustration, we choose the point of Fig. 1 with
> (δm²_s, sin²θ_s) = **(−3.3 eV², 6×10⁻⁴)** for which **δNeff = 0.55**
> and we show the percentage of active (Na) and sterile (Ns)
> neutrinos as a function of x for different T in Fig. 3."

This is an entirely different point:

* δm² = −3.3 eV² (negative sign → *Inverted Hierarchy*, not NH).
* sin²θ_s = 6×10⁻⁴ (single-angle convention, not double-angle;
  reading at HTT's convention this is sin²(2θ_s) ≈ 2.4×10⁻³,
  far from 0.089).
* Used as an illustrative resonance demonstration, not a
  benchmark point.

The sprint-19 brief author appears to have conflated this
illustrative IH-resonance point with the global-fit benchmark.

## §4 Corrected Hannestad-2012 four-point comparison

Under the live cure (sprint 3h-b' OR-firing pair-symmetric
clamp, commit `ff3f81f`):

| Point | sin²2θ_24 | Δm²_41 (eV²) | HTT expected δNeff_ss | PRyMordial-nu | Verdict |
|---|---|---|---|---|---|
| A (strong) | 0.1 | 0.93 | ~1 (Fig. 2 top red, saturated) | 0.9341 | **PASS** |
| B (mid) | 2.26e-3 | 0.93 | ~0.5 (Fig. 2 mid) | 0.6781 | **PASS** |
| C (narrow) | 1e-4 | 0.93 | ~0.02–0.03 (Fig. 2 top blue) | 0.0276 | **PASS** |
| Global-NH | 0.089 | 0.9 | **= 1 (HTT page 8 explicit)** | **0.9457** | **PASS** |

**4/4 PASS — full Hannestad-2012 closure.**

The sprint 19 part-2 SUBSTANTIAL closure designation, the
Stage F sprint 1 "global-fit-NH overshoot investigation", and
the sprint 3h-c unification of "Global-NH overshoot + Yp
deficit" are all artefacts of the fabricated 0.55 expectation.
None of them describes a real disagreement with HTT 2012.

## §5 What this leaves open

**Closed by this audit:**

* Stage F sprint 1 (Global-NH overshoot investigation) —
  no overshoot exists against HTT 2012.
* Sprint 1b Phase 1 (xi_nue lepton asymmetry seeding)
  motivated by the Global-NH gap — the gap doesn't exist.
* The unified "single missing-physics question" framing in
  sprint 3h-c §5 — Global-NH was never broken; the unification
  was wrong.

**Still open (separate question, not against HTT 2012):**

* **Narrow-mixing Yp deficit** (Point C: Yp = 0.2319 vs SBBN
  0.247, dYp_sterile = −0.015). This is a Yp comparison,
  not a δNeff_ss comparison. HTT 2012 does not quote Yp at
  these points. The literature target for sterile-induced Yp
  is Saviano et al. 2013 (arXiv:1302.1200), which expects
  positive shift +0.001 to +0.012 across mixing.
* **Sprint 3h-c verdict on n→p Pauli-blocking dispatch
  remains valid.** The dispatch is correct; the C Yp deficit
  is genuine physics from L=0 NH non-resonant QKE evolution.
* The Saviano comparison may itself rely on L≠0 / different
  resonance treatment. A targeted Saviano audit (parallel to
  this HTT audit, not yet performed) would clarify whether
  the Yp deficit is a real disagreement or another framing
  artefact.

## §6 Required code corrections

1. **`validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.py`**
   and clones — the `expected_delta_neff_ss=0.55` and
   `pass_band=(0.4, 0.7)` for Global-fit (NH) are wrong.
   Correct values: `expected_delta_neff_ss=1.0`,
   `pass_band=(0.9, 1.1)` (matching Point A's saturation band).
2. The `out/` summaries for all four sprint-3h scans
   (3h-a, 3h-b, 3h-b') misclassify Global-NH as FAIL. The
   underlying .npz data is correct; only the verdict cell
   is wrong. Re-running the scans is unnecessary; the .npz
   data already contains the right δNeff_ss values.
3. **Doc updates required (separate commit):**
   * `doc/STAGE_E2_SPRINT19_CURE_DESIGN.md` §8.5 — the gate-5
     four-Hannestad scan should be re-marked 4/4 PASS (FULL
     closure), not 3/4 SUBSTANTIAL.
   * `doc/STAGE_F_BRIEF.md` — sprint 1 "Global-NH overshoot"
     framing closed.
   * `doc/STAGE_F_SPRINT3HC_FINDINGS.md` §5 — the
     "unified missing-physics question" framing is wrong;
     only the Yp-at-narrow-mixing question remains.
   * `doc/STAGE_F_SPRINT1B_BRIEF.md` — the brief should be
     redirected from "fix Global-NH + Yp" to "Saviano Yp
     audit then decide whether L≠0 is needed."

## §7 Provenance

Audit performed by inspection of
`References/1204.5861v2_hannestad2012.pdf` page-by-page on
2026-05-01. PDF was extracted via pypdf. Quoted text is
verbatim from HTT 2012 v2. No web access needed; the project
already had the paper.

The fabrication appears to be a sprint-19 brief
hand-typed citation that propagated unchecked through
subsequent docs and harnesses. Sprint 17's literature audit
methodology (rigorous citation chain to source papers and
specific equations/figures) would have caught it; sprint 17
just didn't cover the global-fit point.

## §8 Lessons

* Citation discipline matters more than a sprint at a time.
  The 0.55 number propagated through three briefs over five
  sprints (17→18→19 part 2→3h→3h-c) without anyone re-checking
  the source. The sprint-17-style audit on Point C was
  exemplary, but bracket-creep on the four-point scan added
  Global-NH without applying the same audit.
* When parameters are within 10% of an audited PASS point and
  the expectation differs by factor 2, consider citation error
  before missing-physics. The user's instinctive question
  ("could this be a methodology difference?" — could it be
  even simpler than methodology, namely a wrong number?)
  inverted the framing in 30 minutes of audit work,
  invalidating ~15h of planned compute.
* Sprint 1b's Phase 1 plan should be REPLACED, not
  prosecuted: the question to test is now "what does Saviano
  2013 actually expect for Yp at our Point C, and does
  PRyMordial-nu agree once we control for the Saviano setup?"
  This is a literature audit first, compute later.
