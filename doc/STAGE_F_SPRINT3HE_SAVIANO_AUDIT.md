# Stage F sprint 3h-e Saviano-2013 sterile-Yp audit

Source-tracing audit of the "Saviano 2013 expects positive
sterile-induced Yp shift +0.001 to +0.012" claim that has been
load-bearing across Stage F sprints 3d-3h. Run after sprint 3h-d
(Hannestad audit) closed the Global-NH δNeff phantom and left
the Saviano Yp comparison as the sole remaining literature
residual. User flagged the additional concern that QKE codes
may carry a known systematic Yp shift relative to thermal SBBN
in the pure 3-active SM case that could mask any sterile-Yp
comparison.

## §1 The claims under audit

Three internal docs cite Saviano 2013 as the source for the
expected positive sterile-induced Yp shift:

* `doc/ROADMAP.md:3361-3365` (sprint 3d findings):
  "Literature consensus says Yp should INCREASE under
   active-sterile mixing: Saviano+ 2013 (arXiv:1302.1200,
   closest published QKE+BBN analog at eV-scale Δm²) reports
   Yp = 0.247 → 0.251–0.256 (ΔYp = +0.004 to +0.010)..."
* `doc/ROADMAP.md:3490`: "Saviano 2013 expected ~+0.005."
* `validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.py`
  (and three sister harnesses): `LIT_YP_SHIFT_LO = 0.001;
  LIT_YP_SHIFT_HI = 0.012` — these constants drive the
  POSITIVE/ZERO/NEGATIVE literature-consistency labelling
  in the .out summary files.

## §2 Saviano 2013 specifics (audit findings)

PDF: arXiv:1302.1200 (Saviano, Iocco, Mangano, Miele, Pisanti,
Serpico 2013, "Multi-momentum and multi-flavour active-sterile
neutrino oscillations in the early universe...").

Audit was performed via web fetch (paper not in local
References/). Quotations below are reported by the audit agent
with section/page references.

**Mixing parameters (Sec. II.A, p.4, Eqs. 8-10):**
Saviano fixes a single set of mixings at the Giunti-Laveder
2011 best-fit:
```
sin²θ_eµ = 0.024     (active θ_13)
sin²θ_es = 0.025     (electron-sterile)
sin²θ_µs = 0.023     (muon-sterile)
Δm²_st   = 0.89 eV²
```
NH only. **No scan over (sin²2θ, Δm²) is performed.** All
their sterile-on Yp results are at this single parameter point.

For reference, their sin²θ_µs = 0.023 corresponds to
sin²(2θ_µs) = 4·sin²θ·cos²θ ≈ 0.090 — comparable to our
Global-NH benchmark (sin²2θ = 0.089), and **three orders of
magnitude larger** than our Hannestad Point C
(sin²2θ = 1e-4 ⟹ sin²θ ≈ 2.5×10⁻⁵). Saviano lives in our
saturated-mixing regime, not our narrow-mixing regime.

**BBN integration (Sec. III, p.7) — verbatim:**
> "the values of the yields of 4He mass fraction Yp and
>  deuterium 2H, as obtained from a modified version of
>  the numerical code PArthENoPE [66] for a baryon fraction
>  ω_b = 0.02249 and the neutron lifetime τ_n = 880.1 s."
>
> "we have computed the effect of sterile neutrinos by
>  rescaling the rates implemented in the code PArthENoPE
>  [66] (see also [68]) by Γ/Γ⁰, which has been numerically
>  evaluated and then interpolated. This amounts to a
>  first-order correction in a perturbative approach"

The baseline is **PArthENoPE thermal-SBBN**. The sterile-on
Yp shift is computed as
   ΔYp = Yp(PArthENoPE + Γ/Γ⁰ from QKE) − Yp(PArthENoPE thermal SBBN)
Their baseline row in Table I is "standard BBN: ΔNeff = 0,
Yp = 0.247". This **0.247 is not a Saviano-QKE-active-only Yp;
it is a PArthENoPE thermal-SBBN Yp**. Saviano does NOT report
a QKE-active-only run.

**Lepton asymmetry (intro p.2 + Table I, audit-confirmed):**
All five sterile-on rows in Table I have |ξ_e| = |ξ_µ| ≥ 10⁻³
(equivalently |L_ν| ≳ 10⁻³). Saviano explicitly restricts
the analysis to the regime
> "|L_ν| ≳ 10⁻³ where the distortions of the active neutrino
>  spectra start to become sizable"

**No L = 0 sterile-on Yp value is reported in Saviano 2013.**

The dYp_sterile range visible from Table I (subtracting the
SBBN baseline 0.247):
| ξ config                        | Yp (Saviano)  | ΔYp     |
|---|---|---|
| ξ_e = −ξ_µ = 10⁻³               | 0.259         | +0.012  |
| ξ_e = ξ_µ = 10⁻³                | 0.257         | +0.010  |
| ξ_e = ξ_µ = 10⁻²                | 0.256         | +0.009  |
| ξ_e = −ξ_µ = 10⁻²               | 0.255         | +0.008  |
| (one further row)               | 0.251         | +0.004  |

The famous "+0.001 to +0.012 sterile Yp shift" is therefore
**L ≠ 0 only** — driven by asymmetry-induced spectral
distortion of ν_e and ν̄_e plus late resonant conversion.
Saviano notes (Sec. III, p.8):
> "a significant fraction of the effect on Yp is due to the
>  changes of the weak rates regulating the n↔p chemical
>  equilibrium due to distorted ν_e and ν̄_e distributions"

This mechanism is **inoperative at L = 0**: no asymmetry to
distort the ν/ν̄ spectra differently, and no NH resonance for
L = 0 (HTT 2012 Appendix A).

**Saviano makes no Yp claim at L = 0 NH non-resonant
narrow-mixing.** Hannestad Point C is outside their published
parameter space on TWO axes (mixing magnitude AND lepton
asymmetry).

## §3 The QKE-vs-thermal-SBBN Yp systematic in 3-active SM

Audit task 2 finding:

* Across modern QKE codes (Mangano 2005, de Salas-Pastor
  2016, Akita-Yamaguchi 2020, Bennett 2021, Froustey
  2020/2024, FortEPiaNO, NUDEC_BSM, PRyMordial), the
  active-only ΔNeff is 0.043-0.044 and the corresponding
  active-only Yp shift versus instantaneous-decoupling SBBN
  is ~+(2-4)×10⁻⁴.
* This is well below the 0.001 level.
* PRyMordial-nu's sprint 3d active-only Yp = 0.24717 vs
  thermal SBBN 0.247 is consistent with this ~10⁻⁴ floor.

**PRyMordial-nu has no measurable QKE-vs-thermal Yp systematic
in the 3-active SM case.** The QKE-vs-thermal systematic
cannot explain our −0.015 deficit at narrow mixing.

## §4 Verdict — the Saviano comparison is also misframed

The "Saviano 2013 expects +0.001 to +0.012 sterile-induced
Yp shift" claim has been applied to **the wrong parameter
point with the wrong physics regime**:

* Wrong mixing magnitude — Saviano sin²θ ~ 0.025
  (saturated); we apply it at sin²θ ~ 2.5×10⁻⁵ (narrow).
  Three orders of magnitude apart.
* Wrong lepton asymmetry — Saviano |ξ| ≥ 10⁻³
  (asymmetry-dominated); we apply it at L = 0
  (no asymmetry mechanism).
* Saviano explicitly identifies asymmetry-driven spectral
  distortion as the dominant Yp mechanism, and that
  mechanism is INOPERATIVE in our setup.
* Saviano's QKE solver only modifies the BBN n↔p Born
  rates by ≤ 3% (Sec. III, p.7), described as
  "comparable or lower than neglecting the modification
  to the reheating in the standard scenario" — much
  smaller perturbative corrections than what we'd need
  for a definitive sterile-Yp prediction.

The C Yp deficit (−0.015) is REAL, but it is **not a
disagreement with Saviano** — Saviano makes no claim at our
parameter point. PRyMordial-nu is plausibly the first
published prediction at L = 0 NH non-resonant 2-flavor
mixing at Δm² = 0.93, sin²2θ = 1e-4.

## §5 Updated Stage F open-issue inventory

**Closed by audits (sprint 3h-d + 3h-e):**

* Hannestad Global-NH δNeff: 4/4 PASS, full closure
  (sprint 3h-d).
* Saviano Yp comparison: misframed; no Saviano claim at
  our parameter point (this audit).
* QKE-vs-thermal Yp systematic: not a contributor (~10⁻⁴
  floor; can't explain −0.015).
* The "unified missing-physics question" framed in sprint
  3h-c §5: BOTH halves are now closed. There is no
  unified question.

**Real but reframed:**

* **C Yp deficit (−0.015) at L = 0 NH narrow mixing.**
  Real PRyMordial-nu prediction. No published comparator.
  Two interpretations:
  1. Genuine new physics from full QKE treatment that
     simplified solvers (Saviano's Γ/Γ⁰ perturbative
     rescaling) cannot capture.
  2. PRyMordial-nu-specific bug at narrow mixing that no
     published comparison would have caught.
  The only honest discriminator: **reproduce Saviano's
  actual setup (sin²θ_es ≈ 0.025, Δm² = 0.89 eV², L = 10⁻³
  to 10⁻², NH) and check whether PRyMordial-nu reproduces
  Saviano's +0.001 to +0.012 shifts there.** If yes,
  PRyMordial-nu is calibrated to Saviano in Saviano's
  regime, and the −0.015 at narrow mixing is novel
  trustworthy physics. If no, there is a real pipeline
  disagreement traceable in Saviano's parameter space.
  Either outcome is interpretable.

**Still open and not yet audited:**

* Sprint 4 (FortEPiaNO comparison + V_nunu paradox at
  structural level) — sprint-18 carryover. Gariazzo 2019
  IS in References/, so a parallel audit + reproduction
  is feasible. Strong candidate for cross-solver
  validation in Saviano's regime AND at Hannestad
  benchmarks.

## §6 Sprint 1b redirect (cancels the Phase 1 plan as written)

`doc/STAGE_F_SPRINT1B_BRIEF.md` Phase 1 plan was: ξ_nue scan
at Hannestad benchmark points to test whether L ≠ 0 fixes the
Global-NH overshoot AND the C Yp deficit. The Global-NH
"overshoot" is gone (sprint 3h-d), so half the motivation
disappears. The C Yp deficit motivation is also misframed
(this sprint), so the L ≠ 0 scan at Hannestad points doesn't
test anything Saviano-comparable.

**Replace with:** "Sprint 1b' — Saviano-regime L ≠ 0
reproduction." Run PRyMordial-nu at Saviano's actual setup
(sin²θ_es ≈ 0.025, Δm² = 0.89 eV², L ∈ {10⁻³, 10⁻²}, NH)
and compare Yp directly to Saviano Table I. ~5-10 h
sequential. This is the apples-to-apples literature
calibration test.

If sprint 1b' shows PRyMordial-nu reproduces Saviano in
Saviano's regime, the C Yp at L = 0 narrow mixing is
trustworthy as PRyMordial-nu's first-published prediction
at that point. If it fails to reproduce, the discrepancy is
diagnosable in a regime where literature data exists.

## §7 Required code corrections

1. **Four scan harnesses** (`diag_stage_f3h_a/_b/_b_prime/`,
   plus `diag_sprint19_hannestad_scan.py`) — the
   `LIT_YP_SHIFT_LO = 0.001; LIT_YP_SHIFT_HI = 0.012`
   constants and the
   "POSITIVE (literature-consistent)" /
   "NEGATIVE (literature-inconsistent)" labelling are based
   on the misframed Saviano comparison. Either remove the
   labelling, or replace the citation comment to make
   clear that the +0.001 to +0.012 range is L ≠ 0
   saturated-mixing only and does NOT apply to Hannestad
   benchmark points.
2. **Doc updates** (separate commit):
   * `doc/STAGE_F_BRIEF.md` — sprint-1 framing already
     stale post 3h-d; both literature-comparison residuals
     are now closed. Should be retitled as "Stage F
     priorities post 3h-d/3h-e: Saviano-regime
     reproduction + FortEPiaNO cross-solver validation".
   * `doc/STAGE_F_SPRINT3HC_FINDINGS.md` §5 — the
     unified-missing-physics framing is wrong on both
     halves now; should be replaced with the audit
     verdict.
   * `doc/STAGE_F_SPRINT1B_BRIEF.md` — Phase 1 plan
     cancelled; redirect to Saviano-regime reproduction
     (this audit §6).
3. **`PRyM_init.py` xi flag default** — already 0.0
   (correct for L = 0 closure-config); no change needed.

## §8 Provenance

Audit performed by:
* Sub-agent literature-fetch and section-by-section read of
  arXiv:1302.1200v2 (web fetch — paper not in local
  References/), 2026-05-01.
* Cross-check against the user-flagged QKE-vs-thermal Yp
  systematic literature (Mangano 2005, Akita-Yamaguchi
  2020, Bennett 2021, Froustey 2020/2024, FortEPiaNO,
  NUDEC_BSM, PRyMordial paper arXiv:2307.07061).

The Saviano misframing appears to have entered via a
literature-overview note in sprint 3d (sprint 3d's
ROADMAP.md entry is the earliest internal citation we have;
it presents "Saviano 2013 reports Yp = 0.247 → 0.251–0.256"
as a straightforward citation without specifying the
parameter point or L value). Subsequent sprints inherited
the framing without re-checking the source — same failure
mode as the Hannestad 0.55 fabrication (sprint 3h-d).

## §9 Lessons (incremental over sprint 3h-d)

The Hannestad audit lesson was: when parameters are within
10% of an audited PASS point and the expectation differs by
factor 2, suspect citation error.

The Saviano audit lesson is: when a literature comparison
asserts "+/−" sign opposites between published code and
project code, check that the comparison is at the SAME
parameter point AND the SAME physics regime. Saviano's
+0.001 to +0.012 was at sin²θ ~ 0.025 with L = 10⁻³ to 10⁻²;
we applied it at sin²θ ~ 2.5×10⁻⁵ with L = 0. Three orders
of magnitude in mixing AND a missing dominant mechanism.
The sign of the Yp shift was never expected to be the same
across these two regimes.

**Combined lesson from both audits:** literature-comparison
assertions should be source-traced in writing (citation +
section/page + parameter values + physics regime caveats)
before being encoded into harness pass/fail conditions or
sprint-direction decisions. Both Hannestad and Saviano
phantoms cost multiple sprints of compute and analysis that
were structurally chasing nothing.
