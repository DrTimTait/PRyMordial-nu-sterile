# Stage E.2 brief: structural DW small-mixing bug hunt

Single-file handoff for a fresh context window taking over from Stage
E.1 (commit `18b1fa7`). Read this first. By the end you should know
which three hypotheses are on the table, how to distinguish them
experimentally, and what NOT to touch.

## One-paragraph orientation

Stage E.1 implemented the Mirizzi+2012 pair-specific damping formula
behind the new `PRyMini.qke_damping_formula` flag (now default
`"mirizzi"`) and verified via a minimal 2-level damped Rabi test that
the Lindblad structure is correct. The formula change did NOT close
the small-mixing DW literature gap: at Hannestad Point C
(sin²2θ=10⁻⁴, Δm²=0.93) ΔNeff moved from 0.85 → 0.858, and at the
Gariazzo benchmark (|U_μ4|²=10⁻⁴, Δm²=1.29) PRyMordial gives 0.923
vs their reported 0.09 — a 10× overproduction. Sigl-Raffelt analysis
says this cannot be a damping-coefficient problem (linear-in-D regime,
Gariazzo has *larger* D but reports *less* thermalization). Stage E.2's
job: find where the structural mismatch actually lives. Three
candidate hypotheses below, in order of estimated likelihood.

## What to read, in order

Budget ~45 minutes for reading before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.1 (landed)". The
   full E.1 validation table is there; E.2 is explicitly scoped as
   "open" at the bottom.
3. **`doc/STAGE_E1_BRIEF.md`** — the E.1 brief. Physics context
   (Mirizzi coefficients, Hannestad conventions, per-reference
   Δm²_41 conventions) is reused unchanged; don't re-derive.
4. **`validation/diagnostics/README.md`** — full diagnostic chain
   from Stage D onwards, including the "Summary of localization"
   table and "Revised diagnosis" listing 5 candidate mechanisms. E.2
   picks 3 of them.
5. **`validation/diagnostics/GARIAZZO_DAMPING_FIX.md`** — the
   bug-localization writeup that led to E.1. Section "Why this
   causes ~8× over-production at small mixing" is the key physics
   passage; the saturation argument there is what now rules out
   "just fix the coefficients".
6. Git log: `git show 18b1fa7` — the E.1 landing commit. You need
   the context of what just shipped.
7. `PRyM/PRyM_boltzmann.py` lines **4397** (the `S_gain_si = 0.0`
   line, hypothesis A), **3915–4100** (the QKE temperature-range
   and initial-conditions setup, hypothesis C), and **4300–4345**
   (the diagonal collision integrals `I_total`, hypothesis B).
8. **`References/1905.11290v3_gariazzo2019.pdf`** — Gariazzo et al.
   2019. Appendix A.2 is Eq. A.2 for `I[ρ(y)]`, and A.16 is the
   damping approximation `I_αβ(ρ) = −D_αβ · ρ_αβ`. For hypothesis A,
   read App. A.1 ("Diagonal elements") to see what Gariazzo puts on
   the sterile-active CROSS-TERMS of the diagonal integrals. Their
   FortEPiaNO code is at `https://bitbucket.org/ahep_cosmo/
   fortepiano_public` if a direct comparison is needed (Stage E.2c
   only).
9. **`References/1204.5861_hannestad2012.pdf`** if present — their
   Eqs. 2.7–2.16 for the 1+1 QKE. Section 3 spells out their
   integration window (60 → 1 MeV) — relevant for hypothesis C.

## Stage E.1 inheritance: what NOT to touch

Per the E.1 landing commit, do NOT re-litigate any of these:

- `DensityMatrixSolver._compute_D_pair_matrix` (the Stage E.1
  helper — already correct by 2-level damped-Rabi exactness).
- `PRyMini.qke_damping_formula = "mirizzi"` default. E.2 assumes
  Mirizzi is the baseline; diagnostic scripts should run with the
  default, not flip it back to `"symmetric"`.
- `_build_L_list`, `_etdrk2_expm_phi`, `_apply_unitary`,
  `_build_H_list` — all Stage D.* work, verified correct.
- `C_D` and `C_A` values.
- The D.7.1 driver body `evolve_step_ode_etdrk2`.

## Three candidate hypotheses

### Hypothesis A — Missing active-sterile gain in the collision RHS

**Claim**: `_assemble_collision_N` sets `S_gain_si = 0.0` for
active-sterile pairs at line 4473, on the grounds that "sterile has no
SM vertex". But active neutrinos in a coherent superposition with
sterile DO scatter via their active content — so `ρ_αs` has a NONZERO
gain term proportional to the active-sector partial gain. Gariazzo's
Eq. A.16 `I_αβ(ρ) = −D_αβ · ρ_αβ` uses the damping approximation
identically for active-active AND active-sterile off-diagonals
(no gain for EITHER), whereas PRyMordial provides gain for
active-active but not active-sterile. This asymmetry may be the bug.

**Diagnostic**: write `validation/diagnostics/diag_as_gain.py` that
patches `S_gain_si` for active-sterile pairs to (i) zero (current
behaviour), (ii) a trial "Gariazzo-like" pure-damping form where
the ACTIVE-ACTIVE gain is also zeroed to match (make the whole
_assemble_collision_N use Eq. A.16 across all off-diagonals), and
(iii) a trial where active-sterile gets a gain term proportional to
the active gain `gain_active[2·p_idx]` scaled by some coupling. Run
at Hannestad Point C (sin²2θ=1e-4, Δm²=0.93, PMNS off) and measure
ΔNeff for each variant.

**Expected**: if variant (ii) drops ΔNeff from 0.29 to < 0.1, we've
found the bug — PRyMordial's active-active gain term is incorrectly
enhancing coherence. If variant (iii) drops it, the right fix is to
add an active-sterile gain matching the active gain.

**Likelihood**: **high**. This is the simplest structural asymmetry
between PRyMordial and Gariazzo, and it sits right at line 4473.

### Hypothesis B — Evolution window mismatch

**Claim**: PRyMordial uses `PRyMini.T_start = 10 MeV` as the earliest
BBN temperature. If the QKE solver initializes at T=10 MeV, it
misses the T ∈ [10, 60] MeV window where DW production PEAKS (per
Hannestad+2012 Fig. 2: peak at T ~ 5·(Δm²/1 eV²)^(1/3) MeV ≈ 5·1 =
5 MeV for Δm²=1 eV² — but the code might compute this starting too
low). Hannestad integrates 60 → 1 MeV.

**Diagnostic**: (1) Find where the QKE solver's initial temperature
is set (likely in `PRyM_main.py` or `DensityMatrixSolver.
initial_conditions`). (2) Try extending the range to T_QKE_start
= 60 MeV via a new PRyMini flag, without touching T_start (which
controls the thermo / nuclear path). (3) Re-run Hannestad Point C.

**Expected**: if the T range was the bug, ΔNeff at Point C drops
substantially (maybe 0.858 → 0.3–0.5). This moves the needle but
probably doesn't close the full gap — Hannestad's Fig. 2 peaks around
T ~ 20 MeV for their benchmark, and PRyMordial at T_start=10 MeV
does integrate downward from there, so the missing window is not
the whole factor of 20.

**Likelihood**: **medium**. Worth ~1–2 h to rule out.

### Hypothesis C — y-integration discretization artifact

**Claim**: PRyMordial uses `Ny_boltz = 100` points on a linear grid
from `y_max_boltz = 100 MeV` to `y ≈ 0`. At small y (IR), numerical
aliasing between the oscillation phase `H·dt/E ∝ 1/y` and the damping
rate `D ∝ y` could leave a residual sterile population that escapes
the clamp. Gariazzo uses Gauss-Laguerre quadrature specifically to
avoid IR pile-up.

**Diagnostic**: (1) Run Hannestad Point C at `Ny_boltz ∈ {50, 100,
200, 400}` and check if ΔNeff converges. (2) Optionally try a
log-spaced y grid or Gauss-Laguerre nodes.

**Expected**: if B is the bug, ΔNeff drifts monotonically with Ny
and stabilizes only above some threshold. If it's already stable
at Ny=100 (within 10⁻³ of Ny=400), rule out B.

**Likelihood**: **low**. PRyMordial's existing Ny=100 regression
suite would already be showing convergence failures if this were a
dominant effect. But cheap to rule out as a control.

## Session realism

Three options for scope, in increasing thoroughness:

- **(a) Single-hypothesis sprint**: pick A (highest likelihood),
  write `diag_as_gain.py`, run 3 variants, land a fix if found.
  ~3–4 h.
- **(b) Triage all three**: implement diagnostic scripts for A, B,
  and C; execute; land whichever fix closes the gap. ~5–6 h. Best
  if you want to rule out C and B definitively before committing
  to the A fix.
- **(c) FortEPiaNO cross-check**: clone
  `https://bitbucket.org/ahep_cosmo/fortepiano_public`, compare
  their `collision_terms.f90` (or equivalent) directly against
  PRyMordial's `_assemble_collision_N`, identify the discrepancy
  line-by-line. ~8+ h. Use if (a) and (b) both leave a > 30% gap.

## Validation targets (post-fix)

1. **Import & fast tests** — `pytest tests/test_regression.py -m
   "not slow" -v`. All 4 fast tests must still PASS. Mirizzi
   default is committed, so this is now the baseline.
2. **Minimal 2-level cross-check** — `python
   validation/diagnostics/diag_2level_damped.py`. Still must give
   exact L-expm ≡ analytic damped Rabi (4 sig figs). Any E.2 fix
   to the collision kernel should leave the Hamiltonian+damping
   structure untouched.
3. **Sterile regression suite** — `pytest tests/test_regression.py
   -k sterile -v`. All 3 sterile tests must PASS. Note that the
   SF asymmetry test and the Stage A invariant are not expected to
   shift; `test_sterile_dw_production` (sin²2θ=0.1) is in the
   fully-thermalised regime and should also be unchanged. If any
   of them moves, your fix is doing something you didn't intend.
4. **Hannestad DW literature** —
   `python validation/sterile_DW_literature.py`. Targets vs
   Hannestad+2012:

   | Point | sin²2θ | H+2012 | E.1 baseline | E.2 target |
   |---|---:|---:|---:|---:|
   | A full | 1e-1 | 1.00 | 0.955 | 0.9–1.1 |
   | B partial | 2.26e-3 | 0.50 | 0.970 | 0.3–0.7 |
   | C minimal | 1e-4 | 0.04 | 0.858 | < 0.1 |

5. **Gariazzo benchmark** —
   `python validation/sterile_DW_gariazzo.py` (parked by E.1 for
   E.2 reuse). Target: ΔNeff ∈ [0.05, 0.2] at their benchmark
   (|U_μ4|²=1e-4, Δm²=1.29).

## Minimum viable opening message for the new session

> Read `doc/STAGE_E2_BRIEF.md` end-to-end before writing any code.
> Then use `EnterPlanMode` to propose a concrete Stage E.2
> investigation plan. My target for this session is {ONE of a / b / c}.
> Do not touch `_compute_D_pair_matrix` or flip the
> `qke_damping_formula` default — Stage E.1 landed Mirizzi and that's
> the committed baseline.

## Commit chain for context

- `18b1fa7` — **Stage E.1 landed**: Mirizzi as default, shared
  `_compute_D_pair_matrix` helper, `qke_damping_formula` flag,
  Gariazzo branch stubbed.
- `a11d68a` — doc: STAGE_E1_BRIEF Δm²_41 clarification.
- `9cc09bf` — doc: Add STAGE_E1_BRIEF.md.
- `49e31f4` — validation/diagnostics: localized DW bug to D_pair.
- `2fa1f28` — validation/diagnostics: L-expm verified via 2-level.
- `5d8426b` — validation/diagnostics/ initial.

## Post-fix: downstream opportunities

If Stage E.2 closes the DW gap:

1. **Shi-Fuller literature comparison** (Saviano+2013 or
   equivalent) — same QKE framework, different physics focus.
2. **Lovell 2023 keV-sterile DM** — needs QKE extension to T ~ 1
   GeV and QH-transition physics; separate multi-session project.
3. **Re-open Gariazzo form** — with the structural fix in place,
   Mirizzi vs Gariazzo coefficients might both land within literature
   tolerance. Implement the Gariazzo branch (currently
   `NotImplementedError`) and compare.
4. **Update `doc/ROADMAP.md`** Stage E.2 landing section with the
   fix, validation outcomes, and any new defaults.
