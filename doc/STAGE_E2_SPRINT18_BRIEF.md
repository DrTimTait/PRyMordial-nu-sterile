# Stage E.2 sprint 18 brief: joint (V_nunu, damping) axis bracket and Phase-0 handoff diagnostic

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 17. Read this first. By the end you should know
why sprint 17's single-axis V_nunu and damping-kernel brackets
each move δN_eff_ss in the expected direction but neither alone
reaches the HTT 2012 band [0.02, 0.10], and why the recommended
sprint-18 path is the **joint configuration** (active_only=False,
damping='symmetric') alongside a Phase-0/Phase-B handoff probe to
test whether the sprint-12-localised Tg ≈ 60-64 MeV "resonance
jump" is physical or a numerical artefact (HTT 2012 Appendix A
states no resonance exists for L=0 NH).

## One-paragraph orientation

Sprint 17 identified **two material conventions in our project
notes that diverge from HTT 2012** (the source of the "[0.02,
0.10]" Hannestad target band): (a) HTT uses the legacy "symmetric"
Stodolsky damping kernel, not our default "mirizzi"; (b) HTT's
quantity is δN_eff, not a raw Σρ_ss — reframing in HTT-comparable
units collapses the inherited "200× gap" into a 3-10× gap on
δN_eff_ss. Sprint-17's three-run axis bracket showed that flipping
the damping kernel to HTT-matching "symmetric" reduces δN_eff_ss
from 0.629 (project default) to 0.331 — a 47% reduction in the
right direction but still 3.3× above the band edge 0.10. Flipping
the V_nunu projection from active_only=True (sprint-5 fix) to
active_only=False (sprint-5 legacy) raises δN_eff_ss to 0.963 but
**restores near-SM Neff (3.09) and Yp (0.250)** — paradoxically,
removing the projection that was supposed to "fix" V_nunu actually
restores active-flavor thermalisation. The natural next experiment
is the **joint configuration** `(active_only=False,
damping='symmetric')` which combines the SM-Neff-restoring V_nunu
mode with the HTT-matching damping kernel; if it trips into [0.02,
0.10], Stage E.2 closes with a multi-axis cure. Sprint 18 also
runs a Phase-0/Phase-B handoff diagnostic to localise the
sprint-12 "Tg ≈ 60-64 MeV resonance jump", which HTT 2012 says
should not exist for L=0 NH and may be a numerical artefact of
the temperature-segment driver.

## What to read, in order

Budget ~45 min before writing any code or starting a new
experiment.

1. **`CLAUDE.md`** — project overview.
2. **`doc/STAGE_E2_SPRINT17_LITERATURE.md`** — HTT 2012 reference
   table, unit-convention audit (§1), and FortEPiaNO scope verdict.
3. **`doc/ROADMAP.md`** — the sprint-17 landing record at the
   bottom has the three-run axis bracket data, the unit-mismatch
   structural finding, and the V_nunu paradox (Run B near-SM
   Neff/Yp). Skim sprints 14-16 (driver-side falsifications),
   sprint 13 (Suspect 6 falsified), sprint 5 (the V_nunu
   active-only projection introduction).
4. **`doc/STAGE_E2_SPRINT17_BRIEF.md`** — for the original
   "physics-config divergence" framing and the candidate axis
   ladder (V_nunu, damping, mixing-angle, IC, resonance).
5. **`validation/diagnostics/diag_sprint17_axis_v_nunu.py`** —
   reference harness; sprint 18 mirrors the structure for the
   joint bracket.
6. **`validation/diagnostics/diag_sprint17_axis_v_nunu.out`** —
   numerical baselines for sprint 18's reproducibility check.
7. **`PRyM/PRyM_boltzmann.py`**:
   * `_build_H_list` (line 4333+) and the `qke_v_nunu_active_only`
     projection (lines 4368-4373) — the axis the V_nunu bracket
     toggles.
   * `_compute_D_pair_matrix` (line 3322+) — the damping-kernel
     dispatcher selectable as `symmetric`/`mirizzi`/`gariazzo`;
     "symmetric" is the HTT 2012 Eq. 2.15-16 form, "mirizzi" is
     our default.
8. **`PRyM/PRyM_main.py`**: lines 213-265 set up the Phase A → 0
   → B segment chain. The Phase-0 driver runs from
   T_phase0_start=100 MeV to T_boltz_start=30 MeV; Phase B from
   T_boltz_start to T_boltz_end=0.005 MeV. The sprint-12 jump at
   Tg ≈ 60-64 MeV is **inside Phase B** (T_boltz_start=30 MeV
   means the segment runs from 30 MeV down) — wait, it's actually
   **inside Phase 0** (100 MeV → 30 MeV). Verify which segment
   contains 60-64 MeV before writing the diagnostic.

## What's broken (the framing inherited from sprint 17)

* δN_eff_ss at project default (active_only=True, mirizzi) is
  0.629 — **6× above HTT 2012 band edge 0.10** for sin²(2θ)=1e-4,
  Δm²=0.93 eV². This is the raw "physics-config divergence"
  Suspect 8 from sprint 16, now reframed in HTT-comparable units.
* Single-axis flips reduce or change the gap but neither alone
  closes it:
  - V_nunu axis to legacy (`active_only=False`): δN_eff_ss → 0.963
    (worse), but Neff → 3.09 (near-SM), Yp → 0.250 (near-SM).
  - Damping axis to HTT-matching (`symmetric`): δN_eff_ss → 0.331
    (better, half-way to band), Neff → 4.10, Yp → 0.261.
* Sprint-12 localised an ETDRK2 jump at Tg ≈ 60-64 MeV that was
  framed as a "resonance crossing"; HTT 2012 Appendix A says
  no resonance exists for L=0 NH. The jump may be a Phase-0/Phase-B
  numerical handoff artefact rather than physical.
* The active-only V_nunu projection (sprint-5 fix) appears to
  drive the active sector OUT of thermalisation rather than into
  it. The mechanism is unclear and not in any sprint-5 design
  document. Worth a closer look.

## The joint-axis bracket experiment

The sprint-17 axis bracket data:

| Run | active_only | damping | δN_eff_ss | Neff | Yp |
|---|---|---|---|---|---|
| A | True  | mirizzi  | 0.629 | 9.85 | 0.313 |
| B | False | mirizzi  | 0.963 | 3.09 | 0.250 |
| C | True  | symmetric | 0.331 | 4.10 | 0.261 |

The two single-axis brackets each give a δN_eff_ss / Neff / Yp
delta. The joint configuration (`active_only=False, damping=
'symmetric'`) tests whether these deltas compose:
  - Linear-superposition guess: δN_eff_ss(joint) ≈ A + (B-A) +
    (C-A) ≈ 0.629 + 0.334 + (-0.298) = 0.665. Wrong direction
    if linear; would not close the gap.
  - Multiplicative guess: δN_eff_ss(joint) ≈ A · (B/A) · (C/A) ≈
    0.629 · 1.531 · 0.527 = 0.508. Closer but still above band.
  - Non-linear: Run B's V_nunu legacy mode could plausibly
    interact with Run C's symmetric damping in the resonance
    region to give either much smaller or much larger δN_eff_ss —
    only the experiment determines which.

The joint case is also physically the most HTT-2012-consistent
configuration we can produce with existing flags: HTT uses
symmetric damping AND no live ν-ν integral (their V_nunu is
closed-form thermal `V_1`), and their L=0 NH case has no
asymmetry (xi_*_init=0, which we already match).

## Recommended sequence

Sprint 18 should be a two-experiment sprint: the joint axis
bracket for the Suspect-8 verdict (~25 min), and a Phase-0/Phase-B
handoff diagnostic for the sprint-12 jump (~30 min instrumented
single-run). Total: 1-2 days.

### Phase 1 — Joint axis bracket (~30 min, one new harness)

Write a new ETDRK2 harness:
```
validation/diagnostics/diag_sprint18_joint_axis.py
```

Single run at reduced n_B = 2500 + n_B_phase0 = 1000 (apples-to-
apples with sprint-17 Run A baseline 14.5122 raw and 0.629
δN_eff_ss). Configuration:
- `qke_v_nunu_active_only = False` (legacy V_nunu projection)
- `qke_damping_formula = "symmetric"` (HTT 2012 damping)
- All other flags identical to `diag_sprint17_axis_v_nunu.py`
  Run C / Run B (which are themselves identical except for the
  axis variables).

Output: raw Σρ_ss, δN_eff_ss, Neff, Yp, wall-clock (~25 min
expected from the sprint-17 timings of 23-26 min per run).

Decision tree:
- δN_eff_ss in [0.02, 0.10] → **Stage E.2 closes** with multi-axis
  cure. Document the HTT 2012-matching configuration in
  `PRyM_init.py` and propose default flips for Hannestad-style
  benchmarks.
- δN_eff_ss in [0.10, 0.30] → close, but not in band. Investigate
  the momentum-grid axis (Ny=200 or 300, with the Kainulainen-Sorri
  non-uniform mapping) before declaring Suspect 8 unresolved.
- δN_eff_ss > 0.30 → axes do not compose toward the band; either
  V_nunu has a non-linear interaction with damping that we have
  not understood, or there is a third axis (likely the
  `Phase-0/Phase-B handoff` per Phase 2 below) that is the actual
  divergence. Escalate to sprint 19.
- δN_eff_ss < 0.02 → over-shoots in the right direction; the joint
  case is more aggressive than HTT 2012; isolate via single-axis
  re-confirmation runs.

### Phase 2 — Phase-0/Phase-B handoff diagnostic (~45 min)

HTT 2012 Appendix A: no resonance for L=0 NH. Sprint-12 localised
an ETDRK2 jump at Tg ≈ 60-64 MeV. Three possibilities:
1. The jump is a physical resonance from a sub-leading effect HTT
   doesn't have (e.g. our live-integral V_nunu).
2. The jump is a numerical artefact of the Phase-0 → Phase-B
   handoff at T = 30 MeV (or the Phase A → Phase 0 handoff at
   T = 100 MeV, depending on sub-segment localisation).
3. The jump is internal-to-Phase-0 from grid stiffness around the
   Kainulainen-Sorri non-resonant peak T_max ≈ 9.9 MeV — but 60
   MeV is far from 9.9 MeV, so this is unlikely.

Diagnostic harness (extends `diag_sprint17_axis_v_nunu.py`):
* Re-run Run A configuration with sprint-12 instrumentation
  (`qke_energy_diag_flag=True`, `qke_msw_diag_flag=True`) to
  capture per-step rho_aa, rho_ss, H_aa, H_ss, H_as in the
  Tg ∈ [50, 70] MeV window.
* Run a second pass with `T_phase0_start = 70 MeV` and
  `T_boltz_start = 50 MeV` (shrink the Phase-0 segment so the
  60-64 MeV jump sits at the segment boundary). Compare the jump
  shape between full-segment Phase 0 and the truncated Phase 0:
  if the jump moves with the segment boundary, it's a handoff
  artefact; if not, it's physical.
* (Optional) Re-run with `qke_phase0_flag=False` (skip Phase 0
  entirely and run Phase B all the way from T_phase0_start=100
  MeV down). If the jump disappears, Phase 0 is the suspect; if
  it persists, the jump is internal to Phase B's collision
  damping.

Output: a per-step diagnostic dump and a verdict on the jump's
origin. If the jump is a handoff artefact, sprint 18 may need to
also propose a single-pass driver (Phase 0 + Phase B fused into
one ETDRK2 segment) — that is a structural change worth flagging
as out-of-scope for sprint 18 but on the radar.

### Phase 3 — Verification gates

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default. |
| 2 | `pytest -k sterile -v` | 3/3 at default. |
| 3 | Sprint-18 joint bracket | δN_eff_ss verdict (see decision tree above). |
| 4 | Phase-0/B handoff diagnostic | Identify whether the sprint-12 jump is physical or a handoff artefact. |

### Decision tree at gate 3

| Outcome | Verdict | Action |
|---|---|---|
| δN_eff_ss ∈ [0.02, 0.10] | **Suspect 8 cured by joint configuration** | Document the joint config; default-flip both flags for Hannestad-style benchmarks; close Stage E.2. |
| δN_eff_ss ∈ [0.10, 0.30] | Joint configuration insufficient alone; close to band | Run the momentum-grid bracket (Ny axis) before sprint 19 escalation. |
| δN_eff_ss > 0.30 | Axes do not compose toward the band | Sprint-19 escalation: structural look at V_nunu live-integral vs HTT closed-form; possibly implement the closed-form V_1 as a third option. |
| δN_eff_ss < 0.02 | Joint configuration over-shoots; Stage E.2 ambiguous | Single-axis re-confirmation runs; possibly isolate which axis is the dominant cure. |

## Stage inheritance: what NOT to touch

* `evolve_step_ode_etdrk2`, `evolve_step_ode_etdrk4`,
  `evolve_step_lsoda` — production drivers, untouched.
* `_etdrk_expm_phi_4` (sprint 16) — kept as opt-in infrastructure.
* `_build_H_list`, `_build_L_list`, `_assemble_collision_N`,
  `_compute_D_pair_matrix` — kernels reused. Modifications here
  count as physics-config changes and must be flag-gated.
* `tests/test_regression.py` — must pass 6/6 fast bit-identical
  at default.
* `qke_v_nunu_active_only` and `qke_damping_formula` defaults —
  do **not** change in sprint 18 even if the joint case
  reproduces HTT. Default flips wait for a closure sprint where
  all four Hannestad benchmark points (A/B/C plus the L=10⁻²
  case) are re-run with the candidate defaults.

## Why not another single-axis sprint?

Sprint 17 ruled out single-axis cures for both V_nunu projection
and damping kernel. The bracket showed:

* V_nunu axis flip: δN_eff_ss moves the WRONG way (0.629 → 0.963)
  but Neff/Yp move toward SM. Whatever V_nunu does, it has at
  least two competing effects on the sterile.
* Damping axis flip: δN_eff_ss moves the right way (0.629 → 0.331)
  but only by a factor of 2 — not enough to close a 6× gap alone.

Composing the two is the next-cheapest experiment that has any
plausibility of closure. Higher-priority axes (mixing-angle
convention, IC, full HTT closed-form V_nunu) all involve more
implementation work than a single flag-flip; the joint bracket
is one new harness with no production code changes.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT18_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete sprint-18
> plan. Target for this session is to run the joint
> `(active_only=False, damping='symmetric')` configuration and
> identify whether the sprint-12 Tg ≈ 60-64 MeV jump is physical
> or a Phase-0/Phase-B handoff artefact.

## Commit chain for context

  * **Sprint 17** (this) — HTT 2012 reproduction; two material
    convention divergences identified (damping kernel,
    units-of-Σρ_ss); three-run axis bracket; both axes
    quantitatively sensitive but neither alone closes the gap;
    joint configuration is the surviving candidate.
  * **Sprint 16** — Krogstad ETDRK4 driver; Suspect 7 falsified.
  * `b8dd47a` — Sprint 15 (analytic Jacobian + segment-only
    LSODA dispatcher; LSODA wall-clock-infeasible at Point C).
  * `c6149dd` — Sprint 14 (LSODA driver; gate 5 wall-clock
    infeasible without analytic Jacobian).
  * `db511c5` — Sprint 13 (collision-N refactor falsified Suspect 6).
  * `e699c00` — Sprint 12 (sub-step instrumentation;
    H-iteration variants; localised the Tg ≈ 60-64 MeV jump).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver).
  * `ac1d521` — Sprint 9 (MSW diagnostic).
  * `bb0ea32` — Sprint 8 (Suspect 1 falsified).
  * Sprint 5 — V_nunu active-only projection introduced
    (`qke_v_nunu_active_only` flag, default False, opt-in True).
