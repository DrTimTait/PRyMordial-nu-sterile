# Stage E.2 sprint 17 brief: literature-review reconstruction of a Hannestad 3+1 trajectory

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 16. Read this first. By the end you should know
why three driver-level sprints (14 LSODA, 15 LSODA hardening, 16
ETDRK4) cannot move Σρ_ss out of the [15, 22] saturation regime
toward Hannestad's band [0.02, 0.10] for sin²(2θ)=1e-4, Δm²=0.93
eV², and why the recommended path is a literature-review
reproduction of a published 3+1 trajectory side-by-side with our
QKE — to identify which physics ingredient (V_nunu treatment,
damping kernel, resonance condition, initial condition) diverges
between our setup and Hannestad's.

## One-paragraph orientation

Sprints 14-16 ran the falsifier ladder for Suspect 7 (ETDRK2
corrector order is over-pumping the resonance) to its end. Sprint
14 added an LSODA reference driver and discovered it is
wall-clock infeasible at Hannestad Point C; sprint 15 hardened
LSODA with an analytic Jacobian and a segment-only window and
showed the bottleneck is RHS-bound, not Jacobian-bound; sprint 16
built a Krogstad ETDRK4 corrector on top of ETDRK2's existing
kernels and ran Hannestad Point C through it. The ETDRK4 result
at reduced n_B (Σρ_ss=15.81) is only +9% above ETDRK2 at the
same n_B (Σρ_ss=14.51) — the order-2 → order-4 corrector upgrade
moves Σρ_ss by ~1.3 absolute, nowhere near the 200× gap to the
Hannestad band. Suspect 7 is **falsified**: the time integrator
is not the bottleneck. Suspect 8 (the physics-config diverges
from Hannestad's) is the surviving structural hypothesis. Sprint
17's job is to **reproduce a published 3+1 Hannestad-style
trajectory** in a controlled side-by-side comparison and identify
the divergence point in the physics model.

## What to read, in order

Budget ~60 min before writing any code or starting a literature
search.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — the sprint-16 landing record at the
   bottom has the structural Suspect-7 falsification (ETDRK4 vs
   ETDRK2 at apples-to-apples n_B) and the Suspect-8 promotion.
   Skim sprints 14-15 (LSODA wall-clock evidence) and sprint 13
   (Suspect 6 falsified).
3. **`doc/STAGE_E2_SPRINT16_BRIEF.md`** — for context on the
   ETDRK4 design and the Suspect 7 falsification framing.
4. **The Hannestad reference itself** — sin²(2θ)=1e-4, Δm²=0.93
   eV² Point C is a specific point in
   Hannestad-Tamborra-Tram (2012) "Thermalisation of light
   sterile neutrinos in the early universe" (arXiv:1204.5861)
   or one of its follow-ups. The first task of sprint 17 is to
   identify the exact paper + figure + table cell so the
   side-by-side comparison is unambiguous. Track this in
   `doc/STAGE_E2_SPRINT17_LITERATURE.md` (new file).
5. **`Sabti-BBN.pdf`** — sets the general distribution-function
   framework PRyMordial-nu inherits. Compare its V_nunu treatment
   to Hannestad's.
6. **`PRyM/PRyM_boltzmann.py`**:
   * `_build_H_list` (around line 4000) — builds the per-mode
     Hamiltonian H[s, i] = H_vacuum + V_thermal + V_nc + V_nunu.
     **The V_nunu integrand and projection convention is the most
     likely divergence point from Hannestad** — sprint 5 already
     fixed an active-only projection issue here, and the
     `qke_v_nunu_active_only` flag (line ~250 of PRyM_init.py)
     is opt-in.
   * `_compute_D_pair_matrix` and `qke_damping_formula` —
     pair-damping kernel selectable as `symmetric` (legacy),
     `mirizzi`, or `gariazzo`. Hannestad uses one specific form;
     identify which.
   * `_assemble_collision_N` — collision integrand. Active-active
     gain is computed via `_offdiag_collision_gain` /
     `_offdiag_collision_gain_massive`. Active-sterile pairs have
     zero SM gain by design (line ~4535) — verify this matches
     Hannestad's treatment of active-sterile coherence sourcing.
   * **Untouched in sprint 17**: `evolve_step_ode_etdrk2`,
     `evolve_step_ode_etdrk4`, `evolve_step_lsoda`, all sprint-12
     instrumentation. Sprint 17 is a physics-side investigation.
7. **`PRyM/PRyM_init.py`** — flag map. Pay particular attention to
   `qke_damping_formula`, `qke_damping_scale`, `qke_v_nc_scale`,
   `qke_v_thermal_scale`, `qke_v_nunu_scale`,
   `qke_v_nunu_active_only` — these are the knobs sprint 5
   sprung out for the V_nunu projection investigation, and they
   bracket the most likely divergence axis from Hannestad.

## What's broken (the framing inherited from sprint 16)

* ETDRK2 produces Σρ_ss=22.2 at Hannestad Point C (production n_B);
  ETDRK4 produces Σρ_ss=15.8 at reduced n_B (would extrapolate to
  ≈22 at production). Hannestad's reported band is [0.02, 0.10].
  The 200× gap is robust to:
  - corrector order (sprint 16: 2 vs 4 → 8% change)
  - corrector kind (sprints 13-15: ETDRK2 vs LSODA equivalence
    confirmed in benign regimes; LSODA at Point C wall-clock-
    infeasible but not for accuracy reasons)
  - n_B resolution within the tested range (sprint 16: 14.5 →
    22.2 as n_B goes from reduced to production)
  - collision-N refactor (sprint 13 falsified Suspect 6)
* This makes the physics-config the active suspect. Candidate
  divergence axes:
  1. **V_nunu integrand and projection** — Hannestad's reference
     defines V_nunu as `sqrt(2)·G_F · ∫ d³p/(2π)³ (ρ - ρ̄)`
     with the integrand restricted to the active sector.
     PRyMordial-nu's `qke_v_nunu_active_only` flag toggles this
     restriction. Verify which Hannestad uses and match.
  2. **Damping kernel form** — `qke_damping_formula` selects
     among `symmetric`, `mirizzi`, `gariazzo`; each gives a
     different `D_αβ = (Γ_α + Γ_β)/2` formula. Hannestad uses
     one specific form (likely `mirizzi`); cross-check.
  3. **Mixing-angle convention** — `theta_24 = arcsin(sqrt(1e-4))/2`
     in our harness. The factor of 2 vs 1 in the half-angle
     convention has been known to flip results by 2× in past
     sprints. Verify against Hannestad's `sin²(2θ_24) = 1e-4`
     literally.
  4. **Initial condition** — at T_phase0_start = 100 MeV, do we
     match Hannestad's "thermal active, vanishing sterile" IC,
     and is the active asymmetry initialised correctly under
     `xi_nue_init = 0.0` etc.?
  5. **Resonance crossing condition** — in the Shi-Fuller / MSW
     picture, the resonance is at `omega_res = sqrt(Δm² · cos2θ /
     (2 E_nu))`. Sprint-12 localised the resonance at istep
     906→1035 / Tg ≈ 60-64 MeV; verify Hannestad's reported
     resonance T agrees.

## Recommended sequence

Sprint 17 should be a literature-review-first sprint, NOT a
code-first sprint. The implementation cost depends entirely on
which divergence axis turns up.

### Phase 1 — Identify the exact reference (~4 hours)

Read Hannestad-Tamborra-Tram (1204.5861) and tabulate:
* The exact Δm², sin²(2θ), and α (mixing-flavor index) for
  Point C.
* The reported Σρ_ss / ΔNeff / Yp values, with the units (Σρ_ss
  in our convention is `dy · sum_y y² · ρ_ss(y)` per sector;
  Hannestad may use `ΔNeff = (Σρ_ss / 4) · (T_nu/T_γ)⁴` or
  something equivalent — verify).
* The integration window (T_start to T_end), the n_B equivalent,
  and the integrator they used.
* The V_nunu, damping, and mixing-angle conventions stated in
  the paper.

Record in `doc/STAGE_E2_SPRINT17_LITERATURE.md`. If Hannestad's
exact band [0.02, 0.10] turns out to be in different units than
our Σρ_ss = sum_y y²·ρ_ss, the entire "Suspect 8 vs band"
framing might collapse — that's a possible early closure.

### Phase 2 — Identify candidate FortEPiaNO baseline (~3 hours)

`validation/diagnostics/diag_fortepiano_*` already has some
FortEPiaNO comparison machinery from a sprint-15 prep commit
(see commits 98a5d05 and 141a7d0). Determine whether FortEPiaNO
provides a Hannestad Point C reference trajectory we can
sample against. If yes, use FortEPiaNO directly as the
side-by-side baseline.

### Phase 3 — Single-axis bracketing (~6 hours per axis)

For each candidate divergence axis from the list above, run a
controlled experiment with our QKE under the Hannestad-matched
setting. Log the resulting Σρ_ss and compare to:
- Hannestad's reported band.
- The original ETDRK2 Σρ_ss=22.2.

Bisect: if a single axis flip moves Σρ_ss from 22 to ~0.05,
that's the divergence cause; document and exit. If multiple
axes are needed, document the joint configuration.

### Phase 4 — Verification gates

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default. |
| 2 | `pytest -k sterile -v` | 3/3 at default. |
| 3 | Sprint-17 controlled axis run | At least one axis (or joint configuration) brings Σρ_ss into [0.02, 0.10] AT THE TEST POINT C. |
| 4 | Re-run Hannestad A/B/C with the matched config | A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10]. |

If gates 1-2 pass and gate 3 produces a single-axis cure, sprint
17 closes Stage E.2.

### Decision tree at gate 3

| Outcome | Verdict | Action |
|---|---|---|
| Single axis flip → Σρ_ss in [0.02, 0.10] | **Suspect 8 confirmed and cured** | Default-flip the matched flag; close Stage E.2. |
| Joint configuration → Σρ_ss in band | **Suspect 8 confirmed; multi-axis cure** | Document the joint configuration; default-flip; close Stage E.2. |
| No combination of physics-config flags brings Σρ_ss into band | **Suspect 8 mechanism unclear** | Stage E.2 escalation: contact Hannestad or use FortEPiaNO output as the ground truth and bisect against that. |
| Hannestad's band turns out to be in different units than our Σρ_ss | **Framing collapse** | Resolve units, recompute, possibly close Stage E.2 immediately. |

## Stage inheritance: what NOT to touch

* `evolve_step_ode_etdrk2`, `evolve_step_ode_etdrk4`,
  `evolve_step_lsoda` — production drivers, untouched.
  Sprint-17 is a physics-config investigation, not a driver
  experiment.
* `_etdrk_expm_phi_4` (sprint 16) — kept as opt-in infrastructure
  for any future ETDRK6 or Magnus integrator follow-up.
* `_build_H_list`, `_build_L_list`, `_assemble_collision_N`,
  `_compute_D_pair_matrix` — kernels reused as pure functions.
  Modifications here count as "physics-config changes" and
  should be flag-gated, not in-place.
* Sprint-13 collision-N refactor — Suspect 6 falsified.
* Sprint-15 LSODA infrastructure — opt-in.
* `tests/test_regression.py` — must pass 6/6 fast bit-identical
  at default.

## Why not another driver sprint?

Sprints 14-16 collectively ruled out the time integrator as the
Σρ_ss-saturation driver. The bracket spans:

* Multistep / Newton (LSODA, sprints 14-15): wall-clock-infeasible
  at Hannestad Point C; not the cure even when affordable.
* Order-2 exponential time-differencing (ETDRK2, baseline):
  Σρ_ss = 22 at production n_B.
* Order-4 exponential time-differencing (ETDRK4, sprint 16):
  Σρ_ss = 15.8 at reduced n_B (extrapolated ≈ 22 at production).

The 200× gap to Hannestad's band is robust across the entire
ladder. The bottleneck is structural to the physics setup, not
the integrator. Higher-order ETD (ETDRK6, Magnus) would, by the
sprint-16 evidence, give Σρ_ss within ~5-10% of ETDRK4 at the
same n_B — still on the wrong side of the gap.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT17_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete sprint-17
> plan. Target for this session is to identify the exact Hannestad
> reference, lay out the V_nunu / damping / mixing-angle / IC
> bracket, and run at least one controlled axis experiment.

## Commit chain for context

  * **Sprint 16** (this) — Krogstad ETDRK4 driver shipped;
    Suspect 7 falsified at apples-to-apples comparison;
    Suspect 8 promoted to surviving structural hypothesis.
  * `b8dd47a` — Sprint 15 (analytic Jacobian + segment-only
    LSODA dispatcher; LSODA wall-clock-infeasible at Point C).
  * `c6149dd` — Sprint 14 (LSODA driver; gate 5 wall-clock
    infeasible without analytic Jacobian).
  * `db511c5` — Sprint 13 (collision-N refactor falsified Suspect 6).
  * `e699c00` — Sprint 12 (sub-step instrumentation;
    H-iteration variants).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver).
  * `ac1d521` — Sprint 9 (MSW diagnostic).
  * `bb0ea32` — Sprint 8 (Suspect 1 falsified).
