# Stage E.2 sprint 8 brief: active-sterile energy-balance anomaly at extended-window projection

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 7 (commit `0c28d8d`). Read this first. By the end you
should know why Point A over-heats the sterile and Point B
under-produces at w30 projection even at n_B=10000 (where Point C
lands cleanly in the Hannestad band), what diagnostic shortlists the
active-sterile energy-conservation bug as the prime suspect, and the
path to closing Stage E.2 with a projection-default flip + conditional
n_B heuristic re-pin.

## One-paragraph orientation

Sprint 7 localised the cold-T NaN to `_F_stat_stable`'s upper clamp
(`_hi = 1.0 - 1.0e-20` rounds to exactly 1.0 in float64) and fixed it
with a one-constant change. That resolved the NaN at origin but
**did not resolve** the w30 over-sterilisation that sprint 6 flagged
as "Next structural candidate #2". Sprint 7's Phase 5 n_B convergence
scan refactored the over-sterilisation into two independent bugs:
(i) **n_B under-resolution at weak mixing** — Point C (sin²2θ=1e-4)
at n_B=10000 lands cleanly in the Hannestad [0.02, 0.1] band with
ΔNeff=+0.060 (from +4.212 at n_B=5031). Clean convergence. (ii) **an
active-sterile energy-balance anomaly** at moderate-to-strong
mixing — Point A (sin²2θ=1e-1) at n_B=10000 gives ΔNeff=+1.568 (was
+1.257 at n_B=5031; **worsens** with n_B), Point B (sin²2θ=2.26e-3)
gives ΔNeff=+0.060 (regardless of n_B; Hannestad target 0.50). At
both A and B, Σρ_ss ≈ 29 (fully thermal sterile by population) but
the energy content is wrong by large factors: the sterile is ~11%
hotter than T_ν at A (ΔNeff ≈ 1.11⁴ ≈ 1.52, close to observed 1.57)
and much colder at B. Sprint 8's job: diagnose and fix the
energy-balance bug, then land the conditional n_B re-pin that sprint
7 teed up, then flip the `qke_v_nunu_active_only` default to `True`
and close Stage E.2.

## What to read, in order

Budget ~60 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 7". The sprint-7
   landing record has the full Phase-5 table (n_B=5031 vs n_B=10000
   across A/B/C), the fix rationale, and the ranked next-candidate
   list. Also skim sprint 5 (V_nunu projection), sprint 6 (NaN
   sanitisation), and sprint 2 (n_B auto-scale) — sprint 8 builds
   on all three.
3. **`git show 0c28d8d`** — the sprint-7 landing commit. Message
   covers the probe finding, the `_F_stat_stable` clamp bug, and the
   Phase-5 convergence data table at resolution ready-to-cite.
4. **`validation/diagnostics/diag_hannestad_proj_w30_nB10k.out`** — the
   Phase 5 full suite at n_B=10000 w30 projection. The A/B/C entries
   are the primary data sprint 8 must explain.
5. **`validation/diagnostics/diag_pointC_nB_probe.out`** — the
   single-point n_B=10000 Point C probe. Shows the clean fix.
6. `PRyM/PRyM_boltzmann.py` lines **4314-4370** — `_build_H_list`.
   This is where V_nunu projection, matter potentials (V_NC,
   V_thermal), and vacuum mixing get assembled into H per y-mode. The
   sprint-5 projection is at lines ~4337-4340.
7. `PRyM/PRyM_boltzmann.py` lines **4371-4510** — `_assemble_collision_N`.
   Builds collision-integral RHS. For active-sterile pairs, line
   ~4502 sets `S_gain_si = 0.0` (sterile has no SM vertex); damping
   enters via `rhs_si = -D_pairs[p_idx] * rho_ab_stored` at line
   ~4504. Study the damping-only-on-off-diagonals convention.
8. `PRyM/PRyM_boltzmann.py` lines **3303-3400** —
   `_compute_D_pair_matrix`. The D_αβ formula (Mirizzi / Gariazzo /
   symmetric) that enters the damping rate. For active-sterile:
   `D_αs = 0.5 * (Γ_α + Γ_s) = 0.5 * Γ_α` (legacy symmetric), or
   Mirizzi pair-specific form.
9. `PRyM/PRyM_boltzmann.py` lines **4582-4800** — `evolve_step_ode_etdrk2`.
   The D.7.1 Strang-symmetric driver: ½-diag → predictor → corrector
   → ½-diag. Understand how collision RHS (from `_assemble_collision_N`)
   feeds the off-diagonal predictor, and how the diagonal half-diag
   ingests `I_total` (which has NO sterile-diagonal entry — sterile
   diagonal gains only through the unitary step inside the predictor
   via the off-diagonal commutator).
10. `validation/diagnostics/diag_2level_damped.py` — 2-level analytic
    reference with damping. Sprint 8 may want to extend this to check
    **energy conservation** in a single (α, s) pair under damping,
    since it's the simplest test of the damping-vs-commutator
    energy balance.

## What's probably broken (three suspects, ordered by likelihood)

### Suspect 1 — energy-balance bug in active-sterile off-diagonal damping (HIGH)

At Point A: sterile is fully populated (Σρ_ss=29.96 ≈ thermal) but
the Neff contribution is ~50% over target. That scales like
(T_sterile/T_active)⁴ = 1.11⁴ ≈ 1.52 — the sterile is about 11%
hotter in temperature. At Point B: sterile is also fully populated
(Σρ_ss=28.55 ≈ thermal) but Neff is 10× under target, equivalent to
a much colder sterile. Population is thermal in both cases; what
differs is the y-distribution of the sterile occupation.

Mechanism hypothesis: when the off-diagonal ρ_αs coherence is
damped, the corresponding active-state energy should land in ρ_ss
at the *resonance y-mode*. If the damping rate `D_αs` has the wrong
y-dependence relative to the Hamiltonian commutator that drives the
[ρ_αα ↔ ρ_ss] transfer, the energy gets deposited at different
y-modes than physics dictates. Adiabatic (A): resonance sweeps slow,
code over-deposits in high-y → hot sterile. Semi-adiabatic (B):
code under-deposits → cold sterile. Non-adiabatic (C): tiny mixing,
transfer is small and the distribution doesn't matter (agrees with
Hannestad).

**Diagnostic**: instrument `evolve_step_ode_etdrk2` to log
`∫y² ρ_αα(y) dy + ∫y² ρ_ss(y) dy` (sum of active and sterile
energy densities in the (α, s) 2×2 block) at every step, for Point
A. In a closed two-state system under H-evolution + damping, this
sum should be **conserved** (H is unitary, damping removes coherence
without changing diagonal energy). Drift indicates energy leak. Run
Point A at w30 n_B=10000 with the instrument on; the drift sign
tells us if we're gaining or losing. ~2-3 hours.

**Fix candidates** (contingent on drift direction):
- If drift is **positive** (sterile gains more than active loses):
  a double-counting of energy in the damped coherence → sterile
  pump. Likely in `_assemble_collision_N`'s handling of active-
  sterile pairs vs. active-active pairs.
- If drift is **negative** (total energy leaks out): damping is
  removing coherence without properly updating diagonals. Could be
  a missing "damping back-reaction" term on ρ_ss.
- If drift is **zero**: energy is conserved, so the y-distribution
  error is coming from the Hamiltonian commutator or the collision
  integral's y-weighting. Escalate to Suspect 2.

**Likelihood**: **high**. The Σρ_ss=thermal-but-wrong-Neff pattern
is very specifically an energy-weighted anomaly, and the
active-sterile damping is the only place where sterile interacts
with anything — if that's miswired, everything downstream follows.

### Suspect 2 — MSW-passage over-pumping at adiabatic mixing in ETDRK2 eigenbasis (MEDIUM)

If Suspect 1's diagnostic shows zero energy drift, the y-distribution
bug is in the unitary (Hamiltonian) part. The D.7.1 ETDRK2 driver
uses the eigenbasis of L = -i[H, ·] - D_off, which rotates with the
instantaneous in-medium mixing angle. Near the MSW resonance, the
mixing angle sweeps through π/4 rapidly, and the eigenbasis rotation
can over-resolve or under-resolve the resonance depending on how the
`_etdrk2_expm_phi` regulariser handles the eigenvalue gap collapse.

**Diagnostic**: at Point A, extract the instantaneous in-medium
mixing angle θ_m(T, y) for a representative y-mode and compare the
ETDRK2 trajectory to a direct-RK4 solution (no eigenbasis
acceleration). Difference at the resonance T = diagnostic of the
expm accuracy there. Cost: one hour plus integration harness.

**Fix candidate**: tighten the `_etdrk2_expm_phi` Al-Mohy regulariser
tolerance at the resonance, or fall back to direct-expm when the
mixing angle sweeps through π/4 fast. Sprint 5 flagged this as the
"secondary issue behind Hannestad A undershoot" and sprint 6
carried it forward.

**Likelihood**: **medium**. Plausible but not the most economical
explanation; Suspect 1 is simpler and more consistent with the data
(especially the fact that Σρ_ss is thermal — the resonance DID
convert, it just deposited wrong).

### Suspect 3 — Phase-A thermal-IC inadequacy at T_boltz_start ≫ T_MSW (LOW-MEDIUM)

PRyMordial's Phase A sets `ρ_active(T_boltz_start) = f_FD(E/T_ν)`
and `ρ_sterile = 0`. At T_boltz_start = 30 MeV, the true active
distribution has already seen some MSW pre-processing (if mixing
is large, the resonance crossing happens above 30 MeV for some
y-modes). PRyMordial skips that. FortEPiaNO evolves across 60 → 5
MeV, preserving the history.

For Point A (sin²2θ=1e-1, adiabatic), the resonance span in y and T
is broad; missing the 60 → 30 MeV segment could bias the initial
reservoir. Probably explains some of the discrepancy but not 50%.

**Diagnostic**: compare PRyMordial's Phase A → Phase B handoff state
at T_boltz_start to a pre-evolved Phase-0 state from an independent
reference (or from FortEPiaNO's intermediate output if obtainable).

**Fix candidate**: add a Phase-0 QKE driver that evolves from
T_ν-decoupling (~100 MeV) down to T_boltz_start, feeding Phase B
with the correctly-initialised state. Medium-size scope.

**Likelihood**: **low-medium**. Parked by sprint 4's informal
IC audit (agreement to ~1e-3 per y-mode). But sprint 4's agreement
was without projection; with projection on, the active-sterile
coherence history across 60 → 30 MeV may matter more.

## Stage inheritance: what NOT to touch

- **`_F_stat_stable` upper clamp** at PRyM_boltzmann.py:1060
  (sprint 7). The clamp is now `1.0 - 1.0e-15`; do not revert.
- **`evolve_step_ode_etdrk2` NaN-safe sanitisation** at
  PRyM_boltzmann.py:~4749 (off-diag) and ~4774 (diagonal
  `nan_to_num`-before-clip) (sprint 6). Now mathematical no-ops on
  finite state thanks to sprint 7; keep as defense in depth.
- **V_nunu active-only projection** at PRyM_boltzmann.py:4337-4340
  (sprint 5).
- **`DensityMatrixSolver._compute_D_pair_matrix`** internals
  (Stage E.1) — the `mirizzi` / `gariazzo` / `symmetric` switch is
  stable; sprint 4 ruled out damping magnitude/form as a driver.
- **`PRyMini.qke_damping_formula = "mirizzi"`** default.
- **`_apply_unitary`** (Stage D.3 ν-bar convention).
- **D.7.1 Strang-symmetric sequence** in `evolve_step_ode_etdrk2`
  (½-diag → predictor → corrector → ½-diag).
- **`_etdrk2_expm_phi`** mathematical form (Al-Mohy; validated via
  2-level damped Rabi test — must still pass).
- **Default of `qke_v_nunu_active_only`** (still `False` when this
  sprint starts; the flip to `True` is this sprint's *output*, not
  its input).
- **Any other flag default.**

## Scope options

- **(a) Suspect-1 diagnostic only**: instrument
  `evolve_step_ode_etdrk2` with a per-step energy accounting, run
  Point A at w30 n_B=10000, identify drift direction. Report back.
  No fix landed. ~4-6 hours. Outputs: landable-as-diagnostic
  `diag_energy_balance.py`, a ROADMAP entry, and a precise
  sprint-9 brief.
- **(b) (a) + fix**: close Suspect 1 at origin, re-run Hannestad
  A/B/C at w30 n_B=10000. If all three land in Hannestad bands,
  land the conditional n_B re-pin (k=4 → k=7 when
  `qke_v_nunu_active_only=True`), regenerate w20/w30 artifacts,
  flip the projection default, close Stage E.2. ~12-16 hours.
- **(c) Full Suspect-2 + Suspect-3 audit**: if (a) finds zero
  energy drift, escalate to MSW-passage adiabatic-eigenbasis audit
  and/or Phase-A thermal-IC evolution. ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix, for scope (b))

Same ladder as sprint 7, with gate 6 being the primary gate at
**n_B=10000**:

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must pass bit-identical at default config.
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 must pass at default config.
3. **2-level damped Rabi** —
   `python validation/diagnostics/diag_2level_damped.py`. Ratio 1.000.
4. **Sprint-5 5 MeV baseline** —
   `python validation/diagnostics/diag_vnunu_active_only.py`. Point C
   ΔNeff ≈ −0.012 bit-identical.
5. **Sprint-6 w20 clean completion** —
   `python validation/diagnostics/diag_hannestad_proj_w20.py`. No
   crash; all Yp ∈ [0.24, 0.26].
6. **Hannestad literature at w30 n_B=10000** —
   rebuild or rerun `diag_hannestad_proj_w30_nB10k.py` (transient
   from sprint 7 — resurrect from the `.out` file's configuration
   footprint). **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7],
   C ∈ [0.02, 0.1]**. Yp for all three points ∈ [0.24, 0.26]. No
   line-4755 warnings. All three in-band closes Stage E.2.
7. If gate 6 hits: land conditional n_B re-pin in `PRyM_init.py`
   (`k=7` branch guarded on `qke_v_nunu_active_only=True`,
   bit-identical at projection=False). Regenerate
   `diag_hannestad_proj_w20.out` with the new higher n_B; check
   `diag_hannestad_proj_w15.out` for regression.
8. Flip `qke_v_nunu_active_only` default to `True` in
   `PRyM_init.py`. Re-run gates 1-5. The sterile regression (gate
   2) will shift — `test_sterile_dw_production` expects
   `0.5 < dNeff < 1.1` at default (θ_14, Δm²=1, w5) which
   pre-projection gave 0.93. Post-projection at w5 it drops to
   ~0.17 (undershoot; w5 is below the MSW for Δm²=1). Either
   (a) update the test band to `0.1 < dNeff < 1.1`, or (b) lift
   the test's `T_boltz_start` to 30 MeV with `n_B_override=10000`
   so it exercises the full thermalisation regime. Prefer (b) —
   the test then measures the actual Hannestad-calibrated
   thermalisation rate, not the legacy V_nunu-saturated bug.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT8_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-8 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-5 V_nunu projection physics, the sprint-6
> NaN-safe sanitisation, the sprint-7 `_F_stat_stable` clamp, or the
> D.7.1 Strang-symmetric sequence. The primary suspect is an
> active-sterile energy-balance bug, not a numerical instability.

## Commit chain for context

- `0c28d8d` — **Sprint 7** (cold-T NaN origin fix; `_F_stat_stable`
  clamp). Sprint 8 is built directly on top — the starting state.
- `32230ff` — Sprint 7 brief (context doc for sprint 7).
- `968c936` — Sprint 6 (NaN-safe sanitisation).
- `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
  dependency** — the projection is what exposes the energy-balance
  bug by removing the V_nunu ballast that previously masked it.
- `66ba8ec` — Sprint 4 (damping magnitude ruled out).
- `ca589b0` — Sprint 2 (Phase B stabilisation: auto-scaled n_B,
  QED-table zero-clamp). **Critical dependency** — the n_B
  heuristic here is the one that sprint 8 will re-pin conditionally.

## Post-fix: downstream opportunities

If sprint 8 closes Suspect 1 and lands the default flip + n_B
re-pin:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). Target: ΔNeff ∈ [0.05, 0.2] per
   Gariazzo+2019. Should come for free after Suspect 1 closes.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   resonance-driven physics at non-zero lepton asymmetry. Sprint
   5-8's projection + energy-balance fixes should port over.
3. **Sibling clamp/sanitisation sites at `PRyM_boltzmann.py:3907`
   and `:4256`** — same NaN-silencing pattern as the sprint-6
   D.7.1 sanitisation, on non-default legacy paths. Maintenance
   pass: port the sanitisation so all three evolution paths are
   NaN-safe.
4. **Representation-factor audit** — still parked from sprint 4.
   Likely moot after sprint 8 if energy-balance closes.
5. **Phase A thermal-IC evolution** (Suspect 3) — close out if not
   addressed in sprint 8.
