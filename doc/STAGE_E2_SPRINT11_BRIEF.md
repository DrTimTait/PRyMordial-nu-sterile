# Stage E.2 sprint 11 brief: Point-C narrow-mixing over-adiabatisation at Phase-0 MSW crossing

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 10 (commit **`<sprint-10 hash>`**; will be set by the
sprint-10 landing). Read this first. By the end you should know
exactly which part of the ETDRK2 driver over-pumps the y=0.5
antineutrino MSW resonance at T ≈ 60 MeV for sin²2θ = 1e-4, the
proposed narrow-mixing fix candidates, and the path to closing
Stage E.2 with all three Hannestad points in band.

## One-paragraph orientation

Sprint 10 built a Phase-0 QKE driver that evolves the 4×4 density
matrix from `T_phase0_start = 100 MeV` down to `T_boltz_start = 30 MeV`
with adiabatic-vacuum IC, supplying a history-preserving
`ρ_all(30 MeV)` to Phase B. Gate-6 Hannestad result: Point A moved
from ΔNeff = +1.57 (miss-high) to 0.835 (miss-low by 0.065 — 87% of
the anomaly closed); Point B essentially unchanged; Point C
catastrophically regressed from ΔNeff = +0.06 (in-band) to +5.52
(way over-band, Σρ_ss = 22.22). Sprint 10 landed as scope-(b)-minus-
flip: Phase-0 driver committed as opt-in machinery, defaults preserved.

The sprint-10 post-landing probe
(`validation/diagnostics/diag_phase0_pointC.py`) ran Point C with
`qke_phase0_diag_flag=True`, recording `ρ_ss(istep, a, Tg, y)` after
each of the 2460 Phase-0 steps. Findings:

  * **Only 3% of the final Σρ_ss is deposited during Phase 0**
    (0.67 of 22.22). The other 97% is Phase-B collisional
    thermalisation of the Phase-0 seed.
  * The Phase-0 seed is localised almost entirely at **y = 0.5 in the
    antineutrino sector** (ρ_ss = 0.444; the next y-mode, y=1.5,
    carries only 0.02). Neutrino sector max = 0.168 (factor 2.6
    smaller).
  * Growth is two-staged. Monotone rise from 1e-5 → 0.17 over
    T = 100 → 65 MeV (istep 0 → 906), then a **step-function jump
    from 0.177 → 0.4145 between istep 906 (Tg = 64.1 MeV) and
    istep 1035 (Tg = 60.2 MeV)**. Factor-2.3 jump in one 130-step
    window, sector 1 only. After that, ρ_ss holds flat at ~0.415 for
    1400 steps until Phase-0 exit.
  * Phase B then amplifies the seed 33× (22.22 / 0.67) by DW
    collisional rates applied to the lowest-y modes over the
    3-decade Phase-B window.

**Suspect 2-residual is PROMOTED to prime.** This is sprint-9's
original MSW-passage over-pumping hypothesis resurfacing at narrow
mixing. Sprint 9 falsified it at Point A (large mixing) because the
bias signature ran opposite to prediction; Point A's problem was
Phase-A IC, which Phase 0 fixed. At Point C the eigenbasis over-
pumping finally shows its expected signature: spurious conversion at
the one y-mode whose MSW resonance lies inside the integration
window. Phase 0 surfaced this by moving a previously-silent
(pre-`T_boltz_start`) resonance into the integrated range.

Sprint 11's job: localise the T ≈ 62 MeV step-function jump to its
mechanism inside `_etdrk2_expm_phi` or the predictor-corrector
composition, land a narrow-mixing fix that doesn't regress Point A's
+0.835 closure, re-run gate 6, close Stage E.2.

## What to read, in order

Budget ~75 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 10". The
   sprint-10 landing record has Gate 9's A/B/C table, the MSW-probe
   summary, the Phase-0 three-mechanism ranking (seed at y=0.5, ν̄
   asymmetry, Phase-B collisional amplification), and the sprint-11
   handoff. Also skim sprint 9 (MSW diagnostic Suspect-2 falsification
   at Point A), sprint 8 (L-conservation guard), and Stage D.7.1
   (Strang-symmetric sequence).
3. **`git show <sprint-10 hash>`** — the sprint-10 landing commit.
4. **`validation/diagnostics/diag_phase0_pointC.out`** — the
   definitive probe output with the step-function jump at istep
   906→1035 (Tg 64→60 MeV) clearly visible in the 20-row history
   table. Focus on sector-1 (`max nubar`) which carries the
   signature.
5. **`validation/diagnostics/diag_phase0_pointC.py`** — the probe
   harness; the analysis logic (threshold crossings, top-10 y-modes,
   nu/nubar split) is what a sprint-11 deeper probe will extend.
6. **`PRyM/PRyM_boltzmann.py` lines 4593-4644** — `_etdrk2_expm_phi`.
   The Al-Mohy & Higham augmented-matrix expm that produces
   `(Phi0, Phi1, Phi2)` per y-mode. **Prime fix-site candidate.** The
   regulariser constant near eigenvalue collapse is the sprint-11
   first-edit target.
7. **`PRyM/PRyM_boltzmann.py` lines 4314-4383** — `_build_H_list`.
   The Hamiltonian assembly. For Point C (Δm² = 0.93, sin²2θ = 1e-4 in
   the μ channel) the (α=numu, s) sector's diagonal difference
   `H_αα − H_ss` vanishes at T_res(y). For y = 0.5, ν̄, the probe
   localises T_res between 64 and 60 MeV.
8. **`PRyM/PRyM_boltzmann.py` lines 4736-4805** — the ETDRK2
   predictor+corrector block inside `evolve_step_ode_etdrk2`. Where
   the eigenbasis propagator is applied. The sprint-11 fix, if
   localised to the driver, lands here or in `_etdrk2_expm_phi`.
9. **`validation/diagnostics/diag_2level_damped_energy.py`** — the
   sprint-8 N+E conservation regression guard. Any fix to
   `_build_L_list` or `_etdrk2_expm_phi` that breaks this test
   (|dN/N|, |dE/E| > 1e-8) is wrong.
10. **`validation/diagnostics/diag_phase0_decoupled.py`** — the
    sprint-10 gate-5 guard. Any sprint-11 fix that leaks ρ_ss > 1e-12
    in the fully decoupled configuration is wrong.

## What's probably broken (one suspect, one follow-up)

### Suspect 2-residual — ETDRK2 eigenbasis over-adiabatisation at narrow-mixing MSW passage (PRIME)

At Point C (sin²2θ = 1e-4), the y=0.5 antineutrino MSW resonance
`H_αα − H_ss = 0` is encountered during Phase 0 near T ≈ 62 MeV.
With such narrow mixing, Landau-Zener adiabaticity at the exact
crossing is

```
γ_LZ = |H_αs|² / (|dΔ/dt|)
     ~ (sin(2θ) · Δm²/2E)² / (V̇_thermal)
```

which for sin²2θ = 1e-4 and y = 0.5 should give γ_LZ ≪ 1 (strongly
non-adiabatic, tunneling through with tiny conversion). Physically
the mode should stay in its initial flavour eigenstate with ρ_ss at
most a few percent.

The probe shows ρ_ss jumps from 0.177 → 0.4145 in the 130 steps
between istep 906 (Tg=64.1 MeV) and istep 1035 (Tg=60.2 MeV). That's
a doubling inside a narrow Tg window, opposite to the physical
non-adiabatic expectation. The step-function shape is the tell:
smooth physics at narrow mixing produces smooth rho_ss(T); a sudden
doubling is a **regulariser-branch transition** inside the eigen-
decomposition.

**Candidate mechanism.** `_etdrk2_expm_phi` builds Phi_0, Phi_1,
Phi_2 via an Al-Mohy augmented-matrix trick. Near the MSW turning
point for narrow mixing, two eigenvalues of L (the Lindbladian)
become near-degenerate over an adiabatic width `Δλ ~ sin(2θ)·Δm²/2E`.
The augmented-matrix expm has an ill-conditioned regulariser there;
the constant that decides when to switch branches (linear expansion
vs. exact expm) may straddle the adiabatic width at Point C and
produce the observed step-function. This is the exact hypothesis
sprint 9 falsified at Point A (where the symptom was y-distribution
bias in the opposite direction) but which now re-surfaces at narrow
mixing with the **predicted** signature: over-deposition at the one
y-mode whose resonance lies inside the integration window.

**Primary diagnostic.** Extend
`validation/diagnostics/diag_phase0_pointC.py`:
  1. Log the eigenvalues of L per y-mode per step during Phase 0
     in the y = 0.5 antineutrino channel. Plot
     `|λ₁ − λ₂|(istep)` through the Tg = 64 → 60 MeV window. The
     adiabatic-width minimum should coincide with the ρ_ss jump.
  2. Switch `_etdrk2_expm_phi` to a direct `scipy.linalg.expm`
     fallback for the y = 0.5 antineutrino mode and re-run the
     probe. If the jump disappears, bug localised to the
     augmented-matrix regulariser.
  3. Tighten the regulariser tolerance (hard-coded constant inside
     `_etdrk2_expm_phi`) by 10×, rerun Point C. If the jump softens
     but persists, the constant is the mechanism but not the unique
     fix.

**Secondary diagnostic.** Check the sprint-9 non-Phase-0 MSW
diagnostic at Point C (no such run exists — add one by copying
`diag_msw_passage.py` and setting `theta_24 = arcsin(sqrt(1e-4))/2`).
If Point C without Phase 0 shows step-function features at the y=0.5
antineutrino resonance in Phase B (T_res within Phase B's 30→5 MeV
range), then the mechanism is not Phase-0-specific and the fix must
land in `_etdrk2_expm_phi` unconditionally. If Point C without Phase 0
is smooth, the fix can be Phase-0-only (degraded-adiabatic driver).

**Fix candidates.** In order of least-intrusive:

1. **Direct-expm fallback near eigenvalue collapse** (Stage D.7
   fix candidate #2 from sprint-9 brief). Guard: only trigger when
   `|λ_1 − λ_2| / max(|λ_1|, |λ_2|) < ε_cross` for some small ε_cross
   (start at 1e-3 and tune). Cost: higher per-mode for those y-modes
   only, negligible globally. This is the fastest path to closing.
2. **Regulariser-tolerance tightening** in `_etdrk2_expm_phi`.
   Only needed if (1) is insufficient. Global effect — must be
   validated against Point A's +0.835 (must not worsen).
3. **Phase-0-specific: force non-adiabatic passage at narrow
   mixing.** Crude fix: scale n_B_phase0 *down* (not up) by
   max(sin²2θ, 1e-4) so fewer steps = less adiabatic numerically.
   Physically unprincipled; skip unless 1 and 2 fail.
4. **Resonance-aware step controller.** Detect
   `|H_αα − H_ss| < threshold` per y-mode per step; use a different
   (cheaper) propagator locally. Architecturally correct but 1-2
   sessions of work.

**Likelihood**: **high**. The step-function signature in a quantity
that should be smooth under physical narrow-mixing dynamics is a
textbook numerical transition; the per-sector asymmetry
(ν̄ dominates, factor 2.6) is the MSW sign asymmetry (for Dm²_41 > 0
the antineutrino resonance lies at higher T); the y-mode
localisation (almost all at y = 0.5) is the LZ width narrowness at
small mixing. Three independent features pointing at the same
mechanism.

### Suspect 4 — Phase-B collisional amplification at narrow mixing (FOLLOW-UP)

Even if the Phase-0 seed at y=0.5 is driven to 0.04 (target) instead
of 0.44, Phase B's observed 33× amplification would push the final
to 1.3 instead of 22.2. Still out of band (target 0.04). The
collisional amplification may itself be over-integrating. Investigate
only if Suspect 2-residual closes the Phase-0 seed but the final
Σρ_ss is still above band.

## Stage inheritance: what NOT to touch

Sprint-10 additions (all retained):

- **`qke_phase0_flag` / `T_phase0_start` / `n_B_phase0_override` /
  `qke_phase0_diag_flag`** defaults (all opt-in; defaults all stay
  at sprint-10 landing values).
- **`PRyMclass._run_qke_segment` nested helper** in `PRyM_main.py`.
  Both Phase 0 and Phase B call it.
- **`PRyMclass._phase0_rho_final` / `_phase0_rho_ss_history`** expose
  points on the class for diagnostic harnesses.
- **`validation/diagnostics/diag_phase0_decoupled.py`** — the gate-5
  regression guard. Any sprint-11 fix must keep
  `max|ρ_ss| < 1e-12` in the fully decoupled configuration.
- **`validation/diagnostics/diag_phase0_pointC.py`** — the post-
  landing probe; extend it for sprint-11 diagnostics; do not delete.

From sprint 9's inherited list (all retained):

- Sprint-9 `_msw_snapshot` + `qke_msw_diag_flag` instrumentation.
- Sprint-8 `_energy_snapshot` + `qke_energy_diag_flag` +
  2-level L-conservation regression guard.
- `_F_stat_stable` upper clamp (sprint 7).
- NaN-safe sanitisation (sprint 6).
- V_nunu active-only projection (sprint 5).
- `_build_L_list`, `_build_H_list`, `_assemble_collision_N`,
  `_build_PMNS`, `_apply_unitary`, `_compute_D_pair_matrix` internals.
- D.7.1 Strang-symmetric sequence.
- `PRyMini.qke_damping_formula = "mirizzi"` default.
- `PRyMini.qke_v_nunu_active_only` default `False`.

## Scope options

- **(a) Suspect-2-residual diagnostic + localised fix at
  `_etdrk2_expm_phi`**: extend `diag_phase0_pointC.py` with per-step
  eigenvalue logging; identify the regulariser-branch transition;
  land fix candidate #1 (direct-expm fallback near eigenvalue
  collapse); re-run `diag_phase0_pointC.py` and verify the step-
  function jump disappears. No full gate-6 run. ~6-10 hours.
  Deliverable: Phase-0 Point-C seed < 0.04, gate-5 still PASS.
- **(b) (a) + full gate-6 re-run + conditional default flip**: on
  top of (a), re-run `diag_hannestad_proj_w30_nB10k.py` with
  `qke_phase0_flag=True` defaults. If A (~0.9), B (~0.5), C
  (~0.04-0.10) all in band, **close Stage E.2**: flip
  `qke_phase0_flag` default to True, update
  `test_sterile_dw_production` fixture with a config override
  forcing `qke_phase0_flag=False` for bit-identity (precedent from
  sprint 8/10). ~14-20 hours.
- **(c) (a) falsifies Suspect 2-residual**: escalate to Suspect 4
  (Phase-B collisional amplification). ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix, for scope (b))

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must pass bit-identical at default (Phase 0 off).
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 at default config (Phase 0 off).
3. **2-level damped Rabi** —
   `python validation/diagnostics/diag_2level_damped.py`. Ratio 1.000.
4. **2-level L conservation** —
   `python validation/diagnostics/diag_2level_damped_energy.py`. Both
   ratios within 1e-8.
5. **Phase-0 decoupled-sterile regression guard** —
   `python validation/diagnostics/diag_phase0_decoupled.py`.
   `max|ρ_ss| < 1e-12` (sprint-10 gate).
6. **Point-C seed probe** (sprint-11 addition) —
   `python validation/diagnostics/diag_phase0_pointC.py`.
   **Target: no step-function jump in max|ρ_ss|(istep) between
   istep 906 and 1035; Σρ_ss at Phase-0 exit < 0.1.**
7. **Sprint-5 5 MeV baseline** —
   `python validation/diagnostics/diag_vnunu_active_only.py`.
   Point C ΔNeff ≈ −0.012 bit-identical (Phase 0 off path).
8. **Sprint-6 w20 clean completion** —
   `python validation/diagnostics/diag_hannestad_proj_w20.py`. No
   crash; all Yp ∈ [0.24, 0.26].
9. **Hannestad gate 6 with Phase 0 on** —
   `python validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`.
   **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.1]**.
   Yp for all three ∈ [0.24, 0.26]. All three in-band closes Stage E.2.
10. If gate 9 hits: flip `qke_phase0_flag` default True in
    `PRyM_init.py`; update `test_sterile_dw_production` with a
    config override forcing `qke_phase0_flag=False` for bit-identity.
    Re-run gates 1-5.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT11_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete sprint-11
> plan. My target for this session is {one of a / b / c}. Do not
> touch the sprint-10 Phase-0 driver or its instrumentation, the
> sprint-9 MSW instrumentation, the sprint-8 energy instrumentation
> or 2-level energy regression guard, the sprint-7 `_F_stat_stable`
> clamp, the sprint-6 NaN-safe sanitisation, the sprint-5 V_nunu
> projection, or the D.7.1 Strang-symmetric composition. The primary
> suspect is ETDRK2 eigenbasis over-adiabatisation at the narrow-
> mixing MSW passage, specifically a regulariser-branch transition
> inside `_etdrk2_expm_phi` near eigenvalue collapse at T ≈ 62 MeV,
> y = 0.5, antineutrino sector. The sprint-10 probe localised the
> step-function jump to istep 906-1035 of Phase 0.

## Commit chain for context

- `<sprint-10 hash>` — **Sprint 10** (Phase-0 QKE driver; A partial
  closure +0.835, B unchanged, C regression +5.52; scope (b) minus
  default flip landed; Suspect 2-residual promoted at narrow mixing).
- `ac1d521` — Sprint 9 (MSW diagnostic; Suspect 2 falsified at
  Point A; Suspect 3 promoted). **Critical context** — Suspect 2's
  falsification was large-mixing-specific.
- `bb0ea32` — Sprint 8 (Suspect 1 falsified at machine precision).
- `0c28d8d` — Sprint 7 (cold-T NaN origin fix).
- `968c936` — Sprint 6 (NaN-safe sanitisation).
- `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
  dependency** — projection is what exposes the MSW anomaly.
- `ca589b0` — Sprint 2 (Phase B stabilisation; n_B auto-scale).
- `e2f41f6` — D.7.1 (Strang-symmetric sequence).
- `4d8ab2c` — D.7 (per-mode expm via Al-Mohy augmented matrix; the
  **`_etdrk2_expm_phi` that sprint 11 will edit**).

## Post-fix: downstream opportunities

If sprint 11 closes Suspect 2-residual and lands the default flip:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). ΔNeff ∈ [0.05, 0.2] per Gariazzo+2019.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   resonance-driven physics at non-zero lepton asymmetry.
3. **Representation-factor audit** — still parked from sprint 4.
4. **Point-A residual 0.065** (sprint-10 remainder) — if not closed
   by the Suspect 2-residual fix, investigate via the sprint-10
   probe technique at Point A Phase 0 (check for any ETDRK2
   regulariser-branch transitions in the Phase-A→Phase-B handoff
   y-modes).
5. **Fold sprint-8/9/10/11 per-step instrumentation into a unified
   telemetry facility** that future stages reuse.
