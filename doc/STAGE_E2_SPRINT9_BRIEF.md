# Stage E.2 sprint 9 brief: MSW-passage adiabatic over-pumping audit

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 8 (commit **`<sprint-8 hash>`**; will be set by the
sprint-8 landing). Read this first. By the end you should know why
Point A over-heats the sterile at w30 projection n_B=10000 despite
the driver's trace accounting being correct at machine precision,
why the Point-B undershoot likely shares the same root cause, and
the path to closing Stage E.2 with a localized fix to the MSW
resonance handling in the ETDRK2 predictor.

## One-paragraph orientation

Sprint 7 localised a cold-T NaN at `_F_stat_stable`'s upper clamp
and fixed it with a one-constant change; the Phase-5 n_B scan left
two residual anomalies at w30 projection: Point A over-production
(ΔNeff = +1.57 vs. target 1.0) and Point B under-production
(ΔNeff = +0.06 vs. target 0.5). Σρ_ss is thermal (~29) at both —
population right, y-distribution wrong. Sprint 8 built a per-step
energy-accounting diagnostic and a surgical sub-step trace test;
both showed the driver preserves trace at machine precision (~2e-15
per step). The +50% growth in the (α, s) pair sum seen by the naive
diagnostic is bath-pumped physical thermalisation, not a numerical
leak. **Suspect 1 (active-sterile energy-balance bug) is
falsified.** The Point A dNeff anomaly must therefore be a
**y-distribution** effect: where the bath-pumped resonance deposits
energy, not the total amount of energy deposited. Sprint 9's job:
confirm the mechanism in Suspect 2 (MSW-passage adiabatic over-
pumping in ETDRK2 eigenbasis), fix it at origin, and close
Stage E.2.

## What to read, in order

Budget ~45 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 8". The
   sprint-8 landing record has the falsification of Suspect 1 with
   the full sub-step-decomposition table and surgical-test output,
   and the ranked next-candidate list with Suspect 2 promoted.
   Also skim sprint 7 (Phase-5 n_B scan), sprint 5 (V_nunu
   projection), and Stage D.7.1 (Strang-symmetric sequence).
3. **`git show <sprint-8 hash>`** — the sprint-8 landing commit.
   Covers the Phase-1 instrumentation, the (ρ_αα + ρ_ss)-drift
   diagnostic, the surgical per-step trace test, the ROADMAP
   falsification argument, and this brief.
4. **`validation/diagnostics/diag_energy_balance.out`** — the
   full-run drift summary for Point A at w30 projection n_B=10000
   with the refined verdict at the bottom.
5. **`validation/diagnostics/diag_energy_balance_decomp.out`** —
   the post-processed sub-step decomposition + surgical trace
   test. Shows predictor relative drift = 2.24e-15.
6. `PRyM/PRyM_boltzmann.py` lines **4593-4644** — `_etdrk2_expm_phi`.
   The Al-Mohy & Higham augmented-matrix expm that produces Phi0,
   Phi1, Phi2 per y-mode. Sprint-9's prime fix candidate site if the
   issue is propagator accuracy at resonance.
7. `PRyM/PRyM_boltzmann.py` lines **4314-4383** — `_build_H_list`.
   Where the Hamiltonian (vacuum + V_NC + V_CC + V_thermal + V_nunu)
   is assembled per y-mode. The MSW resonance is where the (α, s)
   sector's diagonal difference `H_αα − H_ss` vanishes. For Point A
   (Δm²=0.93, sin²2θ=1e-1 in the μ channel), this crosses during
   Phase B at a T- and y-dependent locus.
8. `PRyM/PRyM_boltzmann.py` lines **4526-4591** — `_build_L_list`.
   Where the per-mode L = −i[H, ·] − D_off_diag is built. Validated
   N+E-conserving per sub-block by sprint-8's 2-level unit test
   (`diag_2level_damped_energy.py`); do NOT touch.
9. `PRyM/PRyM_boltzmann.py` lines **4736-4805** — the ETDRK2
   predictor+corrector block inside `evolve_step_ode_etdrk2`. Where
   the eigenbasis propagator is applied to the off-diagonal
   coherence. Sprint-9's fix, if localized to the driver, will land
   here or in the `_etdrk2_expm_phi` helper it calls.
10. **`validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`** —
    gate-6 harness (resurrected by sprint 8). Use this for the
    sprint-9 fix validation.

## What's probably broken (two suspects, ordered)

### Suspect 2 — MSW-passage adiabatic over-pumping in the ETDRK2 eigenbasis (HIGH)

At Point A (sin²2θ=1e-1), the in-medium mixing angle θ_m(T, y)
sweeps through π/4 at the resonance temperature T_res(y). Above
resonance: θ_m ≈ 0 (flavor eigenstates ≈ mass eigenstates of the
medium). Below resonance: θ_m ≈ θ_vacuum (flavor reverts to mass
eigenstates of vacuum). The physical adiabatic conversion efficiency
is controlled by the sweep rate vs. the resonance width (Landau-
Zener): adiabatic = efficient conversion at the exact resonance
y-mode, non-adiabatic = most flux tunnels through without
converting.

At Point A, large mixing makes the sweep broadly adiabatic across
most y-modes. Full thermalisation is the correct result, and that's
what we see in the integrated Σρ_ss. But the y-distribution is 11%
hotter than thermal.

**Candidate mechanism.** The ETDRK2 predictor uses
`Phi0 = expm(L·dt)`, `Phi1 = dt·phi_1(L·dt)` built via the Al-Mohy
3N²-augmented matrix trick. Near the MSW turning point, L's
eigenvalues collapse (two flavor eigenvalues become degenerate over
an adiabatic width), and the augmented-matrix expm has an
ill-conditioned regularisation there. If the regulariser biases
toward a particular eigenvector in the collapse — say, toward high-y
modes where the L entries are larger (E_eV ∝ y/a) — then the
resonance sweeps over-deposits at high y and under-deposits at low
y. That's exactly the observed signature.

**Primary diagnostic.** At each time step of Point A, extract the
in-medium mixing angle θ_m per y-mode and flag the y-modes crossing
π/4 that step. Record the ETDRK2 propagator's action on a
delta-peaked ρ_αs at each resonance y-mode and compare to a direct
RK4 reference integrated at much smaller dt. Difference located in
y is the bug. Build `validation/diagnostics/diag_msw_passage.py` for
this.

**Secondary diagnostic.** Disable the ETDRK2 eigenbasis and fall
back to flavor-basis Strang-splitting (the pre-D.7 path) for a
single Point A run. If that reproduces sprint-6's flat Yp ≈ 0.299
pattern, the bug is in ETDRK2's eigenbasis handling. If it produces
different (but still wrong) Point A numbers, the bug is elsewhere.

**Fix candidates.**
- Tighten `_etdrk2_expm_phi` regulariser tolerance near eigenvalue
  collapse (scipy `expm` already uses Al-Mohy; the regulariser
  constant in `_etdrk2_expm_phi` may need to be y-mode adaptive).
- Fall back to direct `scipy.linalg.expm` (not the augmented-matrix
  trick) for y-modes where |λ_1 − λ_2| < threshold; accept the
  higher per-mode cost locally.
- At deeper scope: switch the D.7.1 Strang composition to use
  direct-expm of the full ETDRK2 operator when any y-mode is within
  an adiabatic-width fraction of resonance.

**Likelihood**: **high** given that the symptom (y-distribution bias
in a population that is otherwise thermal) is exactly what a
propagator-accuracy-at-resonance bug would produce.

### Suspect 3 — Phase-A thermal-IC inadequacy (LOW-MEDIUM)

Same as the sprint-8 brief's Suspect 3. PRyMordial's Phase A starts
with `ρ_active(T_boltz_start) = f_FD(E/T_ν)` and `ρ_sterile = 0` at
T=30 MeV. For Point A the MSW resonance for some y-modes happens
above 30 MeV; the exact IC is adiabatically pre-processed in reality
but treated as thermal here. That could bias the initial reservoir
at the high-y end (where resonance is first encountered).

Only investigate if Suspect 2 falsifies.

## Stage inheritance: what NOT to touch

Sprint-8 additions to the "do not touch" list (existing items from
sprint-8 brief retained):

- **Sprint-8 Phase-1 instrumentation** at `PRyM_boltzmann.py:4695-4744`
  (`_energy_snapshot` helper) and `:4796-4805`, `:4858-4862`
  (hook points). Flag-gated off at default; do not remove or inline.
- **`PRyMini.qke_energy_diag_flag` / `qke_energy_diag_path`** defaults
  (both opt-in; both stay at the sprint-8 defaults).
- **`PRyMclass._boltz_dm_solver`** expose at `PRyM_main.py:526` —
  the sprint-8 diagnostic pipeline uses this.
- **2-level energy-conservation unit test**
  (`diag_2level_damped_energy.py`). This is a *regression guard*:
  any fix to `_build_L_list` or `_etdrk2_expm_phi` that makes this
  test fail (|dN/N|, |dE/E| > 1e-8) is wrong.

From sprint 8's inherited list (all retained):

- `_F_stat_stable` upper clamp at `PRyM_boltzmann.py:1048` (sprint 7).
- NaN-safe sanitisation at `PRyM_boltzmann.py:4831` (isfinite check)
  and `:4857` (nan_to_num before clip) (sprint 6).
- V_nunu active-only projection at `PRyM_boltzmann.py:4351-4354`
  (sprint 5).
- `_build_L_list` internals (Stage E.1 + D.7 + sprint-8 guard).
- `PRyMini.qke_damping_formula = "mirizzi"` default.
- `_apply_unitary` (Stage D.3).
- D.7.1 Strang-symmetric sequence in `evolve_step_ode_etdrk2` —
  only the predictor/corrector substeps are candidates for sprint-9
  fix; the outer Strang composition stays.
- Default of `qke_v_nunu_active_only` (still `False`).
- Any other flag default.

## Scope options

- **(a) Suspect-2 diagnostic only**: build
  `validation/diagnostics/diag_msw_passage.py`, instrument per-step
  θ_m per y-mode, run Point A at w30 n_B=10000, identify which
  y-modes receive anomalous deposition. No fix landed. ~4-6 hours.
  Outputs: landable-as-diagnostic `.py` + `.out`, ROADMAP entry, and
  a precise sprint-10 brief.
- **(b) (a) + fix**: close Suspect 2 at origin (either via the
  `_etdrk2_expm_phi` regulariser tightening or via a conditional
  direct-expm fallback at near-resonance y-modes), re-run gate 6
  (A/B/C at w30 n_B=10000). If all three land in Hannestad bands,
  land the conditional n_B re-pin (k=4 → k=7 when
  `qke_v_nunu_active_only=True`), regenerate w20/w15 artifacts,
  flip the projection default, close Stage E.2. ~12-18 hours.
- **(c) Full Suspect-3 audit**: if (a) finds no resonance-localized
  anomaly, build a Phase-0 QKE driver from ~100 MeV → 30 MeV to
  supply a history-preserving IC to Phase B, re-run A. ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix, for scope (b))

Same ladder as sprint 8, with the new regression guard at gate 4:

1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
   4/4 must pass bit-identical at default config.
2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
   3/3 must pass at default config (at pre-flip settings).
3. **2-level damped Rabi** —
   `python validation/diagnostics/diag_2level_damped.py`. Ratio 1.000.
4. **2-level L conservation** (sprint-8 addition) —
   `python validation/diagnostics/diag_2level_damped_energy.py`. Both
   ratios within 1e-8.
5. **Sprint-5 5 MeV baseline** —
   `python validation/diagnostics/diag_vnunu_active_only.py`. Point C
   ΔNeff ≈ −0.012 bit-identical.
6. **Sprint-6 w20 clean completion** —
   `python validation/diagnostics/diag_hannestad_proj_w20.py`. No
   crash; all Yp ∈ [0.24, 0.26].
7. **Hannestad literature at w30 n_B=10000** —
   `python validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`.
   **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.1]**.
   Yp for all three points ∈ [0.24, 0.26]. All three in-band closes
   Stage E.2.
8. If gate 7 hits: land conditional n_B re-pin in `PRyM_init.py`
   (`k=7` branch guarded on `qke_v_nunu_active_only=True`,
   bit-identical at projection=False). Regenerate
   `diag_hannestad_proj_w20.out` with the new higher n_B; check
   `diag_hannestad_proj_w15.out` for regression.
9. Flip `qke_v_nunu_active_only` default to `True` in
   `PRyM_init.py`. Re-run gates 1-6. Sterile regression (gate 2)
   will shift — `test_sterile_dw_production` update per sprint-8
   brief's option (b): lift `T_boltz_start = 30` and
   `n_B_override = 10000` inside the test body.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT9_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-9 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-8 instrumentation, the sprint-8 2-level
> energy regression guard, the sprint-7 `_F_stat_stable` clamp, the
> sprint-6 NaN-safe sanitisation, the sprint-5 V_nunu projection, or
> the D.7.1 Strang-symmetric composition. The primary suspect is
> MSW-passage adiabatic over-pumping in the ETDRK2 eigenbasis; sprint
> 8 falsified the active-sterile energy-balance candidate at machine
> precision.

## Commit chain for context

- `<sprint-8 hash>` — **Sprint 8** (Suspect-1 falsified at machine
  precision; per-step trace test; diagnostic scaffolding landed).
- `0c28d8d` — Sprint 7 (cold-T NaN origin fix).
- `968c936` — Sprint 6 (NaN-safe sanitisation).
- `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
  dependency** — projection is what exposes the MSW anomaly.
- `ca589b0` — Sprint 2 (Phase B stabilisation; n_B auto-scale).
- `e2f41f6` — D.7.1 (Strang-symmetric sequence; base of ETDRK2 path).
- `4d8ab2c` — D.7 (per-mode expm via Al-Mohy augmented matrix; the
  `_etdrk2_expm_phi` whose regulariser is sprint-9's prime suspect
  site).

## Post-fix: downstream opportunities

If sprint 9 closes Suspect 2 and lands the default flip + n_B
re-pin:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). Target: ΔNeff ∈ [0.05, 0.2] per
   Gariazzo+2019. Should come for free.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   resonance-driven physics at non-zero lepton asymmetry.
3. **Representation-factor audit** — still parked from sprint 4.
   Likely moot after sprint 9 if MSW closes.
4. **Phase-A thermal-IC evolution** (Suspect 3) — close out if not
   addressed in sprint 9.
5. **Fold `_energy_snapshot` opt-in instrumentation into a small
   per-step telemetry facility** that future stages can reuse.
