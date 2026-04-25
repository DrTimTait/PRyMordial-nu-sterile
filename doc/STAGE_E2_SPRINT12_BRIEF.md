# Stage E.2 sprint 12 brief: Phase-0 step-function source localisation

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 11 (commit `<sprint-11 hash>`; will be set by the
sprint-11 landing). Read this first. By the end you should know why
Phase-0 Point-C still produces a 0.177 → 0.4145 step-function jump
in ρ_ss(y=0.5, ν̄) between istep 906 and 1035 even with sprint-11's
eigendecomposition fallback engaged, what diagnostic surfaces will
localise the mechanism to a sub-step inside `evolve_step_ode_etdrk2`,
and the path to closing Stage E.2 with all three Hannestad points
in band.

## One-paragraph orientation

Sprint 11 landed an opt-in eigendecomposition fallback inside
`_etdrk2_expm_phi` (PRyM_boltzmann.py:4605-4720), gated on small
active-sterile commutator gap AND non-zero off-diagonal H_α,sterile
coupling. The fallback fires correctly: 740 times across Point-C's
12500-step Phase-0+B run, zero κ-guard reverts. Phase-B over-
amplification is fully suppressed — final Σρ_ss drops from 22.22 to
under 1, Neff from 8.52 to **3.0114**, Yp from 0.272 to 0.2485 (in
the [0.24, 0.26] band). But the Phase-0 dynamics are essentially
unchanged: Σρ_ss(Phase-0 exit) = 0.6673 vs sprint-10 baseline
0.6656, and the istep 906→1035 step-function in ρ_ss(y=0.5, ν̄) is
preserved bit-for-bit at 0.177 → 0.4145. Replacing scipy.expm with
np.linalg.eig + per-eigenvalue phi_k reconstruction at exactly that
mode produces equivalent (Phi0, Phi1, Phi2) — the mechanism is
**downstream of `_etdrk2_expm_phi`**, somewhere in the predictor-
corrector composition, the half-diagonal exp-Euler regularisation,
the V_nunu mean-field feedback, or the collisional N_gain term.
Sprint 12's job: localise that mechanism, land a fix, close
Stage E.2 with Hannestad A/B/C all in band.

## What to read, in order

Budget ~60 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 11". The
   full sprint-11 landing record has the gate-by-gate validation
   table, the mechanism reinterpretation, and the Suspect 5
   handoff. Also skim sprint 10 (Phase-0 driver landed), sprint 9
   (MSW Suspect 2 falsified at Point A), sprint 8 (L-conservation
   guard), sprint 5 (V_nunu projection introduced), and the D.7.1
   Strang-symmetric composition.
3. **`git show <sprint-11 hash>`** — the sprint-11 landing commit.
4. **`validation/diagnostics/diag_phase0_pointC_fallback.out`** —
   sprint-11 result with fallback ON. The 20-row Phase-0 history
   table shows the istep 906→1035 jump preserved at 0.1771 →
   0.4145 (sector 1, max nubar). Final Σρ_ss = 0.6673 vs sprint-
   10 baseline 0.6656.
5. **`validation/diagnostics/diag_phase0_pointC.out`** — sprint-10
   baseline at flag off. Same step-function signature. The two
   files diff only in the last digit of Σρ_ss across the full
   history.
6. `PRyM/PRyM_boltzmann.py` lines **4870-5010**:
   `evolve_step_ode_etdrk2`. The Strang-symmetric composition
   (½-diag → predictor → corrector → ½-diag) where Suspect 5 must
   live since `_etdrk2_expm_phi` itself is now ruled out.
   Particularly:
     * **lines 4918-4924** — half-diag #1 advances the diagonal
       populations using `phi_half · I_total(rho_n)`.
     * **lines 4928-4939** — predictor: `rho_star = Phi0 · rho +
       Phi1 · N_off` per sector, then Hermitian symmetrisation
       (line 4938).
     * **lines 4942-4952** — corrector: `rho_new = rho_star + Phi2
       · (N_off_star − N_off_n)`, then Hermitian symmetrisation
       (line 4951).
     * **lines 4958-4995** — half-diag #2 uses the recomputed
       I_total at rho_after_corrector and includes the off-diagonal
       magnitude clamp.
7. `PRyM/PRyM_boltzmann.py::_build_H_list` lines **4326-4400**:
   the V_nunu source. Sprint-11's gate confirms H_α,sterile is
   non-zero at Point C; the *magnitude* of V_nunu in the active-
   sterile entries is the question. As ρ_ss grows during Phase 0,
   ρ − ρ̄ grows, V_nunu grows, the off-diagonal H_α,sterile shifts.
   This is non-linear feedback: small perturbation at istep 905
   could blow up at istep 1035.
8. `PRyM/PRyM_boltzmann.py::_assemble_collision_N` (search for the
   def): the N_gain collisional source. The `D_off · ρ_ab` add-
   back in `_build_L_list:4596-4601` cancels the −D·ρ piece in
   `_assemble_collision_N`. If the cancellation has a subtle
   numerical leak at narrow mixing (where ρ_α,sterile is many
   orders of magnitude smaller than ρ_α,α), the leak compounds.

## What's probably broken (one new suspect, three sub-suspects)

### Suspect 5 — Phase-0 step-function source downstream of `_etdrk2_expm_phi` (PRIME)

Sprint-11's eigendecomposition fallback fires at the y=0.5 ν̄ mode
during the istep 906→1035 window (Suspect 2 mechanism), but the
ρ_ss trajectory is bit-for-bit identical to Al-Mohy. Whatever
drives the 0.177 → 0.4145 jump is therefore not in the per-mode
matrix exponential — it is in the surrounding ETDRK2 driver code.

The step-function shape (sharp jump in 130 steps, then flat for
1400 steps until Phase-0 exit) is the tell: a smooth physical
process gives a smooth ρ_ss(istep). A discontinuity suggests
either a *single-step* bifurcation that locks in a new equilibrium,
or a *threshold-crossing* in the non-linear V_nunu feedback that
saturates rapidly.

#### Sub-suspect 5a — V_nunu non-linear feedback (HIGH likelihood)

V_nunu in `_build_H_list` is built from `(ρ − ρ̄)`. With the
sprint-5 active-only projection, V_nunu's sterile row/column are
zeroed, but its active 3×3 block changes whenever `ρ` shifts.

At istep 906, ρ_ss(y=0.5, ν̄) = 0.177; ρ_ss(y=0.5, ν) = 0.167.
At istep 1035, ρ_ss(y=0.5, ν̄) = 0.4145. The ν/ν̄ asymmetry in
ρ_ss grows from |0.177−0.167| = 0.01 to |0.4145−0.168| = 0.247.
That is a 25× growth in the (ρ − ρ̄) at one y-mode in 130 steps.

V_nunu is an integral over the y-grid weighted by y² (or similar).
A 25× growth at y=0.5 alone may not dominate the integral, but
it changes the *direction* of the V_nunu term in ν vs ν̄ sectors.
That feedback shifts H_α,α relative to H_ss in the active block,
which can either accelerate or suppress further conversion.

**Diagnostic.** Extend `diag_phase0_pointC_fallback.py` to log,
per Phase-0 step in the istep 880→1080 window:
  * V_nunu_active_block(y=0.5) magnitudes, both sectors.
  * (ρ − ρ̄)(y=0.5) vector component-wise.
  * H_α,α(y=0.5) − H_ss(y=0.5) per α, both sectors.
  * H_α,sterile(y=0.5) per α, both sectors.

If V_nunu(y=0.5, ν) and V_nunu(y=0.5, ν̄) show a step-function
crossover at istep ~1000, the feedback is the mechanism.

**Fix candidate.** Iterate H over the predictor-corrector cycle:
re-evaluate H at rho_star (after the predictor) before applying
Phi2 in the corrector. This converts the frozen-coefficient
ETDRK2 to a fully implicit step at narrow mixing, smoothing the
non-linear feedback. Cost: one extra `_build_H_list` call per
step. ~1 ms/step × 12500 steps = 12 s overhead.

#### Sub-suspect 5b — N_gain D·ρ cancellation at narrow mixing (MEDIUM)

`_build_L_list:4591-4601` adds `D_off · ρ_ab` back to N_gain to
cancel the `−D·ρ` damping in `_assemble_collision_N`. The
cancellation is exact at the level of the L-N pair, but the
arithmetic order matters when ρ_α,sterile is 10−10−10−1 of
ρ_α,α. At the MSW pass for narrow mixing, ρ_α,sterile is changing
rapidly while D_off is stiff — a small relative error in the
cancellation could feed a step-function via the predictor (which
sees the residual through `Phi1 · N_off`).

**Diagnostic.** Per Phase-0 step in the istep 880→1080 window,
log:
  * N_off[y=0.5, α=1, sterile] before and after the D·ρ add-back.
  * Phi1 · N_off contribution to rho_star at the y=0.5 ν̄ mode.
  * Same for the corrector's `Phi2 · (N_off_star − N_off_n)`.

If the D·ρ residual jumps at istep ~1000, sub-suspect 5b is
likely.

**Fix candidate.** Reformulate `_assemble_collision_N` and
`_build_L_list` to never produce the `−D·ρ` piece in the first
place — move the off-diagonal damping into L only, and have
N_gain be a pure gain term. This removes the cancellation
entirely. Larger refactor scope (~½ session).

#### Sub-suspect 5c — Half-diagonal phi_half regulariser at stiff damping (LOW)

`evolve_step_ode_etdrk2:4900-4913` computes phi_half =
phi_1(Γ_α · dt/2) · (dt/2) for the diagonal exp-Euler step. At
narrow mixing the diagonal damping rates are similar to non-
narrow cases — Γ_α scales with thermal potential, not with mixing
angle — so the regulariser should not behave differently at Point
C vs Point A. But the *threshold* `z_h < 1e-4` that switches
between Taylor and (1 − e^−z)/z formulae is hardcoded; if the
threshold straddles a particular y-mode at Point C, it could
flicker between formulae step-to-step and produce a step-function
in the diagonal increment.

**Diagnostic.** Log z_h at the y=0.5 mode per Phase-0 step. If
z_h crosses 1e-4 around istep 1000, sub-suspect 5c is the
mechanism.

**Fix candidate.** Replace the hard threshold with a smoothed
selector (e.g. a Padé fraction that matches both limits to
machine precision); or tighten the threshold by 100× so the
flicker is below numerical noise.

**Likelihood**: low — phi_half has been stable since D.7.1, and
sprint-5/-6/-7/-8/-9/-10 baselines show no Point-A step-function
at the Phase-A handoff. But check it is cheap.

## Stage inheritance: what NOT to touch

Sprint-11 additions (all retained):

  * **`qke_expm_fallback_near_degeneracy` / `qke_expm_fallback_eps_cross`**
    defaults (both opt-in; defaults at False / 1e-3).
  * **`_etdrk2_expm_phi` H_sector kwarg + per-mode dispatch + κ-guard.**
  * **`_expm_fallback_eig_count` / `_expm_fallback_kappa_high_count`**
    counters on the solver instance.
  * **`validation/diagnostics/diag_phase0_pointC_fallback.py`** —
    extend it for sprint-12 sub-step instrumentation; do not delete.
  * **`validation/diagnostics/diag_phase0_decoupled_fallback.py`** —
    sprint-11 gate-5 guard. Any sprint-12 fix must keep
    `max|ρ_ss| < 1e-12` in the decoupled configuration.
  * **`validation/diagnostics/diag_vnunu_active_only_fallback.py`** —
    sprint-11 gate-7 reference (5 MeV window, fallback ON).

From sprint 10's inherited list (all retained):

  * Sprint-10 `_run_qke_segment` helper, `qke_phase0_flag` /
    `T_phase0_start` / `n_B_phase0_override` / `qke_phase0_diag_flag`
    defaults, `_phase0_rho_final` / `_phase0_rho_ss_history` exposers.
  * Sprint-9 `_msw_snapshot` + `qke_msw_diag_flag` instrumentation.
  * Sprint-8 `_energy_snapshot` + `qke_energy_diag_flag` +
    2-level L-conservation regression guard.
  * `_F_stat_stable` upper clamp (sprint 7).
  * NaN-safe sanitisation (sprint 6).
  * V_nunu active-only projection (sprint 5).
  * `_build_L_list`, `_build_H_list`, `_assemble_collision_N`,
    `_build_PMNS`, `_apply_unitary`, `_compute_D_pair_matrix`
    internals.
  * D.7.1 Strang-symmetric sequence (the *outer* composition shape;
    *what* each substep does is the sprint-12 investigation site).
  * `PRyMini.qke_damping_formula = "mirizzi"` default.
  * `PRyMini.qke_v_nunu_active_only` default `False`.

## Scope options

  * **(a) Sub-step diagnostic + localised fix** at whichever sub-
    suspect localises (5a / 5b / 5c). Extend
    `diag_phase0_pointC_fallback.py` with the sub-step logging
    described above; identify the bifurcation; land the fix
    candidate for the localised sub-suspect; re-run the probe and
    verify the step-function disappears. No full gate-6 run.
    ~10-14 hours. Deliverable: Phase-0 Σρ_ss at exit < 0.05;
    gate-5 still PASS.
  * **(b) (a) + full gate-6 + conditional default flip**: on top
    of (a), re-run `diag_phase0_pointC_fallback.py` (and a
    Hannestad A/B/C variant if needed) with the fix on. If A
    (~0.9-1.1), B (~0.3-0.7), C (~0.02-0.10) all in band, **close
    Stage E.2**: flip both `qke_phase0_flag` and
    `qke_expm_fallback_near_degeneracy` defaults to True; update
    `test_sterile_dw_production` fixture with config overrides
    forcing both False for bit-identity (precedent from sprint
    10/11). ~18-24 hours.
  * **(c) Full DLSODA path** (radical, sprint-11 brief option (c)
    carried over). Write a parallel driver via
    `scipy.integrate.solve_ivp` with `method='LSODA'` or `'BDF'`
    on the full vectorised state. Compare against the Strang-split
    driver in the default 5 MeV regime for correctness, then use
    it as the production driver for extended windows. ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix, for scope (b))

  1. **Fast tests** — `pytest tests/test_regression.py -m "not slow" -v`.
     4/4 must pass bit-identical at default config (both new flags
     off).
  2. **Sterile regression** — `pytest tests/test_regression.py -k sterile -v`.
     3/3 at default config.
  3. **2-level damped Rabi** —
     `python validation/diagnostics/diag_2level_damped.py`. Ratio
     1.000 at default.
  4. **2-level L conservation** —
     `python validation/diagnostics/diag_2level_damped_energy.py`.
     \|dN/N\|, \|dE/E\| within 1e-8 at default.
  5. **Phase-0 decoupled-sterile guard** —
     `python validation/diagnostics/diag_phase0_decoupled_fallback.py`.
     `max|ρ_ss| < 1e-12` (sprint-11 gate). Must hold under any
     sprint-12 fix.
  6. **Sprint-11 fallback Point-C probe** —
     `python validation/diagnostics/diag_phase0_pointC_fallback.py`.
     **Target: no step-function jump in max\|ρ_ss(istep)\| between
     istep 906-1035; Σρ_ss(Phase-0 exit) < 0.05.**
  7. **5 MeV V_nunu projection** —
     `python validation/diagnostics/diag_vnunu_active_only_fallback.py`.
     Point C with projection ΔNeff within 0.005 of sprint-5
     baseline ΔNeff = −0.012. Sprint-11 drift was 0.013; fix
     should narrow it to ULP-level if the Phase-0 mechanism is
     correctly localised.
  8. **Hannestad gate 6** —
     `python validation/diagnostics/diag_hannestad_proj_w30_nB10k.py`
     with both new flags on. **Target: A ∈ [0.9, 1.1],
     B ∈ [0.3, 0.7], C ∈ [0.02, 0.10]**. Yp for all three ∈
     [0.24, 0.26]. All three in-band closes Stage E.2.
  9. If gate 8 hits: flip `qke_phase0_flag` AND
     `qke_expm_fallback_near_degeneracy` defaults to True; update
     `test_sterile_dw_production` with config overrides forcing
     both False for bit-identity. Re-run gates 1-5.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT12_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete
> sprint-12 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-11 eigendecomposition fallback or its
> H-coupling gate, the sprint-10 Phase-0 driver or instrumentation,
> the sprint-9 MSW instrumentation, the sprint-8 energy
> instrumentation or 2-level energy regression guard, the sprint-7
> `_F_stat_stable` clamp, the sprint-6 NaN-safe sanitisation, the
> sprint-5 V_nunu projection, or the D.7.1 Strang-symmetric
> composition. The primary suspect is the predictor-corrector
> composition or V_nunu non-linear feedback at the istep 906→1035
> window of Phase-0 Point C — eigendecomposition cannot fix it
> because the mechanism lives downstream of `_etdrk2_expm_phi`.

## Commit chain for context

  * `<sprint-11 hash>` — **Sprint 11** (eigendecomposition fallback;
    Phase-B over-amplification closed; Phase-0 step-function
    persists; Suspect 2-residual partially confirmed). Sprint 12 is
    built directly on top of this — the starting state.
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver; Point-A 87% closed,
    Point-C narrow-mixing regression introduced; Suspect 2-residual
    promoted at narrow mixing).
  * `ac1d521` — Sprint 9 (MSW diagnostic; Suspect 2 falsified at
    Point A; Suspect 3 promoted).
  * `bb0ea32` — Sprint 8 (Suspect 1 falsified at machine precision).
  * `0c28d8d` — Sprint 7 (cold-T NaN origin fix).
  * `968c936` — Sprint 6 (NaN-safe sanitisation).
  * `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
    dependency** — projection is what exposes the MSW anomaly.
  * `ca589b0` — Sprint 2 (Phase B stabilisation; n_B auto-scale).
  * `e2f41f6` — D.7.1 (Strang-symmetric sequence — the outer
    composition shape sprint 12 investigates the *substeps* of).
  * `4d8ab2c` — D.7 (per-mode expm via Al-Mohy augmented matrix).

## Post-fix: downstream opportunities

If sprint 12 closes Suspect 5 and lands the default flip:

  1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
     (|U_μ4|²=1e-4, Δm²=1.29). ΔNeff ∈ [0.05, 0.2] per
     Gariazzo+2019.
  2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
     resonance-driven physics at non-zero lepton asymmetry.
  3. **Representation-factor audit** — still parked from sprint 4.
  4. **Performance** — eigendecomposition fallback cost (~1.4×
     wall on Point-C n_B=10000) is acceptable for diagnostic but
     worth profiling for production. Cache "stable" mode classes
     across repeated steps.
  5. **Fold sprint-8/9/10/11/12 per-step instrumentation into a
     unified telemetry facility** that future stages reuse.
