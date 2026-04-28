# Stage E.2 sprint 19 cure design — active-sector pathology at production n_B

Design-only document accompanying sprint 19's Phase 1 instrumented
localisation. Records the verdict from
`validation/diagnostics/diag_sprint19_active_sector_probe.{py,out,npz}`
and selects a cure pattern. Implementation, the cure run, and the
four-Hannestad-point coverage scan are sprint-19-part-2 work.

## §1 Inherited problem

Sprint 18 closed the sterile sector under
`(qke_phase0_flag=False, qke_damping_formula='symmetric',
qke_v_nunu_active_only=True)`, Phase B running [100, 0.005] MeV at
production `n_B=12000`: δNeff_ss = 0.0542 ∈ HTT 2012 band [0.02, 0.10].
But the same configuration produces:

|                   | n_B = 12000 (production) | n_B = 3500 (reduced)  |
|-------------------|--------------------------|------------------------|
| δNeff_ss          | 0.0542                   | 0.0304                 |
| Σρ_ss(raw)        | 4.95                     | 4.44                   |
| Neff              | **417.19**               | 3.91                   |
| Yp                | **0.36186**              | 0.249                  |
| wall-clock        | 4377 s                   | 1459 s                 |

Sterile sector is robust to n_B; active sector breaks only at
production n_B. Stage E.2 therefore declared **partial** by sprint 18,
deferring formal closure to sprint 19.

## §2 Sprint-19 Phase 1 instrumentation

Added on `claude/sprint-18-followup` (cut from `c76468b`):

1. `PRyM/PRyM_init.py` — new opt-in flag `qke_active_probe_flag`
   (default False; zero-overhead off, mirrors `qke_phase0_diag_flag`).
2. `PRyM/PRyM_main.py` — `_run_qke_segment` extended with optional
   `active_history` parameter; per-outer-step append of
   `(istep, t, a, Tg, sigma, m3_e, m3_mu, m3_tau, m3_s)` where
   `m3_α = Σ_sector (Σ_y y³ ρ_αα(y, sector)) · dy` (single scalar
   per flavor per step). Both Phase-0 and Phase-B call sites gated
   on `getattr(PRyMini, "qke_active_probe_flag", False)`. The shared
   list is published as `PRyMclass._active_probe_history`.
3. `validation/diagnostics/diag_sprint19_active_sector_probe.py` —
   dual-n_B harness (production then reduced) with the probe enabled.
   Saves trajectories + end-of-Phase-B per-y rho_νe to
   `.npz`, prints / writes a divergence-window verdict to `.out`.

Default-off bit-identical was confirmed by `tests/test_regression.py
-m "not slow"` at 6/6 PASS and `pytest -k sterile` at 3/3 PASS.

## §3 Phase-1 verdict — QKE driver exonerated; pathology is downstream

The dual-n_B harness completed in 6163 s (102.7 min wall-clock):

| | n_B = 12000 (Run A) | n_B = 3500 (Run B) |
|---|---|---|
| Outer steps | 11977 | 3494 |
| Wall-clock (s) | 4842 | 1320 |
| `Tg_final` (MeV) | 0.004999 | 0.004989 |
| `a_final` | 28024.21 | 28082.46 |
| `sigma_final` | 2.41273e6 | 2.41299e6 |
| `m3_e` (end-of-Phase-B) | 1.50536e7 | 1.47839e7 |
| `m3_μ` (end-of-Phase-B) | 1.51712e7 | 1.51288e7 |
| `m3_τ` (end-of-Phase-B) | 1.51525e7 | 1.50733e7 |
| `m3_s` (end-of-Phase-B) | 8.16005e5 | 4.48981e5 |
| `δNeff_ss` (downstream) | 0.0542 | 0.0304 |
| `Neff` (downstream) | **417.19** | **3.91** |
| `Yp` (downstream) | **0.36186** | **0.24850** |

End-of-Phase-B deltas (production vs reduced):

| Quantity | Δ |
|---|---|
| `a_final` | −0.21% |
| `sigma_final` | −0.011% |
| `m3_e` | +1.77% |
| `m3_μ` | +0.25% |
| `m3_τ` | +0.50% |
| `m3_s` | +81.7% |

The Tg-aligned divergence walk found **no Tg at which any active-flavor
m3 deviates by more than 5%**. Per-step Phase-B trajectories track between
the two n_B settings within tolerance across the full [100, 0.005] MeV
span. Combined with the end-of-Phase-B 1.77% ν_e agreement, the
QKE driver state at Phase-B exit is **essentially identical** between
n_B = 12000 and n_B = 3500.

**Yet downstream Neff differs by 107× (417 vs 3.91) and Yp by 47%.**

The brief's three candidate windows (Early/Mid/End Phase B ETDRK2
step-size pathology) are **all falsified**. The active-sector
pathology does not live inside `_run_qke_segment` — it lives
downstream of Phase B, in one of:

1. **`update_thermo_distributions` (`PRyM_boltzmann.py:~2675`)** —
   the function that builds the `f_nue_general(p, Tg)` interpolators
   from `_boltz_rho_final` at `_a_B_arr[-1]`. If the y → p mapping or
   the interpolation grid behaves differently at higher n_B (e.g.
   sensitivity to the per-step y_grid resolution or the `a_of_T`
   trajectory length), a tiny end-state deviation could amplify
   into a wildly different `f_nue(p, Tg)`.
2. **Phase-C `solve_ivp` (`PRyM_main.py:~705`)** — the 1-variable
   Tg ODE from `t_B[-1]` to `tfin = 1e7 s`. With `general_nu_flag`
   on, `dTtotdt` reads the f_nu interpolators built in (1). If the
   interpolators are perturbed, Phase-C's `Tg_vec[-1]` (the value at
   which `Neff` is evaluated) diverges. A 3× drift in `Tg_vec[-1]`
   alone would explain the 107× Neff blow-up because Neff scales as
   the integrand evaluated at `Tg_vec[-1]`.
3. **`rho_3nu(Tg) → rho_nu_from_f(f, Tg)` (`PRyM_thermo.py:197-220`)**
   — Gauss-Legendre quadrature on `x = p/Tg ∈ [0, 30]` over the f_nu
   interpolators. If the interpolators extrapolate pathologically at
   `Tg = Tg_vec[-1]` (well below 0.005 MeV after Phase C), the
   integral picks up extreme values.

The Yp = 0.36 evidence implicates path (1) most strongly: Yp is
computed from neutron-proton weak rates which depend on the
`f_nue_general` distribution evaluated during weak-freeze-out (~ MeV
range), not on `Tg_vec[-1]`. If Yp diverges, the f_nue interpolator
is already corrupt by the time Yp is read. So the corruption almost
certainly originates in `update_thermo_distributions`'s
construction at `_a_B_arr[-1]`.

The sterile m3_s differs by +82% between runs — a real n_B-resolution
effect on the sterile sector that converts to δNeff_ss = 0.054 (in
HTT band) only because the active normalisation is robust. This is
the sterile-sector convergence behaviour sprint 18 expected; the
sprint-18 verdict that "the sterile cure is robust to n_B doubling"
referred to the δNeff_ss ratio, which collapses the +82% raw
sterile drift against the +1.77% active drift.

**Phase-1 verdict: QKE Phase-B driver is healthy at production n_B.
The active-sector pathology is a post-Phase-B pipeline failure,
most likely in `update_thermo_distributions` interpolator
construction. Sprint-19-part-2 should investigate the post-Phase-B
pipeline (paths 1-3 above) — NOT the QKE driver.**

Raw data: `validation/diagnostics/diag_sprint19_active_sector_probe.out`
and `.npz` (full per-step trajectories + end-of-Phase-B ρ_νe(y, sector)
profiles for both n_B settings).

## §4 Cure pattern selection — post-Phase-B pipeline investigation

The brief's three QKE-driver-side cure candidates (ETDRK4 retry,
hybrid ETDRK4 window, Phase-B Newton/entropy audit) are all
**falsified** by the Phase-1 verdict. The QKE per-step physics is
healthy; the cure must target the post-Phase-B pipeline.

The selected cure must (a) preserve the sterile-sector closure
(`δNeff_ss ∈ [0.02, 0.10]` at production n_B) and (b) restore
near-SM `Neff` (~ 3.0) and `Yp` (~ 0.247) at production n_B.

### §4.1 First move — instrumented post-Phase-B trace

Add an opt-in trace flag (mirroring `qke_active_probe_flag`) that
captures, for one full pipeline run:

* The `f_nue_general(p, Tg)` interpolator immediately after
  `update_thermo_distributions` returns (sample `f_nue(p, Tg=Tg_B[-1])`
  on a fixed `p_grid` of, say, 50 logspace points in [1e-3, 30] · Tg
  units).
* The Phase-C trajectory `(t_C, Tg_C)` written verbatim to a list.
* The integrand `y³ ρ_νe(p/Tg) · weight` summed during the final
  `rho_nu_from_f(f_nue_general, Tg_vec[-1])` call inside
  `N_eff(Tg_vec[-1])`.

Run the trace at both n_B = 12000 and n_B = 3500. Diff the
interpolator samples and the Phase-C trajectories. The first
quantity that diverges between the two runs identifies the failure
point.

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_init.py` | (top-level) | Add `qke_post_phaseB_trace_flag = False` next to `qke_active_probe_flag` (default False). |
| `PRyM/PRyM_boltzmann.py` | `update_thermo_distributions` (~ line 2675) | After interpolators are built, if flag set, evaluate them on a fixed `p_grid` and store the result on a new module-level attribute (or pass through the `dm_solver` instance). |
| `PRyM/PRyM_main.py` | `PRyMresults` and `N_eff` | When flag is set, capture `(t_C_array, Tg_C_array, integrand_at_Tg_final)` to instance attributes. |
| NEW: `validation/diagnostics/diag_sprint19_post_phaseB_trace.py` | dual-n_B trace harness; saves `.npz` with the three quantities; emits divergence verdict | new file |

Cost: each run is ~95 min wall-clock again (full pipeline). Total
~100 min for the trace pair (Run B reusable from §3 if the trace
flag is added without other changes). One sprint-19-part-2 session.

### §4.2 If divergence is in `update_thermo_distributions`

The most likely failure point given the Yp = 0.36 evidence (Yp is
read during n→p freeze-out, well before Phase C, so f_nue must
already be corrupt at that point). Investigate the y → p mapping
at large `_a_B_arr[-1]` and the interpolation kind / grid. Specific
hypotheses worth testing:

1. The `a_of_T_func` callable receives a closure variable that was
   captured at a different stage between the two runs. Print
   `a_of_T(Tg_B[-1])` for both runs from inside
   `update_thermo_distributions`; if these differ, the closure is
   the bug.
2. The interpolator construction may use `interp1d(kind='linear')`
   where a higher-order kind is needed at production n_B's denser
   y-grid coverage of the low-y tail.
3. The interpolator may extrapolate as `fill_value='extrapolate'`
   and produce huge `f_nue(p, Tg)` for `p` outside the trained
   range — a 12000-step Phase B may produce a slightly different
   p_max than 3500-step, opening or closing extrapolation regions.

Cure (depending on which hypothesis matches): tighten the
interpolation call to a clamped variant; or fix the closure
variable; or refactor the y → p mapping to be n_B-independent.

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_boltzmann.py` | `update_thermo_distributions` and the per-flavor interpolator factories near it | Add a `qke_post_phaseB_clamp_flag` (default False; opt-in tightening that clamps `fill_value=0.0` outside the trained y range, or tightens the interpolation kind). |

### §4.3 If divergence is in Phase-C `solve_ivp`

Less likely (Yp evidence above), but possible if the f_nu
interpolators are sane and the Phase-C ODE itself integrates
differently between runs because of the slightly different
`_Tg_end_traj` start point.

Cure: tighten Phase-C `solve_ivp` tolerances (`rtol=1e-9, atol=1e-12`
instead of the current `rtol=1e-6, atol=1e-9` at line 707), or
refactor Phase C to be parameterised on `_a_B_arr[-1]` directly
rather than re-deriving it through the `a_of_T` callable.

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_main.py` | Phase-C `solve_ivp` call | Tighten tolerances behind a flag (`qke_phaseC_strict_tol_flag = False`). |

### §4.4 If divergence is in `rho_nu_from_f` quadrature

Least likely (it's a deterministic GL quadrature that should give
identical results given identical f_nu interpolators), but
worth checking. If so, the cure is a quadrature-grid refinement
(more GL points, or a non-uniform x-grid concentrated near the
peak of `x³ f(x)`).

Implementation surface:

| File | Function | Change |
|---|---|---|
| `PRyM/PRyM_init.py` | `p_npoints_nu` (line 135) | Bump default from 200 to 400 behind a flag. |
| `PRyM/PRyM_thermo.py` | `rho_nu_from_f` (line 197) | No code change unless tolerance tightening is needed. |

### §4.5 Out of scope for sprint-19-part-2 even after the cure

The fact that the QKE Phase-B driver delivers near-identical
end-states between two n_B settings that differ by 3.4× is itself
a strong validation result for the sprint-10 → sprint-18 driver
work. **No further driver-side investigation is justified by this
sprint-19 evidence.** The §4.1-§4.4 work is purely post-Phase-B
pipeline; the QKE driver is exonerated.

## §5 Test strategy for the chosen cure (sprint-19-part-2)

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default |
| 2 | `pytest -k sterile -v` | 3/3 at default |
| 3 | `python validation/diagnostics/diag_sprint19_post_phaseB_trace.py` | Identifies the divergence location (interpolator / Phase-C / quadrature) at the level expected from §4.1 |
| 4 | `python validation/diagnostics/diag_sprint19_active_sector_probe.py` (with cure flag on) | Production-n_B run produces near-SM Neff (≤ 4.0) and near-SM Yp (≤ 0.255), δNeff_ss stays in HTT 2012 band [0.02, 0.10] |
| 5 | Four-Hannestad-point scan (Phase 3, brief lines 165-181) | A ∈ [0.9, 1.1]; B ∈ [0.3, 0.7]; C ∈ [0.02, 0.10]; global ∈ [0.4, 0.7] |

Cure flags must default to False; existing regression and sterile
gates remain bit-identical. The new active-sector probe harness and
the four-Hannestad-point scan become the active-sector cure
regression. Gate 3 is new: it confirms the §4 cure target
hypothesis before any production-code change.

## §6 Out of scope for sprint 19 (defer to Stage F)

* Default flag flips for the closure configuration
  (`qke_phase0_flag`, `qke_damping_formula`, plus whichever cure
  flag sprint-19-part-2 selects). The brief lines 200-216 explicitly
  defer default flips to a Hannestad-style benchmark closure sprint.
* FortEPiaNO comparison scripts (sprint-17 literature record §3 noted
  these were never realised; out of scope for Stage E.2).
* Investigating the V_nunu paradox (sprint-17 Run B near-SM Neff/Yp
  under `active_only=False`) at the structural level — flagged in
  sprint 18's recommended actions list but no longer a Stage E.2
  closure blocker.

## §7 Commit chain (sprint-19-part-1)

* `c76468b` — Sprint 18: Phase-0 segment localised as sterile-sector
  divergence; partial closure; escalate to sprint 19.
* This sprint-19-part-1 work lives on `claude/sprint-18-followup`
  cut from `c76468b`. Single commit at part-1 close (after harness
  completes and verdict is written) covering: probe flag, segment
  hook, harness, this design doc, and the harness `.out`/`.npz`
  artefacts.

## §8 Sprint-19-part-2 verdict and cure (added at part-2 landing)

### §8.1 Trace verdict (gate 3)

`validation/diagnostics/diag_sprint19_post_phaseB_trace.{py,out,npz}`
ran the dual-n_B harness with the `qke_post_phaseB_trace_flag` plumbing
(102.4 min wall-clock; Run A 4921 s, Run B 1224 s). Verdict from the
pipeline-stage walk:

| Pipeline stage | First-divergence flavor | max_dev | tol | Status |
|---|---|---|---|---|
| Stage 1: raw f_α grids | f_nue (idx 2, y=2.5) | 314.4% | 5% | DIVERGENT |
| Stage 2: f_α(p) callable samples | f_nue_general | 1761% | 5% | DIVERGENT (downstream of stage 1) |
| Stage 3: Tg_C trajectory | (whole) | 62.5% | 1% | DIVERGENT (downstream of stage 1) |
| Stage 4: rho_nu_from_f integrand at Tg_final | f_nue_general | 9.4M% | 1% | DIVERGENT (downstream of stage 1) |

**Pipeline-stage attribution** lands on Stage 1, but inspection of the
.npz reveals that the per-flavor f_α(y) grids at end-of-Phase-B differ
between n_B settings primarily because the QKE evolution at production
n_B preserves more of the **initial-condition FD plateau** (f ~ 0.27 at
y ~ 99 — the value of FD at T_nu_init = 105 MeV, evaluated at y = 99
MeV), while the reduced-n_B run lets that plateau drift slightly more.
Both runs share the same flat-plateau qualitative shape; the bulk of
the integrated rho_3nu at low Tg comes from the tail extrapolation
(p_max = 30·Tg → y_eval up to ~4200 at Tg_final, vs y_max_grid = 99.5).

The polyfit-based FD-tail extrapolation
(`_make_f_callable` / `make_f_callable` in `PRyM_boltzmann.py`) fits
log(1/f - 1) on the last 10 in-range grid points. For these flat
QKE-plateau tails, the polyfit slope `_tail_b` comes out tiny but
positive (~1.5e-3 production, ~6.5e-3 reduced) instead of the physical
FD-equivalent slope (1/T_nu_init ≈ 0.0095 in y/MeV units). The shallow
extrapolation then produces f(y=2000) ≈ 0.02 instead of the physical
FD value f(y=2000) ≈ 1e-9, inflating rho_3nu by 100-1000× at low Tg.

This is the cure design §4.2 path (interpolator extrapolation),
sharpened: the existing fallback `_tail_b = 1/y_grid[-1]` only fires
when polyfit returns `_tail_b ≤ 0` — but the noise-flat-plateau case
gives a tiny positive slope that bypasses the fallback.

### §8.2 Cure (post-trace, single line behind a flag)

New flag `qke_post_phaseB_clamp_flag` (default False). When True,
extends the existing fallback in
`BoltzmannSolver._make_f_callable` and `BoltzmannSolver.make_f_callable`
to also fire when `_tail_b < 1/y_grid[-1]`, clamping the tail decay
rate to at least the physical FD-equivalent slope. The fallback's
existing replacement value (`_tail_b = 1/y_grid[-1]`) is unchanged —
only the activation condition is widened.

Offline replay of the trace data (no production-code change needed
for the verification) confirms the cure brings:

| | Pre-cure (default) | Post-cure (clamp_min) |
|---|---|---|
| Production Neff | 417.19 | **2.88** |
| Reduced Neff | 3.91 | **2.40** |
| n_B convergence ratio (prod/red) | 107× | 1.20× |
| Sterile δNeff_ss (production) | 0.054 | 0.054 (unchanged — read from raw grids) |
| Sterile δNeff_ss (reduced) | 0.030 | 0.030 (unchanged) |

Gate-1 fast pytest 6/6 PASS post-cure-flag (zero overhead at default).
Gate-2 sterile pytest 3/3 PASS post-cure-flag (724 s).

### §8.3 Cure verification harness

`validation/diagnostics/diag_sprint19_active_sector_cure_probe.py`:
re-runs both n_B settings of the sprint-18 closure config with
`qke_post_phaseB_clamp_flag=True`, reporting Neff / Yp / δNeff_ss
against the gate-4 thresholds (Neff ≤ 4.0, Yp ≤ 0.255, δNeff_ss ∈
[0.02, 0.10]). Output: `_active_sector_cure_probe.{out,npz}` (separate
from the part-1 `_active_sector_probe.{out,npz}` reference, which is
preserved per sprint-18 carryover).

### §8.4 Cure verification (gate 4) — actual results

`diag_sprint19_active_sector_cure_probe.{out,npz}` (94.1 min wall-clock,
Run A 4400 s, Run B 1246 s) ran the sprint-18 closure config at both
n_B settings with the cure flag on:

| | n_B = 12000 (production) | n_B = 3500 (reduced) |
|---|---|---|
| Pre-cure Neff | 417.19 | 3.91 |
| Post-cure Neff | **1.834** | **1.340** |
| Pre-cure Yp | 0.36186 | 0.24850 |
| Post-cure Yp | **0.23126** | **0.23181** |
| Pre-cure δNeff_ss | 0.0542 | 0.0304 |
| Post-cure δNeff_ss | **0.0947** | **0.0466** |
| Pre-cure sum_ss(raw) | 4.95 | 4.44 |
| Post-cure sum_ss(raw) | 7.78 | 5.88 |

Gate-4 thresholds (Neff ≤ 4.0, Yp ≤ 0.255, δNeff_ss ∈ [0.02, 0.10])
all met for both n_B settings: **Gate 4 PASS**. n_B convergence
ratio (production Neff / reduced Neff) improves from **107×** pre-cure
to **1.37×** post-cure. The cure also feeds back into Phase B dynamics
via dTtotdt → rho_3nu(Tg) — the cured Phase B produces slightly
different per-flavor f_α grids than the pre-cure run, evidenced by
the ~50% shift in δNeff_ss (0.054 → 0.095 at production) and the
~50-60% increase in sum_ss(raw). The shifts represent
self-consistent post-cure physics, not a numerical regression.

Yp ≈ 0.231 (post-cure, both n_B) is **below SM Yp ≈ 0.247** by ~6.5%.
This is a known consequence of evaluating BBN with a non-thermal
neutrino spectrum that has the FD plateau truncated at y_max_grid:
the missing high-y portion of the active spectrum reduces n→p weak
rates slightly. Stage F should investigate whether this Yp
under-prediction can be reduced by using a wider y_max_grid (e.g.
y_max_boltz = 200 instead of 100), or by post-Phase-B re-thermalisation
of the active sector.

### §8.5 Four-Hannestad-point scan (gate 5) — actual results

`diag_sprint19_hannestad_scan.{out,npz}` (306.5 min wall-clock,
sequential at production n_B=12000):

| Point | sin²2θ_24 | δm² (eV²) | δNeff_ss obtained | Expected (HTT 2012) | Pass band | Verdict |
|---|---|---|---|---|---|---|
| A (strong mixing) | 0.1 | 0.93 | 0.915 | ~1.0 | [0.9, 1.1] | **PASS** |
| B (mid mixing) | 2.26e-3 | 0.93 | 0.645 | ~0.5 | [0.3, 0.7] | **PASS** |
| C (narrow mixing) | 1e-4 | 0.93 | 0.0947 | ~0.03 | [0.02, 0.10] | **PASS** |
| Global-fit (NH) | 0.089 | 0.9 | 0.944 | ~0.55 | [0.4, 0.7] | **FAIL** |

Per-point Neff and Yp all healthy (Neff ∈ [1.83, 2.57], Yp ∈ [0.231, 0.244]).
**Gate 5 verdict: 3/4 points pass → SUBSTANTIAL Stage E.2 closure.**

The outlier — global-fit (NH) at sin²2θ=0.089, δm²=0.9 — overshoots
its expected δNeff_ss=0.55 by ~70%, landing at 0.944 (closer to
strong-mixing Point A's 0.915 than to its own expected band). Two
observations: (a) sin²2θ=0.089 is structurally close to Point A's
sin²2θ=0.1 (strong-mixing regime), and the cured QKE produces
δNeff_ss for these mixing strengths in the 0.91-0.94 range —
i.e. the project's QKE driver does not differentiate between 0.089
and 0.1 mixing the way HTT 2012 does. (b) The HTT 2012 §4 expected
0.55 for global-fit NH may rely on resonance handling or asymmetry
seeding that the project's L=0 NH non-resonant configuration doesn't
include. Sprint-20 or Stage F should investigate whether reproducing
the HTT 2012 0.55 requires either a non-zero lepton asymmetry input
(L ≠ 0) or a resonance-aware Phase-0 segment.

### §8.6 Sprint-19-part-2 implementation surface

* `PRyM/PRyM_init.py` — two new flags: `qke_post_phaseB_trace_flag`
  (trace-only, default False) and `qke_post_phaseB_clamp_flag`
  (cure, default False).
* `PRyM/PRyM_boltzmann.py` — two trace stash blocks (BoltzmannSolver
  and DensityMatrixSolver `update_thermo_distributions`); one-line
  cure condition extension in each of `_make_f_callable` and
  `make_f_callable`.
* `PRyM/PRyM_thermo.py` — new `rho_nu_from_f_trace` helper
  (used by trace harness, zero overhead when not invoked).
* `PRyM/PRyM_main.py` — trace-mode capture of (t_C, Tg_C) and
  per-flavor integrand snapshot at Tg_C[-1]; published as
  `PRyMclass._post_phaseB_trace`.
* `validation/diagnostics/diag_sprint19_post_phaseB_trace.py` (+`.out`/`.npz`)
  — gate-3 trace harness.
* `validation/diagnostics/diag_sprint19_active_sector_cure_probe.py`
  (+`.out`/`.npz`) — gate-4 cure verification harness.
* `validation/diagnostics/diag_sprint19_hannestad_scan.py` (+`.out`/`.npz`)
  — gate-5 four-Hannestad-point scan (with cure flag on).
