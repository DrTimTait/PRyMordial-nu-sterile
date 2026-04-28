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
