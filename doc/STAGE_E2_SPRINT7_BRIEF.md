# Stage E.2 sprint 7 brief: cold-T collision-integral hazard and extended-window over-sterilisation

Single-file handoff for a fresh context window taking over from Stage
E.2 sprint 6 (commit `968c936`). Read this first. By the end you
should know why extending the Phase B window to T_boltz_start ≥ 20
MeV with the V_nunu active-only projection produces unphysical
sterile over-population (sum ρ_ss ≈ 30 at w30 vs ~10 at w5), what
to land to close it, and the path to flipping the
`qke_v_nunu_active_only` default to `True`.

## One-paragraph orientation

Sprint 5 identified the dominant small-mixing DW bug (V_nunu applied
to the full 4×4 ρ − ρ̄ matrix; sterile has no NC charge) and landed
an opt-in projection fix. Sprint 6 made the ETDRK2 driver NaN-safe,
resolving the sprint-5 `scipy.linalg.expm` crash that blocked
extended-window runs. Sprint 6's Phase-1 probe, critically, gave us
precise localisation: **the first non-finite entry in
`evolve_step_ode_etdrk2` is a single value at step=4099, T≈5.5 keV,
sector=0, comp=0, y_idx=0, introduced by half-diag-2** — i.e. the
cold-T, lowest-momentum-mode, neutrino-ρ_ee diagonal, from the
collision integral `I_total_post` evaluated inside
`_assemble_collision_N`. Sprint 6's fix sanitised this NaN
downstream via `nan_to_num` before the diagonal clip, which prevents
the crash and keeps the run numerically stable — but does not fix
the underlying cold-T collision-integral hazard. At extended windows
the hazard fires more often, and each sanitisation event biases the
collision integral's value at its neighbouring steps, which
accumulates into window-dependent over-sterilisation: Point C's
ΔNeff goes from −0.032 at w20 to +4.212 at w30 (non-monotonic, target
0.040); sum ρ_ss grows monotonically with window size. Sprint 7's
job: fix the `_assemble_collision_N` cold-T hazard at origin,
re-verify the Hannestad A/B/C suite at w30 lands within 10-20% of
literature, flip the projection default to `True`.

## What to read, in order

Budget ~45 min before writing any diagnostic code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — search for "Stage E.2 sprint 6". The sprint-6
   landing record has the full validation-ladder table, the
   Phase-1 probe's exact first-NaN report, and the sprint-7
   candidate list. Also review sprint 5 (V_nunu projection) and
   sprint 2 (n_B auto-scale + QED-table zero-clamp) — sprint 7
   builds directly on all three.
3. `git show 968c936` — the sprint 6 landing commit. Read the
   commit message end to end; it contains the Phase-1 probe finding
   at the resolution of (step, T, sector, comp, y_idx), and the
   extended-window Hannestad table that defines the target.
4. **`validation/diagnostics/diag_hannestad_proj_w30.out`** — the
   post-sprint-6 w30 run. **Yp=0.30046 at Point A, Yp=0.29320 at
   Point C** (unphysical; target [0.24, 0.26]). sum ρ_ss 20.4-29.3
   across the three points. Two RuntimeWarnings at line 4755 remain.
   **`validation/diagnostics/diag_hannestad_proj_w20.out`** — the
   same suite at w20; Yp all in [0.24, 0.26] (physical). Compare the
   two side-by-side to calibrate how the hazard scales with window
   size.
5. `PRyM/PRyM_boltzmann.py` lines **4371-4500** — `_assemble_collision_N`.
   The origin of the cold-T NaN is inside one of:
   `_collision_integral_nu_nu` (line 4413), `_collision_integral_nu_e`
   (line 4430, or `_massive` variant at 4418), or one of the
   `fnu_*_scat/ann` interpolants (lines 4426-4429). The
   `tail_params` call at line 4411 is another candidate.
6. `PRyM/PRyM_boltzmann.py` lines **4725-4777**: the D.7.1 half-diag-2
   and Phase-2 sanitisation. The write point of the NaN is at
   line 4731 (`rho_all[0, 0] += phi_half[0] * I_total_post[0]`).
   Phase 2's sanitisation is at line 4749 (off-diag) and 4774
   (diagonal nan_to_num before clip). **Do not revert** — the
   sanitisation is a keeper; sprint 7 removes its need by fixing
   the origin.
7. `PRyM/PRyM_init.py` lines **726-753**: the `n_B_override`
   auto-scale. The heuristic `n_B = int(2000 * (decades/3)**4)` was
   pinned by sprint 2 for T_boltz_start=30 MeV; at w30 it gives
   n_B=5031. Is this enough to resolve the MSW passage at T_MSW≈10
   MeV when the window starts at T_boltz_start=30 MeV? A convergence
   scan is the first-order check.

## What's probably broken (three suspects, ordered by likelihood)

### Suspect 1 — cold-T edge-case in `_assemble_collision_N` kernels (HIGH)

The Phase-1 probe's localisation is surgically specific: **sector=0
(ν), comp=0 (ρ_ee diagonal), y_idx=0 (lowest y-mode), T≈5.5 keV**.
The origin is in `I_total_post[0]` — the nue collision integral at
the lowest y-mode. At T=5.5 keV, massless neutrino energy E = y/a
can be tiny (the `np.maximum(..., 1.0e-4)` floor at
`_build_H_list:4312` is eV not MeV, and may not apply here). Likely
mechanisms:

- **Fermi-Dirac exponent overflow**: `f_eq(E/T) = 1/(exp(E/T) + 1)`.
  At T=5.5 keV and the high-y end of the grid, E/T can be ~30+,
  producing `exp(E/T)` near float64 overflow. Division-by-Inf
  yields 0 (fine), but if the derivative `df_eq/dT` appears as
  `E/T² · exp(E/T) · f_eq²`, the product is `0 · Inf = NaN`.
- **`tail_params` extrapolation**: `_compute_all_tail_params` at line
  4411 presumably fits an analytic tail beyond `Ny_coll`. At cold T
  the fit may be extrapolating into a regime where the fit function
  passes through zero or turns singular.
- **`f_eq` division in a `_collision_integral_*` kernel**: a rate
  divided by `f_eq(y_min)` when y_min is the lowest-energy mode
  could go as `1/0` at T=5.5 keV if f_eq underflows.
- **`_fnu_*_scat/ann` interpolants**: these are `scipy.interpolate`
  interpolators of pre-computed tables. If T=5.5 keV is below the
  table's domain, `interp1d` with its default `fill_value` (which
  used to be `np.nan` in older scipy) will inject NaN.

**Diagnostic**: add a probe inside `_assemble_collision_N` that
checks each intermediate (`f_all`, `tail_params`, `I_nu_nu`, `I_nu_e`,
`I_total`, the four `fnu_*` scalars) for non-finite values and
reports the first failure with (T, y_idx). Run at Point A w20 (same
config as sprint 6 Phase 1) and follow the chain. ~2-3 hours.

**Fix candidate**: depends on mechanism. Likely a single
`np.where(x < floor, floor, x)` or a similar guard in the offending
kernel. Should eliminate both the rare NaN and the line-4755
warnings in the sprint-6 w30 output.

**Likelihood**: **high**. The Phase-1 probe localisation is
surgical, and the mechanism is a classical cold-T edge case in
collision-integral code.

### Suspect 2 — `n_B_override` auto-scale under-resolving MSW passage at large windows (MEDIUM)

Sprint 2's `n_B = int(2000 * (decades/3)**4)` was pinned by a
convergence scan at T_boltz_start=30 MeV, but **without the V_nunu
projection**. Sprint 5's projection removes the "V_nunu ballast"
that dominated the Hamiltonian at high T, unmasking the tiny vacuum
mixing. The MSW resonance at T_MSW ≈ 10 MeV now has a much steeper
crossing rate (smaller effective mixing → sharper Landau-Zener
transition) that the sprint-2-calibrated n_B may not resolve.

**Diagnostic**: at T_boltz_start=30 MeV, projection on, Point C,
manually override `n_B_override` to {1×, 2×, 4×, 8× default} and
watch ΔNeff. If ΔNeff converges toward 0.04 (Hannestad target) as
n_B grows, the auto-scale is the bug. Cost: one convergence run ≈ 4
points × ~35 min ≈ 2-3 hours.

**Fix candidate**: re-pin the heuristic exponent (maybe `**5` or
`**6`), or make it a function of `decades` and
`qke_v_nunu_active_only`.

**Likelihood**: **medium**. The window-dependent over-sterilisation
*could* be MSW under-resolution, but the Phase-1 probe finding
argues the origin is specifically the cold-T hazard (Suspect 1), not
a hot-T integration error. If Suspect 1 closes the hazard and Yp
normalises but ΔNeff still misses literature, Suspect 2 is the
second line of attack.

### Suspect 3 — Phase A thermal-IC inadequacy at T_boltz_start ≫ T_MSW (LOW)

Phase A of PRyMordial sets `ρ_active(T_boltz_start) = f_FD(E/T)` and
`ρ_sterile = 0`. At T_boltz_start = 30 MeV, the true active
distribution has already been slightly depleted by the ν-ν and ν-e
collisions over the T=∞ → 30 MeV evolution; PRyMordial skips that
and assumes exact thermal. FortEPiaNO evolves the deviation across
60 → 5 MeV, preserving the depletion history. This IC mismatch
could bias all three Hannestad points upward by overestimating the
initial ν reservoir available for conversion.

**Diagnostic**: compare PRyMordial's Phase A state at T_boltz_start
with FortEPiaNO's state at the same T. If they agree to ~1e-3 per
y-mode, Suspect 3 is falsified.

**Likelihood**: **low**. Sprint 4 already looked at ICs informally
and found agreement. Also the w20 Yp is physical — if Phase A were
broken we'd expect Yp to be off at every window size, not just w30.

## Stage inheritance: what NOT to touch

- **`PRyM_boltzmann.py:4749` off-diag NaN-safe pass** (sprint 6,
  required for crash-free runs).
- **`PRyM_boltzmann.py:4774` diagonal `nan_to_num` before clip**
  (sprint 6). These are the current safety net. Sprint 7 removes
  the *need* for them by fixing the origin; the sanitisation itself
  stays (defense in depth).
- **`_build_H_list` projection** at lines 4337-4340 (sprint 5).
- **`DensityMatrixSolver._compute_D_pair_matrix`** (Stage E.1).
- **`PRyMini.qke_damping_formula = "mirizzi"`** default.
- **`_apply_unitary`** (Stage D.3 ν-bar convention).
- **The D.7.1 Strang-symmetric sequence** in
  `evolve_step_ode_etdrk2` (½-diag → predictor → corrector →
  ½-diag).
- **`_etdrk2_expm_phi`** mathematical form (Al-Mohy; validated via
  2-level damped Rabi test, must still pass).
- **Default of `qke_v_nunu_active_only`** (still `False` when this
  sprint starts; the flip to `True` is the sprint's *output*, not
  its input).
- **Any other flag default.** If you add new flags (e.g. a cold-T
  floor for f_eq), default them to current behaviour.

## Scope options

- **(a) Cold-T origin fix only** (Suspect 1): instrument
  `_assemble_collision_N`, find the mechanism, fix at origin.
  Re-run w30. If the Hannestad table lands within tolerance and Yp
  is physical, flip the default and close Stage E.2. If the fix
  eliminates the NaN and line-4755 warnings but ΔNeff at w30 still
  misses literature, escalate to (b). ~4-6 hours.
- **(b) (a) + n_B convergence re-pin** (Suspects 1+2): land (a),
  then run the n_B convergence scan at w30 Point C. If ΔNeff
  converges monotonically toward 0.04 with n_B, re-pin the
  heuristic. ~8-10 hours.
- **(c) Full Phase-A IC audit** (Suspect 3, radical): cross-check
  against FortEPiaNO at T_boltz_start, build a Phase-A evolution
  driver if ICs disagree. ~2 sessions.

State the choice explicitly in the opening message.

## Validation targets (post-fix)

Same ladder as sprint 6, with gate 6 being the primary gate:

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
   crash; all Yp ∈ [0.24, 0.26] (still holds).
6. **Hannestad literature at w30** —
   `python validation/diagnostics/diag_hannestad_proj_w30.py`.
   **Target: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.1]**. Yp
   for all three points ∈ [0.24, 0.26]. No line-4755 warnings.
   Hitting all three closes Stage E.2.
7. If gate 6 hits: flip `qke_v_nunu_active_only` default to `True`
   in `PRyM_init.py`. Re-run 1-5 and verify
   `sterile_DW_literature.py` now gives Hannestad agreement. Update
   `validation/sterile_DW_literature.out.txt` accordingly.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT7_BRIEF.md` end-to-end before writing any
> code. Then use `EnterPlanMode` to propose a concrete Stage E.2
> sprint-7 plan. My target for this session is {one of a / b / c}.
> Do not touch the sprint-5 V_nunu projection physics, the sprint-6
> NaN-safe sanitisation, or the D.7.1 Strang-symmetric sequence. The
> bug is a cold-T collision-integral edge case, not a physics scope
> change.

## Commit chain for context

- `968c936` — **Sprint 6** (NaN-safe clamp fix). Sprint 7 is built
  directly on top — the starting state.
- `db6b7cf` — Sprint 6 brief (context doc for sprint 6).
- `a2a975c` — Sprint 5 (V_nunu projection opt-in). **Critical
  dependency** — the projection is what exposes the cold-T hazard
  by removing the V_nunu ballast that previously masked it.
- `66ba8ec` — Sprint 4 (damping magnitude ruled out).
- `f403442` — Sprint 3 (y-grid discretisation falsified).
- `ca589b0` — Sprint 2 (Phase B stabilisation: auto-scaled n_B,
  QED-table zero-clamp). **Critical dependency** — enables the
  extended-window runs sprint 7 needs to validate, and its n_B
  heuristic is the subject of Suspect 2.

## Post-fix: downstream opportunities

If sprint 7 closes the extended-window over-sterilisation and lands
the default flip:

1. **Gariazzo benchmark** — `validation/sterile_DW_gariazzo.py`
   (|U_μ4|²=1e-4, Δm²=1.29). Target: ΔNeff ∈ [0.05, 0.2] per
   Gariazzo+2019.
2. **Shi-Fuller literature** (Saviano+2013). Same QKE framework,
   different physics focus; sprint-5/6/7 fixes should carry over.
3. **Sibling clamp sites at `PRyM_boltzmann.py:3907` and `:4256`**
   — same NaN-silencing pattern as the sprint-6 D.7.1 fix, on
   non-default legacy paths. Maintenance pass: port the sanitisation
   so all three evolution paths are NaN-safe.
4. **Representation-factor audit** — still parked from sprint 4.
   Likely moot after sprint 7.
5. **Phase A thermal approximation audit** (Suspect 3) — close out
   if not addressed in sprint 7.
