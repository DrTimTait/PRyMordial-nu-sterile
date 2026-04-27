# Stage E.2 sprint 16 brief: ETDRK4 prototype as a Suspect-7 falsifier

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 15. Read this first. By the end you should know
why two LSODA-based attempts (sprints 14, 15) cannot close the
Suspect 7 question and why the recommended path is ETDRK4 — a
higher-order exponential time-differencing corrector built on
ETDRK2's existing `_build_L_list` / `_etdrk2_expm_phi` kernels.

## One-paragraph orientation

Sprint 14 added `evolve_step_lsoda` (per-outer-step
`scipy.integrate.solve_ivp(method='LSODA')`) and discovered it
cannot complete one Hannestad Point C run in tractable wall-clock.
Sprint 15 attributed that to dense FD-Jacobian rebuilds and built
both an analytic Jacobian (dense + banded variants) and a
segment-only LSODA dispatcher (LSODA inside a Tg window, ETDRK2
outside). Both landed cleanly and both were correctness-verified.
But running gate 5 still does not finish: a single LSODA call at
Tg ≈ 80 MeV, 4-flavor sterile, dt ≈ 1.7e-3 s does not complete
even with rtol = 1e-2, atol = 1e-4. Sample-based stack inspection
shows LSODA is in `cb_f_in_lsoda` (RHS callback) repeatedly with
no `cb_jac` frame — the Jacobian is never consulted, the cost is
adaptive sub-stepping on a system that is simultaneously stiff and
oscillatory. That regime is exactly where multistep Adams/BDF/LSODA
is fundamentally weak; the textbook cure is exponential time-
differencing or a symplectic / Magnus integrator. Sprint 16's job
is to build an **ETDRK4** corrector on top of ETDRK2's existing
`_build_L_list` and `_etdrk2_expm_phi` machinery and use the
ETDRK4-vs-ETDRK2 disagreement on Σρ_ss as a substitute Suspect-7
falsifier. Cost: ~3× ETDRK2 wall-clock (well within budget).

## What to read, in order

Budget ~45 min before writing any driver code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — the sprint-15 landing record at the
   bottom has the structural-finding analysis (multistep methods
   ill-suited to oscillatory stiff QKE) and the option ladder
   (a)/(b)/(c) carried over from the sprint-15 brief, both of
   which sprint 15 ruled out. Skim sprint 14 (sprint-15 brief
   recapped above), sprint 13 for the falsified Suspect-6 framing,
   and sprint 12 for the Suspect-5 sub-step probe and the
   resonance localisation at istep 906→1035 / Tg ≈ 60-64 MeV.
3. **`doc/STAGE_E2_SPRINT15_BRIEF.md`** — for context on what
   options (a)/(b)/(c) were and why they're now ruled out.
4. **`PRyM/PRyM_boltzmann.py`**:
   * `_build_L_list` (line 4545) — per-mode-per-sector
     superoperator. **Reused as-is by ETDRK4.**
   * `_etdrk2_expm_phi` (line 4612) — Al-Mohy & Higham augmented
     matrix exponential producing `Phi0 = e^{Lt}`,
     `Phi1 = t·φ_1(Lt)`, `Phi2 = t²·φ_2(Lt)` simultaneously per
     mode. **ETDRK4 needs a Phi3 = t³·φ_3(Lt) augmented matrix —
     same Al-Mohy structure, one extra block.**
   * `evolve_step_ode_etdrk2` (line ~4810) — the ETDRK2 driver.
     ETDRK4 lands as a sibling `evolve_step_ode_etdrk4` (or
     ETDRK4S as the higher-order Krogstad variant).
   * `evolve_step_lsoda` (line ~5400) — sprint-14/15 driver,
     **untouched** in sprint 16.
   * `_lsoda_jacobian_dense` / `_lsoda_jacobian_banded` /
     `_lsoda_compute_jblocks` (line ~5260-5340) — sprint-15
     additions, **untouched** in sprint 16.
5. **`PRyM/PRyM_init.py`** — flags. Sprint 16 adds an
   `qke_etdrk4_flag` (default False) controlling the new driver.
   Existing `qke_ode_etdrk2_flag` and `qke_lsoda_*` flags remain.
6. **`validation/diagnostics/diag_hannestad_pointC_lsoda.py`** —
   the trimmed Point-C-only harness. Sprint 16 mirrors this with
   `diag_hannestad_pointC_etdrk4.py` (or extends in place to
   accept a driver-selection flag).

## What's broken (the framing inherited from sprint 15)

* ETDRK2 at Hannestad Point C produces Σρ_ss ≈ 22.225 (≫ Hannestad's
  band [0.02, 0.10] for sin²(2θ) = 1e-4, Δm² = 0.93 eV²).
* Suspect 7 says: ETDRK2's Strang split (or its order-2 corrector
  truncation) over-integrates the resonance crossing, driving Σρ_ss
  to the resonance equilibrium instead of the Landau-Zener band.
* Suspect 8 says: ETDRK2 is fine, but Hannestad's target band is
  incompatible with our self-consistent V_nunu setup, and the true
  saturation here really is ≈ 22.

A higher-order ETDRK driver (ETDRK4, Krogstad or Cox-Matthews
formulation) tests Suspect 7 directly: if Σρ_ss collapses toward
[0.02, 0.10] when the corrector is upgraded from order 2 to order
4, the order-2 truncation was the source. If Σρ_ss stays ≈ 22, the
truncation isn't the issue — Suspect 8 wins.

## The ETDRK4 design (Krogstad's variant, recommended)

Krogstad's ETDRK4 uses 4 stages per step:

```
A_n = exp(L·dt/2) · u_n + (dt/2) · φ_1(L·dt/2) · N(u_n, t_n)
B_n = exp(L·dt/2) · u_n + (dt/2) · φ_1(L·dt/2) · N(A_n, t_n + dt/2)
C_n = exp(L·dt/2) · A_n + (dt/2) · φ_1(L·dt/2) · (2·N(B_n, t_n + dt/2) - N(u_n, t_n))
u_{n+1} = exp(L·dt) · u_n
       + dt · [φ_1(L·dt) - 3·φ_2(L·dt) + 4·φ_3(L·dt)] · N(u_n, t_n)
       + dt · [2·φ_2(L·dt) - 4·φ_3(L·dt)] · (N(A_n, t_n + dt/2) + N(B_n, t_n + dt/2))
       + dt · [-φ_2(L·dt) + 4·φ_3(L·dt)] · N(C_n, t_n + dt)
```

Three matrix exponentials per outer step (`exp(L·dt/2)` cached;
`exp(L·dt)`; the augmented-matrix form gives all four φ functions
simultaneously). Four `N` evaluations per outer step — same per-
evaluation cost as ETDRK2's two, so total wall-clock ~3× ETDRK2.

The Al-Mohy & Higham augmented-matrix scheme (already used by
`_etdrk2_expm_phi` for Phi0/Phi1/Phi2) extends naturally to Phi3
by adding one more block of size N²:

```
M = [[L·dt, I, 0,    0],
     [0,    0, I,    0],
     [0,    0, 0,    I],
     [0,    0, 0,    0]]
exp(M)[0:N², 0:N²]      = Phi0
exp(M)[0:N², N²:2N²]    = Phi1 / dt
exp(M)[0:N², 2N²:3N²]   = Phi2 / dt²
exp(M)[0:N², 3N²:4N²]   = Phi3 / dt³
```

For 4-flavor sterile: the augmented block is 64 × 4 = 256 per
mode, vs ETDRK2's 192. Per-mode `expm` cost scales O((4N²)³) =
O(N⁶) ≈ 6× ETDRK2's. For 100 modes: still ~ms per outer step.

## Recommended sequence

Sprint 16 should attempt the Krogstad ETDRK4 first. The
implementation cost is 1-2 days; the upside is a higher-order
falsifier for Suspect 7 with the same wall-clock-tractable
character as ETDRK2.

### Phase 1 — Extend `_etdrk2_expm_phi` to produce Phi3 (~3 hours)

Add `_etdrk_expm_phi_4(self, L, dt_nat)` returning all four phi
functions per mode. The single-`expm` augmented form is exact and
the cleanest implementation. Verify on a closed-form L (diagonal,
known eigenvalues): each Phi_k should equal the closed-form
`(e^{λ dt} - sum_{j<k} (λ dt)^j / j!) / λ^k` per eigenvalue.

### Phase 2 — Add `evolve_step_ode_etdrk4` (~6 hours)

Implement Krogstad's 4-stage scheme using the existing
`_build_L_list` and `_assemble_collision_N` kernels, mirroring
`evolve_step_ode_etdrk2`'s structure. Reuse the post-step
sanitisation (off-diagonal magnitude clamp, NaN-safe diagonal
clip).

### Phase 3 — Wire the driver flag (~1 hour)

Add `qke_etdrk4_flag` (default False) in `PRyM/PRyM_init.py`.
Modify `PRyM/PRyM_main.py:_run_qke_segment` to dispatch:

```python
if PRyMini.qke_etdrk4_flag:
    dm_solver.evolve_step_ode_etdrk4(...)
elif PRyMini.qke_ode_etdrk2_flag:
    dm_solver.evolve_step_ode_etdrk2(...)
else:
    ...
```

### Phase 4 — Verification gates

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 6/6 bit-identical at default (ETDRK4 off). |
| 2 | `pytest -k sterile -v` | 3/3 at default. |
| 3 | ETDRK4 3×3 SM (`cfg_3x3` style) | Neff matches ETDRK2 to <1e-4, Yp <1e-5; wall-clock <5× ETDRK2. |
| 4 | ETDRK4 decoupled-sterile (`cfg_4x4_decoupled`) | Σρ_ss < 1e-12; wall-clock <5× ETDRK2. |
| 5 | ETDRK4 Hannestad Point C | Σρ_ss verdict (see decision tree). Wall-clock <90 min. |
| 5b | ETDRK4 Hannestad A/B/C | A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10] OR consistent ETDRK2-like saturation. Wall-clock <6 h. |

### Decision tree at gate 5

| Σρ_ss | Verdict | Action |
|---|---|---|
| ≈ [0.02, 0.10] | **Suspect 7 confirmed** (order-2 truncation was the bottleneck) | Default-flip `qke_etdrk4_flag=True`; close Stage E.2. |
| ≈ 22 (matches ETDRK2) | **Suspect 8 confirmed** (Hannestad target wrong / V_nunu setup mismatch) | Stage E.2 closure deferred to literature-review sprint. Document the ETDRK4-vs-ETDRK2 agreement. |
| Crashes / diverges / wall-clock blow-up | ETDRK4 design issue | Cross-check against a closed-form linear test problem; debug the augmented-matrix `expm` or Krogstad coefficient signs. |

## Stage inheritance: what NOT to touch

* `evolve_step_ode_etdrk2` and all its sprint-12 instrumentation
  (substep snapshots, energy probes, MSW probes, eigendecomp
  fallback). ETDRK2 stays as production.
* `_build_H_list`, `_build_L_list`, `_assemble_collision_N`,
  `_compute_D_pair_matrix` — kernels are reused as pure functions.
* `evolve_step_lsoda`, `_lsoda_jacobian_dense`,
  `_lsoda_jacobian_banded`, `_lsoda_compute_jblocks` — sprint-14/15
  infrastructure, untouched.
* `evolve_step`, `evolve_step_ode` — legacy.
* Sprint-13 collision-N refactor — Suspect 6 falsified.
* `tests/test_regression.py` — must pass 6/6 fast bit-identical at
  default (ETDRK4 flag off).

## Why not LSODA / BDF / Radau?

Sprint 14 demonstrated LSODA wall-clock infeasibility with FD
Jacobian. Sprint 15 added the analytic Jacobian (dense + banded)
and the segment-only window dispatcher and **still** could not
complete one outer step at Tg ≈ 80 MeV, 4-flavor sterile. Sample-
based stack inspection of the stuck process showed LSODA was in
`cb_f_in_lsoda` repeatedly with no Jacobian callback frame — the
cost is RHS evaluations during adaptive sub-stepping on a system
that is simultaneously stiff and oscillatory. BDF and Radau share
the same multistep / Newton machinery and would be subject to the
same RHS-eval budget. Exponential integrators (which is what
ETDRK2/ETDRK4 are) cure exactly this regime by exponentiating the
linear part exactly and only quadrature-integrating the nonlinear
part — which is why ETDRK2 already produces a Hannestad Point C
result in 27 minutes, where LSODA could not finish in 6 hours.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT16_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete sprint-16
> plan. Target for this session is to extend the augmented-matrix
> `expm` to Phi3, implement Krogstad ETDRK4, and run gate 5
> (Hannestad Point C) with it. Do NOT retry LSODA — sprint 15
> already showed the cost is structural to the multistep family,
> not specific to FD vs analytic Jacobian.

## Commit chain for context

  * **Sprint 15** (this) — analytic Jacobian (dense + banded) +
    segment-only LSODA window dispatcher; both correct, neither
    enables a Suspect 7 verdict. Sprint 16 builds on top.
  * `c6149dd` — Sprint 14 (LSODA driver landed; gate 5 wall-clock
    infeasible without analytic Jacobian).
  * `db515c5` — Sprint 13 (collision-N refactor falsified Suspect 6).
  * `e699c00` — Sprint 12 (sub-step instrumentation; H-iteration
    variants).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver).
  * `ac1d521` — Sprint 9 (MSW diagnostic).
