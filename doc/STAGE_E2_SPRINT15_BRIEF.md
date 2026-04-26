# Stage E.2 sprint 15 brief: stiff-solver hardening for the LSODA driver

Single-file handoff for a fresh context window taking over from
Stage E.2 sprint 14 (commit `c6149dd`). Read this first. By the end
you should know why `evolve_step_lsoda` works correctly but is
unusable as-is at Hannestad Point C, why three remediation paths
exist with very different scopes, and why the recommended path is
(a) — analytic Jacobian — even though it's the largest of the three.

## One-paragraph orientation

Sprint 14 landed `DensityMatrixSolver.evolve_step_lsoda`, a per-
outer-step `scipy.integrate.solve_ivp(method='LSODA')` driver that
reuses `_build_H_list` and `_assemble_collision_N` so kernel-level
physics matches ETDRK2. Default-path bit-identity confirmed (gate 1:
4/4 fast pytest, gate 2: 3/3 sterile pytest). Gate 5 (LSODA Hannestad
Point C) **did not finish in tractable wall-clock**: killed at 5h
28min @ rtol=1e-6, atol=1e-10 (strict) and 6h 17min @ rtol=1e-4,
atol=1e-8 (loose 100× each), both with no result. CPU 99-100%, RSS
stable, no errors — LSODA's adaptive multistep is doing what it's
asked to do, just enormously slowly. Loosening tolerances 100× saved
nothing meaningful, which is the diagnostic signature: **the cost is
structural (dense FD Jacobian rebuilds in stiff resonance), not
accuracy-bound**. The Suspect 7 vs Suspect 8 verdict is still open;
sprint 15's job is to make the LSODA driver fast enough to actually
run gate 5.

## What to read, in order

Budget ~45 min before writing any driver code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — the sprint-14 landing record at the
   bottom has the structural failure-mode analysis and the three-
   option remediation tree. Skim sprint-13 for the falsified
   Suspect-6 framing and sprint-12 for the H-iteration variants.
3. **`doc/STAGE_E2_SPRINT14_BRIEF.md`** — the sprint that just
   ran. Suspects 7 (Strang split structurally wrong) and 8
   (Hannestad target wrong) are still both open.
4. **`PRyM/PRyM_boltzmann.py`**:
   * `evolve_step_lsoda` (after the `evolve_step_ode_etdrk2`
     method, ~5247-5359 in the sprint-14 commit) — the new driver.
     Sprint-15 modifies this in place, or adds variant siblings,
     but must NOT touch `evolve_step_ode_etdrk2`.
   * `_build_H_list`, `_build_L_list`, `_assemble_collision_N` —
     the kernels the LSODA RHS reuses. Pure, no instrumentation
     coupling, no state mutation.
5. **`PRyM/PRyM_init.py`** — flags `qke_lsoda_driver_flag`,
   `qke_lsoda_rtol`, `qke_lsoda_atol`. New flags for sprint-15
   variants land here.
6. **`validation/diagnostics/diag_hannestad_pointC_lsoda.py`** —
   the trimmed Point-C-only harness. Sprint-15 should reuse it
   (with potentially tightened tolerances back to 1e-6 / 1e-10
   once the speed problem is resolved).
7. **`validation/diagnostics/diag_hannestad_proj_w30_nB10k.out`** —
   the ETDRK2 baseline (n_B=10000): Point C Neff=8.52219, ΔNeff=
   5.5219, Σρ_ss=22.225. The sprint-15 verdict criterion is
   whether LSODA's Point-C ΔNeff lands in [0.02, 0.10] (Suspect 7
   confirmed → ship LSODA) or near 5.52 (Suspect 8 confirmed →
   escalate to literature re-read).

## What's broken (the new framing)

The empirical evidence:

  * LSODA at rtol=1e-6, atol=1e-10: 5h 28min, no result → killed.
  * LSODA at rtol=1e-4, atol=1e-8: 6h 17min, no result → killed.
  * Loosening tolerances 100× saved nothing meaningful. **If the
    bottleneck were tolerance-bound, loosening would cut wall-
    clock 10-50×.** It didn't.

The structural-cost diagnosis:

  * State size at Point C: `2 * n_components * Ny = 2 * 16 * 100
    = 3200` real DOFs (sterile_flag=True ⇒ 4×4 ⇒ 16 components).
  * `solve_ivp(method='LSODA')` builds Jacobians by finite
    differences — no `jac=` argument was provided. **Each FD
    Jacobian rebuild costs `n_dof = 3200` RHS evaluations.**
  * Each RHS evaluation calls `_assemble_collision_N`, which
    runs `_collision_integral_nu_nu` (numba, O(Ny²)) and
    `_collision_integral_nu_e` (numba, O(Ny)). Wall-clock per
    RHS call: ~10-50 ms.
  * In a stiff regime LSODA may rebuild the Jacobian several
    times per outer step. With 12500 outer steps, this dominates.

**The LSODA RHS itself is correct** — gate 1 (4/4) and gate 2 (3/3)
pass at default flag-off, and the SM 3×3 reference (no resonance)
runs were progressing at sane CPU rates before being killed. The
problem is purely the FD-Jacobian cost across the resonance.

## Three remediation paths (scope ladder)

### Option (a) — Analytic Jacobian (RECOMMENDED)

Pass `jac=J(t, y)` to `solve_ivp`. The QKE Jacobian has closed-form
block structure that exploits per-mode independence:

  * The RHS `dρ/dt = -i [H, ρ] + N_collision(ρ)` couples y-modes
    only through `_assemble_collision_N` (which integrates over
    f_all across all y-modes for the diagonals) and through
    `_build_H_list`'s V_nunu term (which integrates the active-
    sterile asymmetry across modes).
  * The unitary part `-i [H, ρ]` is **diagonal in y-mode index**:
    `dρ_i/dt|_unitary = -i [H_i, ρ_i]` for mode i, where H_i
    depends only on (a, Tg, V_nunu(ρ)).
  * The collision diagonal damping is **diagonal in y-mode index**:
    `D_off,i ⊙ ρ_i`.
  * The collision integrals (gain on off-diag, I_total on diag)
    are non-local in y-mode index but smooth (low-rank tail
    parametrisation). **Approximation: hold the collision-integral
    Jacobian frozen at the start-of-outer-step state, treating
    only the unitary/damping part as ρ-dependent.** Validity: the
    unitary part is what oscillates fast in the resonance; the
    collision-integral Jacobian varies slowly across an outer dt.

Block-diagonal Jacobian under this approximation:

```
J(y_flat) = block_diag([J_i for i in range(Ny)])
J_i shape: (2 * 2*N², 2 * 2*N²)   # 2 sectors, real+imag pack
```

For Point C, that's `100` blocks of size `64×64` each — **`6400`
non-zero entries vs `3200² = 10.24M` for the dense Jacobian**, a
1600× sparsity factor. Even without exploiting block-diagonality,
just providing `jac=` (dense, but evaluated once per Jacobian
rebuild instead of `n_dof` RHS calls) is a `n_dof`-fold win at the
solver's Jacobian-rebuild cadence.

**Estimated wall-clock**: 50-500× faster than gate-5's failed
runs. Could land Point C in 30-90 min, A/B/C in ~3-5 hours.

**Implementation cost**: 1 day. Need to derive the Jacobian for
the unitary `-i [H, ρ]` term (linear in ρ, so `J = -i (H ⊗ I -
I ⊗ H^T)` in row-major vec form — same kron pattern that
`_build_L_list` already builds), plus the diagonal damping term,
plus the (frozen) collision Jacobian piece. Verify against FD
reference on a 3-flavor SM run (no sterile, fast) before
attempting Point C.

**Verification before Point C**:

  * 3-flavor SM with `jac=` should reproduce 3-flavor SM with
    no `jac=` (FD path) to LSODA's accuracy tolerance.
  * Wall-clock: should be 5-50× faster than the FD path on the
    same problem.

### Option (b) — `method='BDF'` with `jac_sparsity` hint

Pass `jac_sparsity=` (a sparse pattern matrix) to `solve_ivp` with
`method='BDF'`. BDF uses sparse FD Jacobians: only `nnz` RHS
evaluations per Jacobian rebuild, where `nnz` is the number of
non-zero entries in the sparsity pattern.

For the QKE under the same per-mode-block approximation as (a),
`nnz = 100 * 64² = 409,600` entries — still 25× more than (a)'s
analytic-Jacobian-action cost, but 8× cheaper than the dense FD
(3200²) used today.

**Estimated wall-clock**: 5-25× faster than today.

**Implementation cost**: half a day. No analytic derivation needed
— just construct the sparsity pattern (block-diagonal across y-
modes, dense within each `(2 × 2N²)` block).

**Validity caveat**: BDF and LSODA differ in step-size strategy.
BDF is purely implicit-multistep, where LSODA auto-switches
between Adams (non-stiff) and BDF (stiff). For a uniformly stiff
regime, BDF is fine; for a problem that's stiff only across the
resonance crossing, BDF may be slower than LSODA in the bulk.

### Option (c) — Segment-level LSODA over the resonance only

Add `evolve_segment_lsoda(self, T_start, T_end, ...)` that runs
LSODA over a narrowed Phase-0 sub-window across the resonance
crossing (sprint-12 sub-step probe localised this to istep ~926-
960 at the y_target=0.5 mode). Outside that sub-window, stay on
ETDRK2.

**Estimated wall-clock**: bulk integration cost stays at ETDRK2
levels (~25 min); sub-window cost is small relative to bulk
because the window is short (~30 outer steps out of 12500).

**Implementation cost**: half a day. Need to define the sub-
window selection criterion (a, Tg cuts? istep range?), bridge the
state between drivers (the sub-window's terminal ρ becomes
ETDRK2's input), and ensure the entropy bookkeeping in
`PRyM_main.py:_run_qke_segment` is preserved across the boundary.

**Validity caveat**: this is a *test-of-Suspect-7* path, not a
production path. If Suspect 7 wins, the question becomes whether
the sub-window LSODA dynamics are physically equivalent to what
production LSODA would produce — which we can't verify without
production LSODA.

## Recommended sequence

Sprint 15 should attempt option (a) first. The implementation cost
is 1 day; the upside is a production-grade LSODA driver that closes
gate 5 cleanly. If (a) hits a snag (e.g., the frozen-collision-
Jacobian approximation breaks LSODA accuracy), fall back to (b) or
(c). Do **not** attempt (b) or (c) before (a) — the analytic
Jacobian work is reusable, the others are dead ends if (a) succeeds.

### Phase 1 — Derive the Jacobian (~3 hours)

Deliverable: a new method `DensityMatrixSolver._lsoda_jacobian(t,
y_flat, a, Tg)` that returns the (3200, 3200) Jacobian of the LSODA
RHS at point `y_flat`, using the existing `_build_L_list` machinery
to construct the unitary + damping block per mode, and freezing the
collision-integral contribution at start-of-outer-step.

The unitary + damping block per mode is **already computed by
`_build_L_list`** (line 4545): `L_list[s][i]` shape `(N²×N²)` is
exactly `-i [H_i, ·] - D_off,i` in row-major vec(ρ) form for sector
s, mode i. Sprint 15 just needs to assemble these into the full
block-diagonal Jacobian (with appropriate real+imag stacking).

### Phase 2 — Wire `jac=` into the LSODA call (~2 hours)

Pass `jac=lambda t, y: self._lsoda_jacobian(t, y, a, Tg)` into
`solve_ivp`. Test on a 3-flavor SM run (gate 3 from sprint 14:
LSODA-mode 3×3 → Neff matches ETDRK2 to <1e-4).

### Phase 3 — Run gate 5 with analytic Jacobian (~1-3 hours wall-
clock)

Re-run `diag_hannestad_pointC_lsoda.py` (rtol back to 1e-6, atol
back to 1e-10). Decision tree as in sprint 14:

| Σρ_ss | Verdict | Action |
|---|---|---|
| ≈ [0.02, 0.10], ΔNeff matches Hannestad | **Suspect 7 confirmed** | Default-flip `qke_lsoda_driver_flag=True`; update `test_sterile_dw_production` fixture with explicit `qke_lsoda_driver_flag=False` for sprint-11 oracle bit-identity. Stage E.2 closes. |
| ≈ 0.65, ΔNeff matches ETDRK2 | **Suspect 8 confirmed** | Stage E.2 closure deferred to literature-review sprint. Document the LSODA+ETDRK2 agreement on Σρ_ss=0.65 vs Hannestad's [0.02, 0.10] target. |
| LSODA crashes/diverges | Jacobian bug | Debug `_lsoda_jacobian` against FD reference. |

### Phase 4 — Full gate-5 sweep (~3-6 hours wall-clock)

If Point C lands, run the full A/B/C harness
(`diag_hannestad_proj_w30_nB10k_lsoda.py`) for the complete sprint-
15 result. Targets unchanged: A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈
[0.02, 0.10].

## Stage inheritance: what NOT to touch

  * `evolve_step_ode_etdrk2` and all its sprint-12 instrumentation
    (substep snapshots, energy probes, MSW probes, eigendecomp
    fallback). Production stays on ETDRK2 until LSODA proves out.
  * `_build_H_list`, `_build_L_list`, `_assemble_collision_N`,
    `_compute_D_pair_matrix` — kernels are reused as pure functions.
  * `evolve_step`, `evolve_step_ode` — legacy, untouched.
  * Sprint-13 collision-N refactor — explicitly do not retry in any
    variant; Suspect 6 falsified at gate 6.
  * `tests/test_regression.py:278` (direct `evolve_step_ode_etdrk2`
    call) — defaults keep this path bit-identical.

## Validation targets (post-implementation)

| Gate | Command | Target |
|---|---|---|
| 1 | `pytest tests/test_regression.py -m "not slow" -v` | 4/4 bit-identical at default (LSODA off). |
| 2 | `pytest -k sterile -v` | 3/3 at default. |
| 3 | LSODA 3×3 SM with `jac=` | Neff matches ETDRK2 to <1e-4, Yp <1e-5; wall-clock <2× FD path. |
| 4 | LSODA decoupled-sterile (`cfg_4x4_decoupled`) with `jac=` | Σρ_ss < 1e-12; wall-clock <2× FD path. |
| 5 | `diag_hannestad_pointC_lsoda.py` with `jac=` | C ∈ [0.02, 0.10] OR ≈ 5.52; either way, **finishes in <90 min**. |
| 5b | `diag_hannestad_proj_w30_nB10k_lsoda.py` with `jac=` | A ∈ [0.9, 1.1], B ∈ [0.3, 0.7], C ∈ [0.02, 0.10]; finishes in <6 hours. |
| 6 | Wall-clock from gate 5b | <2× ETDRK2 baseline (1605s picard). |

If gates 1-5 pass and Suspect 7 wins: flip `qke_lsoda_driver_flag`
default to True, update fixtures, close Stage E.2.

## Minimum viable opening message

> Read `doc/STAGE_E2_SPRINT15_BRIEF.md` end-to-end before writing
> any code. Then use `EnterPlanMode` to propose a concrete sprint-15
> plan. Target for this session is to derive the analytic Jacobian
> and run gate 5 (Hannestad Point C) with it. Do NOT retry rtol /
> atol tuning — sprint 14 already showed the cost is structural,
> not accuracy-bound.

## Commit chain for context

  * `c6149dd` — **Sprint 14** (LSODA driver landed; gate 5 wall-
    clock infeasible without analytic Jacobian). Sprint 15 builds
    directly on top.
  * `db511c5` — Sprint 13 (collision-N refactor falsified Suspect 6).
  * `e699c00` — Sprint 12 (sub-step instrumentation; H-iteration
    variants).
  * `11eea23` — Sprint 11 (eigendecomp fallback).
  * `5fa6ff5` — Sprint 10 (Phase-0 QKE driver).
  * `ac1d521` — Sprint 9 (MSW diagnostic).
