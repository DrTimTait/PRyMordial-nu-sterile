# PRyMordial-nu vs FortEPiaNO: solver and physics comparison

Written before sprint 16 to identify why FortEPiaNO (Gariazzo, de
Salas, Pastor 2019; arXiv:1905.11290) successfully integrates
Hannestad-style 3+1 sterile thermalisation with the same DLSODA
family that defeated us in sprints 14-15. Source for FortEPiaNO
side: arXiv:1905.11290v3, appendix B and section 2.

The headline: FortEPiaNO solves a **smaller, less stiff version of
the same problem** with a different choice of independent variable,
state representation, and collision-integral structure. Their
DLSODA call is far cheaper per RHS than ours, and the timeline of
the resonance crossing in their independent variable does not
require LSODA to take pathological sub-steps. Most of the cure is
not in the solver — it is in everything *around* the solver.

## Side-by-side at the highest level

| Axis | FortEPiaNO | PRyMordial-nu (current) |
|---|---|---|
| Independent variable | `x = m_e · a` (comoving, monotone with scale factor) | Temperature `T` in MeV (Phase A → Phase 0 → Phase B → Phase C handoffs) |
| Density-matrix sectors | **One** (assume no lepton asymmetry, ν = ν̄) | **Two** (separate ν and stored ν̄*; supports lepton asymmetry) |
| Off-diagonal collision form | **Damping only**: `I_αβ = -D_αβ ρ_αβ`, with `D_αβ` an `O(G_F² T⁵ y / 7π⁴/135)` constant per pair (Eq. A.16-A.20) | **Full gain + damping**: `_offdiag_collision_gain[_massive]` evaluates the same nested 2D integrals as the diagonal kernel, then subtracts `D_αβ ρ_αβ`. Sprint-13 attempted to drop this — Suspect 6 falsified. |
| Diagonal collision form | Full 2D integral over `dy_2 dy_3` (or `dy_4`) per Eq. A.3-A.4 | Same conceptual form (`_collision_integral_nu_nu`, `_collision_integral_nu_e`); numba-compiled |
| Momentum grid | **Gauss-Laguerre, ~50 nodes** truncated to `y_i < 20` from a `N ~ 350` Laguerre family. Exponentially weighted; concentrates points where the FD distribution is non-trivial. | **Linear, 100 nodes** in `y ∈ (dy/2, y_max − dy/2)`, `y_max = 100` MeV. Even spacing across the entire support. |
| 1D auxiliary integrals (Eq. B.6-B.11 J,Y,K,G) | Pre-computed at init on a 110-pt Gauss-Laguerre grid; **interpolated** during evolution | Recomputed live (or pre-tabulated for some, e.g. `D_k0/D_k2` kernel arrays) |
| Hamiltonian — vacuum | `M_F / (2y)` with `M_F = U M U†`, `M = diag(m_1², ..., m_4²)` (we set `m_1 = 0`) | Identical structure (`self._Omega_nu = self.U_PMNS @ Dm2_half @ self.U_PMNS.conj().T`) |
| Hamiltonian — matter (CC + thermal) | **Single term** `−(8√2 G_F y · m_e⁶ / 3 x⁶) · (E_ℓ/m_W² + E_ν/m_Z²)` (Eq. 2.4). `E_ℓ = diag(ρ_e, ρ_µ, 0, 0)`, `E_ν` is the active-only sandwich `S_a (∫ y³ ρ dy) S_a` with `S_a = diag(1,1,1,0)`. **No separate thermal `T⁵ y` correction.** | **Three components**: vacuum + thermal `T⁵ y / m_W²` (Sigl-Raffelt) + V_CC (`G_F · η_b · n_γ`) + V_nunu = (G_F / √2) ∫ y² (ρ_ν − ρ̄*) dy with weight `y² w` (not `y³`). The `qke_v_nunu_active_only` flag projects onto active block. |
| Photon-temperature evolution `dz/dx` | Coupled ODE solved with the same DLSODA call (Eq. B.5) | Phase A solves a separate `dT/dt` ODE; Phase B solves `_run_qke_segment` with adiabatic Tg(a) inside; Phase C is the nuclear network |
| Solver | **DLSODA** (ODEPACK Fortran), single call from `x_in ≈ 10⁻³` (i.e. `T ~ 1 GeV`) to late times, automatic stiff/non-stiff switching, optional banded analytic Jacobian | Multi-segment dispatcher: Phase A `solve_ivp(method=LSODA)` (1D thermal); Phase 0 + Phase B = ETDRK2 (sprint-15 added optional LSODA in a Tg window via `qke_lsoda_window_*`); Phase C `solve_ivp(method=BDF)` for nuclear network |
| Tolerances | `rtol = atol = 10⁻⁶` default; 10⁻⁵ stable to better than 0.1‰; 10⁻⁴ stable to ~few ‰ | `rtol=1e-6, atol=1e-10` for ETDRK2 corrector accuracy; sprint-15 LSODA defaults match. Loosening to 1e-2/1e-4 saved nothing. |
| State DOFs (4×4 sterile, Hannestad) | `N² · N_y + 1 = 16 · 50 + 1 = 801` real | `2 · N² · N_y = 2 · 16 · 100 = 3200` real (4× FortEPiaNO) |
| Total RHS evaluations / outer step (rough) | Single solve_ivp adapts; ~few hundred RHS calls total over the **entire run** for a typical 4×4 case | One ETDRK2 outer step costs ~1 RHS-equivalent (`_assemble_collision_N` + 2 `_build_H_list`); LSODA outer step costs 3-5 RHS calls; total RHS calls per Hannestad Point C run is ~10⁴ for ETDRK2 |
| Wall-clock for typical 4×4 case | "few minutes on four cores" with `N_y = 20` | ETDRK2: ~27 min Hannestad Point C (single-threaded, n_B = 10000). LSODA: did not finish 6 hours sprint-14, did not finish 3 hours sprint-15 windowed. |

## What FortEPiaNO does that we do not

The five differences ranked by *probable wall-clock impact*:

### 1. Off-diagonal collisions are damping-only (Eq. A.16-A.20)

Their `I_αβ = -D_αβ ρ_αβ` for `α ≠ β`. `D_αβ` is a fixed constant
times `G_F² T⁵ y`, no nested `dy₂ dy₃` integrals. This is the
single biggest per-RHS speedup vs us — the off-diagonal collision
gain is the heaviest component of `_assemble_collision_N`, and
they skip it entirely.

The catch: the damping-only approximation drops the off-diagonal
*production* term (the gain), keeping only the destruction. For
active-active oscillations near equilibrium that's fine. For
active-sterile mixing across a resonance it might not be. FortEPiaNO
get away with it because they only damp; the resonance-driven
production happens through the unitary commutator, and the
damping just bleeds coherence at the rate `D_αβ`. Sprint 13
attempted scope (b) of the equivalent refactor and **falsified
Suspect 6** because the variant we tried also broke Phase B; the
FortEPiaNO form is subtly different (constant `D_αβ` per pair, no
gain term at all). Worth re-attempting under sprint 13's framing
with the FortEPiaNO formula explicitly.

### 2. Single density matrix instead of two sectors

FortEPiaNO assumes no lepton asymmetry ⇒ `ρ_ν = ρ̄_ν`, so they
evolve **one** `(N×N)` Hermitian per momentum mode. We carry a
second `ρ̄*` sector for the antineutrino, doubling the state vector
and roughly doubling the RHS evaluation cost.

For Hannestad Point C with `xi_*_init = 0`, the assumption holds
exactly at `t = 0` and is preserved by the unitary + collision
dynamics in the absence of CC asymmetry. We could in principle
short-circuit sector 1 when the run is initialised with no
asymmetry — that would halve our RHS cost.

### 3. Independent variable `x = m_e · a` instead of `T`

FortEPiaNO's RHS has explicit `x⁻⁴` and `x⁻⁶` factors (Eq. 2.4),
which monotonically suppress the collision and thermal-matter
terms as the universe expands. The integrator sees a problem that
gets *less* stiff with time. Our independent variable is `T` (or
`a` indirectly via the temperature-driven outer-step grid), and we
also have an explicit segment split at `T_phase0_start = 100 MeV`
and `T_boltz_start = 30 MeV` because each segment's stiffness
profile is different.

The DLSODA cost in FortEPiaNO is therefore concentrated near the
start of the run (`x = 10⁻³`, `T = 1 GeV`); by the time DLSODA
reaches the resonance the sub-step size has been adaptively
relaxed by orders of magnitude. We start LSODA cold inside the
resonance window (Tg ~ 80 MeV in our diag) where the dynamics are
already maximally stiff, denying LSODA the warm-start that
FortEPiaNO benefits from.

### 4. Gauss-Laguerre momentum grid (~50 nodes vs our 100)

Halving `N_y` (from 100 to 50) is a **4× speedup on the 2D
collision integral** (`O(N_y²)` per matrix entry per RHS call).
Halving `N_y` also halves the state vector. Combined: ~8× on
total wall-clock at fixed `N_y`-precision, because Gauss-Laguerre
weights concentrate where the integrand actually lives. We use
linear spacing because our `_compute_D_pair_matrix` and other
kernel pre-computes index by `dy`-uniform offsets — switching
would touch a lot of plumbing.

### 5. Pre-computed and interpolated 1D auxiliaries (J, Y, G₁, G₂)

FortEPiaNO pre-computes the QED + leptonic phase-space functions
on a fine grid at init and interpolates during evolution. We
either recompute live or use scipy `interp1d` over different
tables; the per-RHS cost of these pieces is small for both, but
their cumulative cost across `~10⁴` RHS calls is non-trivial.

## What we do that FortEPiaNO does not

* We have **two density-matrix sectors** that allow asymmetry runs
  (Stage C cosmological-asymmetry depletion). FortEPiaNO would
  need a code change to support `ξ_νe ≠ 0`.
* We support **multiple thermal-matter formulae** (`qke_damping_formula
  ∈ {symmetric, mirizzi, gariazzo}`) and the full Sigl-Raffelt
  thermal correction `T⁵ y / m_W²`. FortEPiaNO uses the simpler
  Eq. 2.4 form (no `T⁵` term in V_CC; they treat the thermal
  contribution as part of `E_ℓ / m_W²`).
* We have an explicit **multi-segment driver** with phase-aware
  bookkeeping (entropy / energy continuity at each handoff). This
  gives us tighter control over each segment but introduces the
  driver-swap overhead and the phase-B cold-start problem.
* We use **ETDRK2** (exponential time-differencing) as the
  production driver for Phases 0 and B. FortEPiaNO does not need
  this — DLSODA suffices because the per-RHS cost is so much
  lower for them.

## What we should consider for sprint 16

In priority order (highest expected impact first):

### Tier 1 — physics changes that make our problem look more like theirs

* **(P1) Damping-only off-diagonal collision form.** Re-implement
  `_assemble_collision_N` with a `qke_offdiag_damping_only_flag`
  that uses the FortEPiaNO closed-form `D_αβ` constants
  (Eq. A.17-A.20) and drops the gain term entirely. Pair this
  with the existing `_build_L_list` damping bookkeeping. Expected
  per-RHS speedup: 5-10×. Sprint 13 falsified Suspect 6 with a
  different refactor; this is a different formula and worth a
  re-test under sprint 13's gates. **If this works, the LSODA
  driver as-shipped (sprint 14) might already be tractable.**

* **(P2) Single-sector mode** for runs without lepton asymmetry.
  Add a `qke_single_sector_flag` that detects `xi_* = 0` at init
  and evolves only one `(2, n_components, Ny)` state, reconstructing
  ν̄ = ν at use-sites. Expected speedup: 2× on RHS evaluation cost
  and state size (so 2× on LU and 4× on FD-Jacobian if we ever
  fall back).

### Tier 2 — solver-side improvements that benefit ETDRK2 too

* **(P3) Switch to Gauss-Laguerre momentum grid.** Implement
  `qke_grid_type ∈ {linear, gauss_laguerre}` with `N_y_laguerre`
  defaulting to 50. The collision-kernel pre-tables
  (`_compute_D_pair_matrix`, `_precompute_D_tables`) need to
  re-index by quadrature node rather than `dy`-uniform offset,
  but the kernel formulae are unchanged. Expected speedup:
  ~4× on collision-integral-bound RHS.

* **(P4) Re-use FortEPiaNO's `dz/dx` formulation** instead of our
  Phase-A → Phase-B handoff. Couple the photon-temperature ODE
  into the same `solve_ivp` call as the QKE so the integrator
  sees a single stiff system on a monotone independent variable.
  Risky — touches the entropy bookkeeping and the n→p weak rate
  freeze-out timing — but eliminates the cold-start problem at
  Phase B. Better deferred until P1 and P2 land.

### Tier 3 — driver experiments (sprint-16 brief's original direction)

* **(P5) ETDRK4** as previously planned. Still useful as a
  Suspect-7 falsifier independent of the LSODA path, even if P1+P2
  make LSODA viable.

## Concrete next move

The lowest-cost / highest-impact path is **P1**: reimplement the
off-diagonal collision term as the FortEPiaNO damping-only form.
This is a contained edit to `_assemble_collision_N` and a new
flag, gateable so the existing tests can pin the legacy behaviour.
If it works (Hannestad Point C completes in tractable wall-clock
under LSODA + analytic banded Jacobian + damping-only), it
collapses sprints 13-16 into a single positive result and gives
us the Suspect 7 / Suspect 8 verdict immediately. If it doesn't,
it's still a small reversible commit.

P2 (single-sector) is the natural follow-up for additional
speedup once P1 establishes that the damping-only form is
quantitatively acceptable.

P5 (ETDRK4) keeps its original sprint-16 priority as an
independent cross-check on Suspect 7 even if P1 cures the LSODA
wall-clock.

## What we have not yet checked

* Whether **FortEPiaNO is publicly available** at the URL given
  in the appendix (`https://bitbucket.org/ahep_cosmo/fortepiano_public`).
  If yes, pulling the source would let us cross-check the exact
  damping coefficients and the Gauss-Laguerre weight construction
  byte-for-byte. Recommend a quick clone before sprint 16 starts.
* Whether their `D_αβ` constants reproduce the same Hannestad
  Point C result we expect. The Hannestad target band [0.02, 0.10]
  for `Σρ_ss` was derived under their damping-only formulation.
  If we adopt P1, we are integrating the *same* approximation that
  defines the target — which is a feature, not a bug, for the
  Suspect-7 / Suspect-8 verdict.
