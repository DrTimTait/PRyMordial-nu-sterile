# Stage D.7 brief: ETDRK2 with damping-in-L, per-mode matrix exponential

This is a single-file handoff for a fresh context window picking up the
Stage D ODE-driver work on `PRyMordial-nu-sterile` where Stage D.6 left
it. Read this first. By the end you should know: what D.6 tried and
why it failed, what D.7 actually needs to do, and what to leave alone.

## One-paragraph orientation

D.6 built an eigenbasis-aware ETDRK2 scaffold behind
`qke_ode_etdrk2_flag` to restore O(dt²) at the Shi-Fuller MSW
resonance. It works numerically for 3×3 SM and sterile-DW (ξ=0), but
the Cox-Matthews corrector **breaks ν-ν̄ symmetry analytically** in
the current formulation: sector-1 rotations `(V*, Vᵀ)` vs sector-0
`(V, V†)` produce φ-factor outputs that are not complex conjugates of
each other, and the corrector's `φ_2 · (N(ρ*) − N(ρ_n))` term
accumulates the mismatch to an O(1) drift over a Phase-B trajectory.
D.6 shipped as first-order predictor-only. D.7 is the structural
rewrite: absorb the flavor-basis damping operator into the linear
part `L`, then compute `e^{L dt}`, `φ_1(L dt)`, `φ_2(L dt)` via a
per-mode per-sector matrix exponential on the **vectorised
superoperator**, not via an H-eigenbasis decomposition.

## What to read, in order

Budget ~20 minutes for this before writing anything.

1. `CLAUDE.md` — project overview.
2. `doc/ROADMAP.md` — Stage D section. The Stage D.6 subsection
   contains:
   * The full rotation-mismatch derivation with the `A_0_out` vs
     `A_1_out` expressions that show ν-ν̄ symmetry is not preserved
     by the corrector.
   * The four empirical variants tried (full corrector, `k=l`
     eigendiag zeroed, eigenvector phase-fix, predictor only) and
     their measured `n_ξe` drift on sterile DW.
3. `doc/STAGE_D6_BRIEF.md` — the D.6 design doc. Useful only for the
   "Conventions and gotchas" section (ν̄ storage convention, flavor-
   pair layout, collision-operator parts). Its "Concrete
   implementation sketch" is superseded by this brief.
4. `git log --oneline -15` — commit headlines.
5. `git show 943acaf` — the Stage D.6 landing commit with the
   symmetry-breaking summary in the message body.
6. `validation/stage_d6_sf_convergence.out.txt` — the SF
   convergence sweep under the D.6 predictor-only path. Shows
   Neff(n_B=2400,4800,9600) oscillating (3.87→3.89→3.84) rather
   than converging; n_ξe drifts from +2.2 (Strang) to −19/−34
   (D.6 predictor). **This is what D.7 must fix.**
7. `PRyM/PRyM_boltzmann.py`:
   * `evolve_step_ode_etdrk2` (~4432) — the D.6 predictor-only
     implementation. Its docstring contains the "Why no corrector"
     analysis.
   * `_assemble_collision_N` (~4263) — full collision RHS
     assembly. Returns `(N_list, I_total)` where `N_list[s]` is
     the per-sector Hermitian `(Ny, N, N)` matrix in eV natural
     units (already usable by D.7).
   * `_build_H_list` (~4161) — per-sector H in eV. Unchanged by D.7.
   * `_apply_unitary_from_eigs` (~4232) — retained for cheap pure-
     unitary work; D.7 replaces it with the full L-exponential
     per sector.
   * `_etdrk2_phi_apply` (~4337) — the H-eigenbasis φ-factor routine
     that fails in D.6. D.7 will likely not reuse this; the
     superoperator approach computes φ_k via the augmented-matrix
     exponential trick (Al-Mohy & Higham 2011) instead.
   * `evolve_step` and `_apply_unitary` — **do not touch** (pre-D
     Strang references depend on them).

## The core idea of D.7

The linear operator acting on the density matrix is

```
L[ρ]_αβ = −i [H, ρ]_αβ − D_αβ · ρ_αβ
```

where `H` is the flavor-basis Hamiltonian (eV) and `D_αβ = ½(Γ_α +
Γ_β)` is the flavor-basis pair damping matrix (real, so acts as an
entry-wise Hadamard multiplication on the flavor-basis elements of ρ).

`L` contains both the oscillatory MSW physics (through −i[H,·]) and
the stiff damping (through −D⊙·). Because D is diagonal in flavor-
basis pairs and H is **not** H-eigenbasis-diagonal when its off-
diagonal vacuum mixing is nonzero, `L` is **not diagonal in any
single basis**. This is exactly why D.6's eigenbasis-of-H trick
couldn't absorb D into the phi factors without breaking ν-ν̄
symmetry.

The fix is to stop trying to diagonalise L. Instead, vectorise
the Hermitian matrix ρ as a complex length-N² column (standard
Liouville-space lift), represent L as an explicit N²×N² complex
matrix, and compute `e^{L dt}`, `φ_1(L dt)`, `φ_2(L dt)` via
`scipy.linalg.expm` directly. For N=3 this is a 9×9 matrix; for
N=4, a 16×16 matrix. Per mode, per sector, per step.

## Superoperator structure

For any matrix A, `vec(A)_{α·N+β} = A_αβ` (row-major flatten).

```
vec(H A)     = (H ⊗ I_N) · vec(A)
vec(A H)     = (I_N ⊗ H^T) · vec(A)
vec([H, A])  = (H ⊗ I − I ⊗ H^T) · vec(A)
vec(D ⊙ A)   = diag(vec(D)) · vec(A)
```

### Sector 0 (physical ρ, rho_all[0])

```
L_0 = −i (H_ν ⊗ I − I ⊗ H_ν^T) − diag(vec(D))
```

### Sector 1 (stored ρ̄*, rho_all[1])

The stored-convention equation is `d(ρ̄*)/dt = +i[H_ν̄*, ρ̄*] +
C̄_stored`. Using `conj(H) = H^T` for Hermitian H:

```
L_1 = +i (H_ν̄^T ⊗ I − I ⊗ H_ν̄) − diag(vec(D))
```

(The damping part is unchanged between sectors because D is real.)

You can verify the sector-1 expression by rederiving from the
physical ν̄ equation and taking complex conjugate — same recipe D.3
used for `_apply_unitary`'s sector-1 fix.

## Computing e^{L dt}, φ_1, φ_2 in one shot

Al-Mohy & Higham (2011), "Computing the Action of the Matrix
Exponential, with an Application to Exponential Integrators" — Eq.
(2.1) and surrounding text. The augmented matrix

```
    [ L·dt    I_{N²}    0     ]
M = [ 0       0         I_{N²} ] · dt         (block 3(N²) × 3(N²))
    [ 0       0         0     ]
```

has `exp(M)` with the following top-row blocks:

```
exp(M)[0:N², 0:N²]      = e^{L dt}
exp(M)[0:N², N²:2N²]    = dt · φ_1(L dt)
exp(M)[0:N², 2N²:3N²]   = dt² · φ_2(L dt)
```

One call to `scipy.linalg.expm(M)` — done. (For N=4 the augmented
matrix is 48×48 complex; negligible per-mode cost.)

Standard reference: Krogstad (2005), "Generalized integrating
factor methods for stiff PDEs," expresses the same ETDRK2 scheme
with matching conventions.

## ETDRK2 step with the superoperator

```
Predictor:  vec(ρ*)      = e^{L dt} · vec(ρ_n) + dt · φ_1(L dt) · vec(N(ρ_n))
Corrector:  vec(ρ_{n+1}) = vec(ρ*) + dt · φ_2(L dt) · vec(N(ρ*) − N(ρ_n))
```

Here N(ρ) is now **only the nonlinear collision source** — the
gain terms (and any piece not captured by the linear L). Damping
has moved into L, so N's flavor-basis diagonals are
`I_total_α − (−Γ_α · ρ_αα)` = `I_total_α + Γ_α · ρ_αα`, and its
off-diagonals are `S_gain_αβ` (no `−D_αβ · ρ_αβ` term, because that
is now in L).

That identity `I_total_α + Γ_α · ρ_αα` is the "gain" piece: in
the detailed-balance limit `I_total_α = −Γ_α (ρ_αα − ρ_eq_α)`, so
`N_diag_α = Γ_α · ρ_eq_α`, which is the thermal-equilibrium source.
Check: if the collision integrals are consistent with the C_D
damping coefficients, the subtraction is numerically sensible.

If the subtraction produces an ill-conditioned residual (because
I_total's detailed-balance structure uses the full collision
kernel, not just the schematic −Γ_α relaxation), consider an
alternative split where `L` includes the _full_ diagonal damping
rate estimated from `I_total` at ρ_n. But starts the implementation
with the clean `Γ_α · ρ_αα` subtraction.

## Concrete implementation sketch

1. **New method `_build_L_list(self, rho_all, a, Tg) → (L_list,
   N_list)`** in `DensityMatrixSolver`:
   * Call `_build_H_list` to get H_nu, H_nubar (eV).
   * Compute `Gamma[α] = C_D[α] · G_F² · T⁴ · E` in eV.
   * Build `D_mat[α, β] = ½(Γ_α + Γ_β)` (real, shape (N, N)).
   * For each (sector s, mode i):
     - Form the N²×N² linear part in eV:
       sector 0: `L_si = −i·kron(H_i, I) + i·kron(I, H_i^T) − diag(vec(D_mat))`
       sector 1: `L_si = +i·kron(H_i^T, I) − i·kron(I, H_i) − diag(vec(D_mat))`
     - Stack into shape (2, Ny, N², N²).
   * Derive `N_list` by calling `_assemble_collision_N` and then
     adding back the damping term (so the returned N is "gain only"
     per the split):
       `N[s][i, α, α] += Γ_α · rho_all[s,α,i]`          # add back diagonal damping
       `N[s][i, α, β] += D_αβ · rho_all[s, re_αβ, i] + 1j * ... ` # off-diag damping added back
     (or equivalently, subtract the damping contributions at
     assembly time inside `_assemble_collision_N` via a new flag).

2. **New method `_etdrk2_expm_phi(self, L, dt_nat) → (Phi0, Phi1, Phi2)`**
   returning three `(Ny, N², N²)` complex arrays via the augmented
   matrix trick. Loop over Ny modes (maybe over Ny × 2 sectors
   combined). `scipy.linalg.expm` is not batched; loop is fine at
   ~200 small matrix exps per step.

3. **Rewrite `evolve_step_ode_etdrk2`:**
   ```
   L_list, N_list = self._build_L_list(rho_all, a, Tg)
   dt_nat = dt * self._eV_to_secm1
   # Per sector:
   for s in (0, 1):
       Phi0, Phi1, Phi2 = self._etdrk2_expm_phi(L_list[s], dt_nat)
       rho_vec = vec(self._to_mat(rho_all[s]))       # (Ny, N²)
       N_vec_n = vec(N_list[s])                       # (Ny, N²)
       # Predictor:
       rho_vec_star = einsum('ijk,ik->ij', Phi0, rho_vec) \
                    + einsum('ijk,ik->ij', Phi1, N_vec_n)
       rho_all[s] = self._to_vec(unvec(rho_vec_star))
   # Recompute N at rho_star
   _, N_star_list = self._build_L_list(rho_all, a, Tg)  # or just the N part
   # Corrector:
   for s in (0, 1):
       dN_vec = vec(N_star_list[s] - N_list[s])
       rho_vec_star = vec(self._to_mat(rho_all[s]))
       rho_vec_new = rho_vec_star + einsum('ijk,ik->ij', Phi2_s[s], dN_vec)
       rho_all[s] = self._to_vec(unvec(rho_vec_new))
   # Hermitise, clamp off-diag, clip diag (same as evolve_step_ode).
   ```

4. **Utility `vec`/`unvec`** — flatten `(Ny, N, N)` Hermitian complex
   to `(Ny, N²)` complex and back. Careful with row-major convention
   matching the H ⊗ I ordering chosen above.

5. **Flag gating unchanged:** `qke_ode_etdrk2_flag = True` routes to
   the rewritten `evolve_step_ode_etdrk2`. D.6's existing scaffold
   structure (dispatcher, smoke test) stays.

6. **Reset flag in `tests/test_regression.py::_reset_flags`** — already
   there from D.6.

## ν-ν̄ symmetry check (the D.6 failure mode this must avoid)

Add an explicit unit test: start from a ν-ν̄ symmetric state (ρ =
ρ̄*, thermal FD, ξ=0, V_CC=0 via monkeypatch), run 10 ETDRK2 steps
with realistic dt and H_nu = H_nubar (force V_CC=0 during the test),
then assert that `max |rho_all[0] - conj(rho_all[1])| < 1e-10`
pointwise. D.6's predictor passes this, the D.6 full corrector
fails it by ~O(dt · mixing). D.7 must also pass.

## Validation targets

- **Pass all existing slow regression tests** (modes 1, 2, 5, 5b,
  5c, 6, sterile_stage_a_invariant, sterile_dw_production,
  sterile_sf_asymmetry_depletion).
- **Tighten `test_mode5c_qke_ode_etdrk2`** with a frozen reference
  that matches `test_mode5b_qke_full_ode` to 10⁻³ on Neff.
- **Sterile DW (ξ=0)** under `qke_ode_etdrk2_flag=True` should match
  Strang to 10⁻³ on Neff and 10⁻² on Σρ_ss.
- **SF (ξ=5e-2) convergence sweep** via
  `validation/stage_d6_sf_convergence.py` (rename to stage_d7):
  at `n_B ∈ {2400, 4800, 9600}` the drift ratio should be ~0.25
  (O(dt²)), and the Richardson-extrapolated Neff should match the
  Strang Richardson limit 3.968 to 10⁻³. Default-n_B Neff must
  land within 10⁻³ of that limit — this is the whole point.
- **No runaway in n_ξe**: under SF ξ=5e-2, final |n_ξe| should be
  O(1) (initial scale), not O(10). Strang lands at +2.2; D.7 should
  land in the same ballpark.

## Cost budget

- 48×48 complex `expm` per (mode, sector, step). ~0.5 ms each on
  a laptop. Ny=100 × 2 sectors × 2400 steps = 480k expms × 0.5 ms
  ≈ 4 min of overhead per Phase-B call. Full Phase B (including
  collision kernels) currently ~130 s under Strang; D.7 adds
  ~4 min → ~6-min runs. Acceptable for validation; can be
  optimised later (batch via `jax.vmap` + `jax.scipy.linalg.expm`,
  or Padé-and-squaring by hand for small matrices).
- Memory: 3·N² = 48 columns × 48 rows × 8 bytes (complex128) ≈
  18 KB per (mode, sector). 200 modes × 2 × 18 KB = 7 MB live.
  Fine.

## Do-not-touch list

- `evolve_step` — Strang / quasi-static. Pre-Stage-D tests depend
  on it.
- `evolve_step_ode` — Stage D.1 frozen reference (mode 5b).
- `_apply_unitary` — D.3 ν-bar convention fix baked in.
- Default values of any existing flag. `qke_ode_etdrk2_flag` is
  False by default.
- D.6 auxiliary scaffolding (`_build_H_list`, `_apply_unitary_from_eigs`,
  `_assemble_collision_N`, `_etdrk2_phi_apply`) can be reused or
  replaced, but breaking them requires updating every call site.

## Session realism

ROADMAP scopes D.7 as ~1 week. In one context window, pick ONE of:

- **(a) Plan + scaffold + ν-ν̄ symmetry unit test + 3×3 smoke.**
  `_build_L_list`, `_etdrk2_expm_phi`, `evolve_step_ode_etdrk2`
  rewritten. Passes `test_mode5c_qke_ode_etdrk2` and the new
  symmetry unit test. Defer SF validation.
- **(b) Full implementation, sterile DW green, SF validation.**
  Runs `validation/stage_d7_sf_convergence.py` and demonstrates
  O(dt²) at the resonance. Leaves a frozen-reference pytest for
  follow-up.
- **(c) Debugging a partial D.7 implementation handed off from a
  prior session.**

State the choice explicitly in the opening message of the new
session.

## Minimum viable opening message for the new session

> Read `doc/STAGE_D7_BRIEF.md` end-to-end before writing any code.
> Then use `EnterPlanMode` to propose a concrete Stage D.7
> implementation plan. My target for this session is {ONE of the
> three in "Session realism"}. Do not touch `evolve_step`,
> `evolve_step_ode`, or `_apply_unitary`, and do not change any
> existing default flag.
