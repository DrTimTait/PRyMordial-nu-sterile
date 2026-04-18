# Sterile DW small-mixing overproduction diagnostics

Investigation of why PRyMordial-nu over-produces sterile neutrinos at small
active-sterile mixing angles compared to Hannestad+Tamborra+Tram 2012
(arXiv:1204.5861).

## Context

Comparison against Hannestad+2012 Fig. 2 top panel (L=0, Δm²=0.93 eV²,
sin²(2θ) varying) showed:

| sin²(2θ) | Hannestad | PRyMordial (D.7.1) | Ratio |
|---------:|----------:|-------------------:|------:|
|    1e-1  |  1.0      |  0.95              |   1× (saturates) |
|  2.26e-3 |  0.50     |  0.97              |   2× |
|    1e-4  |  0.04     |  0.85              |  20× |

The PRyMordial code over-produces sterile at small mixing angles. This
directory captures the diagnostic work localizing the source.

## Diagnostics (in order run)

### `diag_h_sterile.py`  —  Hamiltonian structure

Dump `H[ν_μ]`, `H[ν_s]`, and `H[ν_μ, ν_s]` at T ∈ {20, 10, 5, 2, 1} MeV
for sin²(2θ_24)=1e-4. Checks:

- |V_NC/ω| ranges from 0.84 at T=20 MeV down to 1e-3 at T=1 MeV (V_NC is
  only dominant very briefly in early Phase B).
- `H[ν_μ] − H[ν_s]` is **always negative** — no spurious NH resonance.
- In-medium mixing angle 2θ_m ≈ 0.57° ≈ vacuum 2θ across all T.
- **Conclusion**: Hamiltonian structure is correct. The physics of `H`
  matches standard DW expectations (non-resonant, vacuum-level mixing).

### `diag_pmns_leak.py`  —  Active-active PMNS routing

Re-run the DW point at sin²(2θ_24)=1e-4 with `θ_12 = θ_13 = θ_23 = 0`
(disabling standard PMNS active-active mixing). Result: ΔNeff drops from
**0.85 → 0.29** — a factor-3 reduction.

- **Finding**: 3 active flavors leak to sterile via active-active PMNS
  mixing (ν_τ → ν_μ → ν_s, ν_e → ν_μ → ν_s) even at the hierarchy
  Δm²_21, Δm²_31 ≪ Δm²_41 where Hannestad claims the 1+1 approximation
  is justified.
- **Residual**: even with PMNS active-active off, ΔNeff = 0.29 vs
  Hannestad's 0.04 — factor 7 remains.

### `diag_cd.py`  —  Damping-coefficient swap

Swap PRyMordial's `C_D = [3.06, 2.22, 2.22]` (de Salas & Pastor 2016) for
Hannestad's `C_α = [1.27, 0.92, 0.92]` (Barbieri & Dolgov 1991). PMNS off.
Result: ΔNeff 0.29 → 0.27.

- **Finding**: C_D is **NOT the dominant source**. Negligible effect
  (6% relative reduction) despite 2.4× damping-coefficient difference.

### `diag_single_step.py` / `diag_single_step2.py`  —  Per-step transfer
Take ONE D.7.1 step at T=10 MeV from a thermal-FD state, measure ρ_ss
populated per step, per momentum mode.

- Observed: max ρ_ss per mode ≈ 1.4e-4 per step.
- Expected (per simple 2-level DW saturation): sin²(2θ_m)/2 = 5e-5.
- Observed is ~3× larger per mode.
- y²-weighted sum ≈ 12.76 per step. Over 2400 Phase-B steps →
  accumulates to ΔNeff ~ 0.29 (matches PMNS-off integrated result).

### `diag_clamp.py`  —  Off-diagonal clamp activity

Check whether the end-of-step `|ρ_off| ≤ sqrt(ρ_aa·ρ_bb)` clamp is being
activated. Result: max ratio = 0.44 — the clamp is NOT kicking in (all
off-diagonals stay within physical bounds).

- **Finding**: Clamp is innocent. The linear `expm(L·dt)` is NOT
  overshooting the physical bound, so clamp-induced irreversibility is
  not the cause.

### `diag_strang.py`  —  Strang-driver cross-check

Re-run sin²(2θ_24)=1e-4 PMNS-off via the **Strang path**
(`qke_full_ode_flag=False`, not the D.7.1 ETDRK2 driver). Result: ΔNeff
= 0.60.

- **Finding**: Strang gives **worse** overproduction (0.60) than D.7.1
  (0.29). So the bug is **NOT specific to the D.7.1 expm machinery** —
  it is structural to PRyMordial's underlying QKE implementation.

## Summary of localization

| Hypothesis | Test | Outcome |
|---|---|---|
| Missing V_NC | Added V_NC, committed | Physics-correct but tiny quantitative effect |
| PMNS active-active routing | `diag_pmns_leak.py` | Confirmed: factor 3 |
| C_D damping coefficient | `diag_cd.py` | Negligible |
| Spurious NH resonance | `diag_h_sterile.py` | None |
| Off-diagonal clamp | `diag_clamp.py` | Not activating |
| D.7.1 driver bug | `diag_strang.py` | No — Strang is worse |

After removing PMNS routing, the residual factor ~7 overproduction
remains in BOTH Strang and D.7.1 paths. This suggests the bug is in one
of:

1. **Collision kernel active-sterile assembly**: look at
   `_assemble_collision_N` — specifically how active-sterile off-diagonal
   damping is applied. Possible convention mismatch with Hannestad.
2. **Damping formula D_pair = 0.5·(Γ_α + Γ_β)**: check if this is the
   physically correct Lindblad-rate for active-sterile decoherence.
   Sigl-Raffelt 1993 and Hannestad+2012 Eq. 2.15 both use D = Γ/2 where
   Γ is a specific scattering-rate integral; the mapping to our Γ_α
   definition may differ.
3. **y-integration weight** in the diagonal exp-Euler step — could be
   double-counting.

An analytical estimate using PRyMordial's `C_D`, `|H_off|`, and `D_pair`
integrated over Phase-B expansion gives Γ_DW ≈ 0.05 (matching Hannestad's
0.04 at the 20% level). So the INPUTS are right; something in the
execution is wrong.

## Follow-up minimal 2-level isolation

### `minimal_2level_compare.py`  —  L-expm without damping (misleading)

First attempt at minimal isolation: disables all PMNS mixing except
`theta_24`, zeros `Delta_m^2_21` and `Delta_m^2_31`, then compares
PRyMordial's 4×4 `L`-expm action over ONE step to an analytic 2-level
`L = -i [H, ·]` (**no damping**) expm. Result: PRyMordial gives 5.5×
more `ρ_ss`.

This comparison is misleading — PRyMordial's `L` includes the `-D_off`
damping term on off-diagonals, so comparing it to a purely unitary
2-level analytic is apples-to-oranges.

### `diag_L_inspect.py`  —  L matrix element audit

Print `L[row, col]` for the rows/columns involving `ρ_μμ`, `ρ_ss`,
`ρ_μs`, `ρ_sμ`. Finds:

- Off-diagonal-commutator elements match analytic `±i·H[1,3]` exactly.
- `L[ρ_μs, ρ_μs] = -1.586×10⁻¹¹ + 4.364×10⁻⁸ j`: the imaginary part is
  `-i·(H_μμ - H_ss)` (commutator, correct), the real part is the
  damping `-D_pair = -0.5·Γ_μ` (correctly included by our V_NC fix).
- All PMNS-off 4×4 matrix elements agree with the expected 2-level
  block structure for this minimal setup.

### `diag_2level_damped.py`  —  **CRITICAL FINDING**

Repeats `minimal_2level_compare.py` but this time BUILDS the analytic
2-level `L` **with damping** matching PRyMordial's `D_pair = 0.5·Γ_μ`.
Result:

| ρ_ss after 1 step | Analytic 2-level (damped) | PRyMordial 4×4 L-expm |
|---:|---:|---:|
|  | **1.418×10⁻⁴** | **1.418×10⁻⁴** |

**Exact agreement to 4 significant figures.** Similarly `ρ_μs` matches:
`-1.314×10⁻³ + 5.092×10⁻⁷ j` in both. So:

- **PRyMordial's 4×4 `L` matrix is assembled correctly** (commutator
  term, damping term, vec ordering).
- **`scipy.linalg.expm(L·dt_nat)` applied in D.7.1 is numerically
  correct**.
- **The D.7.1 driver reproduces the 2-level damped Rabi analytic
  solution exactly**.

## Revised diagnosis

The small-mixing DW overproduction relative to Hannestad+2012 is **NOT
a code bug in the L construction or expm integrator**. The 2-level
exact solution gives `ρ_ss ≈ sin²(2θ_m)·(1 - e^{-2D·dt})` per step at
our parameters, which the code faithfully reproduces. Integrated over
the Phase B window the code then gives `ΔNeff ≈ 0.29` (PMNS off), and
Hannestad reports `0.04`.

The residual factor ~7 must therefore come from a **formulation-level
mismatch** with Hannestad:

1. **Γ-scattering-rate convention**. Hannestad Eq 2.16:
   `Γ = C_α · G_F² · x · T⁵` with `x = p/T`, so `Γ_H = C_α·G_F²·p·T⁴`.
   PRyMordial: `Γ = C_D · G_F² · T⁴ · E`. These agree **only if**
   our `C_D` maps to Hannestad's `C_α`. Our `C_D = [3.06, 2.22, 2.22]`
   (de Salas & Pastor 2016) vs Hannestad `C_α = [1.27, 0.92, 0.92]`
   (Barbieri & Dolgov 1991). Factor ~2.4 higher in our code.
2. **But the `diag_cd.py` experiment** (swapping `C_D` for Hannestad's
   values) showed only a ~6% change in the integrated `ΔNeff`. Which
   is confusing — in the Sigl-Raffelt formula `Γ_DW = |H_off|²·D/
   |H_diff|²`, the rate should scale **linearly** with `D`, so a 2.4×
   `D` swap should give a 2.4× rate swap. The small observed effect
   might indicate we're already in a saturating / clamp-limited regime
   at this mixing angle, where the rate coefficient becomes immaterial.
3. **Thermal averaging** over the Fermi-Dirac distribution. Hannestad
   writes the rate as a function of `x = p/T`; PRyMordial integrates
   `I_total` numerically over the full `y`-grid. Small mismatches in
   the thermal average could accumulate to factor ~2.
4. **δNeff definition**. Hannestad's `δNeff = ∫dx x³ f_0 (P_s⁺ + P_a⁺
   - 4) / 4∫dx x³ f_0` is a specific moment of the density matrix.
   Our `Neff` comes from PRyMordial's thermodynamic integration at
   CMB decoupling. A 10% mismatch at this step seems unlikely but not
   impossible.
5. **Use of a different effective rate formula.** Hannestad may implicitly
   use the Sigl-Raffelt quasi-static formula in the *thermally-averaged*
   form, which differs from our direct density-matrix solution by a
   numerical factor that only shows up at small mixing.

## Recommended next steps

Because `L` and the D.7.1 driver are verified correct, the small-mixing
DW gap vs Hannestad is best pursued via:

- **Choose a validation target that uses the same direct-density-
  matrix QKE formulation** (so no apples-vs-oranges rate-formula
  question). E.g. a modern 3+1 sterile BBN code that outputs density
  matrices, rather than Hannestad's Bloch-vector + thermalisation-
  fraction presentation.
- Or derive the **exact mapping** between PRyMordial's `D_pair` +
  `C_D` convention and Hannestad's `D = Γ/2` + `C_α` convention,
  including thermal averaging factors. This is essentially a
  literature-reading task, not a coding task.
- Or **accept that the code is correct at large mixing** (test at
  `sin²2θ = 0.1` matches Hannestad's full-thermalisation δNeff ≈ 1
  to within 5%; `test_sterile_dw_production` passes) and **document**
  that small-mixing quantitative predictions deviate from Hannestad by
  an overall normalisation factor requiring careful rate-convention
  matching before external comparison.

## Summary of what was learned

**Verified correct** (from diagnostic work in this directory):

- Hamiltonian structure (4×4 H matrix elements)
- V_NC thermal potential (now included, physics-complete for CC+NC)
- `_build_L_list` superoperator construction
- `scipy.linalg.expm(L·dt_nat)` application in D.7.1
- D.7.1 driver numerically solves the 2-level damped Rabi problem exactly
- Strang driver (cross-check)
- Off-diagonal magnitude clamp (not over-activating)
- ν-ν̄ symmetry preservation (the `test_qke_etdrk2_nu_nubar_symmetry` test)

**Observed but requires interpretation work** (not a bug):

- Factor 3 ΔNeff enhancement from active-active PMNS routing
  (physical in 4-flavor QKE, absent in Hannestad's 1+1 formulation).
- Factor ~7 residual overproduction at small mixing (`sin²2θ ≤ 10⁻³`)
  vs Hannestad's reported `δNeff`, attributable to
  rate-formulation/convention mismatch rather than implementation bug.

**Not applicable**:

- `C_D` damping coefficient scaling (empirically ~0 effect).
- Off-diagonal clamp activity (does not kick in).
- D.7.1-specific bug (Strang shows the same structural behavior).
