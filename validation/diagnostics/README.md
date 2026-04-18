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

## Open work

- Isolate the residual factor-7 overproduction. Likely requires a
  line-by-line audit of how active-sterile off-diagonal dynamics couple
  to the sterile diagonal in the QKE formulation. A minimal 2-level test
  case (1 active, 1 sterile, bypassing PRyMordial's 4-flavor plumbing)
  would help isolate driver-vs-formulation.
- Once fixed, re-run `validation/sterile_DW_literature.py` to validate
  quantitatively, then proceed to the Shi-Fuller literature comparison
  (Paper 2: Mirizzi+2012 / Saviano+2013).
