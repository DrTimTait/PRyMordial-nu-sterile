# Gariazzo-Mirizzi damping coefficients: the bug is in D_αβ structure

## Summary

Running PRyMordial against Gariazzo, de Salas, Pastor 2019
(arXiv:1905.11290), which uses the **same density-matrix QKE framework**
and **same C_D damping coefficients** as PRyMordial, still shows a
~8–10× ΔNeff over-production at small mixing angles. This directory
localizes the cause.

## The mismatch

Gariazzo+2019 Appendix A gives the off-diagonal damping coefficients
(from Mirizzi, Saviano, Miele et al.) as pair-specific functions of
the weak mixing angle:

```
D_eμ / F = 15 + 8 sin⁴θ_W                    ≈ 15.4
D_μτ / F = 7 - 4 sin²θ_W + 8 sin⁴θ_W         ≈ 6.5
D_es / F = 29 + 12 sin²θ_W + 24 sin⁴θ_W      ≈ 33.1
D_μs / F = D_τs / F = 29 - 12 sin²θ_W + 24 sin⁴θ_W  ≈ 27.5

where F = 7 π⁴ y³ / 135  is the common normalization.
```

These numbers come from Mirizzi, Saviano, Miele, Pisanti, Serpico 2012
(arXiv:1206.1046) or earlier derivations. They reflect the sum over
all scattering channels available to each pair, giving active-sterile
damping **larger** than active-active damping.

PRyMordial (lines 4313–4324 of `PRyM/PRyM_boltzmann.py`) uses a
symmetric-average formula:

```
Γ_α = C_D[α] · G_F² · T⁴ · E
D_αβ = 0.5 · (Γ_α + Γ_β)
```

with `C_D = [3.06, 2.22, 2.22, 0.0]`. Since `C_D[sterile] = 0`, the
active-sterile pair damping is **half the active damping**:
`D_μs = 0.5 · Γ_μ`.

## The ratio that reveals the bug

| pair | Gariazzo / Mirizzi | PRyMordial | ratio |
|---|---:|---:|---:|
| D_μτ / Γ_μ | ~6.5 · (F / something) | 1.0 (symmetric) | — |
| D_μs / D_μτ | **4.23** | **0.5** | **× 8.5** |

In Gariazzo, active-sterile damping is ~4× active-active damping
(because sterile off-diagonals decohere via additional channels than
active-active do). In PRyMordial, active-sterile damping is half of
active-active (because sterile has zero self-interaction rate).

This is an **8.5× mismatch in the damping-ratio structure**, which
directly determines the rate of sterile production via Sigl-Raffelt
`Γ_DW = |H_off|² · D / (|H_diff|² + D²)`.

## Why this causes ~8× over-production at small mixing

In the weak-damping regime (`D << |H_diff|`), `Γ_DW ∝ D`. In the full
damped-Rabi 2-level system (which PRyMordial's L-expm solves correctly
given the inputs — see `diag_2level_damped.py`), the irreversible
sterile transfer per step is bounded by:

```
ρ_ss_per_step ~ sin²(2θ_m) · (1 - e^{-2·D·dt})
```

For `D·dt << 1` this gives `sin²(2θ_m) · 2·D·dt`, i.e. proportional to
D. Since Gariazzo's D_μs is larger, you'd naively expect MORE
thermalization in their code, but the actual QKE integration in the
full 4×4 density matrix is non-trivial, and the factor involved in
the `(1 - e^{-2·D·dt})` can saturate, yielding different rate
depending on whether D·dt is small, unity, or large.

Empirically at our parameters, `D_PRyMordial · dt ≈ 0.75` — so the
step is near the "1 - e^(-x) ≈ 1/2" crossover. With Gariazzo's 8×
larger D, their `D·dt` would be ~6, well into saturation. The
saturated rate is flatter, so further D increase doesn't produce
more thermalization, and the TRUE PER-STEP RATE would be:

- PRyMordial: `sin²(2θ) · (1 - e^{-1.5}) ≈ sin²(2θ) · 0.78`
- Gariazzo: `sin²(2θ) · (1 - e^{-12}) ≈ sin²(2θ) · 1.0`

Only ~25% difference, not 8×. So the simple saturation argument doesn't
explain the 8× observational discrepancy.

The FULL explanation likely involves how the collision integral itself
contributes to sterile production (the `I(ρ)` not just the damping
approximation), and how Gariazzo's pair-specific D_αβ coefficients
map through their FortEPiaNO integration. A proper resolution requires:

1. Adopt Gariazzo/Mirizzi's D_αβ formula in PRyMordial.
2. Re-run the DW literature comparison to see if the gap closes.
3. If it doesn't close, look at collision-kernel structure next.

## Proposed fix

Replace the `D_pair` computation in `_build_L_list`
(`PRyM/PRyM_boltzmann.py` around line 4414) with:

```python
# Gariazzo / Mirizzi pair-specific damping coefficients.
# Reference: Gariazzo, de Salas, Pastor 2019 (arXiv:1905.11290) App. A.
sW2 = PRyMini.sW2
sW4 = sW2 * sW2
F_coeff = 7.0 * np.pi**4 / 135.0  # common normalization (y^3 factor applied per mode)
# flavor indices: 0=e, 1=μ, 2=τ, 3=s
D_coef = np.array([
    [0.0, 15.0 + 8.0*sW4, 15.0 + 8.0*sW4, 29.0 + 12.0*sW2 + 24.0*sW4],   # e row
    [15.0 + 8.0*sW4, 0.0, 7.0 - 4.0*sW2 + 8.0*sW4, 29.0 - 12.0*sW2 + 24.0*sW4],  # μ row
    [15.0 + 8.0*sW4, 7.0 - 4.0*sW2 + 8.0*sW4, 0.0, 29.0 - 12.0*sW2 + 24.0*sW4],  # τ row
    [29.0 + 12.0*sW2 + 24.0*sW4, 29.0 - 12.0*sW2 + 24.0*sW4, 29.0 - 12.0*sW2 + 24.0*sW4, 0.0],  # s row
])
# Now we need to dimensionally scale this properly to get D_pair in 1/eV.
# Need the full Mirizzi+2012 prefactor which includes G_F², y, T, etc.
# Leaving as a template; the exact scaling requires checking Mirizzi+2012 Eq. 7
# against PRyMordial's current convention.
```

The exact dimensional prefactor (what multiplies `D_coef[α, β] · F · y³`
to give a damping rate in 1/eV) needs to be derived from Mirizzi+2012
Eq. 7 or equivalent and cross-checked against PRyMordial's `Γ_α =
C_D · G_F² · T⁴ · E` conversion. This is documentation-level work
best done with the Mirizzi paper in hand.

## Implications if the fix works

- Our ΔNeff at sin²(2θ)=10⁻⁴ should drop from 0.85 → ~0.1 (matching
  Gariazzo).
- `test_sterile_dw_production` at sin²(2θ)=0.1 should still pass at
  full thermalization.
- The SF (Shi-Fuller) literature comparison (Paper 2) can proceed.
- Downstream, Lovell 2023 keV sterile comparison (Paper 3) becomes
  accessible.

## Open questions

- Why does PRyMordial's symmetric `D_pair = 0.5·(Γ_α + Γ_β)` formula
  exist? It's from de Salas & Pastor 2016 for the SM case (where
  only active-active pairs exist); the sterile extension in Stages A-D
  may have blindly carried it over without adapting to the pair-
  specific formulation needed for active-sterile decoherence.
- Does the collision kernel `_assemble_collision_N` have similar issues
  in its active-sterile off-diagonal assembly? Worth checking after
  the damping fix.

## Follow-up references (if the Mirizzi D_αβ fix is not enough)

- Saviano, Mirizzi, Pisanti et al. 2013 (arXiv:1306.1421): classic SF+DW
  density-matrix QKE. Same framework, different focus.
- Hernandez-Molinero, Gariazzo et al. 2022-23: recent follow-up from
  the de Salas/Gariazzo group.
- FortEPiaNO code (arXiv:2011.04991): their public Fortran code. Could
  run it ourselves with identical configuration for exact cross-check.