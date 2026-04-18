# Stage E.1 brief: fix D_αβ pair-specific damping formula

Single-file handoff for a fresh context window taking over from the
DW-literature investigation that ended in commit `49e31f4`. Read this
first. By the end you should know exactly what to change in the code,
what validation targets to hit, and what regression tests to preserve.

## One-paragraph orientation

A long chain of diagnostics during Stage D comparison against
Hannestad+2012 and Gariazzo+2019 localized a small-mixing DW
over-production bug (ΔNeff ~ 0.85 vs Hannestad 0.04 at
sin²(2θ)=10⁻⁴, Δm²=1 eV²) to **one specific line** in
`PRyM/PRyM_boltzmann.py`: the symmetric damping formula
`D_pair = 0.5·(Γ_α + Γ_β)` used in `_build_L_list` (and the
mirrored form embedded in `_assemble_collision_N` off-diagonal
assembly). This formula is physically **wrong** for active-sterile
pairs. The correct formulation is Mirizzi+2012 Eq. (28) with
Hannestad-style scattering/annihilation coefficients, which for
active-sterile pairs gives damping ~2–8× larger than our current
half-average. Stage E.1's job: replace the symmetric formula with
the pair-specific Mirizzi form and verify the DW literature
comparison against Gariazzo drops to < 20% relative.

## What to read, in order

Budget ~30 minutes for reading before writing any code.

1. **`CLAUDE.md`** — project overview.
2. **`doc/ROADMAP.md`** — Stage D section plus Stage D.7.1 landing.
   Skip the D.7.2 attempt (null result). The Stage D work is done;
   this brief is Stage E.1 (physics validation and corrections).
3. **`validation/diagnostics/README.md`** — full diagnostic chain:
   PMNS routing (factor 3), C_D swap (null), L-expm correctness
   (proved via minimal 2-level), Strang cross-check (worse than
   D.7.1), clamp activity (not firing).
4. **`validation/diagnostics/GARIAZZO_DAMPING_FIX.md`** — the
   bug-localization write-up. This is the precursor to the present
   brief and contains the key ratio mismatch:
   D_μs/D_μτ = 4.23 (Gariazzo) vs 0.5 (PRyMordial).
5. **`References/1206.1046v2_mirizzi2012.pdf`** — Mirizzi, Saviano,
   Miele, Pisanti, Serpico 2012. **Key equation: Eq. (28)**, with
   coefficients from Eq. (29).
6. **`References/1905.11290v3_gariazzo2019.pdf`** — Gariazzo, de
   Salas, Pastor 2019. **Appendix A.17–A.20** for a second
   (FortEPiaNO-specific) form of D_αβ with explicit sin²θ_W
   dependence. Fallback if Mirizzi's form doesn't match them
   numerically.
7. `PRyM/PRyM_boltzmann.py` lines **4395–4443** (`_build_L_list`
   body) and lines **4313–4372** (`_assemble_collision_N`
   off-diagonal pair-damping block, which MUST be fixed
   consistently).
8. `validation/sterile_DW_literature.py` + `.out.txt` — the
   comparison script we use for acceptance tests.

## The core equation (Mirizzi Eq. 28)

Mirizzi writes the collision operator in a Lindblad-like form:

```
Ĉ[ρ] = -(i/2) · G_F² · m⁴ · ({S², ρ−I} − 2S(ρ−I)S + {A², ρ−I} + 2A(ρ̄−I)A)
Ĉ[ρ̄] = -(i/2) · G_F² · m⁴ · ({S², ρ̄−I} − 2S(ρ̄−I)S + {A², ρ̄−I} + 2A(ρ−I)A)
```

with the flavor-diagonal matrices

```
S = diag(g_e^s, g_μ^s, g_τ^s, 0)
A = diag(g_e^a, g_μ^a, g_τ^a, 0)
```

and numerical values (Mirizzi Eq. 29, citing Hannestad):

```
(g_e^s)² = 3.06         (g_e^a)² = 0.50
(g_μ^s)² = (g_τ^s)² = 2.22     (g_μ^a)² = (g_τ^a)² = 0.28
sterile: all zeros
```

The `(g_e^s)² = 3.06` match with PRyMordial's `C_D[0] = 3.06`
confirms that our `C_D[α] = (g_α^s)²` — PRyMordial currently uses
the scattering coefficient only and ignores annihilation.

### Working out the off-diagonal damping rate

At L = 0 (ρ = ρ̄), for (α, β) with α ≠ β:

```
[{S²,ρ} - 2SρS]_αβ = (g_α^s - g_β^s)² · ρ_αβ            (scattering contribution)
[{A²,ρ} + 2A(ρ)A]_αβ = (g_α^a + g_β^a)² · ρ_αβ            (annihilation contribution)
```

Giving:

```
Γ_αβ^{damp} = (G_F² · m⁴ / 2) · [(g_α^s − g_β^s)² + (g_α^a + g_β^a)²]
```

Converting Mirizzi's dimensionless `m⁴/x⁴` to PRyMordial's natural
units (m is an arbitrary reference mass; in Mirizzi `T = 1/a` and
`x = m·a`, so `m/x = T` and `m⁴/x⁴ = T⁴`), we get

```
Γ_αβ^{damp} = (1/2) · G_F² · T⁴ · [(g_α^s − g_β^s)² + (g_α^a + g_β^a)²] · (...E factor)
```

The momentum-dependent factor in PRyMordial is `E` (not `<y>` or
`<p/T>`). Mirizzi uses momentum-averaged approximation with `<y>`
but the FULL QKE uses the mode-by-mode E. Keep E-dependence.

### Numerical coefficients (sanity)

Using `(g_α^s)² + (g_α^a)²` notation:

| pair | (g_α^s−g_β^s)² | (g_α^a+g_β^a)² | total | PRyMordial 0.5·(Γ_α+Γ_β) | Mirizzi / PRy |
|---|---:|---:|---:|---:|---:|
| e–μ | 0.068 | 1.528 | **1.596** | 2.64 | 0.60 |
| e–τ | 0.068 | 1.528 | **1.596** | 2.64 | 0.60 |
| μ–τ | 0 | 1.120 | **1.120** | 2.22 | 0.50 |
| e–s | 3.06 | 0.50  | **3.56**  | 1.53 | **2.33** |
| μ–s | 2.22 | 0.28  | **2.50**  | 1.11 | **2.25** |
| τ–s | 2.22 | 0.28  | **2.50**  | 1.11 | **2.25** |

So Mirizzi's formula gives active-sterile damping ~2.3× larger
than our current formula. Active-active damping is ~0.5–0.6× our
current. **Net effect: stronger DW sterile damping, weaker
active-active damping.**

## Expected outcome

From the Sigl-Raffelt quasi-static formula `Γ_DW =
|H_off|² · D / (|H_diff|² + D²)` in the weak-damping limit
(D << |H_diff|, our regime), Γ_DW ∝ D. With Mirizzi's formula,
D_μs is 2.3× larger, so Γ_DW should be 2.3× higher, meaning MORE
thermalization — the wrong direction for fixing our over-production.

**But** Gariazzo+2019 uses an even more detailed formula
(App. A.17–A.20) with sin²θ_W-specific coefficients, and reports
ΔNeff ≈ 0.09 at sin²(2θ_μ4) = 4×10⁻⁴, Δm² = 1.29 eV² — which is
MUCH less than PRyMordial's 0.85. So the story is more subtle than
the simple linear scaling.

The likely full picture: Mirizzi and Gariazzo's formulas include the
SCATTERING + ANNIHILATION combined in a way that depends on WHICH
bilinear channels couple. The annihilation term `2A(ρ̄−I)A` has a
positive sign that partially cancels (if ρ̄ is in the opposite
direction) or reinforces the damping. PRyMordial's symmetric
average doesn't resolve this structure.

**Validation target: after the fix, ΔNeff at sin²(2θ_24)=10⁻⁴,
Δm²_41=0.93 eV² (4-flavor full PMNS) should drop from 0.85 to
somewhere in [0.1, 0.3] — consistent with Gariazzo scaling.**
If it drops to < 0.1, Mirizzi's formula is too strong and we need
Gariazzo's more-detailed form. If it stays > 0.5, there is still
a bug.

## Concrete implementation plan

### Step 1: Update `_build_L_list` (around line 4405-4414)

Replace the symmetric `D_off_eV` construction with a pair-specific
formula. The new code:

```python
# Mirizzi+2012 Eq. 28 pair-specific damping coefficients (L=0 case).
# C_D[α] = (g_α^s)² is the scattering coefficient (de Salas & Pastor 2016).
# C_A[α] = (g_α^a)² is the annihilation coefficient.
# Active-sterile and active-active damping have different structures
# because the commutator {S², ρ} - 2SρS gives (g_α^s - g_β^s)² on
# off-diagonals, and the annihilation gives (g_α^a + g_β^a)².
# See doc/STAGE_E1_BRIEF.md for the derivation.
C_A = np.array([0.50, 0.28, 0.28, 0.0])   # annihilation coefficients
# (square of g_α^a; ν_e = 0.50, ν_μ = ν_τ = 0.28, sterile = 0)
sqrt_C_D = np.sqrt(self.C_D)
sqrt_C_A = np.sqrt(C_A)

D_off_eV = np.zeros((N, N, Ny))
for alpha in range(N):
    for beta in range(N):
        if alpha == beta:
            continue
        scat_coeff = (sqrt_C_D[alpha] - sqrt_C_D[beta])**2
        anni_coeff = (sqrt_C_A[alpha] + sqrt_C_A[beta])**2
        total_coeff = scat_coeff + anni_coeff
        # Dimensional form: rate = (1/2) * G_F^2 * T^4 * E * total_coeff
        D_off_eV[alpha, beta] = 0.5 * total_coeff * (GF_eV**2) * (T_eV**4) * E_eV
```

Replace the existing:
```python
Gamma_eV = np.zeros((N, Ny))
for alpha in range(N):
    Gamma_eV[alpha] = self.C_D[alpha] * GF_eV**2 * T_eV**4 * E_eV
D_off_eV = np.zeros((N, N, Ny))
for alpha in range(N):
    for beta in range(N):
        if alpha != beta:
            D_off_eV[alpha, beta] = 0.5 * (Gamma_eV[alpha] + Gamma_eV[beta])
```

### Step 2: Update `_assemble_collision_N` (around line 4313-4372)

This function currently uses the same symmetric `D_pairs` formula:

```python
Gamma = np.zeros((self.n_flavor, Ny))
for alpha in range(self.n_flavor):
    Gamma[alpha] = self.C_D[alpha] * GF_eV**2 * T_eV**4 * E_eV * self._eV_to_secm1

n_pairs = len(self._all_pair_flavors)
D_pairs = np.zeros((n_pairs, Ny))
for p_idx, (_fa, _fb) in enumerate(self._all_pair_flavors):
    D_pairs[p_idx] = 0.5 * (Gamma[_fa] + Gamma[_fb])
```

Must be replaced with the same pair-specific Mirizzi form. Note
this one is in units of 1/s (not eV), because `_assemble_collision_N`
builds N in eV by subsequently multiplying by `inv_rate`.

The subtraction pattern `rhs_si = -D_pairs[p_idx] * rho_ab_stored +
S_gain_si` on line 4369 uses this D_pairs. The D.7.1 driver then
adds D_pairs back (in eV units) via `_build_L_list` to make
"gain-only N". If both are updated consistently with the new
Mirizzi formula, the arithmetic still cancels correctly.

### Step 3: Introduce C_A as an instance attribute

In `__init__`, alongside `self.C_D = np.array([3.06, 2.22, 2.22, 0.0])`,
add:

```python
# Annihilation coefficients (Hannestad-style (g_α^a)²).
# ν_e from e+e- → ν_e ν̄_e has g_V² + g_A² = 0.50.
# ν_μ,τ similar with weak mixing: 0.28 each.
# See Mirizzi+2012 Eq. 29 and Hannestad ref [33] therein.
if self.n_flavor == 4:
    self.C_A = np.array([0.50, 0.28, 0.28, 0.0])
else:
    self.C_A = np.array([0.50, 0.28, 0.28])
```

### Step 4: Verify the `-D_pair · ρ_ab` cancellation

The D.7.1 driver moves damping into L (subtracts from off-diagonal
coherence). Both the forward `-D·ρ` in `_assemble_collision_N` and
the "add-back" in `_build_L_list` must use the SAME pair-specific
formula. If they diverge, you double-count or cancel incorrectly.

Check: after the fix, run `diag_2level_damped.py` from
`validation/diagnostics/`. The minimal 2-level check should still
show exact agreement between PRyMordial's `L`-expm and the analytic
damped Rabi — but with a DIFFERENT ρ_ss value reflecting the new
D_μs coefficient.

## Sterile-mass convention

PRyMordial treats `Dm2_41` as a free parameter set via
`PRyMini.Dm2_41`. Each reference paper uses its own benchmark:

| Reference | Benchmark Dm² | Figures | Notes |
|---|---:|---|---|
| Mirizzi+2012 Eq. 9 | 0.89 eV² | Fig. 2 | momentum-averaged QKE |
| Hannestad+2012 Fig. 2 | 0.93 eV² | sin²(2θ) sweep | full density-matrix QKE |
| Gariazzo+2019 Fig. 3 | 1.29 eV² | \|U_α4\|² sweep | full 4×4 QKE w/ FortEPiaNO |
| PRyMordial existing tests | 1.0 eV² | test_sterile_dw_production | ours |

**CRITICAL**: for each acceptance test below, set `PRyMini.Dm2_41`
equal to the specific reference's value — do NOT carry over the
0.93 used in the current `validation/sterile_DW_literature.py`.
The DW rate scales roughly as `Δm² × f(T_max/T_decoup)`, so a
factor-1.4 `Δm²` mismatch injects ~40% systematic. Use per-benchmark
matched `Dm²_41`:

  - Hannestad Point A/B/C benchmarks: `Dm2_41 = 0.93` (keep).
  - Gariazzo benchmark: `Dm2_41 = 1.29` (fix from our current 0.93).
  - Mirizzi benchmark: `Dm2_41 = 0.89`.

Similarly, Gariazzo's `|U_α4|² = 10⁻⁴` corresponds to
`sin²(2θ_α4) = 4·|U_α4|²·(1 − |U_α4|²) ≈ 4×10⁻⁴`, not `10⁻⁴`.
Convert carefully. Our previous "Gariazzo comparison" informally
conflated `|U|²` with `sin²(2θ)` — at the factor-4 level this
matters a lot.

## Validation targets (in order)

1. **Import & fast tests** — `pytest tests/test_regression.py -m
   "not slow" -v`. All 4 fast tests (mode1, mode2, mode6,
   qke_etdrk2_nu_nubar_symmetry) must PASS. The first three should
   bit-identically match (SM 3-flavor, the pair-specific change is
   observable-neutral there because g_α^a and g_α^s differences
   cancel in SM's flavor-symmetric case... CHECK THIS). The symmetry
   test should still pass (D is real, pair-specific or not).
2. **Minimal 2-level cross-check** — `python
   validation/diagnostics/diag_2level_damped.py`. PRyMordial's
   L-expm must equal the analytic 2-level expm with NEW D_μs. If
   not, the change isn't consistently applied.
3. **Mode 5c + sterile DW** — `pytest
   tests/test_regression.py::test_mode5c_qke_ode_etdrk2
   tests/test_regression.py::test_sterile_dw_production -v`. Both
   MUST PASS. test_mode5c may shift slightly (3-flavor SM; check
   tolerance); sterile_dw_production at sin²(2θ)=0.1 should still
   land in [0.5, 1.1] band.
4. **Sterile slow suite** — full 3 slow sterile tests. May need
   to update `test_sterile_sf_asymmetry_depletion` tolerance if the
   SF regime shifts (it will — this is the physics fix for SF).
5. **DW literature probe (n_B=2400 only)** — quick probe at
   sin²(2θ_24)=10⁻⁴, Δm²=0.93, PMNS off (from
   `validation/diagnostics/diag_pmns_leak.py`). Target: ΔNeff drops
   from 0.29 to < 0.1 (closer to Hannestad's 0.04 / Gariazzo's
   ~0.02-0.04 extrapolated).
6. **Full DW literature comparison** —
   `python validation/sterile_DW_literature.py`. Target for all
   three Hannestad benchmarks (A, B, C): relative agreement < 30%
   vs Hannestad's values. Point A (full therm.) should stay at
   ~1.0, Point C (sin²(2θ)=1e-4) should drop below ~0.2. Keep
   `Dm2_41 = 0.93` for this comparison — that matches Hannestad
   Fig. 2.
7. **Gariazzo cross-check** — at their benchmark `Dm2_41 = 1.29`,
   run `|U_μ4|² = 10⁻⁴` (i.e. `sin²(2θ_24) = 4·(1-10⁻⁴)·10⁻⁴ ≈
   4×10⁻⁴`). Target: ΔNeff ≈ 0.09 (matching Gariazzo Fig. 3 violet
   curve). If we land in [0.05, 0.2], the fix is working.

## What NOT to touch

- `_build_H_list` (V_NC, V_thermal are physics-complete).
- `_etdrk2_expm_phi` (L-expm is verified correct).
- `evolve_step_ode_etdrk2` body (D.7.1 Strang-symmetric structure).
- `_apply_unitary` (D.3 ν-bar convention fix).
- `C_D` values (they're from de Salas & Pastor 2016 and match
  Mirizzi's g_α^s²).
- Default value of any flag.

## Session realism

ROADMAP scopes Stage E.1 as ~1 afternoon if the fix is simple and
the validation suite doesn't throw surprises. In one context window:

- **(a) Mirizzi-only fix**: update the two damping-coefficient
  locations, run validation, accept or iterate. Probably 2–3 h.
- **(b) Mirizzi + Gariazzo fallback**: implement the Mirizzi form
  first; if it doesn't match Gariazzo's numbers within 30%, also
  implement the Gariazzo App. A.17–A.20 form behind a flag. Slower
  but covers more bases. Probably 4+ h.
- **(c) Root-cause deep dive**: if BOTH Mirizzi and Gariazzo forms
  still leave a factor-2+ gap vs the reference, the bug is
  elsewhere (annihilation kernel handling? y-integration weight?).
  This becomes a separate multi-session investigation.

State the choice explicitly in the opening message.

## Minimum viable opening message for the new session

> Read `doc/STAGE_E1_BRIEF.md` end-to-end before writing any code.
> Then use `EnterPlanMode` to propose a concrete Stage E.1
> implementation plan. My target for this session is {ONE of the
> three in "Session realism"}. Do not touch `_build_H_list`,
> `_etdrk2_expm_phi`, or the D.7.1 evolve_step_ode_etdrk2 body. Do
> not change the C_D values or any default flag.

## Commit chain for context

- `49e31f4` — current HEAD: "validation/diagnostics: localize DW
  bug to D_pair damping-formula" (this session's final commit).
- `2fa1f28` — L-expm verified correct via minimal 2-level test.
- `5d8426b` — initial diagnostic scripts + investigation README.
- `0e71464` — V_NC thermal potential fix + DW literature script.
- `1d927d7` — D.7.2 null result recorded.
- `e2f41f6` — Stage D.7.1 Strang-symmetric diagonal split (the
  current driver; not touched by this brief).

## Post-fix: downstream opportunities

If Stage E.1 closes the DW gap:

1. **Shi-Fuller literature comparison (Paper 2)**: Saviano+2013
   or equivalent. The same D_αβ fix should propagate.
2. **Lovell 2023 keV-sterile DM**: needs QKE extension to T ~ 1 GeV
   and QH-transition physics, separate project.
3. **Update `doc/ROADMAP.md`** with Stage E.1 landing: the fix,
   validation outcomes, any remaining gaps.