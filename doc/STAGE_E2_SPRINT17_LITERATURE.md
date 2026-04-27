# Stage E.2 sprint 17 literature record: Hannestad-Tamborra-Tram 2012 reproduction

Single-file record of the literature-review work for sprint 17. The
sprint's structural verdict (Suspect 8 vs framing collapse) flows
through the unit-convention audit (§1) and the side-by-side
configuration table (§2). Sprint-16 escalated here on the basis that
"no time-integrator change cures the ≈ 200× gap to Hannestad's
[0.02, 0.10] band"; this file documents that **the gap is at least
partly a units mismatch** (§1) and that **Hannestad's damping kernel
is the legacy "symmetric" Stodolsky form, not our default "mirizzi"**
(§2 row 4) — a divergence axis which sprint 17 brackets directly
in `validation/diagnostics/diag_sprint17_axis_v_nunu.py`.

## §1 Unit-convention audit

There are three quantities in flight, each correctly described in
its own context but trivially confused for one another in the
project notes inherited from sprints 12-16.

### 1.1 The harness "Σρ_ss" — raw dimensionless sum

What the existing diagnostic harnesses report:

```python
sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
# Reference: validation/diagnostics/diag_hannestad_pointC_etdrk4.py:113
```

This is a raw sum across all `Ny = 100` momentum modes and both
sectors `(ν, ν̄)` of the sterile-flavor diagonal density-matrix
element `ρ_ss(y, sector)`. `ρ_ss(y, s)` is bounded in `[0, 1]` and
is initialised to identically zero (see
`PRyM/PRyM_boltzmann.py:3215-3217` — `'nus'` and `'nusbar'` slots
start at 0.0, while active flavors start at the thermal Fermi-Dirac
`1 / (exp(y/T_nu_com) + 1)`).

Bounds at `Ny=100, dy=1`:
- **Maximum-fraction saturation** `ρ_ss(y, s) ≡ 0.5` everywhere:
  Σ_raw_max = `2 × 100 × 0.5 = 100`.
- **Thermal-saturation-equivalent** `ρ_ss(y, s) = f_FD(y)`:
  Σ_raw_thermal = `2 × Σ_y f_FD(y) ≈ 2 × ln(2) / dy ≈ 1.386`.

The reported sprint-16 baselines:
| Run | Σ_raw_ss | per-mode mean (= Σ/200) | × thermal-saturation |
|---|---|---|---|
| ETDRK2 production (n_B=10000) | 22.225 | 0.111 | 16.0× |
| ETDRK4 reduced (n_B=2500) | 15.805 | 0.079 | 11.4× |
| ETDRK2 reduced (n_B=2500) | 14.512 | 0.073 | 10.5× |

**Σ_raw_ss = 22.225 indicates ρ_ss(y) is far above the thermal
Fermi-Dirac envelope at high y** — the sterile fills high-y modes
where the active distribution has long since exhausted (`f_FD(y=99)
≈ exp(-99) ≈ 0`). This is unphysical for a sterile sourced via
oscillation from a thermal active and is the strongest hint that
the saturation is a numerical artefact (or, more carefully, that
some flag in the production harness drives ρ_ss at high y above
its physical ceiling).

### 1.2 Number-density-weighted "Σρ_ss" — the brief's convention

The sprint-17 brief (and earlier sprint briefs) describes the
quantity as:

> Σρ_ss = `dy · sum_y y² · ρ_ss(y)` per sector

This is a **comoving number density** in dimensionless `(T_ν · a)³`
units, single-sector. The conversion from Σ_raw to a single-sector
number-weighted Σρ_ss:

```
Σρ_ss_brief = (dy / N_sectors) · Σ_s Σ_y y² ρ_ss(y, s)        # per-sector convention
            = 0.5 · dy · Σ_s Σ_y y² ρ_ss(y, s)
```

For an arbitrary ρ_ss(y) profile this is **not** algebraically
related to Σ_raw without knowing the y-distribution of ρ_ss. If we
assume ρ_ss(y) is roughly flat at value α across y ∈ [0, y_max=100]:
`Σ_y y² · α ≈ α · y_max³/3 ≈ α · 333333`. With α = Σ_raw / 200:

| Σ_raw | implied α | Σρ_ss_brief (per-sector, flat) |
|---|---|---|
| 22.225 | 0.111 | 0.5 · 1 · 0.111 · 333333 ≈ 18500 |
| 0.10  (Hannestad target band, reinterpreted) | 5e-4 | 0.5 · 1 · 5e-4 · 333333 ≈ 83 |
| 0.02  (Hannestad target band, reinterpreted) | 1e-4 | 0.5 · 1 · 1e-4 · 333333 ≈ 17 |

So under the brief's convention, the production raw 22.225 maps to
~ 18500, vs Hannestad-reinterpreted target ~ 17-83 — a ~ 200-1000×
gap. **Worse than the raw-number framing.** The brief's convention
is therefore not the right one for matching Hannestad either.

### 1.3 The published Hannestad observable — δN_eff

Hannestad-Tamborra-Tram 2012 (arXiv:1204.5861) reports **δN_eff**
(Eq. 3.1, 3.2 of HTT 2012), the dimensionless ratio of the sterile
energy density to one fully-thermalised neutrino species:

```
δN_eff = ρ_ss / ρ_ν,thermal,1-species
       = (sum_s ∫ y³ ρ_ss(y, s) dy / (2π²)) / (7π² / 120 · T_ν,com⁴)
       = (120 / (7π⁴)) · Σ_s ∫ y³ ρ_ss(y, s) dy / (T_ν,com⁴ · (2π² / 2π²))
```

In practice, with a uniform y-grid at `dy = 1`, the discrete
formula is:

```
δN_eff = (60 / (7π⁴)) · dy · Σ_s Σ_y y³ ρ_ss(y, s) / T_ν,com⁴
```

(factor 60 instead of 120 because the sum runs over both sectors
and the denominator normalises to one fully thermal species which
has both sectors).

Hannestad's "barely rises off zero" curve in HTT 2012 Fig. 2 (top
panel, blue, sin²(2θ) = 1×10⁻⁴, δm² = 0.93 eV²) shows **δN_eff ≈
0.02-0.03 at T = 1 MeV** — the visual lower edge of the panel.
**The literal string "[0.02, 0.10]" does NOT appear in the paper**;
that band is a project-internal target accumulated from briefing
notes and likely reflects (a) a rounded reading of HTT 2012 Fig. 2
plus (b) a tolerance margin to cover the four-curve panel. The
upper edge of the band (~0.10) probably corresponds to the
sin²(2θ) = 2.26×10⁻³ curve of Fig. 2 (top panel, second-from-bottom).

### 1.4 The cheapest reconciliation arithmetic

For Σ_raw_ss = 22.225 to translate into δN_eff in the
order-of-magnitude [0.02, 0.10] range, the y³ moment of ρ_ss(y)
needs to be computed from the live profile. **The existing harnesses
do not save the y-resolved profile** — they save only the raw sum.
Sprint 17's axis-bracket harness (§3 below) extends the output to
include both `Σ_y y³ ρ_ss · dy / (2π²)` and the converted δN_eff,
so subsequent comparisons against Hannestad are apples-to-apples.

### 1.5 Verdict on Stage E.2 framing

The "200× gap to Hannestad's band" framing inherited from sprint 16
is **partly correct** (the raw-sum number is genuinely large at
22.225) but **partly a units artefact**: Hannestad's reported
quantity is δN_eff, not the raw sum, and our raw sum is unbounded
in `Ny` and `y_max`. The comparison must be done on δN_eff.

Whether that comparison reveals Stage E.2 closure or genuine
physics divergence depends on the live `Σ_y y³ ρ_ss` profile, which
sprint-17's axis bracket (§3) will harvest for the first time.

## §2 Hannestad-Tamborra-Tram 2012 reference table

Source: arXiv:1204.5861 (Hannestad, Tamborra, Tram 2012, "Thermalisation
of light sterile neutrinos in the early universe", JCAP 07 (2012) 025).
Fetched and tabulated by literature-review agent on 2026-04-27.

| Item | HTT 2012 | PRyMordial-nu (current) | Match? |
|---|---|---|---|
| "Point C" label | **No labelled "Point C"** in the paper. The (Δm², sin²(2θ)) = (0.93 eV², 1×10⁻⁴) configuration is the **lowest-mixing curve of Fig. 2 (top panel, blue)** — Eq. 3.4a parameter scan, L=0, NH | "Point C" is project-internal | Identification: **Fig. 2 top, blue curve** |
| Δm², sin²(2θ) | 0.93 eV², 1×10⁻⁴ (Fig. 2 top blue) | 0.93 eV², 1×10⁻⁴ (in `diag_hannestad_pointC_etdrk4.py:62-64`) | ✓ |
| Reported observable | **δN_eff** (HTT 2012 Eq. 3.1, 3.2). Visual reading of Fig. 2 top blue curve at T=1 MeV: **δN_eff ≈ 0.02-0.03** | Σρ_ss = `c._boltz_rho_final[:, 3, :].sum()` (raw sum, no weight) | **Mismatched** — see §1 |
| Damping kernel | **"Symmetric" Stodolsky form** (HTT 2012 Eq. 2.15, 2.16): `D = (1/2) Γ` with `Γ = C_a G_F² T⁵ E`, `C_e ≃ 1.27`, `C_{μ,τ} ≃ 0.92`. **Not** the Mirizzi-style `[(g_α^s − g_β^s)² + (g_α^a + g_β^a)²]` form | Default `qke_damping_formula = "mirizzi"` (`PRyM_init.py:171`); the existing Point-C harness keeps `"mirizzi"` | **Mismatched** — sprint 17 brackets this axis |
| V_nunu form | Closed-form thermal `V_1` (HTT 2012 Eq. 2.10): `V_1^(a) = -(7π²/(45√2))·(G_F/M_Z²)·x·T⁵·[n_νa + n_ν̄a]·g_a` — assumes thermal active distributions, sterile NOT included. Plus background lepton/asymmetry `V_L` (Eq. 2.11) — also active-only | Live integral `√2·G_F·∫y²(ρ−ρ̄)dy/(2π²·a³)` (`PRyM_boltzmann.py:4368`); active-only when `qke_v_nunu_active_only=True` (Point C harness sets this `True`) | **Compatible**: HTT's closed-form thermal V₁ is the thermal limit of our `active_only=True` live integral when ρ ≈ ρ̄ ≈ thermal active. We use the live integral; they use the closed form. Live form should reduce to closed form pre-saturation. |
| α channel | Generic 2-flavor (active α paired with sterile s); no specific α | numu (idx 1) ↔ sterile (idx 3) per `PRyM_init.py:232` | Compatible (Hannestad does single active-sterile mixing) |
| Mixing-angle convention | HTT 2012 Eq. 2.1-2.2: `ν_a = cos θ_s · ν_1 - sin θ_s · ν_2` (standard half-angle); `sin²(2θ_s) = 10⁻⁴ ⟹ θ_s ≈ 0.005 rad` | `theta_24 = arcsin(sqrt(1e-4)) / 2 ≈ 0.005 rad` (`diag_hannestad_pointC_etdrk4.py:64`) | ✓ |
| Number of sectors | **Two** (ν and ν̄ separately, HTT 2012 Eq. 2.3, 2.17) | Two (`_boltz_rho_final` shape `(2, n_components, Ny)`) | ✓ |
| Initial sterile | `ρ_ss(T_init) = 0` (P_s±(T_init) = 0 implicit; sterile starts unpopulated, HTT 2012 §3.1) | `'nus'` and `'nusbar'` slots start at 0.0 (`PRyM_boltzmann.py:3215-3217`) | ✓ |
| T_init, T_final | T_init = **60 MeV**, T_final = **1 MeV** (HTT 2012 §3.1) | T_phase0_start = **100 MeV**, T_boltz_end = **0.005 MeV** (Phase 0 → Phase B → frozen) | Compatible: starting 100 MeV ≥ 60 MeV (active still thermal at both); ending 0.005 MeV ≪ 1 MeV (post-decoupling, no active dynamics in either case) |
| Resonance crossing | **No resonance for L=0 NH** (HTT 2012 Appendix A). Generic non-resonant peak `T_max ∼ 10·(δm²)^(1/6) MeV ≈ 9.9 MeV` for δm²=0.93 (HTT 2012 §3.2) | Sprint-12 localised an ETDRK2 jump at Tg ≈ 60-64 MeV (`doc/ROADMAP.md` sprint-12 entry) | **Suspicious mismatch**: HTT says no resonance and a peak at ~10 MeV; we see a jump at ~60 MeV. Could be Phase-0/Phase-B handoff artefact or numerical pathology. Out of scope for sprint 17; flagged for sprint 18. |
| Solver | **ndf15** (Shampine, NDF order 1-5) and **RADAU5** (5th-order implicit RK), both stiff multistep, sparse linear algebra (HTT 2012 §3.1, refs [52, 53]) | ETDRK2 (production) / ETDRK4 (sprint-16 opt-in) / LSODA (sprint-15 opt-in, wall-clock-infeasible) | Different family. HTT's stiff multistep works because their Bloch-vector RHS (`P_i±`, 4 real per mode) is much cheaper than our 16 complex DOFs per mode and they use closed-form V_1. |
| Momentum grid | **Kainulainen-Sorri non-uniform map** (HTT 2012 Eq. 3.3): `u(x) = (x − x_min)/(x_max − x_min) · (x_max + x_ext)/(x + x_ext)` with `x_min=10⁻⁴, x_ext=3.1, x_max=100`; sample `u` uniformly. **"a few hundred points"** (HTT 2012 §3.1) | Linear, `Ny=100`, `y_max=100`, `dy=1` (`PRyM_init.py:338-339`, `PRyM_boltzmann.py:1988-1989`) | Different. HTT concentrates points near the resonance, we use uniform spacing. Possibly material if there is sharp y-structure in ρ_ss(y); to be revisited if §3 axis brackets do not close the gap. |

## §3 FortEPiaNO baseline scope

Quick git audit (commits 98a5d05 "FortEPiaNO vs PRyMordial-nu comparison"
and 141a7d0 "pre-compilation analysis added to FortEPiaNO comparison")
shows that **only the comparison document** `doc/STAGE_E2_FORTEPIANO_COMPARISON.md`
was added in those two commits. **No `validation/diagnostics/diag_fortepiano_*`
script exists** — neither in the worktree nor in the git history of any
branch.

The sprint-17 brief mentioned these scripts as if they existed; the
mention was forward-looking and never realised. **FortEPiaNO offers
no prebaked Hannestad reference trajectory in our setup.**

Sprint 17 therefore proceeds with **HTT 2012 directly as the ground
truth**, supplemented by the FortEPiaNO design notes in
`doc/STAGE_E2_FORTEPIANO_COMPARISON.md` for context on what an
external code does differently. Cloning FortEPiaNO and running a
Point C trajectory through it is **out of scope for sprint 17** and
should be its own future-sprint project (estimate: 1-2 weeks of
build, install, harness, format-conversion work).

## §4 Sprint-17 axis-bracket plan (pointer)

The actual numerical experiment for sprint 17 lives in
`validation/diagnostics/diag_sprint17_axis_v_nunu.py`. It runs three
ETDRK2 jobs at reduced `n_B = 2500 + n_B_phase0 = 1000` (apples-to-
apples with the sprint-16 gate-5b baseline `Σ_raw_ss = 14.512`):

1. **Run A** — `qke_v_nunu_active_only = True`, `qke_damping_formula
   = "mirizzi"` (matches existing Point C harness; expected `Σ_raw_ss
   ≈ 14.5` per gate 5b)
2. **Run B** — `qke_v_nunu_active_only = False`, `qke_damping_formula
   = "mirizzi"` (V_nunu axis flip; sprint-5 legacy projection)
3. **Run C** — `qke_v_nunu_active_only = True`, `qke_damping_formula
   = "symmetric"` (damping axis flip; **matches HTT 2012 directly per
   §2 row 4**)

Each run also reports the converted **δN_eff** quantity (per §1.3
formula) so the verdict is in HTT-comparable units rather than the
raw sum. Wall-clock budget: 3 × 21 min ≈ 63 min total.

Decision tree at completion:
- Run C produces **δN_eff in [0.02, 0.10]** ⇒ damping-kernel axis
  is the divergence point; sprint 17 confirms Suspect 8 with a
  single-axis cure (HTT-matching damping formula).
- Run B produces **δN_eff in [0.02, 0.10]** ⇒ V_nunu projection
  is the divergence point.
- All three runs produce the same δN_eff outside the band ⇒
  neither axis cures the gap; escalate to mixing-angle / IC /
  resonance / momentum-grid axes in sprint 18.
- Any δN_eff(A) ≠ δN_eff_existing_baseline (after applying the §1
  conversion to gate-5b's `Σ_raw=14.512`) ⇒ unit-conversion bug;
  audit the harness output before any structural verdict.

## §5 Loose ends for follow-up sprints

1. **Resonance crossing mismatch**: HTT 2012 says no resonance for
   L=0 NH; sprint-12 localised an ETDRK2 jump at Tg ≈ 60-64 MeV.
   Either project sprint-12's "resonance" was a numerical artefact
   (Phase-0/Phase-B handoff signature) or there is a physical
   mechanism we are sourcing that HTT does not have. Worth a
   targeted diagnostic in sprint 18+: turn `xi_nue_init` away from
   zero to introduce a real lepton asymmetry (HTT's "L = 10⁻²"
   case) and see whether the sprint-12 jump moves with the
   asymmetry as predicted by HTT Appendix A.

2. **Momentum grid resolution**: HTT uses ~300 nodes on a
   non-uniform grid concentrated near the resonance; we use 100
   uniform. If the §3 axis brackets do not close the gap, the next
   axis to test is `Ny = 200` (or 300) and a quick re-run at
   reduced `n_B` to see whether the saturation Σ_raw_ss drops with
   `Ny`.

3. **Solver-side**: HTT's ndf15 / RADAU5 work because their RHS is
   cheap (Bloch-vector P± plus closed-form V_1 / V_L). If sprint 18
   pursues a "match HTT's RHS structure" path, the natural design
   is to add an `qke_bloch_vector_flag` that reduces our 16-complex-
   DOF density matrix to HTT's 4-real-DOF Bloch vector when the
   active-sterile system is 2-flavor — orthogonal to the present
   work.

## Citation

* Hannestad, Tamborra, Tram 2012, "Thermalisation of light sterile
  neutrinos in the early universe", JCAP 07 (2012) 025;
  arXiv:1204.5861.
* Mirizzi, Mangano, Saviano, Borriello, Giunti, Miele, Pisanti 2012,
  "Light sterile neutrino production in the early universe with
  dynamical neutrino asymmetries", PRD 86 (2012) 053009; arXiv:1206.1046.
  Reference for the "mirizzi" damping kernel option (PRyM_init.py:171).
* Gariazzo, de Salas, Pastor 2019, "Thermalisation of sterile neutrinos
  in the early Universe in the 3+1 scheme with full mixing matrix",
  JCAP 07 (2019) 014; arXiv:1905.11290 (FortEPiaNO source).
