# Stage F sprint 3h-c findings: n→p Pauli-blocking inspection

Inspection-only sprint. No code change. Falsifies sprint 3g's hypothesis
that the Yp sign-flip mechanism identified in sprint 3e is a *symptom*
of an upstream sign or normalisation slip in `PRyM_eval_nTOp.py`'s
`(1 − f_νebar)` consumer.

## §1 Scope

Sprint 3g identified three sprint 3h candidate cures for the Yp
sign+magnitude bug:

* 3h-a (full-uniform clamp four-Hannestad scan) — FALSIFIED, 1/4 PASS.
* 3h-b (avg-pair-symmetric per_flavor clamp) — FALSIFIED, 1/4 PASS,
  active-sector regression at mid/narrow mixing.
* 3h-b' (OR-firing pair-symmetric per_flavor clamp) — SHIPPED as live
  cure, 3/4 PASS matching gate-5 coverage with right-sign Yp at
  saturation (A and Global-NH); Yp at narrow mixing C regressed from
  gate-5's −0.0086 to 3h-b' OR's −0.0152.
* 3h-c (this sprint): inspect upstream `PRyM_eval_nTOp.py` for sign /
  normalisation slip in the `(1 − f_νebar)` consumer that could
  produce sign-flipped Yp shifts under sterile mixing.

3h-c was motivated by sprint 3e's finding that asymmetric per_flavor
clamp firing on (ν, ν̄) pairs produces a sign-opposite Yp shift via
Pauli-blocking imbalance. Sprint 3g flagged the asymmetry as
possibly a *symptom* rather than the *root cause*.

## §2 Inspection methodology

Cross-checked the dispatch from `f_α_general` callables (constructed
in `PRyM_boltzmann.py`'s `_make_f_callable` / `make_f_callable`) into
the n→p / p→n weak-rate integrands across all four backends:

| Backend | flag combination |
|---|---|
| general + numba | `general_nu_flag=True`, `numba_flag=True` |
| general + python | `general_nu_flag=True`, `numba_flag=False` |
| thermal + numba | `general_nu_flag=False`, `numba_flag=True` |
| thermal + python | `general_nu_flag=False`, `numba_flag=False` |

For each backend, traced `sgnq=±1` dispatch through:

1. `_f_eff_nu_tab_nb` / `f_eff_nu` (the crossing-symmetry switch
   between absorption/emission and ν/ν̄).
2. `_ChiFunc_std_nb` / `ChiFunc` (Born response function consuming
   `f_eff_nu` with kinematic FD2 weight).
3. `_ChiFunc_FM_std_nb` / `_ChiFunc_FM_general_tab_nb` (finite-mass
   response with f_1/f_2/f_3 form-factor swap between sgnq=±1).
4. `_Born_integrand_*` / `_CCR_integrand_*` / `_FMCCR_integrand_*`
   (rate integrands).
5. `L_nTOpBORN` / `L_pTOnBORN` (rate exports — confirm sgnq=+1 used
   for n→p, sgnq=−1 used for p→n).

## §3 Verdict

**No sign or normalisation slip found.** The crossing-symmetry
dispatch is mathematically correct and structurally consistent
across all four backends.

| Site | File / line | Dispatch verdict |
|---|---|---|
| general crossing-sym | `_f_eff_nu_tab_nb` ~364–378 | correct |
| general crossing-sym (python) | `f_eff_nu` ~722–739 | correct |
| thermal crossing-sym | `_FD_nu3_nb` ~152 + `_ChiFunc_std_nb` ~230 | correct (FD at negative argument gives 1−FD via algebra) |
| FM form-factor swap | `_ChiFunc_FM_std_nb` ~236–265 vs `_ChiFunc_FM_general_tab_nb` ~421–452 | structurally identical |
| sgnq for n→p / p→n | `L_nTOpBORN_int` / `L_pTOnBORN_int` ~850–916 | correct |

The four channels are dispatched as:

| Channel | Process | sgnq | E_nu sign | Pauli factor |
|---|---|---|---|---|
| 1 | n + ν_e → p + e⁻ | +1 | ≥ 0 | f_νe |
| 2 | n → p + e⁻ + ν̄_e | +1 | < 0 | 1 − f_νebar |
| 3 | p + e⁻ → n + ν_e | −1 | < 0 | 1 − f_νe |
| 4 | p + ν̄_e → n + e⁺ | −1 | ≥ 0 | f_νebar |

This matches Brown & Sawyer Eq. 2.29–2.30 / Pitrou et al. 2018
§2.6 and is unchanged between the thermal and general-distribution
paths.

## §4 Re-derived mechanism of the 3h-b' OR-rule Yp regression at C

The Yp regression at narrow mixing C under sprint 3h-b' OR
(−0.0152 vs gate-5's −0.0086) is **genuine physics**, not a code
bug. Mechanism:

1. At narrow mixing (sin²2θ=1e-4) the QKE deforms ν_e modestly
   and ν̄_e barely.
2. Under gate-5 per_flavor, only ν_e gets clamped (slope < target).
   ν̄_e tail stays polyfit-flat — i.e. f_νebar at high y is mildly
   *elevated above FD-equivalent*.
3. The elevation reduces `(1 − f_νebar)` at high y → channel 2
   (n decay) is partially Pauli-blocked → fewer n decays → more
   neutrons preserved → **higher** Yp.
4. Under 3h-b' OR, ν̄_e is also forced to FD-equivalent. The
   `(1 − f_νebar)` Pauli-protection vanishes → channel 2 unblocked
   → more n decays → fewer neutrons → **lower** Yp.

So gate-5's polyfit-flat ν̄_e tail at narrow mixing was providing
**accidental Pauli-blocking protection** on n decay. 3h-b' OR
removes this artifact and exposes the genuine sterile-induced Yp
deficit that PRyMordial-nu's QKE evolution produces at L=0 NH
non-resonant.

## §5 Implication: redirect sprint focus upstream

The expectation in the literature (Saviano et al. 2013;
Mirizzi et al. 2012) is a **positive** sterile-induced Yp shift
of order +0.001 to +0.012. PRyMordial-nu, post 3h-b' OR cure,
gives positive at saturation (A: +0.0038, NH: +0.0026 — both
inside Saviano's range) but negative at narrow mixing (C:
−0.0152) and near-zero at mid mixing (B: −0.0010).

This is not a numerical bug. It reflects modeling-physics
choices that PRyMordial-nu currently lacks:

* **Lepton asymmetry seeding (L≠0).** HTT 2012 §4 computes
  L=0 NH and L=0 IH separately; the global-fit point's expected
  δNeff_ss=0.55 (NH) is asymmetry-dependent. The same physics
  applies to Yp: a non-zero L_νe seed at decoupling shifts the
  ν_e–ν̄_e asymmetry into n→p rates, biasing Yp.
* **Resonance-aware Phase-0 segment.** PRyMordial-nu's
  `qke_phase0_flag=False` Phase-B-only path skips the resonance
  crossing that occurs at high T_nu. HTT 2012 includes resonance
  handling, which can produce ν̄_e excess via MSW-flipped
  conversions, elevating channel-4 absorption and biasing Yp.

Both are Stage F sprint 1b carryover items, already noted in the
Stage F brief. **The Yp narrow-mixing deficit is the same
question as the Global-NH overshoot**: both originate in the
PRyMordial-nu QKE configuration's omission of L≠0 / resonance
physics, not in the BBN pipeline downstream.

## §6 What this means for the live cure

Sprint 3h-b' OR remains the live cure shipped:

* SUBSTANTIAL closure (3/4) — matches gate-5 coverage.
* Right-sign Yp at saturation (A, NH).
* Active sector universally healthy.
* Default-off bit-identical (gate-1 6/6 PASS in 34s).

The C Yp regression (−0.0152) is exposed *honest physics*: it
is what PRyMordial-nu's L=0 NH non-resonant configuration
produces. Gate-5's smaller deficit (−0.0086) was the same
physics partially masked by an asymmetric numerical artifact.

If a positive Yp shift is desired at narrow mixing, the path is
not in the n→p code (which is correct) but in adding L≠0 or
resonance physics to the QKE Phase-0/Phase-B segments
(Stage F sprint 1b).

## §7 Open issues remaining

Same as sprint 3h commit:

* Sprint 1b — global-fit-NH overshoot + narrow-mixing Yp deficit.
  Now unified as a single missing-physics question (lepton
  asymmetry / resonance handling).
* Sprint 4 — FortEPiaNO comparison + V_nunu paradox at structural
  level (sprint-18 carryover).
* Sprint 3i — QED-table extension above 40 MeV (every run still
  emits the warning).
