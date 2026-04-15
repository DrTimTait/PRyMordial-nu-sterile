# PRyMordial-nu: Future work

Ideas identified after Task 1 (O(e⁴) QED) and Task 2 (μ-τ / ν-ν̄ symmetry
breaking) landed. Roughly ordered by bucket; priority/effort are rough
estimates.

## Physics extensions

1. **NLO weak corrections (Z-exchange in ν-e) in Phase A**
   ~10⁻³ shift on Neff at T > 3 MeV. Froustey+2020, Akita & Yamaguchi 2020.
   Moderate effort — new matrix elements in `PRyM_eval_nTOp.py`.

2. **Exact 3-flavor PMNS in n=4/n=6** ✓ (done)
   Fix landed in commit `a844a62`: `_setup_collision_mixing` now computes
   the full 3×3 `P_αβ = Σᵢ |V_αi|² |V_βi|²` matrix from the PDG 2024
   oscillation parameters (including δ_CP = 1.36π). The n=4 and n=6
   branches of `_apply_collision_mixing` use this directly. The earlier
   maximal-θ₂₃/no-CP approximation (P_μe = P_τe = (1-P_ee)/2, P_μμ =
   P_ττ = P_μτ = (1+P_ee)/4) deviated from PDG by ~14%.

3. **Massive-electron n=6 nu-e integral** ✓ (done)
   Fix landed in commit `a844a62`: new `_collision_integral_nu_e_massive_asym6`
   ports the massive kinematics from the n=3 function with proper
   ν/ν̄-per-flavor-distinct annihilation. Smoke test gives Neff = 3.03986,
   D/H = 2.4688 (−0.08% shift from the massless n=6 path, consistent
   with finite-m_e corrections at T ~ m_e).

4. **Sigl-Raffelt relaxation for n=4/n=6**
   `_oscillation_relaxation` and `apply_oscillation_mixing` raise error
   for n=4/n=6. Generalize the relaxation target and conservation law.
   Medium effort. (Users currently can use `nu_oscillation_method =
   'collision_mixing'` or the QKE path instead.)

5. **Sterile neutrino production via MSW resonance**
   Genuine new BSM capability via the QKE path. Add sterile Hamiltonian
   terms (Dodelson-Widrow / Shi-Fuller). Opens a big literature.

## Validation & literature comparison

6. **Reproduce Bennett+2021 Table 4 / Fig 4 explicitly** ✓ (done)
   `validation/literature_comparison.py` runs 5 SM configurations and
   prints deltas against Bennett recommended Neff = 3.0440 ± 0.0002.
   QKE reproduces to 10⁻⁴. Expected output: `literature_comparison.out.txt`.

7. **Reproduce Froustey+2020 Fig 5 or Table 2** ✓ (done, same script)
   Includes Froustey's Full QKE = 3.04397, ATAO = 3.04397, and
   w/o mean-field = 3.04407 reference values for comparison.

8. **Plot spectral distortions** ✓ (done)
   `validation/spectral_distortion.py` runs one QKE BBN and saves
   `spectral_distortion.png` showing the 6 per-species Δf/f_FD(y)
   signatures. Matches Dolgov-style hot-tail excess with ν_e ~3%,
   ν_μ/ν_τ ~2.3% at y/T_ν_com = 10.

## Demos / notebooks

9. **BSM demo scenarios** ✓ (done, as script — notebook conversion
   still available if desired)
   `validation/bsm_demos.py` exercises three toy BSM setups:
   (A) L_μ-L_τ anomaly with n=4 diagonal, (B) lepton asymmetry with
   n=6 diagonal, (C) ν_τ bump injection showing plasma absorption at
   T=5 MeV (absorbed at T_boltz_start when ν-e still coupled, so
   LOWERS Neff). Docs the thermalization subtlety.

10. **Update `PRyMdemoSM.ipynb` / `PRyMdemoNP.ipynb`** ✓ (done)
    PRyMdemoSM has a new "PRyMordial-nu extensions" section at the end:
    markdown overview + 4 cells exercising the `two_loop_QED_flag`
    (ΔNeff ≈ −10⁻⁵) and the full QKE density matrix mode (Neff = 3.0446,
    Bennett+2021 match). PRyMdemoNP has a "Task 2: μ-τ / ν-ν̄ asymmetric"
    section at the end with a runnable lepton-asymmetry (ξ_μ = ±0.1,
    n=6 diagonal) example. Both new blocks cross-reference
    `validation/` for more BSM scenarios.

## Code quality

11. **Pytest-style regression tests** ✓ (done)
    `tests/test_regression.py` freezes mode-1, mode-2, mode-3, mode-5, and
    mode-6 outputs with tight tolerances. Fast tests run in ~15 s;
    full suite (`pytest`) in ~4-5 min. See `tests/README.md`.

12. **Refactor the nu-e dispatcher in `collision_integrals`** ✓ (done)
    Extracted six helper methods: `_fnu_corrections(Tg)` (4 float lookups
    once), `_nu_e_n3 / _nu_e_n4 / _nu_e_n6` (per-n ν-e integrals),
    `_nu_nu_dispatch` / `_nu_e_dispatch` (route by self.n_species), and
    `_apply_NP_collisions` (generalized NP collision dispatch — previously
    only wrote to the first 3 slots regardless of n). `collision_integrals`
    body dropped from 119 lines to ~25. Full pytest suite (5/5 tests)
    passes unchanged in 236 s.

13. **Task 0 residual** ✓ (closed)
    Fix landed in commit `fd4fb0e`: new
    `PRyMthermo.distributions_are_thermal_fd()` helper runs the smart-
    dispatch pattern on the six general_nu callables and keeps the aTid
    a(T) correction active when distributions are thermal FD. Residual
    is now at the GL-quadrature convergence floor (~0.006% D/H).

## Performance

14. **Numba AOT or persistent cache** ✓ (done)
    `@njit(cache=True)` added to all njit-decorated functions in
    `PRyM_boltzmann.py` and `PRyM_thermo.py`. `PRyM_eval_nTOp.py`
    already had caching. Saves ~6 s per cold-start on a full Boltzmann
    run (110 s → 104 s); most of the remaining run time is the actual
    collision-integral summation, not JIT.

## Open research questions

15. **Full QKE as an independent ODE driver**
    Currently the QKE path uses Strang splitting (exact unitary osc
    interleaved with explicit collision + damping). A fully-coupled
    implicit driver would be a research-grade contribution but has
    stiffness challenges (osc timescale ≪ collision timescale).

---

## Item 13 details (Task 0 residual)

After upgrading the QED baseline tables from NUDEC_BSM v1 to v2 in
Task 1, the previously-documented +0.176% D/H shift in the "General nu
(FD)" validation mode dropped to +0.007%. That residual is well below
the 0.1% oracle tolerance in CLAUDE.md, so it no longer blocks anything.

The *mechanism* of the original bug is still live (see commit
`dc05a67`'s plan file at `.claude/plans/iterative-strolling-lighthouse.md`
for the full description). In short:

* `PRyM/PRyM_main.py:~720` disables the `aTid_flag` (incomplete-decoupling
  correction to a(T)) whenever `general_nu_flag=True`.
* For the default-FD path, `Tnu_eff_e(Tg) = Tg` identically, so the
  comment-level justification for the disable no longer applies.
* Effect: a(T) differs by ~10⁻⁴–10⁻³ at T < 0.1 MeV between the two
  modes, biasing `rhoB_BBN(a_of_t(t))` and hence D/H (which is set at
  T ~ 0.07 MeV). Yp (set at T ~ 0.8 MeV) is less affected — matches the
  observed fingerprint.

**Planned fix** (for when you decide to close item 13):
- Add a `distributions_are_thermal_fd()` helper in `PRyM_thermo.py`
  (mirroring the smart-dispatch pattern at `PRyM_eval_nTOp.py:619–640`).
- Replace the `general_nu_flag` guard at `PRyM_main.py:~720` with a
  combined test: `aTid_flag and (not general_nu_flag or thermal_fd)`.
- For genuinely non-thermal distributions, generalize the aTid
  correction using the same entropy trajectory the Boltzmann phase
  already uses at `PRyM_main.py:~411–438`.

Low priority until (or unless) someone cares about permille-level D/H
reproducibility across the `general_nu`-but-not-`boltzmann_nu` path.
