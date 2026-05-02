# Stage F sprint 4 brief: FortEPiaNO cross-solver comparison

Single-file handoff for the agent (or human) taking over sprint 4
after the sprint 1b'-Phase 2 closure. By the end of reading this
you should know why we need an independent QKE solver to
discriminate the remaining factor-3 magnitude floor in the
Saviano calibration, what the FortEPiaNO build path is, which
calibration points to run on the FortEPiaNO side, and how to
read the cross-comparison verdict.

## §1 One-paragraph orientation

Sprint 1b' Phase 1 (commit `d3fa045`) established that
PRyMordial-nu reproduces Saviano-2013 sterile-induced Yp shifts
qualitatively (sign correct at all L != 0 points; sign-flipped
pair enhancement captured) but under-predicts magnitude by a
factor of 2-3. Phase 2 (commit `97cc5c4`) tested whether
re-enabling Phase 0 closes the gap; verdict was MIXED — Phase 0
helps S1, doesn't change S2, hurts S3 (sign-flipped pair) and
inverts the S1/S3 ordering relative to Saviano. **The factor-3
magnitude floor is not Phase 0.** Two suspects remain:

* **Suspect 12: live cure flag (post-Phase-B clamp + pair-symm OR)
  suppresses Saviano-regime asymmetry amplification.** The cure
  was designed for L=0 narrow-mixing artifact; may over-fire in
  L != 0 saturated regime. Phase 3 directly tests (Phase 0 ON +
  cure OFF, ~9 h sequential).
* **Suspect 13: solver-level perturbative-vs-self-consistent
  systematic.** Saviano uses PArthENoPE BBN driven by perturbative
  Γ/Γ⁰ rescaling from a 3+1 QKE solver; PRyM uses full
  self-consistent QKE+BBN. The factor-3 may be the irreducible
  cost of doing physics differently. Sprint 4 (this brief) tests
  by running a *different* full QKE solver — FortEPiaNO — at the
  same parameters and checking whether FortEPiaNO matches
  Saviano (then PRyM has a structural issue) or matches PRyM
  (then the factor-3 is genuinely solver-level vs Saviano's
  perturbative method).

Sprint 4 also gives us a clean independent calibration of the
**Hannestad benchmark suite** — PRyM is currently 4/4 PASS
against HTT 2012 (sprint 3h-d audit) but the only published
cross-check at those points has been HTT 2012's own results.
A FortEPiaNO Hannestad scan provides a second independent QKE
data point.

## §2 What to read, in order

Budget ~60 min before fetching FortEPiaNO.

1. **`CLAUDE.md`** — project overview.
2. **`doc/STAGE_E2_FORTEPIANO_COMPARISON.md`** — the existing
   solver/physics comparison written before sprint 16. Already
   identifies all the solver-level differences (independent
   variable, single-sector vs two-sector, off-diagonal damping
   approximation, Gauss-Laguerre vs linear momentum grid, etc.).
   ESSENTIAL reading. The doc identifies FortEPiaNO source URL
   as `https://bitbucket.org/ahep_cosmo/fortepiano_public`.
3. **`doc/STAGE_E2_SPRINT17_LITERATURE.md`** — HTT 2012
   reference table; confirms 4/4 Hannestad PASS targets after
   sprint 3h-d correction.
4. **`doc/STAGE_F_SPRINT3HE_SAVIANO_AUDIT.md`** — Saviano-regime
   parameter point (sin²θ_es = 0.025, sin²θ_µs = 0.023,
   Δm² = 0.89 eV², L != 0 NH).
5. **`validation/diagnostics/diag_stage_f1bp_saviano_calibration.{py,out,npz}`**
   — Phase 1 (Phase 0 OFF) Saviano calibration. Reference for
   PRyM's 3-point Yp under live cure.
6. **`validation/diagnostics/diag_stage_f1bp_phase2_phase0_on.{py,out,npz}`**
   — Phase 2 (Phase 0 ON) Saviano calibration. Reference for
   PRyM under Phase 0.
7. **`validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.{py,out,npz}`**
   — PRyM's 4/4 Hannestad scan under live cure.
8. **`References/1905.11290v3_gariazzo2019.pdf`** —
   FortEPiaNO source paper. Section 2 (equations of motion)
   and Appendix B (build / numerical methods). The PDF is in
   the local References/ directory.

## §3 FortEPiaNO build prerequisites

Per Gariazzo et al. 2019 (arXiv:1905.11290) Appendix B:

* **Source**: `https://bitbucket.org/ahep_cosmo/fortepiano_public`
  (the URL is in the existing comparison doc; verify it still
  resolves — if the bitbucket repo has moved, search for
  "FortEPiaNO" on github / inspire-hep).
* **Compilation**: Fortran 90/95 with intel `ifort` recommended;
  `gfortran` should also work. The repo includes a Makefile.
* **Dependencies**: ODEPACK (DLSODA in Fortran), LAPACK/BLAS
  (for the active-mass-eigenstate diagonalisation). On macOS:
  `brew install gfortran lapack` typically sufficient. On Linux:
  `apt install gfortran liblapack-dev libopenblas-dev`.
* **Output format**: binary or ASCII (look for `&output` namelist
  in the input file). Default emits per-y row data on a Gauss-
  Laguerre grid; reads of "ρ_α(y)" need the same Laguerre weights
  for fair comparison to PRyM's linear-y grid.

**Build step** (write into a sprint-4 setup script when actually
running):

```bash
# Place outside the PRyMordial-nu tree to avoid contaminating
# the project repo (FortEPiaNO is not vendored):
cd ~/code  # or wherever
git clone https://bitbucket.org/ahep_cosmo/fortepiano_public.git
cd fortepiano_public
make           # or: make COMPILER=gfortran
# Test build with the supplied "default" input:
./fortepiano default.nml
```

If the build fails on macOS due to `ifort`-only flags, edit the
Makefile to use `gfortran` and remove `-mkl` linkages. Allow
~15-30 min for first build.

## §4 What FortEPiaNO can and cannot do (per the existing
comparison doc + 2019 paper)

| Feature | FortEPiaNO supports? |
|---|---|
| 3+1 sterile mixing | ✓ (3+1 PMNS, all four mixing angles) |
| Hannestad-style (sin²2θ, Δm²) scans | ✓ (their headline result) |
| L=0 NH / L=0 IH | ✓ |
| **L ≠ 0 lepton asymmetry** | **✗ (single-sector code; ρ_ν = ρ̄_ν built in)** |
| Gauss-Laguerre y-grid | ✓ (default ~50 nodes) |
| QED plasma corrections to T_γ evolution | ✓ (their dz/dx coupled ODE) |
| BBN abundance evaluation | **✗ (FortEPiaNO outputs ρ_α(y), Neff; no BBN reaction network)** |
| Match Saviano L != 0 | ✗ (would require code modification) |
| Match Hannestad (L=0) | ✓ |

**Critical asymmetry.** FortEPiaNO does NOT support L != 0. So
the apples-to-apples Saviano calibration (sprint 1b' style)
cannot be done in FortEPiaNO without code modification.
**Sprint 4 must therefore split into two sub-tests:**

* **Sub-test 4a (L=0 cross-validation, the cheap test)** — run
  FortEPiaNO at the Hannestad benchmark points and compare its
  δNeff_ss against PRyM's 4/4 PASS results. This validates the
  QKE driver only (no BBN abundance comparison since FortEPiaNO
  doesn't compute Yp).
* **Sub-test 4b (Saviano-regime L=0, the diagnostic test)** —
  run FortEPiaNO at Saviano's mixing parameters
  (sin²θ_es = 0.025, sin²θ_µs = 0.023, Δm² = 0.89 eV², NH) but
  with **L = 0** (since FortEPiaNO can't do L != 0). Read
  FortEPiaNO's end-of-evolution ρ_α(y) distributions.
  Independently feed those distributions to PRyM's BBN network
  (replacing PRyM's own QKE end-state) and compute Yp. Compare
  to PRyM's own L=0 Saviano-regime Yp prediction. If the two
  agree within 0.001 Yp, PRyM's QKE driver is calibrated to
  FortEPiaNO. If they disagree, the QKE drivers themselves
  differ structurally.
* **Sub-test 4c (L != 0 — only if a community L != 0 fork
  exists)** — search github/bitbucket for FortEPiaNO forks that
  support L != 0. If found, run the Saviano-regime L != 0
  calibration and check whether FortEPiaNO matches Saviano's
  Table I or matches PRyM. This is the apples-to-apples
  Suspect 13 test. If no fork exists, defer.

## §5 Calibration points to run

### Sub-test 4a: Hannestad cross-validation

Run FortEPiaNO at the four Hannestad benchmark points
(matching the sprint 3h-b' OR scan):

| Point | sin²2θ_24 | Δm²_41 (eV²) | L | PRyM δNeff_ss | FortEPiaNO target |
|---|---|---|---|---|---|
| A (strong) | 0.1 | 0.93 | 0 | 0.9341 | should match within 0.05 |
| B (mid) | 2.26e-3 | 0.93 | 0 | 0.6781 | should match within 0.05 |
| C (narrow) | 1e-4 | 0.93 | 0 | 0.0276 | should match within 0.005 (HTT 2012 visual ~0.02-0.03) |
| Global-NH | 0.089 | 0.9 | 0 | 0.9457 | should match within 0.05 (HTT page 8: =1) |

PRyM's 4/4 PASS (sprint 3h-d) is against HTT 2012's published
visual reads / explicit page-8 statement. FortEPiaNO is an
independent solver running the same physics. Agreement within
the bands above means cross-validated 4/4. Disagreement means
either FortEPiaNO or PRyM has a solver-specific issue.

### Sub-test 4b: Saviano-regime L=0

Run FortEPiaNO and PRyMordial-nu both at:

* sin²θ_es = 0.025 (single-angle convention; theta_14 in PRyM)
* sin²θ_µs = 0.023 (theta_24 in PRyM)
* Δm²_st = 0.89 eV²
* NH (Δm² > 0)
* L = 0 (xi_e = xi_µ = xi_τ = 0)

Compare:

* **FortEPiaNO output**: end-of-evolution ρ_α(y) per flavor
  on Gauss-Laguerre grid; δNeff_ss; if FortEPiaNO outputs Yp,
  use it directly, otherwise feed ρ_α to PRyM BBN network
  (this will require a small adapter: convert FortEPiaNO Gauss-
  Laguerre ρ_α(y) to PRyM linear-y grid via interpolation; the
  PRyM `make_f_callable` interface accepts grid arrays).
* **PRyMordial-nu output**: same point under live cure flags
  (3h-b' OR + post-Phase-B clamp). Yp, Neff, δNeff_ss.

Pass criterion: PRyM and FortEPiaNO Yp predictions agree
within 0.001 (within the QKE-vs-thermal-SBBN systematic floor
~10⁻⁴ documented in sprint 3h-e). If agreement holds, PRyM's
L=0 QKE+BBN result at Saviano-regime parameters is corroborated
by an independent solver. If not, structural disagreement
diagnosable.

### Sub-test 4c: Saviano-regime L != 0 (conditional)

Only run if a community FortEPiaNO fork supporting L != 0 has
been identified. Same parameter point as 4b but with
xi_e = xi_µ = +1e-3 (matching Saviano S1) or xi_e = +1e-2
(S2). Compare directly to Saviano Table I AND PRyM's Phase 1
Saviano calibration. Three-way comparison resolves Suspect 13:

* If FortEPiaNO matches Saviano (~0.257 at S1) and PRyM
  doesn't (0.248): PRyM has a structural issue.
* If FortEPiaNO matches PRyM (~0.248 at S1) and neither
  matches Saviano: the factor-3 is solver-level
  (perturbative-vs-self-consistent), Saviano's PArthENoPE+
  Γ/Γ⁰ approach is the outlier.
* If all three disagree pairwise: deeper structural
  analysis required.

## §6 Workflow

1. **Build FortEPiaNO** outside the PRyMordial-nu tree (~30 min).
2. **Verify FortEPiaNO build** by reproducing one of the
   benchmark cases shipped with the repo (typically a no-sterile
   3-active SM run). Validates Fortran build and ODEPACK
   linkage.
3. **Sub-test 4a (~4-8 h compute)** — run FortEPiaNO at the
   four Hannestad points; record δNeff_ss; tabulate against
   PRyM's 4/4 PASS values. Compute is on FortEPiaNO side
   (much faster per their paper: "few minutes on four cores"
   per 4×4 case). Total wall ~30-60 min for four points.
4. **Sub-test 4b (~2 h compute, mostly setup)** — adapter
   harness from FortEPiaNO ρ_α(y) to PRyM BBN network. Run
   one Saviano-regime L=0 point on each side. Compare Yp.
5. **Sub-test 4c (conditional, ~6 h)** — only if L != 0 fork
   identified.
6. **Write sprint 4 verdict doc** (`doc/STAGE_F_SPRINT4_FINDINGS.md`)
   covering all three sub-tests, with explicit Suspect 13
   verdict.

Total wall-clock estimate: ~10-20 h depending on Sub-test 4c
inclusion and FortEPiaNO build difficulty.

## §7 Implementation surface

* **NEW (outside repo)**: FortEPiaNO clone at e.g.
  `~/code/fortepiano_public`.
* **NEW**: `validation/diagnostics/diag_stage_f4a_fortepiano_hannestad.py`
  — FortEPiaNO input-file generator (writes namelist files for
  each Hannestad point) and output parser (reads FortEPiaNO's
  ρ_α(y) and δNeff_ss; tabulates against PRyM reference). Does
  NOT execute FortEPiaNO from Python; it generates the inputs
  and parses the outputs after manual ./fortepiano runs.
* **NEW**: `validation/diagnostics/diag_stage_f4b_fortepiano_saviano_l0.py`
  — Saviano-regime L=0 input-file generator; FortEPiaNO ρ_α(y)
  → PRyM BBN adapter (interpolates Gauss-Laguerre ρ to linear-y
  PRyM grid, then runs PRyM's BBN with `make_f_callable`-style
  injection); compares end-state Yp.
* **NEW (conditional)**: `validation/diagnostics/diag_stage_f4c_fortepiano_saviano_lnonzero.py`
  — only if L != 0 fork found.
* **NEW**: `doc/STAGE_F_SPRINT4_FINDINGS.md` at sprint 4 close.

No core PRyMordial-nu code changes anticipated. FortEPiaNO is
external.

## §8 Possible outcomes and what they mean

**Outcome A: Sub-test 4a 4/4 cross-validates (PRyM ≈ FortEPiaNO
at Hannestad benchmarks).** PRyM's QKE driver is independently
validated. Sub-test 4b Yp agreement at L=0 Saviano-regime
follows naturally; PRyM's L=0 narrow-mixing -0.015 Yp deficit
becomes a corroborated novel prediction. Sprint 4 closes
positively. Suspect 13 (solver-level systematic) becomes the
likely answer for the factor-3 vs Saviano: it's perturbative-
vs-self-consistent, and PRyM's value is the more rigorous one.

**Outcome B: Sub-test 4a disagrees at one or more Hannestad
points.** PRyM's QKE driver has a solver-specific issue at the
disagreement point(s). Diagnosable: which solver-level
difference (off-diagonal damping form, single vs two sector,
y-grid resolution, ...) drives the disagreement. Sprint 4
closes with a falsified PRyM result + diagnostic path.

**Outcome C: Sub-test 4a agrees but Sub-test 4b disagrees.**
The QKE drivers are equivalent at L=0 thermal active spectra
but disagree once sterile mixing perturbs the active sector.
Specific finding: localise the disagreement at the
post-Phase-B / sterile-mixing-on stage. Likely cure-flag
related (Suspect 12 indirectly tested).

**Outcome D: Sub-test 4c (if available) shows FortEPiaNO L != 0
matches Saviano, PRyM doesn't.** PRyM has a real disagreement
with both independent solvers in the L != 0 regime. Major
finding; structural analysis required.

**Outcome E: All three solvers (PRyM, FortEPiaNO L != 0,
Saviano) give different L != 0 Yp.** The factor-3 spread is
genuinely solver-dependent across self-consistent QKE codes;
no single answer is "right". Drives a community-level
calibration discussion.

## §9 Out of scope for sprint 4

* Re-running PRyM-side scans at Hannestad or Saviano
  parameters. PRyM's reference data already exists in
  `validation/diagnostics/diag_stage_f3h_b_prime_*` and
  `diag_stage_f1bp_*`.
* Modifying PRyMordial-nu core code.
* Implementing L != 0 in FortEPiaNO if no fork exists.
* Sprint 1b'-Phase 3 (cure flag OFF test) — that's a separate
  test of Suspect 12. Sprint 4 tests Suspect 13. Both can run
  in parallel if compute resources allow.

## §10 Success criteria

Sprint 4 closes successfully when at minimum:

* Sub-test 4a is run and a verdict reported (Outcome A or B).
* Sub-test 4b is run and a verdict reported (Outcome A or C).
* The findings doc records exact PRyM vs FortEPiaNO numbers
  for each tested point with PASS/FAIL band classifications.
* Suspect 13 is either upgraded to CONFIRMED, downgraded to
  FALSIFIED, or further-decomposed into sub-suspects.

Sprint 4 closes inconclusively if FortEPiaNO build fails on
the available platform — in that case, document the build
failure and defer.

## §11 Provenance

* Existing solver-comparison doc:
  `doc/STAGE_E2_FORTEPIANO_COMPARISON.md` (pre-sprint-16).
* FortEPiaNO source paper: `References/1905.11290v3_gariazzo2019.pdf`.
* Sprint 1b' Phase 1 + 2 reference data: commits `d3fa045` and
  `97cc5c4` on `claude/sprint-19-followup`.
* Sprint 3h-b' OR Hannestad reference: commit `ff3f81f`.
* Sprint 3h-d Hannestad audit: commit `f6734aa`.
* Sprint 3h-e Saviano audit: commit `19b2b77`.
