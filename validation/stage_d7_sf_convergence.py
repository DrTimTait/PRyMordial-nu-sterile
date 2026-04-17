"""Stage D.7 validation: SF MSW resonance dt-convergence under full ETDRK2.

Companion to `stage_d6_sf_convergence.py` (D.6, ETD1 predictor in the
H-eigenbasis — first-order, non-convergent under SF; see .out.txt there).
D.7 rewrites evolve_step_ode_etdrk2 to absorb the off-diagonal flavor-basis
pair damping into the linear part L and exponentiate the 9x9 vectorised
Liouvillian per mode via the Al-Mohy & Higham (2011) augmented-matrix trick.
This is a genuine ETDRK2 (predictor + corrector, symmetry-preserving) and
is designed to restore O(dt^2) at the Shi-Fuller MSW resonance.

Success criteria (matching the brief § "Validation targets"):

  1. Drift ratio ΔNeff(4800→9600) / ΔNeff(2400→4800) ≈ 0.25 (O(dt^2)).
  2. Richardson-extrapolated Neff matches the D.4 Strang Richardson limit
     3.968 to ~10^-3.
  3. Default-n_B (2400) Neff lands within ~10^-3 of the Richardson limit.
  4. |n_ξe| final is O(1), not O(10) as the D.6 predictor drifted to.

Runtime: ~20 minutes wall (3 runs of ~270, 520, 1040 s). D.7 adds ~4 min of
matrix-exponential overhead per 2400-step run over the Strang baseline.
"""
import os
import sys
import time
import importlib

import numpy as np

# Make the parent worktree importable regardless of where it was cloned.
_WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini


def _configure():
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True   # <-- D.7 path
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = np.arcsin(np.sqrt(1.0e-3)) / 2.0   # sin²(2θ)=1e-3
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 5.0e-2


def _run(label, n_B_override):
    _configure()
    PRyMini.n_B_override = n_B_override
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss = 0.0
    n_asym = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
        if (hasattr(c, "_boltz_solver") and c._boltz_solver is not None
                and hasattr(c._boltz_solver, "y_grid")):
            y = np.asarray(c._boltz_solver.y_grid)
            n_asym = float((y**2 * (rho[0, 0] - rho[1, 0])).sum())
    print(f"{label:30s} Neff={res[0]:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  Σρ_ss={sum_ss:.3f}  "
          f"n_ξe={n_asym:+.3e}  ({dt:.0f}s)", flush=True)
    return res[0], res[4], res[5], sum_ss, n_asym


print("=" * 80)
print("Stage D.7 validation: SF convergence under full ETDRK2 (damping-in-L)")
print("sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2")
print("=" * 80)

results = []
for n_B in (2400, 4800, 9600):
    results.append((n_B, *_run(f"ETDRK2-L SF (n_B={n_B})", n_B)))

PRyMini.n_B_override = None

print()
print(f"{'-'*80}")
print("Convergence metrics:")
print(f"{'-'*80}")
Neffs = [r[1] for r in results]
sss = [r[4] for r in results]
nas = [r[5] for r in results]
for (n_B, Neff, Yp, DoH, ss, na) in results:
    print(f"  n_B={n_B:5d}  Neff={Neff:.5f}  Σρ_ss={ss:.3f}  n_ξe={na:+.3e}")

print()
dN_12 = Neffs[1] - Neffs[0]
dN_24 = Neffs[2] - Neffs[1]
print(f"  ΔNeff (2400→4800) = {dN_12*1e3:+.3f} × 10⁻³")
print(f"  ΔNeff (4800→9600) = {dN_24*1e3:+.3f} × 10⁻³")
print(f"  Ratio = {dN_24/max(abs(dN_12),1e-30):+.3f}   "
      f"(≈ 0.25 for converged O(dt²), ≈ 0.5 for O(dt))")

if results[-1][4] > 0:
    ds_12 = sss[1] - sss[0]
    ds_24 = sss[2] - sss[1]
    print(f"  ΔΣρ_ss (2400→4800) = {ds_12:+.4f}")
    print(f"  ΔΣρ_ss (4800→9600) = {ds_24:+.4f}")

rich = (4 * Neffs[2] - Neffs[1]) / 3.0
print()
print(f"  Richardson extrapolation (4800/9600): Neff_∞ ≈ {rich:.5f}")
print(f"  D.4 Strang Richardson limit (ref):    Neff_∞ ≈ 3.96823")
print(f"  D.4 Strang at default n_B=2400 (ref): Neff   = 3.95297")
print(f"  D.6 ETDRK2 Richardson (non-converged): Neff_∞ ≈ 3.82779")
print(f"  D.7 default-n_B vs D.7 Rich:           {Neffs[0] - rich:+.4f}")
print(f"  D.7 default-n_B vs Strang Rich:        {Neffs[0] - 3.96823:+.4f}")
