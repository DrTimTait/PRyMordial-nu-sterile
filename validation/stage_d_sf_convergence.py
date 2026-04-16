"""Stage D.4 pre-diagnostic: dt-convergence of the SF (ξ=5e-2) run
under the ODE driver.

Motivation: Stage D.2 showed the 3×3 ODE driver is locally converged
(Neff(n_B=2400) vs Neff(∞) differ by 3e-4, well below the 4e-3 gap
from Strang). But the SF regime is different: the MSW resonance sweeps
the in-medium mixing angle sharply through π/4, and the exact unitary
must rotate the density matrix by a large phase per dt near resonance.
If dt is too coarse, the resonance crossing may be under-resolved,
producing an ODE SF answer that drifts with n_B.

If the SF result IS converged at default n_B=2400, the 0.03 ΔNeff gap
vs Strang is physics (quasi-static DW formula overestimates near
resonance); no numerical remedy (ETDRK2, etc.) can close it.

If the SF result DRIFTS with finer n_B, we're under-resolving the
resonance — and ETDRK2 (smaller splitting error) or a denser default
dt near resonance should be considered.

Runtime: ~15 minutes wall (3 runs of ~130, 260, 520 s).
"""
import os
import sys
import time
import importlib

import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

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
print("Stage D.4 pre-diagnostic: SF convergence under ODE driver")
print("sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2")
print("=" * 80)

results = []
for n_B in (2400, 4800, 9600):
    results.append((n_B, *_run(f"ODE SF (n_B={n_B})", n_B)))

# Reset override
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

if results[-1][4] > 0:  # Σρ_ss has meaningful evolution
    ds_12 = sss[1] - sss[0]
    ds_24 = sss[2] - sss[1]
    print(f"  ΔΣρ_ss (2400→4800) = {ds_12:+.4f}")
    print(f"  ΔΣρ_ss (4800→9600) = {ds_24:+.4f}")

# Richardson extrapolation assuming O(dt²) convergence
rich = (4 * Neffs[2] - Neffs[1]) / 3.0
print()
print(f"  Richardson extrapolation (4800/9600): Neff_∞ ≈ {rich:.5f}")
print(f"  Strang reference at default n_B:       Neff   = 3.95300 (from D.3)")
print(f"  Gap vs Strang:                         {rich - 3.953:.4f}")
