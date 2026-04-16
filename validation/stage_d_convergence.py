"""Stage D diagnostic: dt-convergence of mode 5b (full-ODE QKE driver).

Runs the QKE with qke_full_ode_flag=True at n_B ∈ {2400, 4800, 9600} to
check whether the 3.0445 → 3.0406 Neff drift vs mode 5 is splitting
error (cured by finer dt) or a fundamental quasi-static-vs-exact-unitary
difference.

Three interpretations:

  1. Monotonic convergence toward 3.0445: the current evolve_step's
     quasi-static answer IS the correct one, and the ODE driver is
     under-resolving. Implies Stage D.2 (ETDRK2 corrector) will close
     the gap.

  2. Monotonic convergence toward 3.0406 (or lower): the ODE driver is
     already converged, and the 3.0445 value is the quasi-static
     approximation's bias. Implies Stage D.2/D.3 won't help this
     drift; it's a feature not a bug.

  3. No clean trend: something else (Phase A/C, collision integral
     convergence) dominates. Investigate further before implementing.

Runtime: ~20 minutes wall (3 runs of ~130, 260, 520 s).
"""
import os
import sys
import time
import importlib

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
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = False


def _run(label, n_B_override, full_ode):
    _configure()
    PRyMini.qke_full_ode_flag = full_ode
    PRyMini.n_B_override = n_B_override
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    res = PRyMmain.PRyMclass().PRyMresults()
    dt = time.time() - t0
    print(f"{label:45s} Neff={res[0]:.6f}  Yp={res[4]:.6f}  "
          f"D/H={res[5]:.4f}  ({dt:.1f}s)", flush=True)
    return res


print("=" * 80)
print("Stage D dt-convergence diagnostic")
print("=" * 80)

# Baseline Strang path for reference
r_strang = _run("Mode 5 (Strang path, n_B=2400)", None, False)

# ODE driver at three resolutions
r_ode_1 = _run("Mode 5b (ODE, n_B=2400)",  2400,  True)
r_ode_2 = _run("Mode 5b (ODE, n_B=4800)",  4800,  True)
r_ode_4 = _run("Mode 5b (ODE, n_B=9600)",  9600,  True)

# Reset override so later runs aren't affected
PRyMini.n_B_override = None

print()
print(f"{'Neff differences vs Strang (3.0445)':48s}")
print(f"  n_B=2400 :  ΔNeff = {(r_ode_1[0] - r_strang[0])*1e3:+.3f} × 10⁻³")
print(f"  n_B=4800 :  ΔNeff = {(r_ode_2[0] - r_strang[0])*1e3:+.3f} × 10⁻³")
print(f"  n_B=9600 :  ΔNeff = {(r_ode_4[0] - r_strang[0])*1e3:+.3f} × 10⁻³")

# Richardson-style extrapolation for order-2 scheme
# If ΔNeff(n) = C/n², then ΔNeff(∞) ≈ (4·ΔNeff(2n) - ΔNeff(n)) / 3
d1 = r_ode_1[0] - r_strang[0]
d2 = r_ode_2[0] - r_strang[0]
d4 = r_ode_4[0] - r_strang[0]
rich_2 = (4*d2 - d1) / 3.0
rich_4 = (4*d4 - d2) / 3.0
print()
print(f"Richardson extrapolation (assuming O(dt²)):")
print(f"  from n_B=2400/4800 : ΔNeff_∞ ≈ {rich_2*1e3:+.3f} × 10⁻³")
print(f"  from n_B=4800/9600 : ΔNeff_∞ ≈ {rich_4*1e3:+.3f} × 10⁻³")
