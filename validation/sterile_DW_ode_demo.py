"""Stage D.2 validation: Dodelson-Widrow sterile production under the
full-ODE QKE driver (qke_full_ode_flag=True).

Parallels validation/sterile_DW_demo.py but routes through evolve_step_ode
instead of evolve_step. The key check: the exact-unitary ODE driver
should REPRODUCE the near-thermalization ΔNeff ≈ +0.93 signal that the
quasi-static DW block in evolve_step produces for sin²(2θ_14)=0.1. If
the two agree to O(10⁻²) or better, the ODE driver is correctly
handling active-sterile oscillations without needing the separate
Stage B DW quasi-static transfer.

Runtime: ~7 minutes wall.
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
    PRyMini.qke_full_ode_flag = False  # toggled per run
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = False


def _run(label, full_ode, sterile, theta_14_sq):
    _configure()
    PRyMini.qke_full_ode_flag = full_ode
    PRyMini.sterile_flag = sterile
    if sterile:
        PRyMini.Dm2_41 = 1.0
        PRyMini.theta_14 = np.arcsin(np.sqrt(theta_14_sq)) / 2.0 if theta_14_sq > 0 else 0.0
        PRyMini.theta_24 = 0.0
        PRyMini.theta_34 = 0.0
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss = 0.0
    if hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None:
        rho = c._boltz_rho_final
        if rho.shape[1] >= 4:
            sum_ss = float(rho[:, 3, :].sum())
    print(f"{label:40s} Neff={res[0]:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  Σρ_ss={sum_ss:7.3f}  ({dt:.1f}s)",
          flush=True)
    return res, sum_ss


# 3×3 reference, Strang path
r_strang_3x3, _ = _run("Strang 3×3 reference", False, False, 0.0)

# 3×3 reference, ODE path
r_ode_3x3, _ = _run("ODE    3×3 reference", True,  False, 0.0)

# DW production under Strang (Stage B baseline)
r_strang_dw, ss_strang_dw = _run("Strang DW (sin²(2θ)=0.1)", False, True, 0.1)

# DW production under ODE driver
r_ode_dw, ss_ode_dw = _run("ODE    DW (sin²(2θ)=0.1)", True,  True, 0.1)

print()
print(f"3×3 Neff bias (ODE − Strang): {(r_ode_3x3[0] - r_strang_3x3[0])*1e3:+.3f} × 10⁻³")
print(f"DW ΔNeff (Strang):            {(r_strang_dw[0] - r_strang_3x3[0]):+.4f}")
print(f"DW ΔNeff (ODE):               {(r_ode_dw[0]   - r_ode_3x3[0]):+.4f}")
print(f"DW sterile Σρ_ss (Strang):    {ss_strang_dw:.3f}")
print(f"DW sterile Σρ_ss (ODE):       {ss_ode_dw:.3f}")
