"""Stage E.2 sprint-5: Point C only at T_boltz_start=30 MeV with projection.

The full Hannestad A/B/C suite with projection + 30 MeV crashes at
Point A (large theta -> coherence numerics fail without V_nunu ballast).
Point C has sin^2(2theta)=1e-4, so coherences stay small and should
not trigger the same instability. Single solve to get a clean
small-mixing data point at the physically-correct window (well above
T_MSW ~ 10 MeV).

Reuse 3x3 reference at 30 MeV from diag_hannestad_proj_w30.out:
Neff_3x3 = 3.00987 (projection is a no-op at 3 flavors).

Config: PMNS on, Mirizzi, qke_v_nunu_active_only=True, T_boltz_start=30,
sprint-2 auto-scale -> n_B=5031.
"""
import os
import sys
import time
import importlib

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini


def _base_flags():
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
    PRyMini.qke_ode_etdrk2_flag = True
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = None
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0  # Point C
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 30.1 * PRyMini.MeV_to_Kelvin
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True


if __name__ == "__main__":
    print("=" * 88)
    print("Stage E.2 sprint-5: Point C at T_boltz_start=30 MeV + projection")
    print("sin^2 2theta_24=1e-4, Dm2_41=0.93, PMNS on, Mirizzi, qke_v_nunu_active_only=True")
    print("Hannestad target: dNeff = 0.04")
    print("=" * 88)

    _base_flags()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    Neff = res[0]
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())

    # 3x3 reference from diag_hannestad_proj_w30.out (matched-window)
    Neff_3x3_w30 = 3.00987
    dNeff = Neff - Neff_3x3_w30

    print(f"\n  Point C, proj + 30 MeV window")
    print(f"    Neff       = {Neff:.5f}   (runtime {dt:.0f}s)")
    print(f"    Yp         = {res[4]:.5f}")
    print(f"    D/H        = {res[5]:.4f}")
    print(f"    sum_rho_ss = {sum_ss:.3f}")
    print(f"    Neff_3x3   = {Neff_3x3_w30:.5f}  (reused from w30 run)")
    print(f"    dNeff      = {dNeff:+.4f}  vs Hannestad target 0.040")
    print()
    print("Previous Point C values at same config:")
    print(f"    proj +  5 MeV window:  dNeff = -0.012")
    print(f"    proj + 15 MeV window:  dNeff =  0.009")
    print(f"    proj + 30 MeV window:  dNeff = {dNeff:+.4f}  (this run)")
    print(f"    original (no proj, 5 MeV): dNeff = +0.853")
    print(f"    Hannestad+2012: dNeff = 0.040")
