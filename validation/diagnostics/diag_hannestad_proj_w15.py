"""Stage E.2 sprint-5: Hannestad A/B/C with projection + extended window.

Sprint-5 found that V_nunu active-only projection fixes Point C
(dNeff 0.858 -> -0.012) but over-corrects Point A (0.953 -> 0.173).
Physical interpretation: the projection is correct (active-only is
the Sigl-Raffelt tree-level truth, matching FortEPiaNO's matter.f90)
but PRyMordial's default Phase B start at T=5 MeV is essentially
AT the MSW resonance for dm^2 = 0.93 eV^2, so the adiabatic approach
that drives full thermalization at large theta is missed.

Fix candidate: projection + T_boltz_start = 15 MeV (sprint-2 auto-scale
handles n_B). If A, B, C all land on Hannestad targets (1.0, 0.5, 0.04)
the bug is fully resolved.

Config: PMNS on, Mirizzi damping, qke_v_nunu_active_only=True,
T_boltz_start = 15 MeV, n_B auto-scaled via validator.
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
    PRyMini.n_B_override = None  # sprint-2 auto-scale will set via validator
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    # Extended window: 30 MeV -> 0.005 MeV (sprint-2 auto-scale -> n_B ~ 5031)
    PRyMini.T_boltz_start = 15.0
    PRyMini.T_start = 15.1 * PRyMini.MeV_to_Kelvin
    # Sprint-5 fix: V_nunu active-only projection
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True


def _run(label, configure):
    _base_flags()
    configure()
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
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
    Neff = res[0]
    print(f"  {label:48s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss


def configure_3x3():
    PRyMini.sterile_flag = False


def configure_point(sin2_2theta, delta_m2):
    def _c():
        PRyMini.sterile_flag = True
        PRyMini.Dm2_41 = delta_m2
        PRyMini.theta_24 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0
    return _c


if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint-5: Hannestad+2012 WITH projection + T_boltz_start=15 MeV")
    print("qke_v_nunu_active_only = True, T_boltz_start = 15 MeV (auto-scaled n_B).")
    print("=" * 96)

    Neff_3x3, _ = _run("3x3 QKE reference (no sterile, 15 MeV window)", configure_3x3)
    print()

    bench = [
        ("A  full thermalisation",    1.0e-1, 0.93, 1.00),
        ("B  partial thermalisation", 2.26e-3, 0.93, 0.50),
        ("C  minimal thermalisation", 1.0e-4, 0.93, 0.04),
    ]

    results = []
    for label, sin2_2t, dm2, hannestad_dNeff in bench:
        Neff, sum_ss = _run(f"DW ({label}, sin²2θ={sin2_2t:.2e}, δm²={dm2})",
                            configure_point(sin2_2t, dm2))
        dNeff_ours = Neff - Neff_3x3
        results.append((label, sin2_2t, dm2, hannestad_dNeff, dNeff_ours, sum_ss))

    print()
    print("-" * 104)
    print(f"{'Point':30s}  {'sin²2θ':>10s}  {'H+2012':>10s}  "
          f"{'Ours (proj+15)':>15s}  {'Ours (proj+5)':>15s}  {'Ours (orig)':>12s}")
    print("-" * 104)
    # Projection + default 5 MeV window from diag_hannestad_projection.out
    proj5 = {
        "A  full thermalisation":    0.173,
        "B  partial thermalisation": 0.133,
        "C  minimal thermalisation": -0.012,
    }
    orig = {
        "A  full thermalisation":    0.953,
        "B  partial thermalisation": 0.963,
        "C  minimal thermalisation": 0.853,
    }
    for label, sin2_2t, dm2, h_dNeff, ours_dNeff, sum_ss in results:
        print(f"{label:30s}  {sin2_2t:10.2e}  {h_dNeff:10.3f}  "
              f"{ours_dNeff:15.3f}  {proj5[label]:15.3f}  {orig[label]:12.3f}")

    print()
    print("Baseline: 3x3 reference Neff =", f"{Neff_3x3:.5f}")
    print()
    print("Verdict:")
    print("  - A, B, C all land near Hannestad targets -> BUG FULLY RESOLVED.")
    print("  - Some improve, some don't               -> window extension is part of story.")
    print("  - All still low                          -> deeper bug, not just window.")
