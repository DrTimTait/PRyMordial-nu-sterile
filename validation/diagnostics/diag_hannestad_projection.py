"""Stage E.2 sprint-5: full Hannestad+2012 suite with V_nunu active-only projection.

After diag_potential_scope.py identified V_nunu as the saturation driver
and diag_vnunu_active_only.py confirmed the projection fix drops ΔNeff
from +0.858 to -0.012 at Point C, this diagnostic runs the full
A/B/C benchmark suite with the fix on.

Hannestad+2012 benchmarks (Fig. 2, δm²=0.93 eV², ν_μ-sterile mixing):
  A  Full thermalisation      sin²2θ = 1.0e-1    dNeff ≈ 1.00
  B  Partial thermalisation   sin²2θ = 2.26e-3   dNeff ≈ 0.50
  C  Minimal thermalisation   sin²2θ = 1.0e-4    dNeff ≈ 0.04

Pre-fix PRyMordial results: A=0.953 (agree, 5% under), B=0.963 (93%
over), C=0.853 (2031% over). The fix is expected to keep A≈1.0 (large
mixing regime, thermalization saturates regardless of coherence
mechanism), and drop B and C toward literature.

Config: PMNS on, Mirizzi damping, default 5 MeV window.
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


def _reset_scales():
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True  # <-- the fix under test


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
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 5.0
    PRyMini.T_start = 10.0 * PRyMini.MeV_to_Kelvin
    _reset_scales()


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
          f"D/H={res[5]:.4f}  Σρ_ss={sum_ss:.3f}  ({dt:.0f}s)", flush=True)
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
    print("=" * 88)
    print("Stage E.2 sprint-5: Hannestad+2012 suite WITH V_nunu active-only projection")
    print("qke_v_nunu_active_only = True (fix under test)")
    print("=" * 88)

    Neff_3x3, _ = _run("3x3 QKE reference (no sterile)", configure_3x3)
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
    print("-" * 96)
    print(f"{'Point':30s}  {'sin²2θ':>10s}  {'δm²':>6s}  {'H+2012':>10s}  "
          f"{'Ours (new)':>12s}  {'Ours (old)':>12s}  {'Δ new':>9s}")
    print("-" * 96)
    # Pre-fix values from validation/sterile_DW_literature.out.txt
    old = {
        "A  full thermalisation": 0.953,
        "B  partial thermalisation": 0.963,
        "C  minimal thermalisation": 0.853,
    }
    for label, sin2_2t, dm2, h_dNeff, ours_dNeff, sum_ss in results:
        abs_diff_new = ours_dNeff - h_dNeff
        print(f"{label:30s}  {sin2_2t:10.2e}  {dm2:6.3f}  {h_dNeff:10.3f}  "
              f"{ours_dNeff:12.3f}  {old[label]:12.3f}  {abs_diff_new:+9.3f}")

    print()
    print("Baseline: 3×3 reference Neff =", f"{Neff_3x3:.5f}")
