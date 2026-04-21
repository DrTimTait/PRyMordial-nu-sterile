"""Stage E.2 sprint-5: V_nunu active-only projection test.

Sprint-5 potential scope audit (diag_potential_scope.py) showed that
turning off V_nunu drops dNeff from +0.858 to -0.030 at Hannestad
Point C -- V_nunu is the entire driver of the small-mixing DW
overproduction.

Physics diagnosis: the Pantaleone-Sigl-Raffelt ν-ν self-interaction
goes through a Z exchange, which couples only to the SU(2)_L doublet.
Sterile neutrinos are singlets with zero NC charge, so
V_nunu[s, *] = V_nunu[*, s] = V_nunu[s, s] = 0 identically. PRyMordial
computes V_nunu_eV from the full 4x4 (rho - rhobar) matrix and adds
it to every H entry at _build_H_list: this gives spurious
V_nunu[α, s] coupling that bootstraps active-sterile coherence
regardless of the tiny sin(2θ)·Δm²/(2E) vacuum mixing.

Test: flip PRyMini.qke_v_nunu_active_only to True (default False
preserves pre-fix behaviour). With the projection on, V_nunu's
sterile row and column are zeroed before H assembly. Expectation:
dNeff drops from ~0.858 toward the Hannestad target of 0.04.

Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS on, Mirizzi,
window=5 MeV.
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


def _reset():
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nunu_active_only = False


def _point_c_flags():
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
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 5.0
    PRyMini.T_start = 10.0 * PRyMini.MeV_to_Kelvin
    _reset()


def _configure_3x3():
    _point_c_flags()
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0


def _configure_point_c():
    _point_c_flags()
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0


def _run(label, configure, overrides=None):
    configure()
    if overrides:
        for k, v in overrides.items():
            setattr(PRyMini, k, v)
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
    if (hasattr(c, "_boltz_rho_final")
            and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
    Neff = res[0]
    print(f"  {label:60s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss, dt


if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint-5: V_nunu active-only projection (fix test)")
    print("Hannestad Point C (sin^2 2theta_24=1e-4, Dm2_41=0.93, PMNS ON, window=5 MeV)")
    print("Hannestad+2012 target: dNeff ~ 0.04")
    print("=" * 96)

    Neff_3x3_full, _, _ = _run(
        "3x3 reference (V_nunu full 4x4 behaviour, default)",
        _configure_3x3,
    )
    Neff_3x3_proj, _, _ = _run(
        "3x3 reference (V_nunu active-only projection)",
        _configure_3x3,
        {"qke_v_nunu_active_only": True},
    )
    print()

    scenarios = [
        ("baseline (V_nunu full 4x4, default)", {}, Neff_3x3_full),
        ("V_nunu ACTIVE-ONLY projection", {"qke_v_nunu_active_only": True}, Neff_3x3_proj),
    ]
    results = []
    for label, overrides, ref_Neff in scenarios:
        Neff, ss, dt = _run(f"Point C, {label}", _configure_point_c, overrides)
        dNeff = Neff - ref_Neff
        results.append((label, Neff, ss, dNeff, ref_Neff, dt))
        print(f"     dNeff = {dNeff:+.4f}  (ref: {ref_Neff:.5f})")
        print()

    print("=" * 96)
    print(f"{'scenario':45s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s} {'ref_Neff':>10s}")
    print("-" * 96)
    for label, Neff, ss, dNeff, ref, _ in results:
        print(f"{label:45s} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f} {ref:>10.5f}")
    print("=" * 96)
    print()
    print("Verdict:")
    print("  - dNeff drops to O(0.04) with projection -> bug identified and fixed.")
    print("  - dNeff partially drops                  -> projection is part of the story.")
    print("  - dNeff unchanged                        -> V_nunu scope is not the driver.")
