"""Stage E.2 sprint-5: Hamiltonian potential scope audit.

After sprint 4 (damping-magnitude ruled out) the saturation at Point C
is consistent with the effective in-medium mixing angle sin^2 2theta_m
approaching unity regardless of the bare 1e-4. The most economical way
to amplify the effective mixing is to get the Hamiltonian matter
potential wrong. PRyMordial has three matter-type potentials that
FortEPiaNO either doesn't have or handles differently:

  V_NC      -- Notzold-Raffelt NC thermal on active diagonals
               (8√2*G_F*rho_e*E/(3*m_Z^2)). FortEPiaNO's matter.f90
               literally says "!missing: term for NC!".
  V_thermal -- Notzold-Raffelt CC thermal on nu_e only
               (8√2*G_F*rho_e*E/(3*m_W^2)). Both codes include it.
  V_nunu    -- nu-nu self-interaction matrix (Pantaleone-Sigl-Raffelt
               √2*G_F*integral y^2 (rho-rhobar) dy / (2pi^2*a^3)).
               Both codes include it.

This diagnostic scales each of the three potentials to 0.0 at Point C
and compares to the baseline. If ANY of them drops dNeff dramatically,
that potential (or at least our implementation of it) is driving the
amplification.

Point C config copied from diag_qke_window.py: sin^2 2theta_24=1e-4,
Dm2_41=0.93 eV^2, PMNS on, Mirizzi damping, default 5 MeV window.
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
    _reset_scales()


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


def _run(label, configure, scale_overrides=None):
    configure()
    if scale_overrides:
        for name, val in scale_overrides.items():
            setattr(PRyMini, name, float(val))
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
    print("Stage E.2 sprint-5: Hamiltonian potential scope audit")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi, window=5 MeV.")
    print("=" * 96)

    # 3x3 reference (all potentials default; insensitive to 4-flavor V_NC block).
    Neff_3x3, _, _ = _run("3x3 reference (no sterile, all scales = 1.0)", _configure_3x3)
    print()

    # Point C sweeps.
    scenarios = [
        ("baseline (all scales = 1.0)",     {}),
        ("V_NC OFF (qke_v_nc_scale = 0.0)", {"qke_v_nc_scale": 0.0}),
        ("V_thermal OFF (qke_v_thermal_scale = 0.0)", {"qke_v_thermal_scale": 0.0}),
        ("V_nunu OFF (qke_v_nunu_scale = 0.0)", {"qke_v_nunu_scale": 0.0}),
    ]

    results = []
    for label, overrides in scenarios:
        Neff, ss, dt = _run(f"Point C, {label}", _configure_point_c, overrides)
        dNeff = Neff - Neff_3x3
        results.append((label, Neff, ss, dNeff, dt))
        print(f"     dNeff = {dNeff:+.4f}")
        print()

    print("=" * 96)
    print(f"{'scenario':50s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s}")
    print("-" * 96)
    for label, Neff, ss, dNeff, dt in results:
        print(f"{label:50s} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f}")
    print("=" * 96)
    print()
    print("Decision guide:")
    print("  - 'V_NC OFF' drops dNeff dramatically (toward Hannestad 0.04)")
    print("    -> V_NC scope / sign / magnitude is the bug (or at least a major driver).")
    print("  - 'V_thermal OFF' drops dNeff dramatically")
    print("    -> V_thermal is the mechanism (unlikely per FortEPiaNO parity but worth checking).")
    print("  - 'V_nunu OFF' drops dNeff dramatically")
    print("    -> V_nunu self-interaction / trace_nxi path is the driver.")
    print("  - All three scenarios give ~0.85 like baseline")
    print("    -> potentials are innocent; suspect Strang split or representation leak next.")
