"""Stage E.2(b) diagnostic: test hypothesis B (QKE evolution-window mismatch).

Hypothesis A was falsified in diag_as_gain.py (PMNS-on dNeff moves ~1.5-2%
across gain-zero and sym-gain variants; wrong direction too). Moving on
to hypothesis B.

Hypothesis B (per doc/STAGE_E2_BRIEF.md). PRyMordial runs Phase B
(QKE / Boltzmann) from PRyMini.T_boltz_start = 5 MeV down to T_boltz_end.
Phase A (T_start = 10 MeV -> T_boltz_start = 5 MeV) keeps neutrinos in
thermal equilibrium, so all coherences are zero at QKE-start. Hannestad+2012
integrates their QKE from 60 MeV down to 1 MeV. The 5 - 60 MeV window is
entirely absent in PRyMordial. If DW production has significant buildup
in that window, extending QKE-start to 60 MeV should drop dNeff.

Diagnostic. Run Hannestad Point C (sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93,
PMNS on, Mirizzi damping) at three QKE-start temperatures:
  baseline     -- T_boltz_start =  5.0 MeV (default). Expect dNeff ~ 0.858.
  extended_30  -- T_boltz_start = 30.0 MeV.
  extended_60  -- T_boltz_start = 60.0 MeV (matching Hannestad+2012).

T_start is pushed to match so Phase A stays thermal upstream of the QKE.

Expected (per brief). If window was the bug, dNeff ~ 0.858 -> 0.3-0.5 at
extended_60. Monotone progression across 5 / 30 / 60 MeV would be a
stronger signal. Flat profile falsifies hypothesis B.
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


def _set_window(T_qke_start_MeV):
    """Push both T_start and T_boltz_start to T_qke_start_MeV.

    T_start must be >= T_boltz_start (Phase A runs T_start -> T_boltz_start
    thermally). We set T_start very slightly higher so Phase A has a tiny
    thermal-equilibrium interval; the QKE then initialises at T_qke_start.
    """
    PRyMini.T_boltz_start = float(T_qke_start_MeV)
    # T_start is stored in Kelvin.
    PRyMini.T_start = (float(T_qke_start_MeV) + 0.1) * PRyMini.MeV_to_Kelvin


def _run(label, configure, T_qke_start_MeV):
    configure()
    _set_window(T_qke_start_MeV)
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
    print(f"  {label:50s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss


if __name__ == "__main__":
    print("=" * 88)
    print("Stage E.2(b) diagnostic: QKE evolution-window extension (hypothesis B)")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi damping.")
    print("Hannestad target: dNeff ~ 0.04.  PMNS-on T_boltz_start=5 baseline: dNeff ~ 0.858.")
    print("=" * 88)

    # 3x3 reference is insensitive to the window extension for our purposes;
    # compute once with the default window to anchor dNeff.
    Neff_3x3, _ = _run("3x3 reference (no sterile, window=5 MeV)",
                       _configure_3x3, 5.0)
    print()

    results = []
    for T_start_MeV in [5.0, 30.0, 60.0]:
        label = f"(T_boltz_start = {T_start_MeV:>4.0f} MeV)"
        Neff, ss = _run(label, _configure_point_c, T_start_MeV)
        dNeff = Neff - Neff_3x3
        results.append((T_start_MeV, Neff, ss, dNeff))
        print(f"     dNeff = {dNeff:+.4f}")
        print()

    print("=" * 88)
    print(f"{'T_boltz_start (MeV)':>22s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>12s}")
    print("-" * 88)
    for T_start_MeV, Neff, ss, dNeff in results:
        print(f"{T_start_MeV:>22.0f} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>12.3f}")
    print("=" * 88)
    print()
    print("Decision:")
    print("  - dNeff drops substantially with extended window  -> hypothesis B")
    print("    partially confirmed. Land fix as new PRyMini.T_qke_start flag.")
    print("  - dNeff essentially flat across 5 / 30 / 60 MeV   -> hypothesis B")
    print("    falsified. Hand off to hypothesis C (y-grid) or deeper dive.")
