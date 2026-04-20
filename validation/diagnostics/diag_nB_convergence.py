"""Stage E.2 sprint-2 (scope a): n_B convergence scan at T_boltz_start = 30 MeV.

Suspect 1 from doc/STAGE_E2_SPRINT2_BRIEF.md: the Phase B step count
`n_B = max(2000, 2 * n_sampling)` is fixed regardless of the evolution
window. At T_boltz_start = 30 MeV the log-uniform-in-a grid spans ~3.8
decades in T vs 3.0 at defaults, but n_B is unchanged -> coarser steps
at high T where the collision rate scales ~T^5. Sprint 1 saw
`sum rho_ss = 26.17` with dNeff = 0.0496 (internally inconsistent) at
T_boltz_start = 30 MeV. If the inconsistency is a discretisation
artefact, forcing higher n_B should restore consistency and converge
dNeff monotonically.

Point C flags copied from diag_qke_window.py (Mirizzi damping, PMNS on,
sterile 3+1, Dm2_41 = 0.93). T_boltz_start fixed at 30 MeV; vary
PRyMini.n_B_override across {2000, 5000, 10000, 20000}.

Expected decision:
  - dNeff converges monotonically + sum rho_ss becomes consistent
    with dNeff -> Suspect 1 confirmed; the converged n_B pins
    `scale_k` for the validator auto-scale.
  - dNeff unstable / non-monotone -> Suspect 1 falsified; escalate.
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


T_WINDOW_MEV = 30.0


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


def _set_window_30():
    PRyMini.T_boltz_start = T_WINDOW_MEV
    PRyMini.T_start = (T_WINDOW_MEV + 0.1) * PRyMini.MeV_to_Kelvin


def _run(label, configure, n_B_override):
    configure()
    _set_window_30()
    PRyMini.n_B_override = n_B_override
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
    return Neff, sum_ss, dt


if __name__ == "__main__":
    print("=" * 96)
    print(f"Stage E.2 sprint-2 (scope a): n_B convergence at T_boltz_start = {T_WINDOW_MEV:.0f} MeV")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi damping.")
    print("Sprint-1 baseline (n_B = 2000): dNeff = +0.0496, sum_rho_ss = 26.17.")
    print("=" * 96)

    # 3x3 reference once. Uses the default n_B path (override = None),
    # at T_boltz_start = 30 MeV so the window matches the sterile runs.
    Neff_3x3, _, _ = _run(f"3x3 reference (no sterile, window={T_WINDOW_MEV:.0f} MeV, n_B default)",
                          _configure_3x3, None)
    print()

    # Sprint-1 showed ~16 min per solve at n_B=2000; runtime scales ~ linearly in
    # n_B (ETDRK2 per-step cost dominated by O(N^3) expm, N = 9·Ny fixed). The
    # brief's 4-point scan {2000, 5000, 10000, 20000} would be ~5 h; first-pass
    # scan below is {2000, 5000, 10000} (~2.5 h) to see the convergence trend.
    # Add 20000 in a follow-up run if convergence is not yet clear at 10000.
    results = []
    for n_B in [2000, 5000, 10000]:
        label = f"(T_boltz_start={T_WINDOW_MEV:.0f} MeV, n_B_override={n_B})"
        Neff, ss, dt = _run(label, _configure_point_c, n_B)
        dNeff = Neff - Neff_3x3
        results.append((n_B, Neff, ss, dNeff, dt))
        print(f"     dNeff = {dNeff:+.4f}")
        print()

    print("=" * 96)
    print(f"{'n_B':>8s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s} {'runtime(s)':>12s}")
    print("-" * 96)
    for n_B, Neff, ss, dNeff, dt in results:
        print(f"{n_B:>8d} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f} {dt:>12.0f}")
    print("=" * 96)
    print()
    print("Decision:")
    print("  - dNeff monotone + sum_rho_ss consistent -> Suspect 1 confirmed.")
    print("    Pin scale_k for validator auto-scale from the converged n_B.")
    print("  - dNeff unstable / non-monotone           -> Suspect 1 falsified.")
    print("    Escalate to Suspect 3 (ETDRK2 at large |L*dt|) via diag_2level_high_T.")
