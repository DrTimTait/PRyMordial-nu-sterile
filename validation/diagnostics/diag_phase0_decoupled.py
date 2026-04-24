"""Stage E.2 sprint 10: Phase-0 decoupled-sterile regression guard.

Gate-5 unit test. When sterile_flag=True but all active-sterile mixing
angles (theta_14, theta_24, theta_34) and all lepton asymmetries
(xi_*) are zero, the sterile sector is fully decoupled and rho_ss
must stay at zero everywhere. Phase 0 evolves the 4x4 QKE from
T_phase0_start down to T_boltz_start; if Phase 0 leaks sterile
population through numerical noise in _build_L_list, _build_H_list,
or _etdrk2_expm_phi when the physical mixing is identically zero,
this diagnostic catches it before Point A / B / C runs.

Pass criterion: max |rho_ss(T=T_boltz_start)| < 1e-12 across all y-modes
and both sectors after Phase 0 finishes.

This is the Phase-0 analogue of the sprint-8 2-level L-conservation
regression guard (diag_2level_damped_energy.py). Any Phase-0 fix that
makes this test fail is wrong; the decoupled-sterile case must remain
bit-zero under the driver.
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

_OUT_TXT = os.path.join(_WT, "validation/diagnostics/diag_phase0_decoupled.out")


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
    PRyMini.qke_damping_formula = "mirizzi"
    # Decoupled-sterile configuration: sterile present but no mixing.
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # Phase-0 window: 100 -> 30 MeV, 1000 steps (enough to exercise the
    # driver through a representative MSW-region decade without the full
    # gate-9 budget).
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 1000
    # Phase-B at default n_B (auto-scaled by validator) -- we only need
    # Phase 0 to finish for the gate; Phase-B consistency is covered by
    # other gates.
    PRyMini.n_B_override = None
    # Keep the sprint-5 projection off so this gate is orthogonal to the
    # projection-flip decision in sprint 10 Chunk 7.
    PRyMini.qke_v_nunu_active_only = False


def _run():
    _base_flags()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    dt = time.time() - t0
    return c, dt


if __name__ == "__main__":
    header = [
        "=" * 96,
        "Stage E.2 sprint 10 gate 5: Phase-0 decoupled-sterile regression guard",
        "sterile_flag=True, theta_14=theta_24=theta_34=0, xi_*=0.",
        "Phase 0: 100 -> 30 MeV, 1000 steps. Pass if max|rho_ss(30 MeV)| < 1e-12.",
        "=" * 96,
    ]
    for line in header:
        print(line, flush=True)

    c, dt = _run()

    phase0_rho = getattr(c, "_phase0_rho_final", None)
    if phase0_rho is None:
        print("FAIL: c._phase0_rho_final is None; Phase-0 block did not run.",
              flush=True)
        sys.exit(2)
    if phase0_rho.shape[1] < 4:
        print(f"FAIL: c._phase0_rho_final has shape {phase0_rho.shape}; "
              "sterile diagonal (index 3) not present.", flush=True)
        sys.exit(2)

    rho_ss = phase0_rho[:, 3, :]
    max_abs = float(np.max(np.abs(rho_ss)))

    # Also probe the full Phase-B exit rho_ss for completeness -- the
    # fully decoupled case should keep rho_ss = 0 through Phase B too.
    phaseB_rho = getattr(c, "_boltz_rho_final", None)
    max_abs_phaseB = float(np.max(np.abs(phaseB_rho[:, 3, :]))) \
        if phaseB_rho is not None else float("nan")

    tol = 1.0e-12
    verdict = "PASS" if max_abs < tol else "FAIL"

    lines = header + [
        f"Runtime: {dt:.1f}s",
        "",
        f"max |rho_ss| at Phase-0 exit (T = T_boltz_start): {max_abs:.3e}",
        f"max |rho_ss| at Phase-B exit (T = T_boltz_end):   {max_abs_phaseB:.3e}",
        f"tolerance: {tol:.1e}",
        "",
        f"{verdict}: Phase-0 leakage test ({'<' if max_abs < tol else '>='}) "
        f"1e-12 in the decoupled-sterile configuration.",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    with open(_OUT_TXT, "w") as fh:
        fh.write(text + "\n")
    sys.exit(0 if max_abs < tol else 1)
