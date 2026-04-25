"""Stage E.2 sprint 11: Phase-0 decoupled-sterile guard with the
narrow-mixing eigendecomposition fallback turned on. Same configuration
as diag_phase0_decoupled.py except qke_expm_fallback_near_degeneracy=True.

When all mixing angles are zero, _build_H_list produces a Hamiltonian
with no active-sterile coupling, so the eigendecomposition fallback
must produce numerically identical (to bit-zero leakage) results. The
gate triggers based on |H_αα − H_ss| relative to the largest commutator
gap on each y-mode; with theta=0 and Dm2_41=0.93, that diagonal gap is
nonzero (the sterile diagonal is offset by Dm2_41/(2E) regardless of
mixing), so the gate may either fire or not depending on the relative
magnitudes — and either path must keep rho_ss at zero.
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

_OUT_TXT = os.path.join(_WT, "validation/diagnostics/diag_phase0_decoupled_fallback.out")


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
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 1000
    PRyMini.n_B_override = None
    PRyMini.qke_v_nunu_active_only = False
    # Sprint-11 fallback flag.
    PRyMini.qke_expm_fallback_near_degeneracy = True
    PRyMini.qke_expm_fallback_eps_cross = 1.0e-3


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
        "Stage E.2 sprint 11: Phase-0 decoupled-sterile guard with eigendecomp fallback ON",
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

    rho_ss = phase0_rho[:, 3, :]
    max_abs = float(np.max(np.abs(rho_ss)))

    phaseB_rho = getattr(c, "_boltz_rho_final", None)
    max_abs_phaseB = float(np.max(np.abs(phaseB_rho[:, 3, :]))) \
        if phaseB_rho is not None else float("nan")

    boltz = getattr(c, "_boltz_dm_solver", None)
    eig_count = getattr(boltz, "_expm_fallback_eig_count", 0) if boltz is not None else 0
    kappa_high_count = (getattr(boltz, "_expm_fallback_kappa_high_count", 0)
                        if boltz is not None else 0)

    tol = 1.0e-12
    verdict = "PASS" if max_abs < tol else "FAIL"

    lines = header + [
        f"Runtime: {dt:.1f}s",
        "",
        f"max |rho_ss| at Phase-0 exit (T = T_boltz_start): {max_abs:.3e}",
        f"max |rho_ss| at Phase-B exit (T = T_boltz_end):   {max_abs_phaseB:.3e}",
        f"tolerance: {tol:.1e}",
        f"Fallback dispatch: eigendecomp={eig_count}, kappa_guard_revert={kappa_high_count}",
        "",
        f"{verdict}: Phase-0 leakage test ({'<' if max_abs < tol else '>='}) "
        f"1e-12 in the decoupled-sterile configuration WITH fallback ON.",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    with open(_OUT_TXT, "w") as fh:
        fh.write(text + "\n")
    sys.exit(0 if max_abs < tol else 1)
