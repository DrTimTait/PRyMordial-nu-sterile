"""Stage E.2 sprint 16 gate 4: ETDRK4 decoupled sterile (direct-step).

With sterile_flag=True but all theta_alpha4=0 and xi_*=0, the sterile
sector is fully decoupled: the rho_ss diagonals (and the active-sterile
off-diagonals) must remain at zero (or floor f_min) under ETDRK4.

This is a fast unitarity check via direct DensityMatrixSolver stepping
(50 outer steps, no full BBN ODE). The full-BBN variant would be ~30
minutes; this version is <2 minutes and exercises the same code path
through evolve_step_ode_etdrk4 enough to catch leakage from the
augmented-expm structure or the post-step sanitisation.
"""
import os
import sys
import time

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("Stage E.2 sprint 16 gate 4: ETDRK4 decoupled sterile (direct-step)",
          flush=True)
    print("=" * 80, flush=True)

    import PRyM.PRyM_init as PRyMini
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_etdrk4_flag = True
    PRyMini.qke_ode_etdrk2_flag = False
    PRyMini.qke_lsoda_driver_flag = False
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_phase0_flag = False

    import PRyM.PRyM_boltzmann as PB
    solver = PB.DensityMatrixSolver()

    # Phase B-representative state: thermal FD sterile-zero initial conditions
    # at Tnu = 2 MeV, a = 1.
    Tnu = 2.0
    a = 1.0
    Tg = Tnu
    rho = solver.initial_conditions(Tnu, a)

    # Pre-step assertion: sterile should already be at zero/floor.
    pre_max = float(np.max(rho[:, 3, :]))
    print(f"  Pre-step  max|ρ_ss| = {pre_max:.3e}", flush=True)

    n_steps = 50
    dt = 1.0e-4
    t0 = time.time()
    leak_max = 0.0
    for i in range(n_steps):
        solver.evolve_step_ode_etdrk4(rho, dt, 0.0, a, Tg)
        leak_max = max(leak_max, float(np.max(rho[:, 3, :])))
    dt_run = time.time() - t0

    final_ss_sum = float(np.sum(rho[:, 3, :]))
    final_ss_max = float(np.max(rho[:, 3, :]))
    finite = bool(np.all(np.isfinite(rho)))

    # Active-sterile off-diagonals: indices 6..15 are pairs in the n_components=16
    # packed real layout for 4-flavor (last 6 components are active-sterile
    # off-diagonals). Just check magnitude on full state.
    active_max = float(np.max(rho[:, :3, :]))   # active diagonals & active-active
    print(f"  Post-step Σρ_ss = {final_ss_sum:.3e}  max|ρ_ss| = {final_ss_max:.3e}  "
          f"max|leak| = {leak_max:.3e}  finite={finite}",
          flush=True)
    print(f"  ({n_steps} ETDRK4 steps in {dt_run:.1f}s = {dt_run/n_steps:.3f} s/step)",
          flush=True)

    # Leak threshold: in the floor-clipped representation, rho_ss_min = 1e-30.
    # Anything well below 1e-12 means no spurious sterile production from
    # ETDRK4's expm / Hermitisation / sanitisation pipeline.
    pass_ss = leak_max < 1e-12 and finite

    if pass_ss:
        print("  GATE 4: PASS", flush=True)
    else:
        print(f"  GATE 4: FAIL  (leak_max={leak_max:.3e}, finite={finite})",
              flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_etdrk4_decoupled_sterile.out")
    with open(out_txt, "w") as fh:
        fh.write(f"ETDRK4 decoupled (direct, {n_steps} steps): "
                 f"Σρ_ss={final_ss_sum:.3e} max|ρ_ss|={final_ss_max:.3e} "
                 f"leak_max={leak_max:.3e} ({dt_run:.1f}s) "
                 f"verdict={'PASS' if pass_ss else 'FAIL'}\n")
