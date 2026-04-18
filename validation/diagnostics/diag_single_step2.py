"""Second diagnostic: mode-by-mode view of per-step sterile population."""
import os, sys, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT); os.chdir(_WT)
import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True; PRyMini.julia_flag = False; PRyMini.numba_flag = True
PRyMini.verbose_flag = False; PRyMini.general_nu_flag = True; PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True; PRyMini.qke_full_ode_flag = True; PRyMini.qke_ode_etdrk2_flag = True
PRyMini.sterile_flag = True; PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0; PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4))/2.0; PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0
import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()
Tnu, a = 10.0, 1.0
rho_all = solver.initial_conditions(Tnu, a)

# N_off magnitude per mode (sanity)
L_list, N_gain, I_total = solver._build_L_list(rho_all, a, Tnu)
print("N_gain[sector=0] off-diagonal magnitudes per mode (select y values):")
for y_pick in [0.5, 2.5, 5.0, 10.0]:
    i = np.argmin(np.abs(solver.y_grid - y_pick))
    print(f"  y={solver.y_grid[i]:5.2f}: |N[1,3]|={abs(N_gain[0][i, 1, 3]):.3e}   |N[0,1]|={abs(N_gain[0][i, 0, 1]):.3e}   |N[1,1]|={N_gain[0][i, 1, 1].real:.3e}")
print()

dt_s = 4.0e-4
rho_before = rho_all.copy()
solver.evolve_step_ode_etdrk2(rho_all, dt_s, 0.0, a, Tnu)
drho_ss = rho_all[0, 3] - rho_before[0, 3]
drho_mu = rho_all[0, 1] - rho_before[0, 1]
drho_tau = rho_all[0, 2] - rho_before[0, 2]

print(f"Per-mode sterile change after ONE step (sector 0, Tnu=10 MeV, dt=4e-4 s, sin²2θ_24=1e-4):")
print(f"{'i':>3} {'y':>6} {'rho_ss_new':>14} {'drho_ss':>14} {'drho_mu':>14} {'drho_tau':>14}")
for i in range(solver.Ny):
    if i % 5 == 0 or abs(drho_ss[i]) > 0.01:
        print(f"{i:>3} {solver.y_grid[i]:>6.3f} {rho_all[0,3,i]:>14.3e} {drho_ss[i]:>+14.3e} {drho_mu[i]:>+14.3e} {drho_tau[i]:>+14.3e}")

print()
print(f"Total diagonals (sector 0):")
for a_i in range(4):
    print(f"  alpha={a_i}: y^2-sum = {np.sum(solver.y_grid**2 * rho_all[0, a_i]):.3e}   (was {np.sum(solver.y_grid**2 * rho_before[0, a_i]):.3e})")
