"""Measure the clamp scale on the (nu_mu, nu_s) off-diagonal during one
D.7.1 step at sin^2(2theta)=1e-4, T=10 MeV. If the scale << 1, the
clamp is doing heavy lifting (unphysical linear off-diag being cut)."""
import os, sys, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT); os.chdir(_WT)
import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True; PRyMini.julia_flag = False; PRyMini.numba_flag = True
PRyMini.verbose_flag = False; PRyMini.general_nu_flag = True; PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True; PRyMini.qke_full_ode_flag = True; PRyMini.qke_ode_etdrk2_flag = True
PRyMini.sterile_flag = True; PRyMini.Dm2_41 = 0.93
PRyMini.theta_12 = 0.0; PRyMini.theta_13 = 0.0; PRyMini.theta_23 = 0.0
PRyMini.theta_14 = 0.0; PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4))/2.0; PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0

import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()
Tnu, a = 10.0, 1.0
rho_all = solver.initial_conditions(Tnu, a)

# Instrument evolve_step_ode_etdrk2 by calling internal pieces manually
L_list, N_gain_n, I_total_n = solver._build_L_list(rho_all, a, Tnu)
dt_s = 4.0e-4
dt_nat = dt_s * solver._eV_to_secm1

# Compute Phi_cache
Phi_mid = [solver._etdrk2_expm_phi(L_list[s], dt_nat) for s in (0, 1)]

# Run one predictor (off-diag) without clamp
N = solver.n_flavor; Ny = solver.Ny
s = 0
Phi0, Phi1, _ = Phi_mid[s]
rho_mat = solver._to_mat(rho_all[s])
rho_vec = rho_mat.reshape(Ny, N*N)
N_off_vec = N_gain_n[s].reshape(Ny, N*N).copy()
for alpha in range(N):
    N_off_vec[:, alpha*N + alpha] = 0.0
rho_star_vec = np.einsum('ijk,ik->ij', Phi0, rho_vec) + np.einsum('ijk,ik->ij', Phi1, N_off_vec)
rho_star_mat = rho_star_vec.reshape(Ny, N, N)

# Check off-diag magnitude PRE-Hermitisation, PRE-clamp
off_diag_13 = rho_star_mat[:, 1, 3]
diag_1 = rho_star_mat[:, 1, 1].real
diag_3 = rho_star_mat[:, 3, 3].real
phys_bound = np.sqrt(np.abs(diag_1 * diag_3))

print(f"After ONE predictor step (nu_mu - nu_s off-diag, sin^2 2theta=1e-4, T=10 MeV):")
print(f"{'i':>3} {'y':>6} {'|rho_13|':>14} {'sqrt(r11*r33)':>14} {'ratio':>9} {'rho_33':>12}")
for i in [0, 5, 10, 20, 40, 60, 80]:
    r13 = abs(off_diag_13[i])
    bound = phys_bound[i]
    ratio = r13/max(bound, 1e-30)
    print(f"{i:>3} {solver.y_grid[i]:>6.2f} {r13:>14.3e} {bound:>14.3e} {ratio:>9.3e} {diag_3[i]:>12.3e}")

print()
print(f"Max ratio |rho_off|/sqrt(rho_aa*rho_bb): {np.max(np.abs(off_diag_13)/np.maximum(phys_bound, 1e-30)):.3e}")
print(f"(Physical bound is ratio <= 1; large ratios mean the linear predictor overshoots.)")
