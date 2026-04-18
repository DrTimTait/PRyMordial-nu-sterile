"""Minimal 2-level DW isolation test.

Compare PRyMordial's L-matrix exponentiation to an analytic 2-level
(nu_a + nu_s) Rabi+damping solution, over ONE step at fixed (T, a).

If they agree: the L-expm machinery is correct; the bug is elsewhere
(collision kernel, diagonal exp-Euler, y-integration).

If they disagree: the bug is in L assembly or expm application.

Setup:
  - Set Delta_m^2_21 = Delta_m^2_31 = 0 (no active-active vacuum mixing).
  - Set theta_12 = theta_13 = theta_23 = theta_14 = theta_34 = 0.
  - Only theta_24 non-zero (so nu_mu - nu_s is the only active mixing).
  - PRyMordial's 4x4 L should then reduce to a 2x2 block for (nu_mu, nu_s)
    plus decoupled (trivial) blocks for nu_e, nu_tau.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm

_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini

# MINIMAL CONFIG: zero out everything that isn't (nu_mu, nu_s) mixing
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True
PRyMini.qke_full_ode_flag = True
PRyMini.qke_ode_etdrk2_flag = True
PRyMini.sterile_flag = True

PRyMini.Dm2_21 = 0.0
PRyMini.Dm2_31 = 0.0
PRyMini.Dm2_41 = 0.93

PRyMini.theta_12 = 0.0
PRyMini.theta_13 = 0.0
PRyMini.theta_23 = 0.0
PRyMini.theta_14 = 0.0
PRyMini.theta_34 = 0.0
PRyMini.delta_CP = 0.0
PRyMini.delta_14 = 0.0

sin2_2theta = 1.0e-4
theta_24 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0
PRyMini.theta_24 = theta_24

PRyMini.xi_nue_init = 0.0
PRyMini.xi_numu_init = 0.0
PRyMini.xi_nutau_init = 0.0
PRyMini.eta0b = 0.0  # no baryon asymmetry for this diagnostic

# Build solver and a test state
import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()

print(f"theta_24 = {theta_24:.3e} rad,  sin^2(2*theta) = {np.sin(2*theta_24)**2:.3e}")
print(f"Dm2 = [0, 0, 0, {PRyMini.Dm2_41}]   (only nu_mu - nu_s mixing)")
print(f"U_PMNS = \n{solver.U_PMNS}")
print()

Tnu = 10.0   # MeV
a = 1.0
Tg = Tnu

# PRyMordial state: diag = (0, FD, 0, 0), meaning: nu_e empty, nu_mu thermal, nu_tau empty, nu_s empty
# Use initial_conditions to get thermal FD then manually zero other flavors
rho_all = solver.initial_conditions(Tnu, a)
# Zero out nu_e, nu_tau, nu_ebar, nu_taubar to keep this strictly 2-level (nu_mu, nu_s):
rho_all[:, 0, :] = 0.0   # diag 0 = nu_e
rho_all[:, 2, :] = 0.0   # diag 2 = nu_tau
# Keep rho_all[:, 1, :] = FD (nu_mu thermal) and rho_all[:, 3, :] = 0 (nu_s empty)

print(f"Initial state (y-mode i=10, y={solver.y_grid[10]:.3f}):")
print(f"  rho_nu (sector 0):  diag = [{rho_all[0,0,10]:.3e}, {rho_all[0,1,10]:.3e}, {rho_all[0,2,10]:.3e}, {rho_all[0,3,10]:.3e}]")
print()

# Extract PRyMordial's L matrix for this state
L_list, N_gain, I_total = solver._build_L_list(rho_all, a, Tg)
L_nu = L_list[0]  # sector 0, shape (Ny, 16, 16)

# Pick one representative mode
i_mode = 10
y = solver.y_grid[i_mode]
E_eV = y / a * 1.0e6
print(f"Focus mode: i={i_mode}, y={y:.3f}, E = {E_eV:.3e} eV")

# Extract the (nu_mu, nu_s) = (index 1, index 3) block from the 4x4 Hamiltonian
H_4x4 = solver._build_H_list(rho_all, a, Tg)[0][i_mode]  # sector 0
print(f"Full 4x4 Hamiltonian (eV) at this mode:")
print(f"{H_4x4}")
print()

# Extract 2-level block
H2 = np.array([[H_4x4[1, 1], H_4x4[1, 3]], [H_4x4[3, 1], H_4x4[3, 3]]], dtype=complex)
print(f"Extracted 2-level H (eV):")
print(f"  H_mu_mu  = {H2[0,0].real:.3e}    H_mu_s  = {H2[0,1]:.3e}")
print(f"  H_s_mu   = {H2[1,0]:.3e}    H_s_s   = {H2[1,1].real:.3e}")
print(f"  |H_diff| = H_mu - H_s = {(H2[0,0] - H2[1,1]).real:.3e} eV")
print(f"  |H_off|  = {abs(H2[0,1]):.3e} eV")
print()

# Expected from analytics:
expected_H_diff = -PRyMini.Dm2_41 / (2.0 * E_eV) * np.cos(2*theta_24)
expected_H_off = +PRyMini.Dm2_41 / (2.0 * E_eV) * np.sin(2*theta_24) / 2.0
# (factor 1/2 because Omega = U·diag(m²/2)·U† so off-diag is (m²/2)·sin·cos = (1/4)·sin(2θ)·Δm²)
print(f"Analytic expected for 2-level with only theta_24, Dm2=0.93:")
print(f"  H_diff_exp ≈ -Dm2·cos(2θ)/(2E) = {expected_H_diff:.3e} eV")
print(f"  H_off_exp  ≈ +Dm2·sin(2θ)/(4E) = {expected_H_off:.3e} eV")
print()

# Now compute one step of evolution via PRyMordial L exp and via analytical 2-level
dt_s = 4.0e-4
dt_nat = dt_s * solver._eV_to_secm1
print(f"dt = {dt_s} s = {dt_nat:.3e} 1/eV")
print(f"|H_off|·dt_nat = {abs(H2[0,1])*dt_nat:.3e}    |H_diff|·dt_nat = {abs(H2[0,0]-H2[1,1])*dt_nat:.3e}")

# Build 4x4-space 16×16 L matrix at this mode, and apply expm
N = 4
vec_rho = solver._to_mat(rho_all[0])[i_mode].reshape(N*N)
L_mode = L_nu[i_mode]
expL_dt = expm(L_mode * dt_nat)
vec_rho_new = expL_dt @ vec_rho
rho_new_mat = vec_rho_new.reshape(N, N)
print(f"PRyMordial 4x4 L-expm one-step:")
print(f"  rho_nu_s new (diag 3) = {rho_new_mat[3, 3].real:.3e}  (was 0)")
print(f"  rho_nu_mu new (diag 1) = {rho_new_mat[1, 1].real:.3e}  (was {rho_all[0,1,i_mode]:.3e})")
print(f"  rho_mu_s  off  = {rho_new_mat[1, 3]:.3e}  (was 0)")
print()

# Analytical 2-level expm
# L_2 in vec(2x2) space (row-major):
#   L = -i*(H ⊗ I - I ⊗ H^T)  (no damping for this minimal test)
I2 = np.eye(2, dtype=complex)
L2 = -1j * (np.kron(H2, I2) - np.kron(I2, H2.T))
vec_rho2 = np.array([rho_all[0,1,i_mode], 0.0, 0.0, 0.0], dtype=complex)
# Actually vec(rho_2x2) with row-major where rho_2x2 = [[rho_mu_mu, rho_mu_s], [rho_s_mu, rho_s_s]]:
#   vec[0] = rho_mu_mu (was FD), vec[1] = rho_mu_s (was 0), vec[2] = rho_s_mu (was 0), vec[3] = rho_s_s (was 0)
vec_rho2[0] = rho_all[0,1,i_mode]
expL2_dt = expm(L2 * dt_nat)
vec_rho2_new = expL2_dt @ vec_rho2
rho2_new = vec_rho2_new.reshape(2, 2)
print(f"Analytical 2-level L-expm one-step (no damping):")
print(f"  rho_nu_s new  = {rho2_new[1, 1].real:.3e}")
print(f"  rho_nu_mu new = {rho2_new[0, 0].real:.3e}")
print(f"  rho_mu_s  off = {rho2_new[0, 1]:.3e}")
print()

print(f"Comparison:")
print(f"  d(rho_s_s) PRyMordial: {rho_new_mat[3, 3].real:.3e}")
print(f"  d(rho_s_s) Analytical: {rho2_new[1, 1].real:.3e}")
print(f"  Ratio: {rho_new_mat[3,3].real / max(abs(rho2_new[1,1].real), 1e-30):.3e}")
