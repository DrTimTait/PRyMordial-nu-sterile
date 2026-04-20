"""Analytical 2-level Rabi+damping, match the L from PRyMordial exactly."""
import os, sys, numpy as np
from scipy.linalg import expm
_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT); os.chdir(_WT)
import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True; PRyMini.julia_flag = False; PRyMini.numba_flag = True
PRyMini.verbose_flag = False; PRyMini.general_nu_flag = True; PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True; PRyMini.qke_full_ode_flag = True; PRyMini.qke_ode_etdrk2_flag = True
PRyMini.sterile_flag = True
PRyMini.Dm2_21 = 0.0; PRyMini.Dm2_31 = 0.0; PRyMini.Dm2_41 = 0.93
PRyMini.theta_12 = 0.0; PRyMini.theta_13 = 0.0; PRyMini.theta_23 = 0.0
PRyMini.theta_14 = 0.0; PRyMini.theta_34 = 0.0; PRyMini.delta_CP = 0.0; PRyMini.delta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(1e-4))/2.0
PRyMini.eta0b = 0.0
PRyMini.qke_damping_formula = "mirizzi"  # Stage E.1 pair-specific damping

import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()
rho_all = solver.initial_conditions(10.0, 1.0)
rho_all[:, 0, :] = 0.0; rho_all[:, 2, :] = 0.0

L_list, N_gain, I_total = solver._build_L_list(rho_all, 1.0, 10.0)
H = solver._build_H_list(rho_all, 1.0, 10.0)[0][10]

# Extract 2-level H and damping from PRyMordial
H_aa = H[1, 1]
H_ss = H[3, 3]
H_as = H[1, 3]

# D_pair extracted from the shared helper so any formula change in
# PRyMordial auto-propagates to this analytic reference.
T_eV = 10.0 * 1.0e6
E_eV = 10.5 * 1.0e6
D_off = solver._compute_D_pair_matrix(T_eV, np.array([E_eV]), units="eV")
D_pair = D_off[1, 3, 0]  # (μ, s) pair, single momentum mode, eV

print(f"H_aa-H_ss = {(H_aa-H_ss).real:.3e} eV,  H_as = {H_as:.3e} eV,  D_pair = {D_pair:.3e} eV")
print(f"|H_diff|·dt_nat = {abs(H_aa-H_ss).real*6.077e11:.3e}")
print(f"|H_off|·dt_nat  = {abs(H_as)*6.077e11:.3e}")
print(f"D_pair·dt_nat   = {D_pair*6.077e11:.3e}")
print()

# Build analytical 2-level L WITH damping (matching PRyMordial's construction)
I2 = np.eye(2, dtype=complex)
H2 = np.array([[H_aa, H_as], [np.conj(H_as), H_ss]], dtype=complex)
L_comm = -1j * (np.kron(H2, I2) - np.kron(I2, H2.T))
# Damping on off-diag vec-positions only: vec_2 indices are [0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)]
# Off-diag is indices 1 and 2.
L_damp = np.diag([0.0, -D_pair, -D_pair, 0.0]).astype(complex)
L_2 = L_comm + L_damp

# Initial state: rho_aa = 0.259 (FD at y=10.5, T=10 MeV), everything else 0
rho_aa_init = rho_all[0, 1, 10]
print(f"Initial rho_μμ = {rho_aa_init:.3e}  (from PRyMordial FD)")
print()

vec2 = np.array([rho_aa_init, 0.0, 0.0, 0.0], dtype=complex)
dt_nat = 4.0e-4 * solver._eV_to_secm1

# One-step evolution via expm of analytical 2x2
vec2_new = expm(L_2 * dt_nat) @ vec2
rho2_new = vec2_new.reshape(2, 2)
print(f"Analytical 2-level WITH damping, one step:")
print(f"  ρ_μμ new = {rho2_new[0, 0].real:.3e}")
print(f"  ρ_ss new = {rho2_new[1, 1].real:.3e}  (from 0)")
print(f"  ρ_μs new = {rho2_new[0, 1]:.3e}")
print()

# For comparison, extract PRyMordial's 16-dim L and apply to same initial state
L_16 = L_list[0][10]
vec16 = solver._to_mat(rho_all[0])[10].reshape(16)
vec16_new = expm(L_16 * dt_nat) @ vec16
mat16_new = vec16_new.reshape(4, 4)
print(f"PRyMordial 4x4 L-expm one step:")
print(f"  ρ_μμ new (diag 1) = {mat16_new[1, 1].real:.3e}")
print(f"  ρ_ss new (diag 3) = {mat16_new[3, 3].real:.3e}")
print(f"  ρ_μs new          = {mat16_new[1, 3]:.3e}")
print()

print(f"Comparison of ρ_ss growth:")
print(f"  Analytical 2-level (damped): {rho2_new[1, 1].real:.3e}")
print(f"  PRyMordial 4x4:              {mat16_new[3, 3].real:.3e}")
print(f"  Ratio (PRyMordial / 2-level): {mat16_new[3,3].real/max(abs(rho2_new[1,1].real), 1e-30):.3f}")
