"""Inspect PRyMordial's L matrix elements relevant to nu_mu-nu_s coupling."""
import os, sys, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
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

import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()
rho_all = solver.initial_conditions(10.0, 1.0)
rho_all[:, 0, :] = 0.0; rho_all[:, 2, :] = 0.0

L_list, _, _ = solver._build_L_list(rho_all, 1.0, 10.0)
L = L_list[0][10]   # sector 0, mode i=10
H = solver._build_H_list(rho_all, 1.0, 10.0)[0][10]

print("H (4x4, eV):")
print(H.real)
print()
print(f"H[1,3] = {H[1,3]} (expected +H_as)")
print(f"H[3,1] = {H[3,1]} (expected conj(H_as))")
print()

# For 4x4 vec (row-major): vec index α*4+β corresponds to ρ[α, β].
# Diagonals at indices 0, 5, 10, 15. ρ_μs at 1*4+3=7. ρ_sμ at 3*4+1=13.
indices = {
    'ρ_ee': 0, 'ρ_μμ': 5, 'ρ_ττ': 10, 'ρ_ss': 15,
    'ρ_eμ': 1, 'ρ_eτ': 2, 'ρ_es': 3,
    'ρ_μe': 4, 'ρ_μτ': 6, 'ρ_μs': 7,
    'ρ_τe': 8, 'ρ_τμ': 9, 'ρ_τs': 11,
    'ρ_se': 12, 'ρ_sμ': 13, 'ρ_sτ': 14,
}

# Print L[row, col] for rows of interest: ρ_μμ, ρ_ss, ρ_μs, ρ_sμ diagonals and couplings
rows_of_interest = ['ρ_μμ', 'ρ_ss', 'ρ_μs', 'ρ_sμ']
cols_of_interest = ['ρ_ee', 'ρ_μμ', 'ρ_ττ', 'ρ_ss', 'ρ_μs', 'ρ_sμ']

print("L matrix elements (only the ones relevant to μ↔s coupling):")
print(f"{'row':>10s}  ", end='')
for c in cols_of_interest:
    print(f"{c:>16s}", end='')
print()
for r in rows_of_interest:
    print(f"{r:>10s}  ", end='')
    for c in cols_of_interest:
        val = L[indices[r], indices[c]]
        if abs(val) < 1e-18:
            print(f"{'  0':>16s}", end='')
        else:
            print(f"{val:>16.3e}", end='')
    print()

print()
print("Analytic expectation (from 2-level commutator):")
print(f"  L[ρ_μs, ρ_μμ] should be +i·H[1,3] = {1j*H[1,3]:.3e}")
print(f"  L[ρ_μs, ρ_ss] should be -i·H[1,3] = {-1j*H[1,3]:.3e}")
print(f"  L[ρ_sμ, ρ_μμ] should be -i·H[3,1] = {-1j*H[3,1]:.3e}")
print(f"  L[ρ_sμ, ρ_ss] should be +i·H[3,1] = {1j*H[3,1]:.3e}")
print(f"  L[ρ_ss, ρ_μs] should be -i·H[3,1] (from Hbar·ρ - ρ·H)")
print(f"  L[ρ_ss, ρ_sμ] should be +i·H[1,3]")
print(f"  L[ρ_μμ, ρ_μs] should be +i·H[3,1]")
print(f"  L[ρ_μμ, ρ_sμ] should be -i·H[1,3]")
