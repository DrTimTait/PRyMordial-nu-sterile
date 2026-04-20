"""Stage E.2 sprint-2 (scope b): 2-level damped expm at high T (Suspect 3).

doc/STAGE_E2_SPRINT2_BRIEF.md Suspect 3: "ETDRK2 expm regime at
D*dt >> 1". At T_boltz_start = 60 MeV and n_B = 2000, the first
Phase B step has very large |L*dt|. `_etdrk2_expm_phi` computes
Phi0 = exp(L*dt), Phi1 = dt*phi_1(L*dt), Phi2 = dt^2*phi_2(L*dt)
via an augmented 3N^2x3N^2 matrix exponential (Al-Mohy-Higham 2011).
Phi0 is straight scipy.linalg.expm, so the accuracy test is
whether the *augmented* construction agrees with direct expm at
|L*dt| >> 1 (augmentation numerics can degrade at extreme
arguments).

This script:
  (1) Builds a 4x4 sector-resolved L at T=60 MeV with Mirizzi damping
      via DensityMatrixSolver._build_L_list at a single mode.
  (2) Estimates the first Phase B dt at T_boltz_start=60 MeV,
      n_B=2000 from da/a = (log10(a_end/a_ini) / n_B) * ln(10)
      and dt = (da/a) / H. Use the standard radiation-era Hubble at 60 MeV.
  (3) Calls `_etdrk2_expm_phi` at this dt.
  (4) Computes the direct expm(L*dt_nat) for comparison.
  (5) Reports elementwise max |Phi0_aug - expm_direct| and the
      magnitudes of Phi1, Phi2 times a representative collision N
      block.

If Phi0_aug == expm_direct to machine precision AND Phi1/Phi2 are
finite (no overflow), Suspect 3 is ruled out.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini

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
PRyMini.theta_24 = np.arcsin(np.sqrt(1e-4)) / 2.0
PRyMini.eta0b = 0.0
PRyMini.qke_damping_formula = "mirizzi"

import PRyM.PRyM_boltzmann as PRyMboltz

solver = PRyMboltz.DensityMatrixSolver()

# T = 60 MeV, a = 1.0 (normalisation only; doesn't enter |L*dt| directly).
T_MeV = 60.0
a = 1.0
rho_all = solver.initial_conditions(T_MeV, a)

# Phase B first-step dt at n_B=2000, T_boltz_start=60 MeV, T_boltz_end=0.005 MeV:
# a_ini ~ 1/60 vs a_end ~ 200 (entropy-based), log-uniform grid in a.
decades_a = np.log10((200.0 * 1.02) / (1.0 / 60.0))  # ~4.08
n_B = 2000
dlna_per_step = np.log(10.0) * decades_a / n_B
# Hubble in s^-1 at T=60 MeV using PRyMordial's thermo.
import PRyM.PRyM_thermo as PRyMthermo
rho_rad = (PRyMthermo.rho_g(T_MeV) + PRyMthermo.rho_e(T_MeV)
           + 3.0 * PRyMthermo.rho_nu(T_MeV))
H_secm1 = PRyMini.MeV_to_secm1 * np.sqrt(8. * np.pi * rho_rad / (3. * PRyMini.Mpl**2))
dt_first_sec = dlna_per_step / H_secm1
dt_nat = dt_first_sec * solver._eV_to_secm1

print("=" * 88)
print("Stage E.2 sprint-2 (scope b): 2-level damped expm at T=60 MeV (Suspect 3)")
print("=" * 88)
print()
print(f"T                   = {T_MeV} MeV")
print(f"Phase B window      = 60 MeV -> 0.005 MeV ({decades_a:.2f} decades in a)")
print(f"n_B                 = {n_B}")
print(f"d ln a / step       = {dlna_per_step:.4e}")
print(f"Hubble(60 MeV)      = {H_secm1:.3e}  s^-1")
print(f"dt (first step)     = {dt_first_sec:.3e}  s  = {dt_nat:.3e}  1/eV")
print()

# Build L at the full 4x4 structure. _build_L_list returns L for each sector
# (nu, nubar) with shape (Ny, N^2, N^2). Pick a mid-range momentum mode.
L_list, N_gain, I_total = solver._build_L_list(rho_all, a, T_MeV)

mode_idx = solver.Ny // 2  # ~y=5 in a 0-10 grid
E_eV = solver.y_grid[mode_idx] * T_MeV * 1.0e6  # MeV -> eV
print(f"Chosen mode: y_idx={mode_idx}, y={solver.y_grid[mode_idx]:.3f}, E={E_eV:.2e} eV")
print()

L_4x4 = L_list[0][mode_idx]  # nu sector, single mode, shape (N^2, N^2) with N=4
N_sq = L_4x4.shape[0]
print(f"L shape         = {L_4x4.shape}  (N^2 = {N_sq}, so N = {int(np.sqrt(N_sq))})")
print(f"||L||_inf       = {np.max(np.abs(L_4x4)):.3e} eV")
print(f"||L*dt||_inf    = {np.max(np.abs(L_4x4)) * dt_nat:.3e}")
print()

# PRyMordial's augmented expm via _etdrk2_expm_phi.
L_single = L_4x4[np.newaxis, :, :]  # (1, N^2, N^2)
Phi0_aug, Phi1_aug, Phi2_aug = solver._etdrk2_expm_phi(L_single, dt_nat)
Phi0_aug = Phi0_aug[0]
Phi1_aug = Phi1_aug[0]
Phi2_aug = Phi2_aug[0]

# Direct expm of L*dt.
Phi0_direct = expm(L_4x4 * dt_nat)

# Comparison.
max_abs = np.max(np.abs(Phi0_direct))
diff = np.max(np.abs(Phi0_aug - Phi0_direct))
rel = diff / max(max_abs, 1e-300)

print("Phi0 comparison (augmented vs direct expm):")
print(f"  max |Phi0_aug|        = {np.max(np.abs(Phi0_aug)):.3e}")
print(f"  max |Phi0_direct|     = {max_abs:.3e}")
print(f"  max elementwise diff  = {diff:.3e}")
print(f"  relative              = {rel:.3e}")
print()

print("Phi1, Phi2 finite-value check:")
print(f"  max |Phi1|            = {np.max(np.abs(Phi1_aug)):.3e}")
print(f"  max |Phi2|            = {np.max(np.abs(Phi2_aug)):.3e}")
print(f"  any NaN in Phi1/Phi2  = {np.any(np.isnan(Phi1_aug)) or np.any(np.isnan(Phi2_aug))}")
print(f"  any Inf in Phi1/Phi2  = {np.any(np.isinf(Phi1_aug)) or np.any(np.isinf(Phi2_aug))}")
print()

# Asymptotic check for extreme |L*dt|:
# For a pure-damped eigenvalue lambda = -D (real, negative, |D*dt| >> 1):
#   phi_0(lambda*dt) = exp(lambda*dt) -> 0
#   phi_1(lambda*dt) = (exp-1)/lambda_dt -> -1/(lambda*dt) (small)
#   phi_2(lambda*dt) = (phi_1 - 1)/lambda_dt -> -1/(lambda*dt)^2
# Phi1 return value is dt*phi_1 -> -1/lambda (order ~ 1/D).
# Phi2 return value is dt*phi_2 -> -1/(lambda^2 * dt) (order ~ 1/(D^2 dt)).
L_eigs = np.linalg.eigvals(L_4x4 * dt_nat)
print(f"L*dt eigenvalues (first 8):")
for i, ev in enumerate(L_eigs[:8]):
    print(f"  lambda_{i}*dt = {ev.real:+.3e} + {ev.imag:+.3e}j, "
          f"|...| = {abs(ev):.3e}")
print()

print("=" * 88)
print("Suspect 3 decision:")
print("  Phi0_aug ~= Phi0_direct (rel < 1e-12)  AND Phi1/Phi2 finite")
print("    -> augmented expm is innocent; Suspect 3 ruled out.")
print("  Otherwise: augmented construction degrades at |L*dt| >> 1;")
print("    need a reformulation (scaling-and-squaring patch).")
print("=" * 88)
