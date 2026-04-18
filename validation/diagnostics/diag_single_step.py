"""Diagnostic: run ONE D.7.1 step at T=10 MeV with sin^2 2theta_24=1e-4,
starting from thermal active / empty sterile. See how much rho_ss is
populated per step.

Expected from standard DW rate: per coherence cycle ~ sin^2(2theta)·(D/omega)^2
ratio. Actual should be very small (~1e-8 per step) per Hannestad.
"""
import os, sys, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
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
PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0

import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()

# Build initial state at T_nu = 10 MeV, a = 1
Tnu = 10.0
a = 1.0
Tg = Tnu
rho_all = solver.initial_conditions(Tnu, a)

# Sanity: check sterile is initially 0
print("Initial state diagonals (sector 0):")
print(f"  nu_e:     total y^2-sum = {np.sum(solver.y_grid**2 * rho_all[0, 0]):.3e}")
print(f"  nu_mu:    total y^2-sum = {np.sum(solver.y_grid**2 * rho_all[0, 1]):.3e}")
print(f"  nu_tau:   total y^2-sum = {np.sum(solver.y_grid**2 * rho_all[0, 2]):.3e}")
print(f"  nu_s:     total y^2-sum = {np.sum(solver.y_grid**2 * rho_all[0, 3]):.3e}   (should be 0)")
print()

# Take ONE step at Phase-B-like dt
# Phase B dt ~ da/(a*H). At T=10 MeV, H ~ 110 s^-1, so dt ~ (a_next/a - 1)/H with da/a ~ 1/n_B.
# For n_B=2400, dt ~ 4e-4 s. But we're diagnostic-running.
dt_s = 4.0e-4  # seconds
phi1_dt = 0.0  # disable diagonal exp-Euler for this diagnostic; we just want to see what the predictor does

print(f"Taking ONE D.7.1 step at Tg=Tnu={Tg} MeV, a={a}, dt={dt_s} s")
print(f"phi1_dt = 0 (diagonal exp-Euler disabled to isolate L-step effect)")
print()

rho_before = rho_all.copy()
solver.evolve_step_ode_etdrk2(rho_all, dt_s, phi1_dt, a, Tg)

# Delta's
drho = rho_all - rho_before
print("After one step -- changes (sector 0):")
print(f"  Delta nu_e    y^2-sum = {np.sum(solver.y_grid**2 * drho[0, 0]):+.3e}")
print(f"  Delta nu_mu   y^2-sum = {np.sum(solver.y_grid**2 * drho[0, 1]):+.3e}")
print(f"  Delta nu_tau  y^2-sum = {np.sum(solver.y_grid**2 * drho[0, 2]):+.3e}")
print(f"  Delta nu_s    y^2-sum = {np.sum(solver.y_grid**2 * drho[0, 3]):+.3e}   (sterile populated per step)")
print()
print(f"After step   (sector 0 nu_s):")
print(f"  max rho_ss = {rho_all[0, 3].max():.3e}")
print(f"  y^2-weighted sum rho_ss = {np.sum(solver.y_grid**2 * rho_all[0, 3]):.3e}")
print()

# Compare to expected DW rate: per-step d(rho_ss)/dt ~ 0.5 sin^2(2theta_m) * Gamma ~ 0.5 * 1e-4 * (2e-12 eV)
# rho_ss_per_step ~ 0.5 * 1e-4 * 2e-12 eV * 4e-4 s * (1e15 eV^-1/s) ~ 4e-4 * 1e-4 * 2e-12 * 1e15 = 8e-5
Gamma_damp_SI_at_10MeV = 2e-12 * 1.52e15  # eV * (eV -> s^-1)
Gamma_DW_est = 0.5 * 1e-4 * Gamma_damp_SI_at_10MeV
drho_ss_expected_per_step = Gamma_DW_est * dt_s
print(f"Expected d(rho_ss)/step from 0.5 * sin^2(2theta) * Gamma_damp * dt:")
print(f"  Gamma_damp ~ 2e-12 eV ~ {Gamma_damp_SI_at_10MeV:.2e} /s")
print(f"  Gamma_DW  ~ 0.5 * 1e-4 * Gamma_damp ~ {Gamma_DW_est:.2e} /s")
print(f"  drho_ss_per_step ~ Gamma_DW * dt ~ {drho_ss_expected_per_step:.3e}")
