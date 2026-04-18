"""Diagnostic: dump the in-medium Hamiltonian for the small-mixing DW scenario
at representative Phase-B temperatures. Check:

  1. Magnitude of V_NC vs omega = Delta_m^2 / 2E.
  2. Whether H_active - H_sterile changes sign during the evolution (spurious
     NH resonance?).
  3. In-medium mixing angle theta_m vs vacuum angle.
  4. Gamma_damp magnitude and dimensionless ratio to Hannestad's C_a.

No ODE evolution -- just construct H at fixed (T, a) and print.
"""
import os, sys, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False
PRyMini.compute_nTOp_flag = False
PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True
PRyMini.sterile_flag = True
PRyMini.Dm2_41 = 0.93
sin2_2theta = 1.0e-4
PRyMini.theta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0
PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0

import PRyM.PRyM_boltzmann as PRyMboltz
solver = PRyMboltz.DensityMatrixSolver()
rho_all = solver.initial_conditions(2.0, 1.0)  # arbitrary ICs; just need the shape

N = solver.n_flavor
print(f"n_flavor = {N}, Ny = {solver.Ny}")
print(f"theta_24 = {PRyMini.theta_24:.5e} rad  sin^2(2*theta_24) = {np.sin(2*PRyMini.theta_24)**2:.3e}")
print(f"Dm2_41 = {PRyMini.Dm2_41} eV^2")
print()

# PMNS mass-eigenvalue Omega matrix (same for all temperatures)
print("Omega_nu (in flavor basis, units of eV^2 so divide by 2E [eV] to get eV):")
print(np.abs(solver._Omega_nu))
print()
print("|Omega_nu[1,3]| (nu_mu-nu_s off-diagonal): ", abs(solver._Omega_nu[1,3]))
print("Expected for pure theta_24: (1/2)*sin(2*theta_24)*Dm2_41/2 =",
      0.5 * np.sin(2*PRyMini.theta_24) * PRyMini.Dm2_41 / 2)
print()

# Evaluate H at a few representative T during Phase B.
# In Phase B, a = a(T) increases as T decreases (adiabatic), roughly a*T = const_comoving.
# For this diagnostic we just pick some (T, a) pairs covering the range.
test_points = [
    ("start of Phase B",  30.0, 1.0),
    ("Hannestad T_max ~ 10 MeV",  10.0, 3.0),
    ("mid Phase B",              5.0, 6.0),
    ("end of Phase B",           2.0, 15.0),
    ("far below T_max",          1.0, 30.0),
]

# Pick a representative y-mode (y ~ 3 is the peak of the FD distribution)
i_y = np.argmin(np.abs(solver.y_grid - 3.0))
y_peak = solver.y_grid[i_y]
print(f"Probing at y-mode {i_y} (y = {y_peak:.2f}, near FD peak)")
print()

hdr = f"{'label':<35s} {'T [MeV]':>10s} {'E [eV]':>12s} {'omega [eV]':>12s} {'V_NC [eV]':>12s} {'V_therm [eV]':>14s} {'V_NC/omega':>11s}  {'|H_mu_s| [eV]':>14s}  {'H_mu-H_s [eV]':>15s}  {'2theta_m':>11s}"
print(hdr)
print("-" * len(hdr))

for label, Tg, a in test_points:
    H_list = solver._build_H_list(rho_all, a, Tg)
    H_nu = H_list[0]  # sector 0 = nu

    E_eV = y_peak / a * 1.0e6
    omega = PRyMini.Dm2_41 / (2.0 * E_eV)  # vacuum Delta_m^2/2E

    # Our V_NC and V_thermal at this T
    rho_e_th = 7.0 * np.pi**2 / 60.0 * Tg**4
    V_pref = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e_th / 3.0
    V_NC = V_pref / PRyMini.mZ**2 * E_eV
    V_therm = V_pref / solver.mW2 * E_eV

    H_mm = H_nu[i_y, 1, 1].real  # nu_mu diagonal
    H_ss = H_nu[i_y, 3, 3].real  # nu_s diagonal
    H_ms = H_nu[i_y, 1, 3]        # nu_mu-nu_s off-diag (complex in general)

    dH = H_mm - H_ss
    tan_2theta_m = 2.0 * abs(H_ms) / abs(dH) if abs(dH) > 0 else np.inf
    angle_2tm = np.arctan(tan_2theta_m) * 180 / np.pi  # degrees

    print(f"{label:<35s} {Tg:>10.2f} {E_eV:>12.3e} {omega:>12.3e} {V_NC:>12.3e} {V_therm:>14.3e} {V_NC/omega:>11.3e}  {abs(H_ms):>14.3e}  {dH:>+15.3e}  {angle_2tm:>11.3f}")

print()
print("Damping coefficient check:")
print(f"  Our C_D = {solver.C_D}")
print(f"  Hannestad Eq 2.16: C_e = 1.27, C_mu = C_tau = 0.92")
print()

# Effective damping for nu_mu (active) and active-sterile (av of Gamma_mu + Gamma_s = Gamma_mu/2)
for label, Tg, a in test_points:
    E_eV = y_peak / a * 1.0e6
    GF_eV = PRyMini.GF * 1.0e-12
    T_eV = Tg * 1.0e6
    Gamma_mu_1s = solver.C_D[1] * GF_eV**2 * T_eV**4 * E_eV * solver._eV_to_secm1   # 1/s
    Gamma_mu_eV = Gamma_mu_1s / solver._eV_to_secm1  # eV
    D_pair_mus = 0.5 * Gamma_mu_eV  # nu_mu-nu_s damping rate
    # Hubble
    # H = 1.66 sqrt(g*) T^2 / M_Pl; at T~MeV, g* ~ 10.66
    Hubble_1s = 1.66 * np.sqrt(10.66) * (Tg * 1.0e-3)**2 / 1.22e19 * 1.0e9   # 1/s approximate
    Hubble_eV = Hubble_1s * 6.58e-16
    print(f"  T={Tg:5.2f}MeV  Gamma_mu={Gamma_mu_eV:.2e}eV={Gamma_mu_1s:.2e}/s  D_pair_mus={D_pair_mus:.2e}eV  Hubble={Hubble_eV:.2e}eV  Gamma/H={Gamma_mu_eV/Hubble_eV:.2e}")
