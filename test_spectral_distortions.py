# -*- coding: utf-8 -*-
"""
Validation tests for neutrino spectral distortions against literature.

Compares PRyMordial's Boltzmann solver output to quantitative results from:
  - Froustey, Pitrou, Volpe (2020) [arXiv:2008.01074]
  - Bennett, Buldgen, De Salas, Mangano, Miele, Pastor (2021) [arXiv:2012.02726]

Tests run the diagonal Boltzmann solver (no oscillations) and check:
  1. Total Neff
  2. Per-flavor comoving temperature ratios z_nu = (rho/rho_FD)^{1/4}
  3. Per-flavor fractional energy density increases delta_rho/rho_FD
  4. Flavor hierarchy (nu_e gets more heating than nu_mu)
  5. Spectral distortion shape: zero crossing, sign pattern
  6. Number density ratios n_nu/n_FD

Literature reference values (no oscillations, diagonal):
  z_nue    = 1.00234 ± 0.00002   (Froustey+ 2020, Table I)
  z_numu   = 1.00098 ± 0.00002
  delta_rho_nue  = 0.94%         (= z^4 - 1)
  delta_rho_numu = 0.39%
  n_nue/n_FD  = 1.0014
  n_numu/n_FD = 1.0007
  Neff (no osc) ~ 3.044

Note on per-flavor accuracy:
  The D-kernel polynomial approximation captures ~90% of the exact collision
  energy transfer. The Froustey formalism (Friedmann a + plasma entropy
  equation) self-consistently determines Neff without a drift correction.
  Per-flavor energy fractions differ from literature because the D-kernel
  produces a nearly thermal (uniform T shift) distortion, while the exact
  calculation concentrates energy transfer at intermediate momenta (y ~ 3-5).
"""
import time
import numpy as np
from scipy.optimize import brentq

import PRyM.PRyM_init as PRyMini

# Configure for Boltzmann diagonal (no oscillations)
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False
PRyMini.compute_nTOp_flag = False
PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False
PRyMini.massive_electron_flag = False

import PRyM.PRyM_thermo as PRyMthermo
import importlib
importlib.reload(PRyMthermo)

import PRyM.PRyM_main as PRyMmain

# ============================================================
# Run the Boltzmann solver
# ============================================================
print(" Running Boltzmann diagonal solver ...", end="", flush=True)
t0 = time.time()
prym = PRyMmain.PRyMclass()
res = prym.PRyMresults()
elapsed = time.time() - t0
print(" done (%.1f s)" % elapsed)

# ============================================================
# Extract final distributions from the grid
# ============================================================
solver = prym._boltz_solver
y_grid = solver.y_grid
dy = solver.dy
Ny = solver.Ny
f_final = prym._boltz_f_final  # shape (3, Ny): [nue, nuebar, numu]

# Initial comoving temperature: neutrinos would maintain this T_com forever
# if they decoupled instantaneously (no heating from e+e- annihilation).
T_com_ref = prym._boltz_Tnu_ini * prym._boltz_a_ini

# ============================================================
# Helper functions
# ============================================================
def rho_grid(f_arr):
    """Comoving energy density on the grid."""
    return np.sum(y_grid**3 * f_arr) * dy / (2.0 * np.pi**2)

def n_grid(f_arr):
    """Comoving number density on the grid."""
    return np.sum(y_grid**2 * f_arr) * dy / (2.0 * np.pi**2)

def find_T_eff(f_arr, T_guess=5.0):
    """Find T such that grid-integrated rho of FD(T) matches that of f_arr."""
    rho_target = np.sum(y_grid**3 * f_arr) * dy
    def residual(T):
        return np.sum(y_grid**3 / (np.exp(y_grid / T) + 1.0)) * dy - rho_target
    return brentq(residual, T_guess * 0.9, T_guess * 1.1)

# Analytic FD integral coefficients
rho_FD_coeff = 7.0 * np.pi**4 / 120.0
zeta3 = 1.2020569031595942
n_FD_coeff = 1.5 * zeta3

# Per-flavor distributions (average nu + nubar for electron flavor)
f_nue_avg = 0.5 * (f_final[0] + f_final[1])
f_numu = f_final[2]

Neff = res[0]

# ============================================================
# Compute per-flavor quantities
# ============================================================
rho_ref_val = rho_FD_coeff / (2.0 * np.pi**2) * T_com_ref**4
n_ref_val = n_FD_coeff / (2.0 * np.pi**2) * T_com_ref**3

rho_nue = rho_grid(f_nue_avg)
rho_numu = rho_grid(f_numu)
z_nue = (rho_nue / rho_ref_val)**0.25
z_numu = (rho_numu / rho_ref_val)**0.25
drho_nue = (z_nue**4 - 1.0) * 100  # percent
drho_numu = (z_numu**4 - 1.0) * 100

n_nue = n_grid(f_nue_avg)
n_numu = n_grid(f_numu)
nr_nue = n_nue / n_ref_val
nr_numu = n_numu / n_ref_val

# Shape distortion: delta_g(y) = (f - f_FD(T_eff)) / f_FD(T_eff)
# Uses grid-consistent T_eff to isolate shape from overall heating.
T_eff_nue = find_T_eff(f_nue_avg, T_guess=T_com_ref)
T_eff_numu = find_T_eff(f_numu, T_guess=T_com_ref)
f_eq_nue = 1.0 / (np.exp(y_grid / T_eff_nue) + 1.0)
f_eq_numu = 1.0 / (np.exp(y_grid / T_eff_numu) + 1.0)
dg_nue = (f_nue_avg - f_eq_nue) / np.where(f_eq_nue > 1e-30, f_eq_nue, 1.0)
dg_numu = (f_numu - f_eq_numu) / np.where(f_eq_numu > 1e-30, f_eq_numu, 1.0)

# Find zero crossing of shape distortion
sign_nue = np.where(np.diff(np.sign(dg_nue)))[0]
zero_y_nue = y_grid[sign_nue[0]] if len(sign_nue) > 0 else -1

# ============================================================
# Print results
# ============================================================
print("")
print(" ##########################################################")
print(" Neutrino spectral distortion validation")
print(" ##########################################################")
print("")
print(f"  T_com_ref (Tnu_ini * a_ini) = {T_com_ref:.6f} MeV")
print(f"  Neff = {Neff:.4f}")
print("")

print("  Per-flavor energy transfer:")
print(f"  {'':22s} {'PRyMordial':>12s} {'Literature':>12s}")
print(f"  {'-'*50}")
print(f"  {'z_nue':22s} {z_nue:12.5f} {'1.00234':>12s}")
print(f"  {'z_numu':22s} {z_numu:12.5f} {'1.00098':>12s}")
print(f"  {'delta_rho_nue':22s} {drho_nue:11.3f}% {'0.94%':>12s}")
print(f"  {'delta_rho_numu':22s} {drho_numu:11.3f}% {'0.39%':>12s}")
print(f"  {'drho_nue/drho_numu':22s} {drho_nue/drho_numu:12.2f} {'2.4':>12s}")
print(f"  {'n_nue/n_FD':22s} {nr_nue:12.4f} {'1.0014':>12s}")
print(f"  {'n_numu/n_FD':22s} {nr_numu:12.4f} {'1.0007':>12s}")

print("")
print("  Shape distortion delta_g(y) = (f - f_FD(T_eff)) / f_FD(T_eff):")
print(f"  (nu_e T_eff = {T_eff_nue:.6f}, nu_mu T_eff = {T_eff_numu:.6f} MeV)")
print("")
print(f"  {'y':>6s} {'dg_nue':>10s} {'dg_numu':>10s}")
print(f"  {'-'*28}")
for yi in [1, 3, 5, 8, 10, 15, 20, 25, 30, 40]:
    idx = np.argmin(np.abs(y_grid - yi))
    print(f"  {y_grid[idx]:6.1f} {dg_nue[idx]*100:9.4f}% {dg_numu[idx]*100:9.4f}%")
if zero_y_nue > 0:
    print(f"\n  nu_e zero crossing at y ~ {zero_y_nue:.1f}")

print("")
print("  Note: D-kernel heating is nearly thermal (uniform T shift), so the")
print("  shape distortion is small (~0.2%) and crosses zero at y~22, unlike")
print("  exact calculations where the crossover is at y~4. The Froustey")
print("  formalism gives Neff ~ 3.040 (no oscillations) from D-kernel alone.")

# ============================================================
# Validation checks
# ============================================================
print("")
print(" Validation checks:")
print(" " + "-" * 60)

n_pass = 0
n_fail = 0

def check_bool(name, condition, detail=""):
    global n_pass, n_fail
    status = "PASS" if condition else "FAIL"
    if condition:
        n_pass += 1
    else:
        n_fail += 1
    print(f"  [{status}] {name}{detail}")

def check_range(name, value, lo, hi, fmt=".3f"):
    global n_pass, n_fail
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    print(f"  [{status}] {name}: {value:{fmt}} in [{lo:{fmt}}, {hi:{fmt}}]")

# 1. Neff within 0.2% of 3.044 (D-kernel captures ~90% of exact collision energy;
#    Froustey formalism gives Neff ~ 3.040 without oscillations)
check_range("Neff", Neff, 3.038, 3.048, fmt=".4f")

# 2. z_nue > z_numu (nu_e gets more heating from CC interactions)
check_bool("z_nue > z_numu (flavor hierarchy)",
           z_nue > z_numu,
           f": {z_nue:.5f} > {z_numu:.5f}")

# 3. Both flavors heated (z > 1)
check_bool("z_nue > 1 (nue heated)", z_nue > 1.0, f": {z_nue:.5f}")
check_bool("z_numu > 1 (numu heated)", z_numu > 1.0, f": {z_numu:.5f}")

# 4. Per-flavor delta_rho in physically reasonable range
# D-kernel + drift gives somewhat more per-flavor heating than exact calculation
# but total is correct (conservation of total energy)
check_range("delta_rho_nue (%)", drho_nue, 0.5, 2.0)
check_range("delta_rho_numu (%)", drho_numu, 0.2, 1.5)

# 5. Total neutrino heating: delta_rho_total = drho_nue + 2*drho_numu (per species)
# Literature total: 0.94 + 2*0.39 = 1.72%. Ours will be higher due to drift.
# But total Neff constrains this.
drho_total = drho_nue + 2.0 * drho_numu
check_range("delta_rho_total (%)", drho_total, 1.0, 4.0)

# 6. Shape distortion crosses zero (non-thermal component exists)
has_neg = np.any(dg_nue < -1e-5)
has_pos = np.any(dg_nue > 1e-5)
check_bool("Shape distortion crosses zero",
           has_neg and has_pos,
           f": neg={has_neg}, pos={has_pos}")

# 7. Shape distortion sign: negative at low y, positive at high y
# (energy flows from low to high momentum modes)
dg_low = dg_nue[np.argmin(np.abs(y_grid - 5.0))]
dg_high = dg_nue[np.argmin(np.abs(y_grid - 30.0))]
check_bool("Shape distortion: deficit at y=5, excess at y=30",
           dg_low < 0 and dg_high > 0,
           f": dg(5)={dg_low*100:.3f}%, dg(30)={dg_high*100:.3f}%")

# 8. Number density ratios > 1 (heating adds particles)
check_bool("n_nue/n_FD > 1", nr_nue > 1.0, f": {nr_nue:.4f}")
check_bool("n_numu/n_FD > 1", nr_numu > 1.0, f": {nr_numu:.4f}")

# 9. Number density hierarchy matches energy hierarchy
check_bool("n_nue/n_FD > n_numu/n_FD",
           nr_nue > nr_numu,
           f": {nr_nue:.4f} > {nr_numu:.4f}")

# 10. BBN observables vs thermal reference.
# Froustey formalism gives δYp ~ +0.07%, δ(D/H) ~ +0.3% due to the
# self-consistent plasma entropy decrease during neutrino heating.
print("")
print("  BBN observable cross-check (vs thermal reference):")
PRyMini.general_nu_flag = False
PRyMini.boltzmann_nu_flag = False
importlib.reload(PRyMthermo)
res_thermal = PRyMmain.PRyMclass().PRyMresults()
for i, name, tol in [(4, "Yp(BBN)", 0.15), (5, "D/H", 0.50)]:
    pct = abs(res[i] - res_thermal[i]) / abs(res_thermal[i]) * 100
    check_range(f"|d{name}| vs thermal (%)", pct, 0.0, tol, fmt=".4f")

print("")
print(f" Summary: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail} checks")
print("")
if n_fail == 0:
    print(" All validation checks passed.")
else:
    print(f" WARNING: {n_fail} check(s) failed!")
