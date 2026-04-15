# -*- coding: utf-8 -*-
"""
PRyMordial: Neutrino treatment comparison.

Runs all five neutrino treatment modes and compares their predictions
for Neff and BBN observables (Yp, D/H, He3/H, Li7/H).

Modes (in order of increasing physics):

  1. Standard thermal
     Neutrinos described by a single temperature Tnu(t) evolved via
     coupled Tg-Tnu ODEs with thermally-averaged collision rates.
     Fastest (~2 s). This is the legacy PRyMordial approach.

  2. General nu (prescribed Fermi-Dirac)
     Same thermal distributions, but energy densities and pressures
     evaluated via Gauss-Legendre quadrature over f(p) rather than
     analytic T^4 formulas. Validates the GL integration infrastructure
     that non-thermal (BSM) distributions plug into.

  3. Boltzmann diagonal
     Neutrino distributions evolved on a comoving momentum grid
     (Ny=100, y = p*a) with 2-to-2 collision integrals in the D-kernel
     polynomial formalism (nu-nu and nu-e scattering/annihilation).
     Flavor-diagonal: each species evolves independently, no mixing.

  4. Boltzmann + oscillation relaxation
     Same as (3), plus operator-split flavor mixing via Sigl-Raffelt
     relaxation toward the flavor-averaged distribution, using PMNS
     mixing angles from PDG 2024.

  5. QKE density matrix
     Full 3x3 Hermitian density matrix rho(y) for each momentum mode,
     evolved via Quantum Kinetic Equations with combined oscillation +
     collision stepping: exact unitary oscillation (vacuum + thermal
     matter potential) and diagonal collision integrals with off-diagonal
     damping. Captures flavor coherences that the relaxation approximation
     cannot.

All Boltzmann modes use the Froustey et al. (arXiv:2008.01074) formalism:
Friedmann-evolved scale factor with photon temperature from the plasma
entropy equation. Neff agrees with the thermal reference to <0.01% for
oscillation modes. BBN observables differ from the thermal reference
(delta_Yp ~ +0.07%, delta_D/H ~ +0.3%) due to the self-consistent
treatment of the plasma entropy decrease during neutrino heating.
"""
import time
import numpy as np
import importlib

import PRyM.PRyM_init as PRyMini

# Common settings
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False
PRyMini.compute_nTOp_flag = False

labels = ["Neff", "Omega_nu h2 x 10^6", "sum_mnu/Omega_nu h2 [eV]",
          "Yp (CMB)", "Yp (BBN)", "D/H x 10^5", "He3/H x 10^5", "Li7/H x 10^10"]

# Store results: list of (name, description, result_array, elapsed_time)
runs = []

# ============================================================
# 1. Standard thermal
# ============================================================
PRyMini.verbose_flag = False
PRyMini.general_nu_flag = False
PRyMini.boltzmann_nu_flag = False
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False
PRyMini.massive_electron_flag = False

import PRyM.PRyM_main as PRyMmain

print(" Running: Standard thermal ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("Standard thermal",
             "Coupled Tg-Tnu ODEs with thermally-averaged collision rates",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 2. General nu (prescribed Fermi-Dirac)
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = False
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False

import PRyM.PRyM_thermo as PRyMthermo
importlib.reload(PRyMthermo)

print(" Running: General nu (prescribed FD) ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("General nu (FD)",
             "Thermal FD distributions evaluated via GL quadrature",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 3. Boltzmann diagonal (no oscillations)
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False
importlib.reload(PRyMthermo)

print(" Running: Boltzmann diagonal ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("Boltzmann diagonal",
             "D-kernel collision integrals, no flavor mixing",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 4. Boltzmann + oscillation relaxation
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = True
PRyMini.qke_density_matrix_flag = False
PRyMini.nu_oscillation_method = 'relaxation'
importlib.reload(PRyMthermo)

print(" Running: Boltzmann + oscillation relaxation ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("Boltzmann + osc relax",
             "D-kernel collisions + Sigl-Raffelt flavor relaxation (PMNS)",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 5. QKE density matrix
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = True
PRyMini.massive_electron_flag = False
importlib.reload(PRyMthermo)

print(" Running: QKE density matrix ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("QKE density matrix",
             "Full 3x3 QKE: unitary oscillation + collision + damping",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 6. Standard thermal + O(e^4) two-loop QED
# ============================================================
# Same physics as mode 1, but with the NUDEC_BSM v2 / Escudero+2025 two-loop
# QED plasma correction turned on. Expected shifts ~1e-4 on Neff and Yp.
PRyMini.general_nu_flag = False
PRyMini.boltzmann_nu_flag = False
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False
PRyMini.massive_electron_flag = False
PRyMini.two_loop_QED_flag = True
importlib.reload(PRyMthermo)

print(" Running: Standard thermal + O(e^4) QED ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass().PRyMresults()
elapsed = time.time() - t0
runs.append(("Standard + O(e^4) QED",
             "Standard thermal with NUDEC_BSM v2 two-loop QED correction",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# Restore default so subsequent imports don't carry state
PRyMini.two_loop_QED_flag = False

# ============================================================
# 7. QKE asymmetric (Stage 1) — mu_tau_symmetric_flag=False with
#    equal initial distributions. Should reproduce mode 5 (QKE
#    density matrix) since all 6 species start as identical thermal FD.
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = True
PRyMini.massive_electron_flag = False
PRyMini.mu_tau_symmetric_flag = False   # <-- new flag, Stage 1
importlib.reload(PRyMthermo)

# Equal thermal FD callables for all 6 species (asymmetric path with
# symmetric content). Note: initial_conditions uses (p_phys, Tnu).
def _fd_thermal(p, T):
    import numpy as _np
    x = _np.asarray(p, dtype=float) / T
    out = _np.zeros_like(x)
    mask = x < 500.0
    out[mask] = 1.0 / (_np.exp(x[mask]) + 1.0)
    return out

print(" Running: QKE asymmetric (equal IC, Stage 1) ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass(
    my_f_nue=_fd_thermal, my_f_nuebar=_fd_thermal,
    my_f_numu=_fd_thermal, my_f_numubar=_fd_thermal,
    my_f_nutau=_fd_thermal, my_f_nutaubar=_fd_thermal,
).PRyMresults()
elapsed = time.time() - t0
runs.append(("QKE asym equal IC",
             "mu-tau asymmetric path with 6 equal FD inputs (matches QKE)",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# Restore defaults
PRyMini.mu_tau_symmetric_flag = True

# ============================================================
# 8. Diagonal Boltzmann asymmetric, equal IC (Stage 2 regression)
#    n=4 species with identical FD initial distributions.
#    Should match mode 3 "Boltzmann diagonal" (symmetric) within numerics.
# ============================================================
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = False
PRyMini.massive_electron_flag = False
PRyMini.mu_tau_symmetric_flag = False
importlib.reload(PRyMthermo)

print(" Running: Diag Boltz asym (equal IC, Stage 2) ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass(
    my_f_nue=_fd_thermal, my_f_nuebar=_fd_thermal,
    my_f_numu=_fd_thermal, my_f_nutau=_fd_thermal,
).PRyMresults()
elapsed = time.time() - t0
runs.append(("Diag Boltz asym eq IC",
             "n=4 diagonal, mu=tau equal thermal FD (matches mode 3)",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# ============================================================
# 9. Diagonal Boltzmann asymmetric, unequal mu/tau IC (BSM smoke test)
#    numu starts at 1.1 x FD, nutau at 0.9 x FD. Total rho_3nu preserved
#    initially; nu-nu collisions should partially equilibrate mu vs tau.
# ============================================================
def _fd_numu_hot(p, T):
    return 1.1 * _fd_thermal(p, T)
def _fd_nutau_cold(p, T):
    return 0.9 * _fd_thermal(p, T)

print(" Running: Diag Boltz asym (unequal IC, Stage 2) ...", end="", flush=True)
t0 = time.time()
res = PRyMmain.PRyMclass(
    my_f_nue=_fd_thermal, my_f_nuebar=_fd_thermal,
    my_f_numu=_fd_numu_hot, my_f_nutau=_fd_nutau_cold,
).PRyMresults()
elapsed = time.time() - t0
runs.append(("Diag Boltz asym uneq IC",
             "n=4 diagonal, numu=1.1 FD, nutau=0.9 FD (BSM smoke)",
             res, elapsed))
print(" done (%.1f s)" % elapsed)

# Restore defaults
PRyMini.mu_tau_symmetric_flag = True

# ============================================================
# Summary
# ============================================================
ref = runs[0][2]  # Standard thermal as reference

print("")
print(" ##########################################################")
print(" PRyMordial: Neutrino treatment comparison")
print(" ##########################################################")
print("")

# Flag table
print(" Flag settings:")
print(" %-24s %s %s %s %s" % ("", "general_nu", "boltzmann_nu", "nu_oscillation", "qke_dm"))
print(" " + "-" * 76)
flag_table = [
    ("Standard thermal",       "False", "False", "  --",  " --"),
    ("General nu (FD)",        " True", "False", "  --",  " --"),
    ("Boltzmann diagonal",     " True", " True", "False", "False"),
    ("Boltzmann + osc relax",  " True", " True", " True", "False"),
    ("QKE density matrix",     " True", " True", "  --",  " True"),
    ("Standard + O(e^4) QED",  "False", "False", "  --",  " --"),
    ("QKE asym equal IC",      " True", " True", "  --",  " True"),
    ("Diag Boltz asym eq IC",  " True", " True", "False", "False"),
    ("Diag Boltz asym uneq IC"," True", " True", "False", "False"),
]
for name, g, b, o, q in flag_table:
    print(" %-24s %10s %12s %14s %6s" % (name, g, b, o, q))

print("")
print(" Results (small network, pre-stored weak rates):")
print("")
print(" %-24s %7s %10s %10s %10s %8s %8s %8s" % (
    "Configuration", "Time", "Neff", "Yp(BBN)", "D/H x10^5",
    "dNeff%", "dYp%", "dD/H%"))
print(" " + "-" * 97)

for name, desc, res, elapsed in runs:
    dNeff = (res[0] - ref[0]) / abs(ref[0]) * 100
    dYp = (res[4] - ref[4]) / abs(ref[4]) * 100
    dDH = (res[5] - ref[5]) / abs(ref[5]) * 100
    print(" %-24s %6.1fs %10.4f %10.6f %10.4f %+7.3f%% %+7.3f%% %+7.3f%%" % (
        name, elapsed, res[0], res[4], res[5], dNeff, dYp, dDH))

print("")
print(" Physics included in each mode:")
for name, desc, res, elapsed in runs:
    print("   %-24s %s" % (name + ":", desc))

print("")
print(" All Boltzmann modes use the Froustey et al. (arXiv:2008.01074) formalism:")
print(" the scale factor a is evolved via Friedmann, and the photon temperature Tg")
print(" is determined from the plasma entropy equation d(spl*a^3)/dt = -(Q/Tg)*a^3,")
print(" where Q is the collision energy transfer rate to neutrinos. This self-")
print(" consistently tracks the plasma cooling from e+e- annihilation energy flowing")
print(" to neutrinos, producing Neff > 3 without a momentum drift correction.")
print(" BBN observables differ from the thermal reference (delta_Yp ~ +0.07%,")
print(" delta_D/H ~ +0.3%) due to the self-consistent treatment of the plasma")
print(" entropy decrease during neutrino heating.")
