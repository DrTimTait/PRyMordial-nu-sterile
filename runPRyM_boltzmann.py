# -*- coding: utf-8 -*-
"""
PRyMordial: Boltzmann neutrino transport example.

Evolves neutrino distribution functions on a comoving momentum grid
from thermal (Fermi-Dirac) initial conditions at T = 10 MeV through
BBN, using the internal Boltzmann solver with SM 2-to-2 collision
integrals computed from first principles.

Compares results against the standard thermal calculation to validate
the Boltzmann solver.
"""
import time
import numpy as np

import PRyM.PRyM_init as PRyMini

# ============================================================
# Run 1: Standard thermal calculation (reference)
# ============================================================
print(" ")
print(" ##########################################################")
print(" PRyMordial: Standard thermal run (large network, reference)")
print(" ##########################################################")

PRyMini.verbose_flag = True
PRyMini.smallnet_flag = False
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.general_nu_flag = False
PRyMini.boltzmann_nu_flag = False
PRyMini.compute_bckg_flag = False
PRyMini.compute_nTOp_flag = False

import PRyM.PRyM_main as PRyMmain

start_time = time.time()
res_thermal = PRyMmain.PRyMclass().PRyMresults()
t_thermal = time.time() - start_time

print(" ")
print(" Neff --> ",res_thermal[0])
print(" Yp (BBN) --> ",res_thermal[4])
print(" D/H x 10^5 --> ",res_thermal[5])
print(" ")
print("--- running time: %.1f seconds ---" % t_thermal)

# ============================================================
# Run 2: Boltzmann solver with thermal initial conditions
# ============================================================
print(" ")
print(" ##########################################################")
print(" PRyMordial: Boltzmann solver (thermal ICs, large network)")
print(" ##########################################################")
print(" ")
print(" Neutrino distributions initialized to Fermi-Dirac at T_start.")
print(" Evolved on comoving momentum grid with SM collision integrals.")

# Enable Boltzmann solver.
# The default thermal Fermi-Dirac distributions are used as initial conditions.
PRyMini.boltzmann_nu_flag = True
PRyMini.general_nu_flag = True   # must be set before reloading PRyM_thermo
PRyMini.numba_flag = True        # required for collision integral performance

# Reload PRyM_thermo to initialize GL quadrature nodes for general_nu mode.
# This is needed because PRyM_thermo was first imported with general_nu_flag=False
# during the standard thermal run above; the GL nodes are only defined when
# general_nu_flag is True at import time.
import PRyM.PRyM_thermo as PRyMthermo
import importlib
importlib.reload(PRyMthermo)

start_time = time.time()
res_boltzmann = PRyMmain.PRyMclass().PRyMresults()
t_boltzmann = time.time() - start_time

print(" ")
print(" Neff --> ",res_boltzmann[0])
print(" Yp (BBN) --> ",res_boltzmann[4])
print(" D/H x 10^5 --> ",res_boltzmann[5])
print(" ")
print("--- running time: %.1f seconds ---" % t_boltzmann)

# ============================================================
# Comparison
# ============================================================
print(" ")
print(" ##########################################################")
print(" Comparison: Standard thermal vs Boltzmann (thermal ICs)")
print(" ##########################################################")
print(" ")

labels = ["Neff", "Omega_nu h2 x 10^6 (rel)", "sum_mnu/Omega_nu h2 [eV]",
          "Yp (CMB)", "Yp (BBN)", "D/H x 10^5", "He3/H x 10^5", "Li7/H x 10^10"]

print(" %-28s %14s %14s %10s" % ("Observable", "Thermal", "Boltzmann", "Diff %"))
print(" " + "-"*68)
for i, lab in enumerate(labels):
    v1 = res_thermal[i]
    v2 = res_boltzmann[i]
    if abs(v1) > 0:
        pct = (v2 - v1) / abs(v1) * 100
    else:
        pct = 0.0
    print(" %-28s %14.6f %14.6f %+9.4f%%" % (lab, v1, v2, pct))

print(" ")
print(" Thermal time:   %.1f s" % t_thermal)
print(" Boltzmann time: %.1f s" % t_boltzmann)
print(" ")
print(" Note: The Boltzmann solver shows a ~0.2% Neff deficit relative to the")
print(" standard thermal calculation. This is a known discretization effect from")
print(" the comoving momentum grid (Ny=100, dy=1.0) and is acceptable for BSM")
print(" studies where non-thermal effects are typically >> 0.2%.")
