"""Diagnostic: disable active-active PMNS mixing and re-run DW point C
(sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2). If sterile still
over-produces, the PMNS leak hypothesis is ruled out.

With theta_12 = theta_13 = theta_23 = 0 AND theta_14 = theta_34 = 0,
the ONLY mixing is theta_24 (nu_mu - nu_s). This mimics Hannestad's 1+1
scenario exactly.
"""
import os, sys, time, importlib, numpy as np
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
PRyMini.qke_full_ode_flag = True
PRyMini.qke_ode_etdrk2_flag = True
PRyMini.massive_electron_flag = False
PRyMini.sterile_flag = True
PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0
PRyMini.n_B_override = None  # default n_B

# Kill active-active PMNS
PRyMini.theta_12 = 0.0
PRyMini.theta_13 = 0.0
PRyMini.theta_23 = 0.0

import PRyM.PRyM_thermo as PRyMthermo
importlib.reload(PRyMthermo)
import PRyM.PRyM_main as PRyMmain

print("=" * 80)
print("Diagnostic: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, ALL OTHER angles = 0")
print("Expected: Hannestad ~4% thermalisation (delta_Neff ~ 0.04)")
print("If our code gives >> 0.04, the PMNS leak is not the culprit.")
print("=" * 80)

t0 = time.time()
c = PRyMmain.PRyMclass()
res = c.PRyMresults()
wall = time.time() - t0
sum_ss = 0.0
if hasattr(c, '_boltz_rho_final') and c._boltz_rho_final is not None and c._boltz_rho_final.shape[1] >= 4:
    rho = c._boltz_rho_final
    sum_ss = float(rho[:, 3, :].sum())

print(f"\nNeff = {res[0]:.5f}   Yp = {res[4]:.5f}   D/H = {res[5]:.4f}   Sum_rho_ss = {sum_ss:.3f}   ({wall:.0f} s)")

# Reference 3x3 baseline is 3.04070 from prior runs
Neff_3x3_ref = 3.04070
delta_Neff = res[0] - Neff_3x3_ref
print(f"Delta Neff (vs 3x3 reference 3.04070) = {delta_Neff:+.4f}")
print(f"Hannestad expectation: ~0.04")
