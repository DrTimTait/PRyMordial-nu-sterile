"""Diagnostic: Strang driver (not D.7.1 ETDRK2) at sin^2 2theta_24=1e-4,
PMNS active-active off.

If Strang gives ~0.04 (Hannestad), the bug is specific to D.7.1 and we
can use Strang for literature comparison. If Strang also gives ~0.29,
the issue is in the underlying QKE formulation.
"""
import os, sys, time, importlib, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT); os.chdir(_WT)
import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True; PRyMini.julia_flag = False; PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False; PRyMini.compute_nTOp_flag = False; PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True; PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True
PRyMini.qke_full_ode_flag = False    # <-- Strang path (not ODE)
PRyMini.qke_ode_etdrk2_flag = False
PRyMini.massive_electron_flag = False
PRyMini.sterile_flag = True; PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0; PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4))/2.0; PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0; PRyMini.n_B_override = None

# Kill active-active PMNS
PRyMini.theta_12 = 0.0; PRyMini.theta_13 = 0.0; PRyMini.theta_23 = 0.0

import PRyM.PRyM_thermo as PRyMthermo
importlib.reload(PRyMthermo)
import PRyM.PRyM_main as PRyMmain

print("=" * 80)
print("Diagnostic: STRANG path (qke_full_ode_flag=False)")
print("PMNS off, sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2")
print("Hannestad target: ~0.04")
print("D.7.1 PMNS-off gave 0.29")
print("=" * 80)

t0 = time.time()
c = PRyMmain.PRyMclass()
res = c.PRyMresults()
wall = time.time() - t0
sum_ss = 0.0
if hasattr(c, '_boltz_rho_final') and c._boltz_rho_final is not None and c._boltz_rho_final.shape[1] >= 4:
    sum_ss = float(c._boltz_rho_final[:, 3, :].sum())

dNeff = res[0] - 3.04070
print(f"\nNeff = {res[0]:.5f}   Delta Neff = {dNeff:+.4f}   Sum_rho_ss = {sum_ss:.3f}   ({wall:.0f} s)")
print(f"Hannestad ~0.04")
