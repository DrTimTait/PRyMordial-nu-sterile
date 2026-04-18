"""Diagnostic: PMNS off + Hannestad C_alpha damping. If this brings
delta_Neff close to 0.04, we have identified the two key fixes needed.
"""
import os, sys, time, importlib, numpy as np
_WT = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/.claude/worktrees/trusting-pike-2de6ff"
sys.path.insert(0, _WT); os.chdir(_WT)
import PRyM.PRyM_init as PRyMini
PRyMini.smallnet_flag = True; PRyMini.julia_flag = False; PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False; PRyMini.compute_nTOp_flag = False; PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True; PRyMini.boltzmann_nu_flag = True
PRyMini.qke_density_matrix_flag = True; PRyMini.qke_full_ode_flag = True; PRyMini.qke_ode_etdrk2_flag = True
PRyMini.massive_electron_flag = False
PRyMini.sterile_flag = True; PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0; PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4))/2.0; PRyMini.theta_34 = 0.0
PRyMini.xi_nue_init = 0.0; PRyMini.n_B_override = None

# Kill active-active PMNS
PRyMini.theta_12 = 0.0; PRyMini.theta_13 = 0.0; PRyMini.theta_23 = 0.0

# Swap C_D to Hannestad values. Note: PRyMordial also uses C_D internally
# in the collision-integral side, not just H. But for this quick diagnostic
# we monkey-patch the solver's C_D after construction.
import PRyM.PRyM_thermo as PRyMthermo
importlib.reload(PRyMthermo)
import PRyM.PRyM_boltzmann as PRyMboltz

# Patch the DensityMatrixSolver __init__ so any instance built later gets
# Hannestad's C_D. Simplest: rewrite self.C_D after __init__ finishes.
_orig_init = PRyMboltz.DensityMatrixSolver.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if self.n_flavor == 4:
        self.C_D = np.array([1.27, 0.92, 0.92, 0.0])
    else:
        self.C_D = np.array([1.27, 0.92, 0.92])
    print(f"  [patched] solver.C_D = {self.C_D}")
PRyMboltz.DensityMatrixSolver.__init__ = _patched_init

import PRyM.PRyM_main as PRyMmain

print("=" * 80)
print("Diagnostic: PMNS off + Hannestad C_alpha = (1.27, 0.92, 0.92, 0)")
print("sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2")
print("Target: Hannestad delta_Neff ~ 0.04")
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
print(f"Summary: PMNS-off = 0.29, PMNS-off + Hannestad C_D = {dNeff:.3f}")
