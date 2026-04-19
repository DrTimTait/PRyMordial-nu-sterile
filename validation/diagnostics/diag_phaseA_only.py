"""Stage E.2(b) sanity control: extend Phase A (thermal) without extending Phase B (QKE).

Context. diag_qke_window.py showed that pushing T_boltz_start from 5 to 30 MeV
drives dNeff to 0.0496 and to 10.62 at 60 MeV -- the second is clearly
unphysical (Neff > 4 forbidden in 3+1), suggesting numerical breakdown of
Phase B's Froustey formalism at T >> 5 MeV. The 30 MeV run is internally
inconsistent (dNeff=0.05 but sum_rho_ss=26). Before accepting either result
as evidence of hypothesis B, we need to confirm the effect comes from
QKE-phase extension and not from something in Phase A.

This control. T_start = 60 MeV (Phase A runs thermal from 60 -> 5 MeV),
T_boltz_start = 5 MeV (Phase B QKE runs from 5 -> T_boltz_end, same as
baseline). Phase A is thermal equilibrium, so all neutrinos stay in FD
with Tnu=Tg and zero coherences regardless of window width. The QKE
phase should initialise identically to the default run.

Expected: dNeff ~ 0.845 +/- few%, matching the variant (i) PMNS-on baseline
from diag_as_gain.py. If it matches, the 30 MeV dNeff=0.05 result really
was QKE-phase behaviour (numerically contaminated but physically real-ish).
If it DIFFERS, Phase A's extended thermal evolution itself is somehow
coupling to the result -- which would be a surprise and need investigation.
"""
import os
import sys
import time
import importlib
import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
PRyMini.n_B_override = None
PRyMini.xi_nue_init = 0.0
PRyMini.xi_numu_init = 0.0
PRyMini.xi_nutau_init = 0.0
PRyMini.qke_damping_formula = "mirizzi"

PRyMini.sterile_flag = True
PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
PRyMini.theta_34 = 0.0

# KEY: extend Phase A only.
PRyMini.T_start = 60.1 * PRyMini.MeV_to_Kelvin    # Phase A starts at 60.1 MeV
PRyMini.T_boltz_start = 5.0                        # Phase B (QKE) starts at 5 MeV (default)

import PRyM.PRyM_thermo as PRyMthermo
importlib.reload(PRyMthermo)
import PRyM.PRyM_main as PRyMmain
importlib.reload(PRyMmain)

print("=" * 88)
print("Stage E.2(b) sanity control: T_start = 60 MeV, T_boltz_start = 5 MeV (default)")
print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi.")
print("Expected: dNeff ~ 0.845 (matching default-window PMNS-on baseline).")
print("=" * 88)

t0 = time.time()
c = PRyMmain.PRyMclass()
res = c.PRyMresults()
dt = time.time() - t0

sum_ss = 0.0
if (hasattr(c, "_boltz_rho_final")
        and c._boltz_rho_final is not None
        and c._boltz_rho_final.shape[1] >= 4):
    rho = c._boltz_rho_final
    sum_ss = float(rho[:, 3, :].sum())

# Reference from diag_qke_window (3x3 at T_boltz_start=5, same config modulo sterile off):
Neff_3x3_ref = 3.04071

Neff = res[0]
dNeff = Neff - Neff_3x3_ref
print(f"  Phase-A-extended run   Neff={Neff:.5f}  Yp={res[4]:.5f}  "
      f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)")
print(f"     dNeff = {dNeff:+.4f}")
print()
print(f"  Baseline (default window, from diag_as_gain.py variant (i)): dNeff = +0.8453")
print(f"  Delta from baseline: {dNeff - 0.8453:+.4f}")
