# -*- coding: utf-8 -*-
"""
Generate and save weak rate tables for caching.
Uses the standard thermal path (fast, accurate).
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

import PRyM.PRyM_init as PRyMini
PRyMini.general_nu_flag = False
PRyMini.boltzmann_nu_flag = False
PRyMini.compute_nTOp_flag = True
PRyMini.save_nTOp_flag = True
PRyMini.numba_flag = True
PRyMini.verbose_flag = True

import PRyM.PRyM_thermo as PRyMthermo

# Generate Tg_vec and Tnu_vec from standard 2-var ODE
Tstart_MeV = PRyMini.T_start / PRyMini.MeV_to_Kelvin
def Hubble(Tg, Tnu):
    rho_pl = PRyMthermo.rho_g(Tg) + PRyMthermo.rho_e(Tg) - PRyMthermo.PofT(Tg) + Tg*PRyMthermo.dPdT(Tg)
    rho_3nu = PRyMthermo.rho_nu(Tnu) + 2.*PRyMthermo.rho_nu(Tnu)
    return PRyMini.MeV_to_secm1*((rho_pl+rho_3nu)*8.*np.pi/(3.*PRyMini.Mpl**2))**0.5

def dTtotdt(t, T_vec):
    Tg, Tnu = T_vec
    H = Hubble(Tg, Tnu)
    num_g = -(H*(4.*PRyMthermo.rho_g(Tg)+3.*(PRyMthermo.rho_e(Tg)+PRyMthermo.p_e(Tg))+3.*Tg*PRyMthermo.dPdT(Tg)))
    delta = -(PRyMthermo.delta_rho_nue(Tg,Tnu,Tnu)+2.*PRyMthermo.delta_rho_numu(Tg,Tnu,Tnu))
    num_g += delta
    den_g = PRyMthermo.drho_g_dT(Tg)+PRyMthermo.drho_e_dT(Tg)+Tg*PRyMthermo.d2PdT2(Tg)
    num_nu = -12.*H*PRyMthermo.rho_nu(Tnu) + (-delta)
    den_nu = 3.*PRyMthermo.drho_nu_dT(Tnu)
    return [num_g/den_g, num_nu/den_nu]

tini = 1./(2.*Hubble(Tstart_MeV, Tstart_MeV))
tfin = PRyMini.t_end
sol = solve_ivp(dTtotdt, [tini, tfin], [Tstart_MeV, Tstart_MeV],
                t_eval=np.logspace(np.log10(tini), np.log10(tfin), PRyMini.n_sampling),
                method='LSODA', rtol=1.e-6, atol=1.e-9)
Tg_vec = sol.y[0]
Tnu_vec = sol.y[1]

print(f"Tg range: {Tg_vec[0]:.4f} to {Tg_vec[-1]:.6f} MeV")
print(f"Tnu/Tg at end: {Tnu_vec[-1]/Tg_vec[-1]:.6f}")

t0 = time.time()
import PRyM.PRyM_nTOp as PRyMnTOp
rates = PRyMnTOp.RecomputeWeakRates([Tg_vec, Tnu_vec])
dt = time.time() - t0
print(f"\nWeak rate computation: {dt:.1f}s")

# Verify files were saved
import os
files = ["nTOp_frwrd_HT.txt", "nTOp_bkwrd_HT.txt",
         "nTOp_frwrd_MT.txt", "nTOp_bkwrd_MT.txt",
         "nTOp_frwrd_LT.txt", "nTOp_bkwrd_LT.txt"]
for f in files:
    path = PRyMini.working_dir + "/PRyMrates/nTOp/" + f
    if os.path.exists(path):
        print(f"  Saved: {f} ({os.path.getsize(path)} bytes)")
    else:
        print(f"  MISSING: {f}")
print("\nDone!")
