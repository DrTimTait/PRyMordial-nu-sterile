"""Quick probe: does monkey-patching PRyM_boltzmann._offdiag_collision_gain
actually reach _assemble_collision_N? Tracer raises on call."""
import os, sys, importlib, numpy as np
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
PRyMini.sterile_flag = True
PRyMini.Dm2_41 = 0.93
PRyMini.theta_14 = 0.0
PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
PRyMini.theta_34 = 0.0
PRyMini.theta_12 = 0.0
PRyMini.theta_13 = 0.0
PRyMini.theta_23 = 0.0
PRyMini.xi_nue_init = 0.0
PRyMini.xi_numu_init = 0.0
PRyMini.xi_nutau_init = 0.0
PRyMini.n_B_override = None

import PRyM.PRyM_boltzmann as PRyMboltz

_call_count = {"n": 0}

def _tracer(rho_offdiag, f_all, y_grid, quad_w, a, Tg, GF2_prefactor,
            c_emu_scat, c_mutau_scat, fnu_emu_scat_val, fnu_mutau_scat_val,
            B_spectator_e_idx, tail_params, D_k0, D_k2, Ny_coll):
    _call_count["n"] += 1
    return np.zeros((6, len(y_grid)))

def _tracer_massive(rho_offdiag, f_all, y_grid, quad_w, a, Tg, GF2_prefactor,
                    c_emu_scat, c_mutau_scat, B_spectator_e_idx, tail_params,
                    D_k0, D_k2, Ny_coll, me):
    _call_count["n"] += 1
    return np.zeros((6, len(y_grid)))

# Build solver, hand-construct a minimal rho_all, and directly invoke
# _assemble_collision_N to see if our patched gain is called.
dm = PRyMboltz.DensityMatrixSolver()
print(f"[probe] n_flavor={dm.n_flavor}  Ny={dm.Ny}  pair_flavors={dm._all_pair_flavors}")

# Initial conditions: use thermal FD at Tnu
Tnu = 2.0  # MeV, somewhere in Phase B
a = 1.0 / Tnu
rho_all = dm.initial_conditions(Tnu, a)
print(f"[probe] rho_all.shape = {rho_all.shape}")

# Patch AFTER solver is built (same order as diag_as_gain.py's intent)
PRyMboltz._offdiag_collision_gain = _tracer
PRyMboltz._offdiag_collision_gain_massive = _tracer_massive

try:
    N_list, I_total = dm._assemble_collision_N(rho_all, a, Tnu)
    print(f"[probe] _assemble_collision_N returned; stub was called {_call_count['n']} time(s)")
    if _call_count["n"] == 0:
        print("[probe] FAIL: the patched _offdiag_collision_gain was NEVER called.")
        print("[probe]       Monkey-patching strategy is broken.")
    else:
        print("[probe] PASS: patched stub reached _assemble_collision_N.")
        # Also check that N_list's active-active entries reflect the zero gain.
        # N_full off-diagonals should equal -D_pair*rho_ab_stored (no gain term).
        sector = 0
        rho_mat_idx = dm._active_pair_write[0]  # (Re, Im) for e-mu pair
        rho_ab = (rho_all[sector, rho_mat_idx[0]]
                  + 1j * rho_all[sector, rho_mat_idx[1]])
        print(f"[probe] |N_list[0][:,0,1]| avg = {np.abs(N_list[0][:, 0, 1]).mean():.3e}")
        print(f"[probe] |rho_eμ| avg         = {np.abs(rho_ab).mean():.3e}")
finally:
    pass
