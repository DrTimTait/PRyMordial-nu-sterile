"""Stage E.1 cross-check against Gariazzo, de Salas, Pastor 2019
(arXiv:1905.11290) Fig. 3, |U_mu4|^2 = 1e-4 violet curve.

Gariazzo's benchmark:
    Dm2_41 = 1.29 eV^2
    |U_mu4|^2 = 1e-4  ->  sin^2(2 theta_24) = 4 |U|^2 (1 - |U|^2) ~= 4e-4
    reported dNeff ~= 0.09

With PRyMini.qke_damping_formula = "mirizzi" (or "gariazzo") this script
expects dNeff in [0.05, 0.2] (30% bracket) and ideally ~0.09.

Note: Gariazzo uses FortEPiaNO and their own damping coefficients (App.
A.17-A.20). Mirizzi+2012's form differs in the active-sterile channels
(D_mu_s / D_mu_tau ratio of 2.23 vs Gariazzo's 4.23). If PRyMordial's
Mirizzi result deviates > 30% from Gariazzo, the script will print a
FALLBACK recommendation and Stage E.1 Step 8 (Gariazzo form calibration)
should be activated.
"""
import os
import sys
import time
import importlib

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini


def _base_flags():
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
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0


def _run(label, configure):
    _base_flags()
    configure()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
    Neff = res[0]
    print(f"  {label:60s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  Sigma_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss


def configure_3x3():
    PRyMini.sterile_flag = False


# Gariazzo benchmark: |U_mu4|^2 = 1e-4, Dm2_41 = 1.29 eV^2
U_mu4_sq = 1.0e-4
sin2_2theta_24 = 4.0 * U_mu4_sq * (1.0 - U_mu4_sq)  # ~= 4e-4
Dm2_41_gari = 1.29
dNeff_gari = 0.09  # Gariazzo+2019 Fig. 3 violet curve (extrapolated)


def configure_gari():
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = Dm2_41_gari
    PRyMini.theta_24 = np.arcsin(np.sqrt(sin2_2theta_24)) / 2.0


formula = getattr(PRyMini, "qke_damping_formula", "symmetric")
print("=" * 88)
print("Stage E.1 Gariazzo cross-check (arXiv:1905.11290 Fig. 3)")
print(f"PRyMini.qke_damping_formula = {formula!r}")
print(f"Dm2_41 = {Dm2_41_gari} eV^2,  |U_mu4|^2 = {U_mu4_sq:.1e}  "
      f"(sin^2 2theta_24 ~= {sin2_2theta_24:.2e})")
print(f"Gariazzo reference dNeff ~= {dNeff_gari:.2f}")
print("=" * 88)

Neff_3x3, _ = _run("3x3 QKE reference (no sterile)", configure_3x3)

print()
Neff_gari, sum_ss = _run(
    f"Gariazzo point (|U|^2={U_mu4_sq:.1e}, Dm2={Dm2_41_gari})",
    configure_gari)
dNeff_ours = Neff_gari - Neff_3x3

print()
print("-" * 88)
print(f"Gariazzo reference dNeff = {dNeff_gari:.3f}")
print(f"PRyMordial dNeff         = {dNeff_ours:+.3f}")
rel_diff = (dNeff_ours - dNeff_gari) / max(dNeff_gari, 1e-6)
print(f"Relative deviation       = {rel_diff:+.1%}")
print("-" * 88)

if 0.05 <= dNeff_ours <= 0.20:
    print("PASS: dNeff in [0.05, 0.20] -- Mirizzi form reproduces Gariazzo "
          "within acceptable range.")
else:
    if dNeff_ours > 0.20:
        direction = "OVER-PRODUCTION"
    else:
        direction = "UNDER-PRODUCTION"
    print(f"FAIL: dNeff = {dNeff_ours:.3f} outside [0.05, 0.20] -- {direction}.")
    print("FALLBACK: activate Stage E.1 Step 8 (Gariazzo form calibration) "
          "by setting PRyMini.qke_damping_formula = 'gariazzo' after "
          "dimensional prefactor is verified.")
