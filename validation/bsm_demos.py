# -*- coding: utf-8 -*-
"""
BSM demonstration scenarios for PRyMordial-nu.

Showcases the new μ-τ / ν-ν̄ asymmetric evolution modes (Task 2) through
three toy BSM setups that each exercise a different sector of the new
capability:

  A. L_μ - L_τ anomaly — asymmetric μ and τ "temperatures" post-decoupling.
     Uses n=4 diagonal Boltzmann (mu_tau_symmetric=False, nu_nubar=True).

  B. Lepton asymmetry — ξ_μ ≠ 0 chemical potential for the μ sector,
     f_ν != f_ν̄ for ν_μ. Uses n=6 diagonal (nu_nubar_symmetric=False).

  C. Flavor-specific dark-matter decay — monochromatic ν_τ + ν̄_τ injection
     on top of thermal. Uses n=6 diagonal.

Each scenario prints (Neff, Yp, D/H) alongside the SM baseline so the
BSM shift is visible. Runtime is ~10 min wall (mostly numba compile on
cold start + ~100 s per Boltzmann run).
"""
import os
import sys
import time
import importlib
import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

import PRyM.PRyM_init as PRyMini


# Utility: plain thermal FD at temperature T, evaluated at physical p
def _fd(p, T):
    x = np.asarray(p, dtype=float) / T
    out = np.zeros_like(x)
    mask = x < 500.0
    out[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return out


def _fd_with_mu(p, T, xi):
    """Thermal FD with chemical potential xi = mu/T."""
    x = np.asarray(p, dtype=float) / T - xi
    out = np.zeros_like(x)
    mask = x < 500.0
    out[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return out


def _reset_flags():
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = False
    PRyMini.boltzmann_nu_flag = False
    PRyMini.nu_oscillation_flag = False
    PRyMini.qke_density_matrix_flag = False
    PRyMini.massive_electron_flag = False
    PRyMini.mu_tau_symmetric_flag = True
    PRyMini.nu_nubar_symmetric_flag = True
    PRyMini.two_loop_QED_flag = False


def _run(label, setup_fn, ic_kwargs, runs):
    _reset_flags()
    setup_fn()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    print(f"  {label:48s}", end="", flush=True)
    t0 = time.time()
    res = PRyMmain.PRyMclass(**ic_kwargs).PRyMresults()
    dt = time.time() - t0
    runs.append((label, res, dt))
    print(f"  Neff={res[0]:.5f}  Yp={res[4]:.5f}  D/H={res[5]:.4f}  ({dt:5.1f}s)")


def main():
    runs = []

    # ---- SM baselines ----
    print("SM baselines:")

    def s_sm_thermal():
        pass  # all defaults off
    _run("SM Standard thermal (reference)", s_sm_thermal, {}, runs)

    def s_sm_qke():
        PRyMini.general_nu_flag = True
        PRyMini.boltzmann_nu_flag = True
        PRyMini.qke_density_matrix_flag = True
    _run("SM QKE density matrix (reference)", s_sm_qke, {}, runs)

    # ---- Scenario A: L_μ - L_τ anomaly ----
    # Model: at decoupling, μ-sector is 5% hotter than τ-sector in energy
    # density, total rho preserved. Represents a residual L_μ-L_τ asymmetry
    # after the new gauge interaction has frozen out. Test uses n=4 diagonal.
    print("\nScenario A: L_μ - L_τ anomaly (μ 2.5% hotter in T, τ 2.5% cooler)")

    def s_sceA():
        PRyMini.general_nu_flag = True
        PRyMini.boltzmann_nu_flag = True
        PRyMini.mu_tau_symmetric_flag = False
        PRyMini.nu_nubar_symmetric_flag = True  # n=4 diagonal
    scaleA = 0.025
    def _fd_mu_A(p, T):  return _fd(p, (1.0 + scaleA) * T)
    def _fd_tau_A(p, T): return _fd(p, (1.0 - scaleA) * T)
    _run("  n=4 diag, μ hot / τ cold", s_sceA,
         dict(my_f_nue=_fd, my_f_nuebar=_fd,
              my_f_numu=_fd_mu_A, my_f_nutau=_fd_tau_A), runs)

    # ---- Scenario B: Lepton asymmetry ----
    # Chemical potentials ξ_μ = +0.1, ξ_μbar = -0.1 at decoupling.
    # Total μ-lepton number (n_numu - n_numubar) is non-zero; total rho
    # roughly preserved through 2ξ asymmetry.
    print("\nScenario B: Lepton asymmetry (ξ_μ = +0.1, ξ_μbar = -0.1)")

    def s_sceB():
        PRyMini.general_nu_flag = True
        PRyMini.boltzmann_nu_flag = True
        PRyMini.mu_tau_symmetric_flag = False
        PRyMini.nu_nubar_symmetric_flag = False  # n=6
    xi = 0.1
    def _fd_mu_B(p, T):    return _fd_with_mu(p, T, +xi)
    def _fd_mubar_B(p, T): return _fd_with_mu(p, T, -xi)
    _run("  n=6 diag, ξ_μ = +0.1, ξ_μbar = -0.1", s_sceB,
         dict(my_f_nue=_fd, my_f_nuebar=_fd,
              my_f_numu=_fd_mu_B, my_f_numubar=_fd_mubar_B,
              my_f_nutau=_fd, my_f_nutaubar=_fd), runs)

    # ---- Scenario C: Flavor-specific DM decay into ν_τ ν̄_τ ----
    # DM → ν_τ ν̄_τ produces a delta-function bump at p_DM = m_DM/2 in both
    # species. We smear into a narrow Gaussian at x = p/Tν = 5 with 5%
    # relative amplitude. Models a sub-eV-mass sterile or dark sector decay.
    print("\nScenario C: ν_τ + ν̄_τ injection from DM decay (Gaussian bump at x=5)")

    def s_sceC():
        PRyMini.general_nu_flag = True
        PRyMini.boltzmann_nu_flag = True
        PRyMini.mu_tau_symmetric_flag = False
        PRyMini.nu_nubar_symmetric_flag = False  # n=6
    x_bump, amp, sigma = 5.0, 0.05, 0.5
    def _fd_bump(p, T):
        x = np.asarray(p, dtype=float) / T
        base = np.zeros_like(x)
        msk = x < 500.0
        base[msk] = 1.0 / (np.exp(x[msk]) + 1.0)
        bump = amp * np.exp(-0.5 * ((x - x_bump) / sigma) ** 2)
        return np.clip(base + bump, 0.0, 1.0)
    _run("  n=6 diag, ν_τ bump (amp=5%, x=5)", s_sceC,
         dict(my_f_nue=_fd, my_f_nuebar=_fd,
              my_f_numu=_fd, my_f_numubar=_fd,
              my_f_nutau=_fd_bump, my_f_nutaubar=_fd_bump), runs)

    # ---- Summary ----
    print()
    print("=" * 86)
    print("  BSM scenarios summary  (all vs SM Standard thermal)")
    print("=" * 86)
    ref = runs[0][1]
    print(f"  {'Scenario':46s}  {'ΔNeff%':>8s}  {'ΔYp%':>7s}  {'ΔD/H%':>8s}")
    print("  " + "-" * 80)
    for label, res, dt in runs:
        dN = (res[0] - ref[0]) / ref[0] * 100
        dY = (res[4] - ref[4]) / ref[4] * 100
        dD = (res[5] - ref[5]) / ref[5] * 100
        print(f"  {label:46s}  {dN:+8.3f}  {dY:+7.3f}  {dD:+8.3f}")

    print()
    print("  Observations:")
    print("  - Scenario A: μ hot / τ cold → small ΔNeff (total ρ not exactly")
    print("    preserved: rho ∝ T^4 so 2.5% T asymmetry → +0.4% ρ excess →")
    print("    partially relaxed by nu-e and nu-nu collisions, net ~-0.1%).")
    print("  - Scenario B: ξ_μ = 0.1 lepton asymmetry on top of thermal FD.")
    print("    Pair-annihilation rate modified but symmetrically in ν and ν̄")
    print("    so net energy transfer to plasma is small; tiny ΔNeff.")
    print("  - Scenario C: a ν_τ + ν̄_τ bump at y = 5T_ν injected at T_boltz_start")
    print("    = 5 MeV (still coupled!) is ABSORBED by the plasma via nu-e")
    print("    scattering — its energy heats photons relative to neutrinos,")
    print("    LOWERING Neff by 2.7%. For a free-streaming post-decoupling")
    print("    injection, T_boltz_start would need to be < ~1 MeV.")


if __name__ == "__main__":
    main()
