"""Sterile DW literature comparison: Hannestad, Tamborra, Tram 2012.

Reference: arXiv:1204.5861 (JCAP 07 (2012) 025), "Thermalisation of light
sterile neutrinos in the early universe".

Compares PRyMordial-nu's D.7.1 ETDRK2 ODE driver to the full-QKE results
of Hannestad+2012 for Dodelson-Widrow (non-resonant, L=0) sterile neutrino
production. Three benchmark points are read off their Fig. 2 top panel
(fixed δm² = 0.93 eV², varying sin²2θ):

    Point                          δm² [eV²]   sin²2θ     Hannestad δNeff
    A  Full thermalisation          0.93        1.0e-1     ≈ 1.0
    B  Partial thermalisation       0.93        2.26e-3    ≈ 0.50
    C  Minimal thermalisation       0.93        1.0e-4     ≈ 0.04

Mixing is with ν_μ (theta_24) rather than ν_e (theta_14), matching
Hannestad's default setup where "active" gets only neutral-current matter
potential — they note explicitly that ν_e mixing gives a slightly smaller
δNeff=1 region because of the extra V_CC contribution.

Caveats
-------
- Hannestad uses a 1-active + 1-sterile 2x2 QKE; we use the full 4x4
  with active-active mixing present. The two should agree for small
  mixing angles where active-sterile conversion is rate-limiting.
- Hannestad evolves from T = 60 MeV to T = 1 MeV (BBN start) and reports
  δNeff there. PRyMordial evolves further (through BBN and into the
  weak-rate freeze-out) and reports Neff at CMB decoupling. After T=1 MeV
  the sterile is decoupled and comoving occupation is preserved, so the
  two Neff definitions differ only by the O(10^-3) thermodynamic
  conversion factor already baked into our 3x3 reference.
- Agreement at the 10-20% level is expected given the conventions
  differences; anything tighter is a pleasant surprise.

Runtime: ~45-60 min at default n_B (four QKE runs: 3x3 reference + 3
sterile points, each ~10-15 min under D.7.1).
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
    PRyMini.qke_ode_etdrk2_flag = True      # D.7.1 Strang-symmetric driver
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = None
    # Sterile parameters reset to defaults before each run
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
    print(f"  {label:48s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  Σρ_ss={sum_ss:.3f}  ({dt:.0f}s)", flush=True)
    return Neff, sum_ss


def configure_3x3():
    PRyMini.sterile_flag = False


def configure_point(sin2_2theta, delta_m2):
    def _c():
        PRyMini.sterile_flag = True
        PRyMini.Dm2_41 = delta_m2
        # Mix nu_mu with nu_s via theta_24
        PRyMini.theta_24 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0
    return _c


print("=" * 88)
print("Sterile DW literature comparison: Hannestad, Tamborra, Tram 2012")
print("arXiv:1204.5861  Fig. 2 top panel (fixed δm² = 0.93 eV², L=0)")
print("D.7.1 ODE driver, θ_24 mixing (ν_μ-ν_s; matches Hannestad's default)")
print("=" * 88)

Neff_3x3, _ = _run("3×3 QKE reference (no sterile)", configure_3x3)

print()
bench = [
    ("A  full thermalisation",      1.0e-1, 0.93, 1.00),
    ("B  partial thermalisation",   2.26e-3, 0.93, 0.50),
    ("C  minimal thermalisation",   1.0e-4, 0.93, 0.04),
]

results = []
for label, sin2_2t, dm2, hannestad_dNeff in bench:
    Neff, sum_ss = _run(f"DW ({label}, sin²2θ={sin2_2t:.2e}, δm²={dm2})",
                        configure_point(sin2_2t, dm2))
    dNeff_ours = Neff - Neff_3x3
    results.append((label, sin2_2t, dm2, hannestad_dNeff, dNeff_ours, sum_ss))

print()
print("-" * 88)
print(f"{'Point':30s}  {'sin²2θ':>10s}  {'δm²':>6s}  {'δNeff (H+2012)':>14s}  "
      f"{'δNeff (ours)':>14s}  {'Δ (abs)':>9s}  {'Δ/H':>7s}")
print("-" * 88)
for label, sin2_2t, dm2, h_dNeff, ours_dNeff, sum_ss in results:
    abs_diff = ours_dNeff - h_dNeff
    rel_diff = abs_diff / max(h_dNeff, 1e-6)
    print(f"{label:30s}  {sin2_2t:10.2e}  {dm2:6.3f}  {h_dNeff:14.3f}  "
          f"{ours_dNeff:14.3f}  {abs_diff:+9.3f}  {rel_diff:+7.1%}")

print()
print("Baseline: 3×3 reference Neff =", f"{Neff_3x3:.5f}")
print()
print("Interpretation guide:")
print("  - Agreement at 10-20% relative (|Δ/H| < 0.2) is the validation target;")
print("    Hannestad's own numerical conventions differ from PRyMordial at that")
print("    level, so tighter is a pleasant surprise.")
print("  - Absolute difference |Δ| > 0.1 on the fully-thermalised point (A) would")
print("    suggest a thermodynamics-baseline bug, not a sterile-production one.")
print("  - The low-thermalisation point (C) has small absolute values so relative")
print("    differences are easier to inflate; report |Δ (abs)| too.")
