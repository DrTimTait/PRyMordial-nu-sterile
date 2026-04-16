# -*- coding: utf-8 -*-
"""
Stage B validation: Dodelson-Widrow (non-resonant) sterile production.

Three runs:
  1. sterile_flag=False                                       (3×3 reference)
  2. sterile_flag=True,  θ_14 = 0                             (Stage A invariant)
  3. sterile_flag=True,  sin²(2θ_14) = 0.1,  Δm²_41 = 1 eV²   (DW production)

Run #3 exercises the quasi-static Sigl-Raffelt rate added in Stage B:

    Γ_DW = 2 |H_αs|² D / (D² + ω²)

applied independently per momentum mode. For the parameters above the
active-sterile coupling is large enough that the sterile thermalizes
efficiently well above BBN, giving ΔNeff ≈ +0.93 (near the +1 asymptote
of complete thermalization).

Runtime: ~7 minutes (three ~130 s QKE runs).

Reference: Dolgov+2002 (hep-ph/0201287), "Cosmological bounds on neutrino
degeneracy and oscillations".
"""
import os
import sys
import time
import importlib

import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

import PRyM.PRyM_init as PRyMini  # noqa: E402


def _configure():
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.massive_electron_flag = False


def _run(label):
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_rho_ss = 0.0
    if hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None:
        rho = c._boltz_rho_final
        if rho.shape[1] >= 4:
            sum_rho_ss = float(rho[:, 3, :].sum())
    print(f"{label:32s} Neff={res[0]:.6f}  Yp={res[4]:.6f}  "
          f"D/H={res[5]:.4f}  Σρ_ss={sum_rho_ss:.4e}  "
          f"({dt:.1f}s)", flush=True)
    return res, sum_rho_ss


def main():
    _configure()

    # 1. 3×3 reference
    PRyMini.sterile_flag = False
    res0, _ = _run("3×3 QKE reference")

    # 2. 4×4 decoupled — Stage A invariant
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    res1, _ = _run("4×4 decoupled (θ_14=0)")

    # 3. DW production — large mixing, eV-scale sterile
    sin2_2t = 0.1
    PRyMini.theta_14 = np.arcsin(np.sqrt(sin2_2t)) / 2.0
    print(f"θ_14 = {PRyMini.theta_14:.4f} rad, "
          f"sin²(2θ_14) = {np.sin(2.0 * PRyMini.theta_14)**2:.4f}")
    res2, rho_ss = _run("4×4 DW (sin²(2θ)=0.1, Δm²=1)")

    print()
    print("Stage A invariant (3×3 vs 4×4 decoupled):")
    print(f"  ΔNeff = {(res1[0] - res0[0]) * 1e5:+.2f} × 10⁻⁵")
    print(f"  ΔYp   = {(res1[4] - res0[4]) * 1e5:+.2f} × 10⁻⁵")
    print(f"  ΔD/H  = {(res1[5] - res0[5]) * 1e3:+.2f} × 10⁻³")

    print("Stage B DW signal (3×3 vs 4×4 DW):")
    print(f"  ΔNeff = {(res2[0] - res0[0]) * 1e3:+.3f} × 10⁻³  "
          f"({(res2[0] - res0[0]) / res0[0] * 100:+.2f} %)")
    print(f"  ΔYp   = {(res2[4] - res0[4]) * 1e3:+.3f} × 10⁻³")
    print(f"  sterile occupation grew from 0 to Σρ_ss = {rho_ss:.4e}")


if __name__ == "__main__":
    main()
