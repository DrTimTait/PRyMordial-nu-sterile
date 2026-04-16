# -*- coding: utf-8 -*-
"""
Stage A invariant: `sterile_flag=True` with θ_14=θ_24=θ_34=0 reproduces the
3-flavor QKE result.

This is the most important sanity check for the sterile infrastructure:
when the sterile is decoupled (all mixing angles zero), the 4×4 density
matrix evolves as the 3×3 active block embedded in an empty sterile
row/column. The two solvers must agree on Neff, Yp, D/H to within floating-
point noise (dominant observable drift expected < 10⁻⁴ on Neff,
< 1% on D/H).

Runtime: ~4 minutes (two ~130 s QKE runs).
"""
import os
import sys
import time
import importlib

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
    res = PRyMmain.PRyMclass().PRyMresults()
    dt = time.time() - t0
    print(f"{label:30s} Neff={res[0]:.6f}  Yp={res[4]:.6f}  "
          f"D/H={res[5]:.4f}  ({dt:.1f}s)", flush=True)
    return res


def main():
    _configure()

    # Reference: 3-flavor QKE
    PRyMini.sterile_flag = False
    res_3x3 = _run("3×3 QKE reference")

    # 4×4 decoupled: sterile_flag on, all mixing angles zero
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    res_4x4 = _run("4×4 decoupled (θ_14=0)")

    dNeff = res_4x4[0] - res_3x3[0]
    dYp = res_4x4[4] - res_3x3[4]
    dDoH = res_4x4[5] - res_3x3[5]

    print()
    print(f"Invariant check (4×4_decoupled − 3×3):")
    print(f"  ΔNeff = {dNeff*1e5:+.2f} × 10⁻⁵")
    print(f"  ΔYp   = {dYp*1e5:+.2f} × 10⁻⁵")
    print(f"  ΔD/H  = {dDoH*1e3:+.2f} × 10⁻³")

    ok = (abs(dNeff) < 3.0e-4 and abs(dYp / max(res_3x3[4], 1e-30)) < 1.0e-3
          and abs(dDoH / max(res_3x3[5], 1e-30)) < 1.0e-2)
    print("  Stage A invariant:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
