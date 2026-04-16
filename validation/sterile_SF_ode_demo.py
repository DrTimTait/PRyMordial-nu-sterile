"""Stage D.3 validation: Shi-Fuller MSW resonance under the full-ODE
QKE driver (qke_full_ode_flag=True).

Parallels validation/sterile_SF_demo.py but runs each scenario through
BOTH the Strang path (evolve_step) and the ODE path (evolve_step_ode)
to compare.

Physics motivation: SF production is resonant, driven by a matter
potential that cancels the vacuum ω at some T_res. Near resonance the
in-medium mixing angle sweeps rapidly through θ_m = π/4 and Stage B's
quasi-static DW rate Γ_DW = 2|H_αs|²D/(D²+ω²) is most stressed (the ω→0
limit gives Γ_DW → 2|H_αs|²/D, which the quasi-static formula handles
but the underlying off-diagonal ODE resolves explicitly via exp-Euler).

The ODE driver captures the full unitary adiabatic level-crossing
dynamics per mode, without any explicit DW block. Whether it produces
MORE or LESS sterile than the quasi-static path at SF parameters is the
main physical question.

Scenarios (mirroring sterile_SF_demo.py):
  1. 3×3 reference (no sterile)
  2. DW baseline: sin²(2θ_14)=1e-3, ξ_νe = 0
  3. SF small:    sin²(2θ_14)=1e-3, ξ_νe = 1e-2
  4. SF large:    sin²(2θ_14)=1e-3, ξ_νe = 5e-2

Each scenario runs twice: Strang + ODE. Metrics: Neff, Yp, D/H, Σρ_ss,
integrated |n_νe - n_ν̄e| residual at end of BBN.

Runtime: ~18 minutes wall (8 runs × ~130 s).
"""
import os
import sys
import time
import importlib

import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

import PRyM.PRyM_init as PRyMini


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
    PRyMini.qke_full_ode_flag = False
    PRyMini.n_B_override = None
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0


def _metrics(c):
    """Return (sum_rho_ss, integrated |n_νe − n_ν̄e|) from a completed run."""
    sum_ss = 0.0
    n_asym = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
        if (hasattr(c, "_boltz_solver") and c._boltz_solver is not None
                and hasattr(c._boltz_solver, "y_grid")):
            y_grid = np.asarray(c._boltz_solver.y_grid)
            w = y_grid**2
            n_asym = float((w * (rho[0, 0] - rho[1, 0])).sum())
    return sum_ss, n_asym


def _run(label, driver, sterile, sin2_2theta, xi_nue):
    _configure()
    PRyMini.qke_full_ode_flag = (driver == "ODE")
    PRyMini.sterile_flag = sterile
    if sterile:
        PRyMini.theta_14 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0 if sin2_2theta > 0 else 0.0
        PRyMini.xi_nue_init = xi_nue
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss, n_asym = _metrics(c)
    print(f"{label:44s} Neff={res[0]:.4f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  Σρ_ss={sum_ss:6.3f}  "
          f"n_ξe={n_asym:+.2e}  ({dt:.1f}s)", flush=True)
    return {"label": label, "driver": driver, "res": res,
            "sum_ss": sum_ss, "n_asym": n_asym}


def main():
    print("=" * 88)
    print("Stage D.3 validation: Shi-Fuller under full-ODE QKE driver")
    print("=" * 88)
    print()

    results = []

    # ---- 3×3 references ----
    results.append(_run("Strang  3×3 reference",                "Strang", False, 0.0, 0.0))
    results.append(_run("ODE     3×3 reference",                "ODE",    False, 0.0, 0.0))

    # ---- DW baseline (ξ=0) ----
    results.append(_run("Strang  DW  (sin²=1e-3, ξ=0)",         "Strang", True, 1e-3, 0.0))
    results.append(_run("ODE     DW  (sin²=1e-3, ξ=0)",         "ODE",    True, 1e-3, 0.0))

    # ---- SF small asymmetry ----
    results.append(_run("Strang  SF  (sin²=1e-3, ξ_νe=1e-2)",   "Strang", True, 1e-3, 1.0e-2))
    results.append(_run("ODE     SF  (sin²=1e-3, ξ_νe=1e-2)",   "ODE",    True, 1e-3, 1.0e-2))

    # ---- SF large asymmetry (known visible depletion) ----
    results.append(_run("Strang  SF  (sin²=1e-3, ξ_νe=5e-2)",   "Strang", True, 1e-3, 5.0e-2))
    results.append(_run("ODE     SF  (sin²=1e-3, ξ_νe=5e-2)",   "ODE",    True, 1e-3, 5.0e-2))

    # -------------------- Interpretation --------------------
    print()
    print("-" * 88)
    print("Paired Δ(ODE − Strang) per scenario:")
    print("-" * 88)
    for i in range(0, len(results), 2):
        strang = results[i]
        ode = results[i + 1]
        scenario = strang["label"].split("Strang", 1)[1].strip()
        dNeff = ode["res"][0] - strang["res"][0]
        dss = ode["sum_ss"] - strang["sum_ss"]
        dnasym = ode["n_asym"] - strang["n_asym"]
        print(f"  {scenario:40s}  ΔNeff={dNeff:+.4f}  "
              f"ΔΣρ_ss={dss:+.3f}  Δn_ξe={dnasym:+.2e}")

    # Baseline bias (3×3) already established at ~-4e-3 on Neff
    # (validation/stage_d_convergence.py). Isolate the *sterile-specific*
    # shift by subtracting the 3×3 bias from each paired ΔNeff:
    print()
    print("-" * 88)
    print("Sterile-specific ΔNeff (bias-corrected: ODE_sterile − Strang_sterile − (ODE_3×3 − Strang_3×3)):")
    print("-" * 88)
    dNeff_3x3 = results[1]["res"][0] - results[0]["res"][0]
    for i in range(2, len(results), 2):
        strang = results[i]
        ode = results[i + 1]
        scenario = strang["label"].split("Strang", 1)[1].strip()
        dNeff_raw = ode["res"][0] - strang["res"][0]
        dNeff_corr = dNeff_raw - dNeff_3x3
        print(f"  {scenario:40s}  Δ_sterile = {dNeff_corr:+.4f}")

    # Asymmetry depletion ratio: how much of the initial ξ was drained
    # into sterile? A ratio near 0 = full depletion, near 1 = no depletion.
    print()
    print("-" * 88)
    print("Asymmetry depletion (final |n_ξe| / reference):")
    print("-" * 88)
    # Use the Strang 3×3 reference's |n_asym| as a rough zero (should be O(1e-4) or less).
    # What matters is the ratio between SF-initial and SF-final magnitudes.
    for i in range(4, len(results), 2):
        strang = results[i]
        ode = results[i + 1]
        scenario = strang["label"].split("Strang", 1)[1].strip()
        # Rough "initial asymmetry at Phase B start" scales with xi * integrated_FD
        # Use the known Stage C result for comparison
        xi = float(scenario.split("ξ_νe=", 1)[1].rstrip(")"))
        # Asymmetry fraction remaining:
        print(f"  {scenario:40s}  |n_ξe| Strang={abs(strang['n_asym']):.2e}  "
              f"|n_ξe| ODE={abs(ode['n_asym']):.2e}")


if __name__ == "__main__":
    main()
