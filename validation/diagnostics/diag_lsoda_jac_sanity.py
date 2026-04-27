"""Stage E.2 sprint 15 gates 3+4: LSODA-with-analytic-jac sanity runs.

Two checks before pulling the trigger on gate 5 (Hannestad Point C):

  Gate 3 — 3-flavor SM (no sterile). LSODA must reproduce ETDRK2's
  Neff to <1e-4 and Yp to <1e-5. Baseline: test_mode5b_qke_full_ode
  reference values (Neff=3.041, Yp=0.2485).

  Gate 4 — 4-flavor decoupled sterile (all mixing angles zero). LSODA
  must keep Σρ_ss < 1e-12 (sterile must stay empty). Mirrors
  test_sterile_stage_a_invariant's cfg_4x4_decoupled.

The LSODA driver runs with the new analytic Jacobian
(qke_lsoda_analytic_jac_flag=True default) and rtol=1e-6, atol=1e-10
(production-grade tolerances). Both runs should land in <30 min.
"""
import os
import sys
import time
import importlib

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)


def _base_flags():
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
    PRyMini.qke_lsoda_driver_flag = True
    PRyMini.qke_lsoda_analytic_jac_flag = True
    PRyMini.qke_lsoda_rtol = 1.0e-6
    PRyMini.qke_lsoda_atol = 1.0e-10
    # Reset sterile params so each test sets explicitly.
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.delta_14 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0


def _run_pipeline():
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
    return res[0], res[4], res[5], sum_ss  # Neff, Yp, D/H, Σρ_ss


def gate3_3x3_sm():
    """LSODA 3×3 SM. ETDRK2 baseline: Neff≈3.041, Yp≈0.2485, D/H≈2.47."""
    import PRyM.PRyM_init as PRyMini
    _base_flags()
    PRyMini.sterile_flag = False
    t0 = time.time()
    Neff, Yp, DoH, sum_ss = _run_pipeline()
    dt = time.time() - t0
    return {"label": "3×3 SM (LSODA + analytic jac)",
            "Neff": Neff, "Yp": Yp, "DoH": DoH, "Σρ_ss": sum_ss,
            "wall_s": dt}


def gate4_4x4_decoupled():
    """LSODA 4×4 decoupled sterile (all mixing angles zero). Σρ_ss < 1e-12."""
    import PRyM.PRyM_init as PRyMini
    _base_flags()
    PRyMini.sterile_flag = True
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    t0 = time.time()
    Neff, Yp, DoH, sum_ss = _run_pipeline()
    dt = time.time() - t0
    return {"label": "4×4 decoupled (LSODA + analytic jac)",
            "Neff": Neff, "Yp": Yp, "DoH": DoH, "Σρ_ss": sum_ss,
            "wall_s": dt}


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("Stage E.2 sprint 15 gates 3+4: LSODA-with-analytic-jac sanity", flush=True)
    print("=" * 80, flush=True)
    print("ETDRK2 reference (test_mode5b_qke_full_ode): "
          "Neff=3.041, Yp=0.2485, D/H=2.47", flush=True)
    print("Gate 3 target: |Neff-3.041|<1e-3, |Yp-0.2485|<1e-4 — and finishes.",
          flush=True)
    print("Gate 4 target: Σρ_ss < 1e-12 — and finishes.", flush=True)
    print("", flush=True)

    g3 = gate3_3x3_sm()
    print(f"  Gate 3 — {g3['label']}", flush=True)
    print(f"    Neff = {g3['Neff']:.5f}    Yp = {g3['Yp']:.5f}    "
          f"D/H = {g3['DoH']:.4f}    ({g3['wall_s']:.0f} s)", flush=True)
    g3_pass = (abs(g3["Neff"] - 3.041) < 3.0e-3 and
               abs(g3["Yp"] - 0.2485) < 3.0e-4)
    print(f"    Gate 3: {'PASS' if g3_pass else 'FAIL'}", flush=True)
    print("", flush=True)

    g4 = gate4_4x4_decoupled()
    print(f"  Gate 4 — {g4['label']}", flush=True)
    print(f"    Neff = {g4['Neff']:.5f}    Yp = {g4['Yp']:.5f}    "
          f"D/H = {g4['DoH']:.4f}    Σρ_ss = {g4['Σρ_ss']:.3e}    "
          f"({g4['wall_s']:.0f} s)", flush=True)
    g4_pass = g4["Σρ_ss"] < 1.0e-12
    print(f"    Gate 4: {'PASS' if g4_pass else 'FAIL'}", flush=True)
    print("", flush=True)

    out_path = os.path.join(_WT,
        "validation/diagnostics/diag_lsoda_jac_sanity.out")
    with open(out_path, "w") as fh:
        fh.write(f"Gate 3 (3×3 SM):    Neff={g3['Neff']:.5f}  "
                 f"Yp={g3['Yp']:.5f}  D/H={g3['DoH']:.4f}  "
                 f"({g3['wall_s']:.0f}s)  "
                 f"{'PASS' if g3_pass else 'FAIL'}\n")
        fh.write(f"Gate 4 (4×4 dec.):  Neff={g4['Neff']:.5f}  "
                 f"Yp={g4['Yp']:.5f}  D/H={g4['DoH']:.4f}  "
                 f"Σρ_ss={g4['Σρ_ss']:.3e}  "
                 f"({g4['wall_s']:.0f}s)  "
                 f"{'PASS' if g4_pass else 'FAIL'}\n")
    sys.exit(0 if (g3_pass and g4_pass) else 1)
