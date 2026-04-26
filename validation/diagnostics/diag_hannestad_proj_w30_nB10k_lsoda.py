"""Stage E.2 sprint 14 gate 5: Hannestad A/B/C through the LSODA driver.

Parallel to diag_hannestad_proj_w30_nB10k.py, identical configuration except
that PRyMini.qke_lsoda_driver_flag is set True so the QKE per-outer-step
dispatcher routes through DensityMatrixSolver.evolve_step_lsoda instead of
evolve_step_ode_etdrk2. The ETDRK2 baseline output at
diag_hannestad_proj_w30_nB10k.out is the side-by-side reference; this run
writes to diag_hannestad_proj_w30_nB10k_lsoda.out.

Decision tree (sprint-14 brief):
  Σρ_ss(P0 exit) ≈ [0.02, 0.10] for Point C → Suspect 7 confirmed (Strang
    split is the bottleneck); ship LSODA as production.
  Σρ_ss(P0 exit) ≈ 0.65 for Point C (matches ETDRK2)        → Suspect 8
    confirmed (Hannestad target wrong); escalate to literature re-read.

Sprint-8 targets (Hannestad+2012 bands), unchanged:
  A (sin²2θ=1e-1):     [0.9, 1.1]
  B (sin²2θ=2.26e-3):  [0.3, 0.7]
  C (sin²2θ=1e-4):     [0.02, 0.1]
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
    # Sprint-14: route the QKE dispatcher through evolve_step_lsoda. The
    # qke_ode_etdrk2_flag stays True so the dispatcher branch order falls
    # through to ETDRK2 if the LSODA flag is ever flipped off mid-debug.
    PRyMini.qke_lsoda_driver_flag = True
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = 10000
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 30.1 * PRyMini.MeV_to_Kelvin
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 2500


def _run(label, configure):
    _base_flags()
    configure()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)
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
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, res[4], res[5], sum_ss


def configure_3x3():
    PRyMini.sterile_flag = False


def configure_point(sin2_2theta, delta_m2):
    def _c():
        PRyMini.sterile_flag = True
        PRyMini.Dm2_41 = delta_m2
        PRyMini.theta_24 = np.arcsin(np.sqrt(sin2_2theta)) / 2.0
    return _c


if __name__ == "__main__":
    header = [
        "=" * 104,
        "Stage E.2 sprint 14 gate 5: Hannestad A/B/C @ w30 projection, n_B=10000 (LSODA driver)",
        "qke_lsoda_driver_flag = True, qke_v_nunu_active_only = True, T_boltz_start = 30 MeV.",
        "Targets: A [0.9, 1.1]  B [0.3, 0.7]  C [0.02, 0.1].",
        "=" * 104,
    ]
    for line in header:
        print(line, flush=True)

    Neff_3x3, Yp_3x3, DoH_3x3, _ = _run(
        "3x3 QKE reference (no sterile, 30 MeV window)", configure_3x3)
    print(flush=True)

    bench = [
        ("A  full thermalisation",    1.0e-1, 0.93, 1.00, (0.9, 1.1)),
        ("B  partial thermalisation", 2.26e-3, 0.93, 0.50, (0.3, 0.7)),
        ("C  minimal thermalisation", 1.0e-4, 0.93, 0.04, (0.02, 0.1)),
    ]

    rows = []
    for label, sin2_2t, dm2, hannestad_dNeff, band in bench:
        Neff, Yp, DoH, sum_ss = _run(
            f"DW ({label}, sin²2θ={sin2_2t:.2e}, δm²={dm2})",
            configure_point(sin2_2t, dm2))
        dNeff_ours = Neff - Neff_3x3
        in_band = band[0] <= dNeff_ours <= band[1]
        rows.append((label, sin2_2t, dm2, hannestad_dNeff, dNeff_ours, Yp,
                     sum_ss, band, in_band))

    summary = [""]
    summary.append("-" * 112)
    summary.append(f"{'Point':30s}  {'sin²2θ':>10s}  {'H+2012':>8s}  "
                   f"{'Ours @ n_B=10000':>18s}  {'Yp':>8s}  {'Σρss':>8s}  "
                   f"{'band':>14s}  {'verdict':>8s}")
    summary.append("-" * 112)
    for label, sin2_2t, dm2, h, d, Yp, s_ss, band, in_band in rows:
        summary.append(f"{label:30s}  {sin2_2t:10.2e}  {h:8.3f}  "
                       f"{d:18.4f}  {Yp:8.5f}  {s_ss:8.3f}  "
                       f"[{band[0]:.2f}, {band[1]:.2f}]  "
                       f"{'IN BAND' if in_band else 'MISS':>8s}")
    summary.append("")
    summary.append(f"3x3 reference Neff = {Neff_3x3:.5f}")
    n_in = sum(1 for r in rows if r[-1])
    if n_in == 3:
        summary.append("VERDICT: all three in Hannestad band -> Suspect 7 "
                       "confirmed; ship LSODA as production.")
    elif n_in == 0:
        summary.append("VERDICT: no points in band -> Suspect 8 "
                       "confirmed; escalate to literature re-read.")
    else:
        summary.append(f"VERDICT: {n_in}/3 in band -> partial; "
                       "compare line-by-line to ETDRK2 baseline.")

    text = "\n".join(summary)
    print(text, flush=True)
    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_hannestad_proj_w30_nB10k_lsoda.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        for label, sin2_2t, dm2, h, d, Yp, s_ss, band, in_band in rows:
            fh.write(f"  DW ({label}, sin²2θ={sin2_2t:.2e}, δm²={dm2}) "
                     f"Neff={d + Neff_3x3:.5f}  Yp={Yp:.5f}  Σρ_ss={s_ss:.3f}\n")
        fh.write(text + "\n")
