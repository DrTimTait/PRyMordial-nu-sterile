"""Stage E.2 sprint 18 closure run: Run C config + no Phase 0 + production n_B.

Confirms that the sprint-18 closure verdict (δNeff_ss = 0.0304 in HTT
2012 band [0.02, 0.10]) survives n_B resolution doubling. Sprint 16
showed Σρ_ss is n_B-sensitive (production n_B = 10000+2500 gave
Σρ_ss=22.225, reduced n_B = 2500+1000 gave Σρ_ss=14.512).

Configuration: same as `diag_sprint18_phase0_handoff.py` (Run C config,
qke_phase0_flag=False) but with n_B=12000 to match original Phase-B
step density × extended T range factor (100→0.005 MeV span vs 30→0.005
MeV original).

If δNeff_ss stays within [0.02, 0.10] at production n_B, Stage E.2
closes formally. If it drifts above the band, sprint-19 needs to
investigate the n_B convergence behavior of the no-Phase-0 driver.

Cost: ~35 min (ETDRK2 at n_B=12000 over [100, 0.005] MeV).
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


# Reduced-n_B reference (sprint-18 Phase-0 diagnostic)
NO_PHASE0_REDUCED = {"sum_raw": 4.4416, "delta_neff_ss": 0.0304,
                     "Neff": 3.9090, "Yp": 0.24850, "n_B": 3500}


def _set_flags():
    """Run C config (HTT damping, project-default V_nunu) with no Phase 0 at production n_B."""
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
    PRyMini.qke_etdrk4_flag = False
    PRyMini.qke_ode_etdrk2_flag = True
    PRyMini.qke_lsoda_driver_flag = False
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    # Production n_B for the extended Phase B range:
    # original 30→0.005 MeV used n_B=10000 (~1150 steps/decade);
    # new 100→0.005 MeV is 1.14× longer in log a, so n_B = 12000.
    PRyMini.n_B_override = 12000
    PRyMini.n_B_phase0_override = 0


def _delta_neff_ss_from_rho(rho_final):
    if rho_final is None or rho_final.shape[1] < 4:
        return None
    Ny = rho_final.shape[2]
    y_max = PRyMini.y_max_boltz
    dy = y_max / Ny
    y_grid = np.linspace(dy / 2.0, y_max - dy / 2.0, Ny)
    y3 = y_grid ** 3
    rho_ss = rho_final[:, 3, :]
    rho_e = rho_final[:, 0, :]
    m3_ss = float((rho_ss * y3[None, :]).sum())
    m3_e = float((rho_e * y3[None, :]).sum())
    if m3_e <= 0.0:
        return None
    return m3_ss / m3_e


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 18 closure run: production n_B no-Phase-0 confirmation",
        f"  Config: damping='symmetric', active_only=True, qke_phase0_flag=False",
        f"  Phase B from T=100 MeV to T=0.005 MeV at n_B={12000}",
        f"  Reduced-n_B reference (sprint-18 Phase-0 diagnostic, n_B={NO_PHASE0_REDUCED['n_B']}):",
        f"    Σρ_ss(raw)={NO_PHASE0_REDUCED['sum_raw']:.4f}  δNeff_ss={NO_PHASE0_REDUCED['delta_neff_ss']:.4f}  "
        f"Neff={NO_PHASE0_REDUCED['Neff']:.4f}  Yp={NO_PHASE0_REDUCED['Yp']:.5f}",
        f"  HTT 2012 reference band: δNeff in [0.02, 0.10]",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    _set_flags()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    print("\n--- Production-n_B no-Phase-0 run ---", flush=True)
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt_run = time.time() - t0

    Neff = float(res[0])
    Yp = float(res[4])
    sum_ss = 0.0
    delta_neff_ss = None
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)

    summary = [
        "",
        "=" * 110,
        "Sprint 18 closure run result",
        "=" * 110,
        f"  Production n_B=12000: Σρ_ss(raw)={sum_ss:.4f}  "
        f"δNeff_ss={delta_neff_ss:.4f}  Neff={Neff:.4f}  Yp={Yp:.5f}  ({dt_run:.0f}s)",
        f"  Reduced n_B={NO_PHASE0_REDUCED['n_B']}: Σρ_ss(raw)={NO_PHASE0_REDUCED['sum_raw']:.4f}  "
        f"δNeff_ss={NO_PHASE0_REDUCED['delta_neff_ss']:.4f}  "
        f"Neff={NO_PHASE0_REDUCED['Neff']:.4f}  Yp={NO_PHASE0_REDUCED['Yp']:.5f}",
        "",
    ]

    if delta_neff_ss is None:
        summary.append("  VERDICT: δNeff_ss could not be computed; harness output is incomplete.")
    elif 0.02 <= delta_neff_ss <= 0.10:
        summary.append(
            f"  VERDICT: δNeff_ss = {delta_neff_ss:.4f} STILL IN HTT 2012 band [0.02, 0.10] at "
            "production n_B. **STAGE E.2 CLOSED.** The combined cure is robust to n_B "
            "resolution. Recommended actions:")
        summary.append(
            "    1. Re-run all four Hannestad benchmark points (sin²2θ ∈ {0.1, 2.26e-3, "
            "1e-4, 0.089}) under the closure config to confirm coverage of HTT 2012 "
            "Fig. 2 top panel.")
        summary.append(
            "    2. Document the closure config in PRyM_init.py (line 165-200 for damping; "
            "Phase-0 flag) and propose default flips for Hannestad-style benchmarks.")
        summary.append(
            "    3. Investigate the V_nunu paradox (sprint-17 Run B near-SM Neff/Yp under "
            "active_only=False) at the structural level — it is no longer a Stage E.2 "
            "blocker but remains an unexplained behavior worth characterising.")
    elif delta_neff_ss < 0.02:
        summary.append(
            f"  VERDICT: δNeff_ss = {delta_neff_ss:.4f} BELOW HTT 2012 band at production n_B. "
            "Higher n_B over-suppresses sterile production. Sprint 19 should investigate "
            "n_B convergence behavior to find the right operating point.")
    else:
        summary.append(
            f"  VERDICT: δNeff_ss = {delta_neff_ss:.4f} OUTSIDE HTT 2012 band at production n_B "
            f"(reduced-n_B value was {NO_PHASE0_REDUCED['delta_neff_ss']:.4f}, in band). The closure is "
            "n_B-resolution-dependent. Sprint 19 needs to investigate n_B convergence and "
            "may require a momentum-grid axis bracket as well.")
    summary.append("")
    summary.append(f"  Total wall-clock: {dt_run:.0f}s ({dt_run/60:.1f} min)")
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint18_no_phase0_production.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
