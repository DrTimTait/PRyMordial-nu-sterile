"""Stage E.2 sprint 18 Phase-0/Phase-B handoff diagnostic.

Tests whether the sprint-12-localised Tg ≈ 60-64 MeV "resonance jump" is
a physical effect or a numerical artefact of the Phase-0 → Phase-B
segment handoff at T = 30 MeV (or the Phase A → Phase 0 handoff at
T = 100 MeV).

HTT 2012 Appendix A states no resonance exists for L=0 NH (the L=0
Hannestad Point C configuration). If the sprint-12 jump persists when
the Phase 0 segment is disabled and Phase B runs all the way from
T = 100 MeV down to T = 0.005 MeV, the jump is internal to the unified
QKE driver. If the jump disappears, the segment-handoff is the cause
and sprint 19 needs a single-pass (Phase 0 + Phase B fused) driver.

Configuration: same as sprint-17 Run C (HTT-matching damping, project-
default V_nunu projection) but with `qke_phase0_flag = False` and
`T_boltz_start = 100 MeV` so Phase B picks up directly from where
Phase A leaves off.

Cost: ~30 min (Phase B with extended range needs ~3500 outer steps to
match the resolution of the previous 1000+2500 segment chain).

Decision tree (sprint-18 Phase 2):

    δNeff_ss(no-Phase-0) ≈ Run C δNeff_ss=0.331 → Phase 0 is a non-effect;
                                                    handoff is not the cause
                                                    of the sprint-12 jump.
    δNeff_ss(no-Phase-0) substantially different → Phase 0 / Phase B
                                                    handoff IS material;
                                                    sprint 19 needs a
                                                    single-pass driver
                                                    redesign.
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


# Sprint-17 Run C reference (active_only=True, symmetric damping, Phase 0 ON)
SPRINT17_RUN_C = {"sum_raw": 7.1729, "delta_neff_ss": 0.3312,
                  "Neff": 4.1040, "Yp": 0.26131}


def _set_flags():
    """Sprint-17 Run C config with Phase 0 disabled."""
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
    # HTT-matching damping; project default V_nunu projection (Run C config)
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    # Phase 0 OFF: Phase B runs directly from 100 MeV down
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    # n_B for the extended Phase B range — match the previous total
    # (1000 Phase 0 + 2500 Phase B = 3500 steps over [100, 0.005] MeV).
    PRyMini.n_B_override = 3500
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
        "Stage E.2 sprint 18 Phase-0/Phase-B handoff diagnostic",
        f"  Run C config (active_only=True, damping='symmetric') with qke_phase0_flag=False",
        f"  Phase B runs from T = 100 MeV directly to T = 0.005 MeV (n_B = 3500)",
        f"  Sprint-17 Run C reference (Phase 0 ON):",
        f"    Σρ_ss(raw)={SPRINT17_RUN_C['sum_raw']:.4f}  δNeff_ss={SPRINT17_RUN_C['delta_neff_ss']:.4f}  "
        f"Neff={SPRINT17_RUN_C['Neff']:.4f}  Yp={SPRINT17_RUN_C['Yp']:.5f}",
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

    print("", flush=True)
    print("--- No-Phase-0 run: Run C config with qke_phase0_flag=False ---", flush=True)
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
        "Sprint 18 Phase-0/Phase-B handoff diagnostic result",
        "=" * 110,
        f"  No-Phase-0 (Run C config): Σρ_ss(raw)={sum_ss:.4f}  "
        f"δNeff_ss={delta_neff_ss:.4f}  Neff={Neff:.4f}  Yp={Yp:.5f}  ({dt_run:.0f}s)",
        f"  Phase-0 ON (sprint-17 Run C): Σρ_ss(raw)={SPRINT17_RUN_C['sum_raw']:.4f}  "
        f"δNeff_ss={SPRINT17_RUN_C['delta_neff_ss']:.4f}  "
        f"Neff={SPRINT17_RUN_C['Neff']:.4f}  Yp={SPRINT17_RUN_C['Yp']:.5f}",
        "",
    ]

    if delta_neff_ss is None:
        summary.append("  VERDICT: δNeff_ss could not be computed; harness output is incomplete.")
    else:
        ratio = delta_neff_ss / SPRINT17_RUN_C['delta_neff_ss']
        rel_diff = abs(ratio - 1.0) * 100.0
        if rel_diff < 10.0:
            summary.append(
                f"  VERDICT: δNeff_ss matches Run C within {rel_diff:.1f}% — Phase 0 segment "
                "is a NON-EFFECT. The sprint-12 Tg ≈ 60-64 MeV jump is internal to the "
                "unified QKE driver, not a Phase-0/Phase-B handoff artefact. Sprint 19 "
                "can deprioritise the single-pass driver and focus on the V_nunu "
                "closed-form V_1 implementation as the primary structural axis.")
        elif rel_diff > 30.0:
            summary.append(
                f"  VERDICT: δNeff_ss differs from Run C by {rel_diff:.1f}% — Phase 0 segment "
                "is MATERIAL. The handoff at T = 30 MeV (Phase 0 → Phase B) introduces "
                "a numerical discontinuity that affects the sterile saturation. Sprint 19 "
                "needs a single-pass driver design (Phase 0 + Phase B fused into one "
                "ETDRK2 segment) before any further physics-config bracket.")
        else:
            summary.append(
                f"  VERDICT: δNeff_ss differs from Run C by {rel_diff:.1f}% (between 10% and 30%). "
                "Phase 0 is sub-leading but not negligible. Sprint 19 should consider both "
                "the single-pass driver and the V_1 closed-form V_nunu in parallel.")
    summary.append("")
    summary.append(f"  Total wall-clock: {dt_run:.0f}s ({dt_run/60:.1f} min)")
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint18_phase0_handoff.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
