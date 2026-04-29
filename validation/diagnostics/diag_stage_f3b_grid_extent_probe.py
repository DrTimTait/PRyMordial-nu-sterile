"""Stage F sprint 3b: grid-extent vs grid-resolution disentanglement.

Sprint 3 confirmed FD-plateau truncation contributes to the Yp under-prediction
(Yp shifted +0.030 from y_max=100 to y_max=200 at Hannestad Point C, almost
2x the SM gap of +0.016). But sprint 3 held Ny_boltz=100 fixed, so dy doubled
from 1.0 to 2.0 MeV at y_max=200, and the resulting run broke both
active-sector (Neff=4.10 > 4.0 gate) and sterile-sector (delta_neff_ss=0.391
out of HTT band [0.02, 0.10]) closures.

Sprint 3b runs the same Hannestad Point C config at (y_max_boltz=200,
Ny_boltz=200) — keeping dy=1.0 MeV constant — to isolate the "grid-extent"
effect (FD-plateau resolution) from the "grid-resolution" effect (sterile
resonance features at moderate y).

Decision tree at completion:

| Result                                  | Verdict                                | Action |
|----------------------------------------|----------------------------------------|--------|
| All 4 gates PASS (Yp toward 0.247)     | Hypothesis 1 CLEAN CONFIRMED           | Recommend default flip to (y_max=200, Ny=200). |
| Yp recovers but Neff or delta_neff_ss  | Resolution-extent decomposition        | Sprint 3c: investigate which physics piece    |
|   still out of band                     |  partially clean                       |   is grid-resolution-sensitive at moderate y. |
| Yp stays at ~0.231                     | Hypothesis 1 clean-FALSIFIED           | Pivot to hypothesis 2 (post-Phase-B           |
|                                         |  (truncation effect was Ny-coupled)    |   re-thermalisation) or hypothesis 3          |
|                                         |                                        |   (n->p weak-rate QKE consumption).           |

Output:
  diag_stage_f3b_grid_extent_probe.npz  Single-run Neff, Yp, D/H,
                                         delta_neff_ss, sum_ss(raw), wall-clock.
                                         Plus combined comparison vs sprint 3
                                         P0 (y_max=100, Ny=100, dy=1.0) and
                                         P1 (y_max=200, Ny=100, dy=2.0).
  diag_stage_f3b_grid_extent_probe.out  Three-row comparison table + verdict.
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


SM_YP_REFERENCE = 0.247
ACTIVE_NEFF_MAX = 4.0
ACTIVE_YP_MAX = 0.255
HTT_STERILE_BAND = (0.02, 0.10)

# Sprint-3 references (from diag_stage_f3_y_max_probe.{out,npz}).
SPRINT3_P0 = {
    "label": "F3-P0 (y_max=100, Ny=100, dy=1.0)",
    "y_max_boltz": 100.0, "Ny_boltz": 100, "dy": 1.0,
    "Yp": 0.23126, "Neff": 1.834, "DH": 2.0737,
    "delta_neff_ss": 0.0947, "sum_ss_raw": 7.778,
    "wall_clock_s": 4448.0,
}
SPRINT3_P1 = {
    "label": "F3-P1 (y_max=200, Ny=100, dy=2.0)",
    "y_max_boltz": 200.0, "Ny_boltz": 100, "dy": 2.0,
    "Yp": 0.26132, "Neff": 4.1019, "DH": 2.8079,
    "delta_neff_ss": 0.3910, "sum_ss_raw": 14.184,
    "wall_clock_s": 3643.0,
}

# Sprint 3b: only the new run, with dy=1.0 like P0 but extent doubled like P1.
RUN = {
    "label": "F3b-P2 (y_max=200, Ny=200, dy=1.0)",
    "y_max_boltz": 200.0,
    "Ny_boltz": 200,
    "n_B": 12000,
}


def _set_flags(run):
    """Sprint-2 closure-config defaults at production n_B; Hannestad Point C."""
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
    # Closure-config defaults (now production after sprint 2; set explicitly here).
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phase0_flag = False
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(run["n_B"])
    PRyMini.n_B_phase0_override = 0
    PRyMini.qke_active_probe_flag = False
    PRyMini.qke_post_phaseB_trace_flag = False
    # Probe parameters: vary BOTH y_max and Ny in lockstep to hold dy=1.0.
    PRyMini.y_max_boltz = float(run["y_max_boltz"])
    PRyMini.Ny_boltz = int(run["Ny_boltz"])


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


def _run_one(run):
    print(f"\n--- {run['label']}: y_max_boltz={run['y_max_boltz']} MeV, "
          f"Ny={run['Ny_boltz']}, dy={run['y_max_boltz']/run['Ny_boltz']:.3f} MeV, "
          f"n_B={run['n_B']} ---", flush=True)
    _set_flags(run)
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt_run = time.time() - t0

    Neff = float(res[0])
    Yp = float(res[4])
    DH = float(res[5])
    sum_ss = 0.0
    delta_neff_ss = None
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  D/H={DH:.4f}  "
          f"sum_ss(raw)={sum_ss:.4f}  delta_Neff_ss={delta_neff_ss}  "
          f"({dt_run:.0f}s)", flush=True)

    return {
        "label": run["label"],
        "y_max_boltz": run["y_max_boltz"],
        "Ny_boltz": run["Ny_boltz"],
        "dy": run["y_max_boltz"] / run["Ny_boltz"],
        "n_B": int(run["n_B"]),
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "sum_ss_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "wall_clock_s": float(dt_run),
    }


def _all_gates_pass(r):
    """True iff Yp <= 0.255, Neff <= 4.0, delta_neff_ss in [0.02, 0.10]."""
    if not np.isfinite(r["delta_neff_ss"]):
        return False
    in_band = HTT_STERILE_BAND[0] <= r["delta_neff_ss"] <= HTT_STERILE_BAND[1]
    return (r["Yp"] <= ACTIVE_YP_MAX and r["Neff"] <= ACTIVE_NEFF_MAX
            and in_band)


def _verdict(p2):
    """Compare F3b-P2 against F3-P0 and F3-P1, decide hypothesis-1 status."""
    yp_gap_p0 = SM_YP_REFERENCE - SPRINT3_P0["Yp"]   # +0.0157
    yp_gap_p2 = SM_YP_REFERENCE - p2["Yp"]
    if abs(yp_gap_p0) > 1e-6:
        recovery = (SPRINT3_P0["Yp"] - p2["Yp"]) / (SPRINT3_P0["Yp"]
                                                     - SM_YP_REFERENCE)
    else:
        recovery = float("nan")

    gates_pass = _all_gates_pass(p2)
    yp_in_active_band = (abs(p2["Yp"] - SM_YP_REFERENCE) <= 0.01)
    sterile_in_band = (HTT_STERILE_BAND[0] <= p2["delta_neff_ss"]
                       <= HTT_STERILE_BAND[1])
    neff_in_band = (p2["Neff"] <= ACTIVE_NEFF_MAX)

    if gates_pass and yp_in_active_band:
        return ("HYPOTHESIS 1 CLEAN CONFIRMED",
                f"P2 (y_max=200, Ny=200, dy=1.0) recovers {recovery:.1%} of "
                f"Yp gap (Yp={p2['Yp']:.4f}) AND keeps all four gates within "
                f"closure bands (Neff={p2['Neff']:.3f} <= 4.0, "
                f"delta_neff_ss={p2['delta_neff_ss']:.4f} in HTT band, "
                f"D/H={p2['DH']:.3f}). Recommend default flip to "
                f"(y_max_boltz=200, Ny_boltz=200) after sprint 3c verifies "
                f"all four Hannestad points stay in HTT band at the new "
                f"defaults.")
    if not yp_in_active_band and recovery < 0.10:
        return ("HYPOTHESIS 1 CLEAN FALSIFIED",
                f"P2 recovers only {recovery:.1%} of Yp gap "
                f"(Yp={p2['Yp']:.4f}) at constant dy. The +0.030 shift in "
                f"sprint-3 P1 was driven by grid-resolution coarsening "
                f"(dy=2.0), not by FD-plateau truncation per se. Pivot to "
                f"hypothesis 2 (post-Phase-B re-thermalisation) or hypothesis "
                f"3 (n->p weak-rate QKE-distribution consumption).")
    if recovery >= 0.50 and not gates_pass:
        broken = []
        if not neff_in_band:
            broken.append(f"Neff={p2['Neff']:.3f} > 4.0")
        if not sterile_in_band:
            broken.append(f"delta_neff_ss={p2['delta_neff_ss']:.4f} not in "
                           f"[{HTT_STERILE_BAND[0]}, {HTT_STERILE_BAND[1]}]")
        return ("HYPOTHESIS 1 PARTIALLY CLEAN",
                f"P2 recovers {recovery:.1%} of Yp gap (Yp={p2['Yp']:.4f}) "
                f"but {' and '.join(broken)}. Resolution decoupling helps "
                f"but does not fully cure. Sprint 3c should investigate "
                f"which physics piece is grid-resolution-sensitive at "
                f"moderate y.")
    return ("HYPOTHESIS 1 INDETERMINATE",
            f"P2 Yp={p2['Yp']:.4f} (recovery {recovery:.1%}); "
            f"all-gates pass={gates_pass}, "
            f"Yp in active band={yp_in_active_band}. Manual review needed.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3b: grid-extent vs grid-resolution disentanglement",
        "  Hannestad Point C (sin^2(2theta_24)=1e-4, dm^2=0.93 eV^2), n_B=12000",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Probing: does (y_max=200, Ny=200, dy=1.0) recover Yp toward SM "
        "0.247 while keeping all four gates in closure bands?",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    p2 = _run_one(RUN)
    t_total = time.time() - t_total_0

    verdict, verdict_detail = _verdict(p2)

    summary = ["", "=" * 110,
               "Stage F sprint 3b grid-extent disentanglement results",
               "=" * 110]
    # Three-row comparison: P0, P1 (from sprint 3 references), P2 (this run).
    for r in (SPRINT3_P0, SPRINT3_P1, p2):
        gates = "ALL-PASS" if _all_gates_pass(r) else "GATE-BREAK"
        summary.append(
            f"  [{gates:<10s}] {r['label']:42s}  "
            f"Yp={r['Yp']:.5f}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  delta_Neff_ss={r['delta_neff_ss']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  P2 - P0 deltas (extent doubled, resolution preserved):",
        f"    delta(Yp)            = {p2['Yp'] - SPRINT3_P0['Yp']:+.5f}",
        f"    delta(Neff)          = {p2['Neff'] - SPRINT3_P0['Neff']:+.4f}",
        f"    delta(D/H)           = {p2['DH'] - SPRINT3_P0['DH']:+.4f}",
        f"    delta(delta_Neff_ss) = "
        f"{p2['delta_neff_ss'] - SPRINT3_P0['delta_neff_ss']:+.4f}",
        "",
        f"  Stage F sprint 3b verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3b_grid_extent_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3b_grid_extent_probe.npz")
    payload = {
        "p2_label": np.array(p2["label"]),
        "p2_y_max_boltz": np.array(p2["y_max_boltz"]),
        "p2_Ny_boltz": np.array(p2["Ny_boltz"]),
        "p2_dy": np.array(p2["dy"]),
        "p2_Neff": np.array(p2["Neff"]),
        "p2_Yp": np.array(p2["Yp"]),
        "p2_DH": np.array(p2["DH"]),
        "p2_sum_ss_raw": np.array(p2["sum_ss_raw"]),
        "p2_delta_neff_ss": np.array(p2["delta_neff_ss"]),
        "p2_wall_clock_s": np.array(p2["wall_clock_s"]),
        "sprint3_p0_yp": np.array(SPRINT3_P0["Yp"]),
        "sprint3_p0_neff": np.array(SPRINT3_P0["Neff"]),
        "sprint3_p0_delta_neff_ss": np.array(SPRINT3_P0["delta_neff_ss"]),
        "sprint3_p1_yp": np.array(SPRINT3_P1["Yp"]),
        "sprint3_p1_neff": np.array(SPRINT3_P1["Neff"]),
        "sprint3_p1_delta_neff_ss": np.array(SPRINT3_P1["delta_neff_ss"]),
        "SM_Yp_reference": np.array(SM_YP_REFERENCE),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
