"""Stage F sprint 1b' Phase 2: Saviano calibration with Phase 0 RE-ENABLED.

Phase 1 (validation/diagnostics/diag_stage_f1bp_saviano_calibration.{py,out,npz})
showed that PRyMordial-nu reproduces Saviano 2013 Table I qualitatively
(right sign at all three points, sign-flipped pair enhancement
captured) but under-predicts the magnitude by a factor 2-3 with
monotonic L-dependence where Saviano shows a flat L-saturation
plateau. Diagnosis: Phase 0 disabled (the closure default set in
sprint 18 to cure L=0 sterile-sector divergence) skips the high-T
resonance crossing that NH has only when L != 0 (HTT 2012 Appendix A),
which is exactly where Saviano's L-saturation amplification originates.

This Phase 2 re-enables qke_phase0_flag=True at the same three
Saviano points (S1, S2, S3) and re-runs the calibration. Predicted
outcome: PRyM dYp moves toward Saviano's +0.010 plateau across all
three points. If confirmed, sprint 18's Phase-0 disable is established
as a Hannestad-L=0-specific optimization that does NOT generalize to
L != 0 regimes; the closure-config default for L=0 should remain
Phase-0-off, but L != 0 runs should default Phase-0-on.

If Phase 2 does NOT close the magnitude gap, the under-prediction is
structural (perturbative Gamma/Gamma_0 vs full self-consistent QKE),
and PRyMordial-nu's Saviano-regime predictions are defensible at
factor-2 level rather than 1-1 reproductions.

Configuration changes vs Phase 1:

  Flag                    Phase 1            Phase 2
  ---------------------   ----------------   ----------------
  qke_phase0_flag         False              True
  T_boltz_start           100.0 MeV          5.0 MeV (sprint-pre-19 standard;
                                              Phase 0 covers [100, 5], Phase B
                                              covers [5, 0.005])
  T_phase0_start          (irrelevant)       100.0 MeV (default)
  n_B_phase0_override     0                  None (auto-scale)
  T_start                 105 MeV            105 MeV (unchanged; above T_phase0_start)
  n_B_override            12000              12000 (unchanged for Phase B)
  All cure flags          (live cure)        (live cure, unchanged)

Wall-clock estimate: similar to Phase 1 (~80 min per point, ~4h
total). The temperature span [100, 0.005] is the same; just split
between Phase 0 [100, 5] and Phase B [5, 0.005] segments.

Saviano predictions to match (Table I, NH):

  S1  xi_e=xi_mu=+1e-3   Yp = 0.257   dYp = +0.010
  S2  xi_e=xi_mu=+1e-2   Yp = 0.256   dYp = +0.009
  S3  xi_e=-xi_mu=+1e-3  Yp = 0.259   dYp = +0.012

Phase 1 baseline (PRyM, Phase 0 OFF):

  S1  Yp=0.24830  dYp=+0.00130  residual -0.0087
  S2  Yp=0.25094  dYp=+0.00394  residual -0.0051
  S3  Yp=0.25095  dYp=+0.00395  residual -0.0081

Output:
  diag_stage_f1bp_phase2_phase0_on.npz
  diag_stage_f1bp_phase2_phase0_on.out
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


SBBN_YP_BASELINE = 0.247

SAVIANO_SIN2_THETA_ES = 0.025
SAVIANO_SIN2_THETA_MUS = 0.023
SAVIANO_DM2_ST = 0.89

POINTS = [
    {
        "label": "S1: xi_e=+xi_mu=+1e-3",
        "xi_e": 1e-3,
        "xi_mu": 1e-3,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.257,
        "Saviano_dYp": +0.010,
        "Phase1_Yp": 0.24830,
        "Phase1_dYp": +0.00130,
        "n_B": 12000,
    },
    {
        "label": "S2: xi_e=+xi_mu=+1e-2",
        "xi_e": 1e-2,
        "xi_mu": 1e-2,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.256,
        "Saviano_dYp": +0.009,
        "Phase1_Yp": 0.25094,
        "Phase1_dYp": +0.00394,
        "n_B": 12000,
    },
    {
        "label": "S3: xi_e=-xi_mu=+1e-3",
        "xi_e": 1e-3,
        "xi_mu": -1e-3,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.259,
        "Saviano_dYp": +0.012,
        "Phase1_Yp": 0.25095,
        "Phase1_dYp": +0.00395,
        "n_B": 12000,
    },
]


def _set_flags(point):
    """Live cure (3h-b' OR) + Saviano-regime mixing + L != 0 + Phase 0 ENABLED."""
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
    # Saviano-regime sterile mixing.
    PRyMini.Dm2_41 = float(SAVIANO_DM2_ST)
    PRyMini.theta_14 = float(np.arcsin(np.sqrt(SAVIANO_SIN2_THETA_ES)))
    PRyMini.theta_24 = float(np.arcsin(np.sqrt(SAVIANO_SIN2_THETA_MUS)))
    PRyMini.theta_34 = 0.0
    # Saviano-regime lepton asymmetry.
    PRyMini.xi_nue_init = float(point["xi_e"])
    PRyMini.xi_numu_init = float(point["xi_mu"])
    PRyMini.xi_nutau_init = float(point["xi_tau"])
    # Live cure (3h-b' OR) — unchanged from Phase 1.
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phaseB_clamp_anchor = "T_nu_init"
    PRyMini.qke_phaseB_clamp_mode = "per_flavor"
    PRyMini.qke_phaseB_clamp_uniform_at_end = False
    PRyMini.qke_phaseB_clamp_pair_symmetric = True
    PRyMini.qke_phaseB_clamp_pair_symmetric_rule = "min"
    # Phase 2 changes: enable Phase 0 and restore standard temperature
    # window split.
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_boltz_start = 5.0  # Phase B starts at 5 MeV; Phase 0 covers [100, 5].
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin  # above T_phase0_start.
    PRyMini.n_B_phase0_override = None  # auto-scale Phase 0 step count.
    PRyMini.n_B_override = int(point["n_B"])  # Phase B step count.
    # Misc inherited from Phase 1.
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_active_probe_flag = False
    PRyMini.qke_post_phaseB_trace_flag = False
    PRyMini.y_max_boltz = 100.0
    PRyMini.Ny_boltz = 100


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


def _run_one(point):
    print(f"\n--- {point['label']}: xi_e={point['xi_e']:+.0e}, "
          f"xi_mu={point['xi_mu']:+.0e}, xi_tau=0, n_B={point['n_B']} "
          f"(Phase 0 ON) ---", flush=True)
    print(f"    Saviano Table I expects Yp = {point['Saviano_Yp']:.4f} "
          f"(dYp_sterile = {point['Saviano_dYp']:+.4f})", flush=True)
    print(f"    Phase 1 (Phase 0 OFF) baseline: Yp = {point['Phase1_Yp']:.5f} "
          f"(dYp = {point['Phase1_dYp']:+.5f})", flush=True)
    _set_flags(point)
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

    dYp_sterile = Yp - SBBN_YP_BASELINE
    dYp_vs_saviano = Yp - point["Saviano_Yp"]
    dYp_vs_phase1 = Yp - point["Phase1_Yp"]
    saviano_match_within_001 = abs(dYp_vs_saviano) <= 0.001
    saviano_match_within_005 = abs(dYp_vs_saviano) <= 0.005

    print(f"    PRyMordial-nu (Phase 0 ON): Neff={Neff:.4f}  "
          f"Yp={Yp:.5f}  D/H={DH:.4f}  sum_ss(raw)={sum_ss:.4f}  "
          f"delta_Neff_ss={delta_neff_ss}  ({dt_run:.0f}s)", flush=True)
    print(f"    dYp_sterile (vs SBBN 0.247): {dYp_sterile:+.5f}  "
          f"(Saviano expects {point['Saviano_dYp']:+.4f})", flush=True)
    print(f"    Yp residual vs Saviano: {dYp_vs_saviano:+.5f}  "
          f"({'WITHIN 0.001' if saviano_match_within_001 else 'WITHIN 0.005' if saviano_match_within_005 else 'OUT OF 0.005'})",
          flush=True)
    print(f"    Yp shift vs Phase 1 (Phase 0 OFF): {dYp_vs_phase1:+.5f}  "
          f"({'TOWARD Saviano' if dYp_vs_phase1 > 0.001 else 'AWAY from Saviano' if dYp_vs_phase1 < -0.001 else 'NO CHANGE'})",
          flush=True)

    return {
        "label": point["label"],
        "xi_e": float(point["xi_e"]),
        "xi_mu": float(point["xi_mu"]),
        "xi_tau": float(point["xi_tau"]),
        "n_B": int(point["n_B"]),
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "sum_ss_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "Saviano_Yp": float(point["Saviano_Yp"]),
        "Saviano_dYp": float(point["Saviano_dYp"]),
        "Phase1_Yp": float(point["Phase1_Yp"]),
        "Phase1_dYp": float(point["Phase1_dYp"]),
        "dYp_sterile": dYp_sterile,
        "dYp_vs_saviano": dYp_vs_saviano,
        "dYp_vs_phase1": dYp_vs_phase1,
        "saviano_match_within_001": bool(saviano_match_within_001),
        "saviano_match_within_005": bool(saviano_match_within_005),
        "wall_clock_s": float(dt_run),
    }


def _phase2_verdict(results):
    n_total = len(results)
    n_001 = sum(1 for r in results if r["saviano_match_within_001"])
    n_005 = sum(1 for r in results if r["saviano_match_within_005"])
    n_toward = sum(1 for r in results if r["dYp_vs_phase1"] > 0.001)
    n_away = sum(1 for r in results if r["dYp_vs_phase1"] < -0.001)
    if n_001 == n_total:
        return ("CALIBRATED to Saviano (within 0.001) under Phase 0 ON",
                "PRyMordial-nu reproduces Saviano Table I to within 0.001 "
                "at all three points with Phase 0 enabled. Phase 0 was the "
                "missing physics; sprint 18's Phase-0 disable is confirmed "
                "as L=0-specific.")
    elif n_005 == n_total:
        return ("CALIBRATED to Saviano (within 0.005, not 0.001) under Phase 0 ON",
                "PRyMordial-nu reproduces Saviano Table I to within 0.005 "
                "at all three points with Phase 0 enabled. Solver-level "
                "systematic ~0.001-0.005 between PRyM and Saviano-PArthENoPE.")
    elif n_toward == n_total:
        return (f"PARTIAL Phase-0 amplification (all 3 moved TOWARD Saviano, but "
                f"not within 0.005)",
                "Phase 0 enabling moves PRyM Yp consistently toward Saviano's "
                "magnitudes at every point. The amplification mechanism is "
                "captured but not at full Saviano amplitude. Could be Phase 0 "
                "step density, y-grid resolution, or solver-level systematic.")
    elif n_toward >= 1 and n_away >= 1:
        return ("MIXED Phase-0 effect (some points moved toward, others away)",
                "Phase 0 enabling is point-dependent. Suggests resonance "
                "crossings at different L produce different effects in PRyM "
                "than in Saviano. Diagnostic: inspect per-point dYp_vs_phase1 "
                "and Neff/dNss changes.")
    elif n_away == n_total:
        return ("Phase 0 enabling moved AWAY from Saviano at all 3 points",
                "Unexpected. Phase 0 enabling makes the gap larger. Possible "
                "Phase 0 driver instability or sterile-sector pathology "
                "from sprint 18 reasserting itself in the L != 0 regime.")
    else:
        return (f"NEGLIGIBLE Phase-0 effect ({n_toward} toward, {n_away} away, "
                f"rest within 0.001 of Phase 1)",
                "Phase 0 makes ~no difference. The under-prediction is "
                "structural (likely solver-level perturbative-vs-self-"
                "consistent), not a Phase-0 issue. Sprint 4 (FortEPiaNO "
                "cross-solver) becomes the natural next discriminator.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 1b' Phase 2 Saviano calibration with Phase 0 RE-ENABLED",
        f"  Saviano-regime mixing: sin^2 theta_es = {SAVIANO_SIN2_THETA_ES:.3f}, "
        f"sin^2 theta_mu_s = {SAVIANO_SIN2_THETA_MUS:.3f}, "
        f"delta m^2_st = {SAVIANO_DM2_ST} eV^2, NH",
        "  Live cure (3h-b' OR): post-Phase-B clamp + pair_symmetric=True, rule='min'",
        "  Phase 0 ENABLED (qke_phase0_flag=True); T_phase0_start=100, "
        "T_boltz_start=5.0 (sprint-pre-19 standard split)",
        "  Production grid: y_max=100, Ny=100, n_B=12000 (Phase B), "
        "n_B_phase0=auto",
        "  Phase 1 (Phase 0 OFF) baseline available for direct comparison",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for point in POINTS:
        results.append(_run_one(point))
    t_total = time.time() - t_total_0

    verdict, verdict_detail = _phase2_verdict(results)

    summary = ["", "=" * 110,
               "Stage F sprint 1b' Phase 2 Saviano calibration summary",
               "=" * 110]
    summary.append(
        f"  {'config':<28s} {'Phase1 Yp':>10s} {'Phase2 Yp':>10s} "
        f"{'Sav. Yp':>9s} {'P2 dYp':>9s} {'Sav. dYp':>9s} "
        f"{'P2-P1':>9s} {'P2-Sav':>9s} {'verdict':>14s}")
    for r in results:
        v = ("WITHIN 0.001" if r["saviano_match_within_001"]
             else "WITHIN 0.005" if r["saviano_match_within_005"]
             else "OUT OF 0.005")
        summary.append(
            f"  {r['label']:<28s} {r['Phase1_Yp']:>10.5f} {r['Yp']:>10.5f} "
            f"{r['Saviano_Yp']:>9.4f} {r['dYp_sterile']:>+9.5f} "
            f"{r['Saviano_dYp']:>+9.4f} {r['dYp_vs_phase1']:>+9.5f} "
            f"{r['dYp_vs_saviano']:>+9.5f} {v:>14s}")
    summary += [
        "",
        f"  Phase 2 verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Per-point details (Neff, sterile-sector data):",
    ]
    for r in results:
        summary.append(
            f"    {r['label']:<28s}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  sum_ss(raw)={r['sum_ss_raw']:.4f}  "
            f"dNss={r['delta_neff_ss']:.4f}  ({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f1bp_phase2_phase0_on.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f1bp_phase2_phase0_on.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "xi_e": np.array([r["xi_e"] for r in results]),
        "xi_mu": np.array([r["xi_mu"] for r in results]),
        "xi_tau": np.array([r["xi_tau"] for r in results]),
        "n_B": np.array([r["n_B"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "Saviano_Yp": np.array([r["Saviano_Yp"] for r in results]),
        "Saviano_dYp": np.array([r["Saviano_dYp"] for r in results]),
        "Phase1_Yp": np.array([r["Phase1_Yp"] for r in results]),
        "Phase1_dYp": np.array([r["Phase1_dYp"] for r in results]),
        "dYp_sterile": np.array([r["dYp_sterile"] for r in results]),
        "dYp_vs_saviano": np.array([r["dYp_vs_saviano"] for r in results]),
        "dYp_vs_phase1": np.array([r["dYp_vs_phase1"] for r in results]),
        "saviano_match_within_001": np.array(
            [r["saviano_match_within_001"] for r in results]),
        "saviano_match_within_005": np.array(
            [r["saviano_match_within_005"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
