"""Stage F sprint 1b' Saviano-2013 calibration scan.

Apples-to-apples reproduction of Saviano et al. 2013 (arXiv:1302.1200)
Table I sterile-induced Yp shifts at Saviano's actual published
parameter point (sin^2 theta_es ~= 0.025, sin^2 theta_mu_s ~= 0.023,
delta m^2_st = 0.89 eV^2, NH, |xi_e| = |xi_mu| in {1e-3, 1e-2}).

Motivation: sprint 3h-e Saviano audit closed the "C Yp deficit
disagrees with Saviano +0.001 to +0.012" framing as misframed
(Saviano applies at sin^2 theta ~ 0.025 saturated mixing with
L != 0; we were applying at sin^2 2theta = 1e-4 narrow mixing
with L = 0). This harness performs the literature calibration
in Saviano's actual regime, where the published Yp predictions
exist for direct comparison.

Saviano Table I (sterile-on, NH, dominated by asymmetry-driven
nu_e / nubar_e spectral distortion):

  Configuration                     Yp        ΔYp_sterile
  -------------------------------   ------    -----------
  S1: xi_e = xi_mu = +1e-3          0.257     +0.010
  S2: xi_e = xi_mu = +1e-2          0.256     +0.009
  S3: xi_e = -xi_mu = +1e-3         0.259     +0.012  (sign-flipped pair)
  S4: xi_e = -xi_mu = +1e-2         0.255     +0.008
  (further row at lower magnitude:  0.251     +0.004)

This scan runs S1, S2, S3 (the three most-discriminating
configurations: same-sign small, same-sign large, opposite-sign
small). If PRyMordial-nu reproduces these to within ~0.001
(Saviano's quoted precision), then PRyMordial-nu is calibrated
to Saviano in Saviano's regime, and the C Yp at L=0 NH narrow
mixing under sprint 3h-b' OR cure becomes a trustworthy
first-published prediction at that point. If reproduction fails,
the discrepancy is diagnosable in a regime where literature
data exists (rather than in the previously-unpublished L=0
narrow-mixing point).

Configuration:
  * Live cure (sprint 3h-b' OR): qke_post_phaseB_clamp_flag=True,
    qke_phaseB_clamp_anchor='T_nu_init',
    qke_phaseB_clamp_pair_symmetric=True,
    qke_phaseB_clamp_pair_symmetric_rule='min'.
  * Phase 0 OFF (qke_phase0_flag=False) — this is the closure
    default. Stage F sprint 1b'-Phase 2 (re-enable Phase 0)
    is the natural next step if this Phase 1 fails.
  * Active PMNS: PRyMordial-nu defaults (theta_12, theta_13,
    theta_23 from PDG 2021); Saviano's sin^2 theta_e_mu = 0.024
    is consistent with PRyMordial-nu's sin^2 theta_13 = 0.0220
    (within the spread of PMNS measurements).
  * Sterile mixing: theta_14 = arcsin(sqrt(0.025)) ~ 0.159 rad
    (matches Saviano's sin^2 theta_es = 0.025, single-angle
    convention); theta_24 = arcsin(sqrt(0.023)) ~ 0.152 rad;
    theta_34 = 0.
  * Lepton asymmetry: xi_nue_init = +xi (S1, S2, S3),
    xi_numu_init = +xi (S1, S2) or -xi (S3).
  * Production grid: y_max=100, Ny=100, n_B=12000, T_boltz_start=100,
    T_start=105 MeV.

Sequential ~5h wall-clock at production n_B=12000 (3 points
* ~75-95 min each, comparable to Hannestad scan walls since
Saviano's mixings are saturation regime ~ Point A walls).

Output:
  diag_stage_f1bp_saviano_calibration.npz
  diag_stage_f1bp_saviano_calibration.out
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


SBBN_YP_BASELINE = 0.247  # Saviano Table I "standard BBN" baseline


# Saviano's mixing parameters (Sec. II.A, Eqs. 8-10).
# sin^2 theta_es = 0.025 (single-angle convention).
SAVIANO_SIN2_THETA_ES = 0.025
SAVIANO_SIN2_THETA_MUS = 0.023
SAVIANO_DM2_ST = 0.89  # eV^2

# Three most-discriminating Saviano Table I configurations.
POINTS = [
    {
        "label": "S1: xi_e=+xi_mu=+1e-3",
        "xi_e": 1e-3,
        "xi_mu": 1e-3,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.257,
        "Saviano_dYp": +0.010,
        "n_B": 12000,
    },
    {
        "label": "S2: xi_e=+xi_mu=+1e-2",
        "xi_e": 1e-2,
        "xi_mu": 1e-2,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.256,
        "Saviano_dYp": +0.009,
        "n_B": 12000,
    },
    {
        "label": "S3: xi_e=-xi_mu=+1e-3",
        "xi_e": 1e-3,
        "xi_mu": -1e-3,
        "xi_tau": 0.0,
        "Saviano_Yp": 0.259,
        "Saviano_dYp": +0.012,
        "n_B": 12000,
    },
]


def _set_flags(point):
    """Live cure (3h-b' OR) + Saviano-regime mixing + L != 0."""
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
    # Saviano-regime sterile mixing (single-angle convention, matching
    # Saviano Eqs. 8-10).
    PRyMini.Dm2_41 = float(SAVIANO_DM2_ST)
    PRyMini.theta_14 = float(np.arcsin(np.sqrt(SAVIANO_SIN2_THETA_ES)))
    PRyMini.theta_24 = float(np.arcsin(np.sqrt(SAVIANO_SIN2_THETA_MUS)))
    PRyMini.theta_34 = 0.0
    # Saviano-regime lepton asymmetry.
    PRyMini.xi_nue_init = float(point["xi_e"])
    PRyMini.xi_numu_init = float(point["xi_mu"])
    PRyMini.xi_nutau_init = float(point["xi_tau"])
    # Live cure (sprint 3h-b' OR).
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phaseB_clamp_anchor = "T_nu_init"
    PRyMini.qke_phaseB_clamp_mode = "per_flavor"
    PRyMini.qke_phaseB_clamp_uniform_at_end = False
    PRyMini.qke_phaseB_clamp_pair_symmetric = True
    PRyMini.qke_phaseB_clamp_pair_symmetric_rule = "min"
    PRyMini.qke_phase0_flag = False
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(point["n_B"])
    PRyMini.n_B_phase0_override = 0
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
          f"---", flush=True)
    print(f"    Saviano Table I expects Yp = {point['Saviano_Yp']:.4f} "
          f"(dYp_sterile = {point['Saviano_dYp']:+.4f})", flush=True)
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
    saviano_match_within_001 = abs(dYp_vs_saviano) <= 0.001
    saviano_match_within_005 = abs(dYp_vs_saviano) <= 0.005

    print(f"    PRyMordial-nu: Neff={Neff:.4f}  Yp={Yp:.5f}  "
          f"D/H={DH:.4f}  sum_ss(raw)={sum_ss:.4f}  "
          f"delta_Neff_ss={delta_neff_ss}  ({dt_run:.0f}s)", flush=True)
    print(f"    dYp_sterile (vs SBBN 0.247): {dYp_sterile:+.5f}  "
          f"(Saviano expects {point['Saviano_dYp']:+.4f})", flush=True)
    print(f"    Yp residual vs Saviano: {dYp_vs_saviano:+.5f}  "
          f"({'WITHIN 0.001' if saviano_match_within_001 else 'WITHIN 0.005' if saviano_match_within_005 else 'OUT OF 0.005'})",
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
        "dYp_sterile": dYp_sterile,
        "dYp_vs_saviano": dYp_vs_saviano,
        "saviano_match_within_001": bool(saviano_match_within_001),
        "saviano_match_within_005": bool(saviano_match_within_005),
        "wall_clock_s": float(dt_run),
    }


def _saviano_verdict(results):
    n_total = len(results)
    n_001 = sum(1 for r in results if r["saviano_match_within_001"])
    n_005 = sum(1 for r in results if r["saviano_match_within_005"])
    if n_001 == n_total:
        return ("CALIBRATED to Saviano (within 0.001)",
                "PRyMordial-nu reproduces Saviano Table I to "
                "within 0.001 at all three points; the L=0 narrow-mixing "
                "C Yp deficit is a trustworthy first-published prediction.")
    elif n_005 == n_total:
        return ("CALIBRATED to Saviano (within 0.005, not 0.001)",
                "PRyMordial-nu reproduces Saviano Table I to within "
                "0.005 but not 0.001 at all three points. Suggests "
                "minor systematic between solvers; L=0 narrow-mixing "
                "result remains within calibration uncertainty.")
    elif n_005 >= 1:
        return (f"PARTIAL Saviano calibration ({n_005}/{n_total} within 0.005)",
                "Mixed agreement; the deviations may be physically "
                "meaningful (different solver treatments of asymmetry "
                "physics) or signal a pipeline issue.")
    else:
        return (f"FAILED Saviano calibration (0/{n_total} within 0.005)",
                "PRyMordial-nu does not reproduce Saviano Table I in "
                "Saviano's regime. Indicates either a real pipeline "
                "issue (possibly diagnosable now that we have published "
                "data to compare against) OR Phase 0 disabled is "
                "blocking the resonance physics that drives Saviano's "
                "asymmetry mechanism. Sprint 1b'-Phase 2 should "
                "re-enable Phase 0 (qke_phase0_flag=True) and re-test.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 1b' Saviano-2013 (arXiv:1302.1200) calibration scan",
        f"  Saviano-regime mixing: sin^2 theta_es = {SAVIANO_SIN2_THETA_ES:.3f}, "
        f"sin^2 theta_mu_s = {SAVIANO_SIN2_THETA_MUS:.3f}, "
        f"delta m^2_st = {SAVIANO_DM2_ST} eV^2, NH",
        "  Live cure (3h-b' OR): post-Phase-B clamp + pair_symmetric=True, rule='min'",
        "  Phase 0 disabled (closure config); sprint 1b'-Phase 2 will "
        "re-enable Phase 0 if Phase 1 fails calibration",
        "  Production grid: y_max=100, Ny=100, n_B=12000, "
        "T_boltz_start=100, T_start=105 MeV",
        "  Saviano predictions from Table I; baseline Yp = 0.247 "
        "(PArthENoPE thermal SBBN per Saviano Sec. III, p.7)",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for point in POINTS:
        results.append(_run_one(point))
    t_total = time.time() - t_total_0

    verdict, verdict_detail = _saviano_verdict(results)

    summary = ["", "=" * 110,
               "Stage F sprint 1b' Saviano calibration summary",
               "=" * 110]
    summary.append(
        f"  {'config':<28s} {'PRyM Yp':>10s} {'Saviano Yp':>12s} "
        f"{'PRyM dYp':>10s} {'Sav. dYp':>10s} {'residual':>10s} "
        f"{'verdict':>14s}")
    for r in results:
        residual = r["dYp_vs_saviano"]
        v = ("WITHIN 0.001" if r["saviano_match_within_001"]
             else "WITHIN 0.005" if r["saviano_match_within_005"]
             else "OUT OF 0.005")
        summary.append(
            f"  {r['label']:<28s} {r['Yp']:>10.5f} {r['Saviano_Yp']:>12.4f} "
            f"{r['dYp_sterile']:>+10.5f} {r['Saviano_dYp']:>+10.4f} "
            f"{residual:>+10.5f} {v:>14s}")
    summary += [
        "",
        f"  Saviano calibration verdict: {verdict}",
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
        "validation/diagnostics/diag_stage_f1bp_saviano_calibration.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f1bp_saviano_calibration.npz")
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
        "dYp_sterile": np.array([r["dYp_sterile"] for r in results]),
        "dYp_vs_saviano": np.array([r["dYp_vs_saviano"] for r in results]),
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
