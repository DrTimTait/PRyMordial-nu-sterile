"""Stage F sprint 3 hypothesis 1: y_max_boltz under-truncation probe.

Sprint-19 part-2 §8.4 noted Yp ~ 0.231 in the cured 4-flavor sterile config
at production n_B=12000 — about 6.5% below SM Yp=0.247. The Stage F brief
§"sprint 3" hypothesis 1 attributes this to truncation of the active-sector
neutrino spectrum at y_max_boltz=100 MeV: at T_nu_init=105 MeV, the
Fermi-Dirac distribution half-density sits at y~105, right at the grid
edge, so the high-y portion of the active spectrum is replaced by a
polyfit FD-tail extrapolation rather than directly resolved.

This probe runs the cured Hannestad Point C config at production n_B=12000
twice — once at the production y_max_boltz=100 (matching gate-5) and once
at y_max_boltz=200 (resolving the FD plateau out to where f drops below
~e^{-2}). Comparison decides whether widening the grid recovers Yp toward
0.247.

Point C is chosen as the test bed because (a) it's sterile-on, so the cure
flag actually engages; (b) it's narrow-mixing (sin^2(2theta)=1e-4) so the
active-sector evolution is simpler than mid/strong mixing; (c) gate-5 Point
C already reported Yp~0.2419 and delta_Neff_ss=0.0947 (in HTT band), so we
have a clean comparison reference at y_max=100.

The cached thermo / nTOp tables are temperature-tabulated and do not depend
on y_max_boltz directly; both runs use the same cache (compute_bckg_flag=
False, compute_nTOp_flag=False), so the only difference between runs is the
QKE/Boltzmann momentum-grid extent. Ny_boltz is held at 100 — the per-unit-
y resolution halves at y_max=200, which is itself a test of whether the
under-prediction is grid-resolution-limited or grid-extent-limited.

Output:
  diag_stage_f3_y_max_probe.npz  per-run Neff, Yp, D/H, delta_Neff_ss,
                                  sum_ss(raw), wall-clock; delta_Yp.
  diag_stage_f3_y_max_probe.out  comparison table + verdict on whether
                                  hypothesis 1 explains the Yp gap.
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
GATE5_POINT_C_YP = 0.2419  # gate-5 four-Hannestad-point scan, y_max=100
GATE5_POINT_C_DELTA_NEFF_SS = 0.0947


RUNS = [
    {
        "label": "P0 (y_max=100, gate-5 baseline)",
        "y_max_boltz": 100.0,
        "n_B": 12000,
    },
    {
        "label": "P1 (y_max=200, hypothesis 1 test)",
        "y_max_boltz": 200.0,
        "n_B": 12000,
    },
]


def _set_flags(run):
    """Sprint-19 closure config + cure flag at production n_B; vary y_max_boltz only."""
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
    # Hannestad Point C parameters.
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # Stage F sprint 2 production defaults (the closure-config). Set explicitly
    # for clarity even though they now match the post-Stage-F-sprint-2 defaults.
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phase0_flag = False
    # Tunables held at gate-5 values.
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
    # The probe parameter.
    PRyMini.y_max_boltz = float(run["y_max_boltz"])


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
        "n_B": int(run["n_B"]),
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "sum_ss_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "wall_clock_s": float(dt_run),
    }


def _verdict(results):
    p0, p1 = results
    delta_yp = p1["Yp"] - p0["Yp"]
    yp_gap_p0 = SM_YP_REFERENCE - p0["Yp"]
    yp_gap_p1 = SM_YP_REFERENCE - p1["Yp"]
    fraction_recovered = (
        (p0["Yp"] - p1["Yp"]) / (p0["Yp"] - SM_YP_REFERENCE)
        if abs(p0["Yp"] - SM_YP_REFERENCE) > 1e-6 else float("nan")
    )
    # Hypothesis 1 wins if y_max=200 closes >50% of the gap toward SM Yp.
    if not np.isfinite(fraction_recovered):
        return ("INDETERMINATE",
                "Yp gap at baseline is too small to compute recovery fraction.")
    if fraction_recovered >= 0.50:
        return (
            "HYPOTHESIS 1 CONFIRMED",
            f"y_max=200 recovers {fraction_recovered:.1%} of the Yp gap "
            f"({yp_gap_p0:+.4f} -> {yp_gap_p1:+.4f}); the under-prediction "
            f"is dominated by FD-plateau truncation. Stage F sprint 3 "
            f"follow-up: bump y_max_boltz default to 200 MeV (or 250 MeV "
            f"to give margin for the polyfit tail) and verify all gate-5 "
            f"points stay in HTT band.")
    if fraction_recovered >= 0.10:
        return (
            "HYPOTHESIS 1 PARTIAL",
            f"y_max=200 recovers {fraction_recovered:.1%} of the Yp gap "
            f"({yp_gap_p0:+.4f} -> {yp_gap_p1:+.4f}); plateau truncation "
            f"explains some of the under-prediction but not all. Stage F "
            f"sprint 3 follow-up: investigate hypothesis 2 (post-Phase-B "
            f"active-sector re-thermalisation) or hypothesis 3 (n->p weak "
            f"rate consumption of QKE distributions) for the remainder.")
    return (
        "HYPOTHESIS 1 FALSIFIED",
        f"y_max=200 recovers only {fraction_recovered:.1%} of the Yp gap "
        f"({yp_gap_p0:+.4f} -> {yp_gap_p1:+.4f}); plateau truncation is "
        f"NOT the dominant cause. Stage F sprint 3 follow-up: hypothesis 2 "
        f"(post-Phase-B active-sector re-thermalisation) is the next test.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3 hypothesis 1: y_max_boltz under-truncation probe",
        "  Hannestad Point C (sin^2(2theta_24)=1e-4, dm^2=0.93 eV^2), "
        "production n_B=12000",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Probing: does y_max_boltz=200 recover Yp toward SM 0.247 vs "
        "y_max_boltz=100 baseline ~0.242?",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for run in RUNS:
        results.append(_run_one(run))
    t_total = time.time() - t_total_0

    p0, p1 = results
    delta_yp = p1["Yp"] - p0["Yp"]
    delta_neff = p1["Neff"] - p0["Neff"]
    delta_dh = p1["DH"] - p0["DH"]
    delta_dnss = p1["delta_neff_ss"] - p0["delta_neff_ss"]

    verdict, verdict_detail = _verdict(results)

    summary = ["", "=" * 110,
               "Stage F sprint 3 hypothesis 1 results",
               "=" * 110]
    for r in results:
        summary.append(
            f"  {r['label']:42s}  "
            f"Yp={r['Yp']:.5f}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  delta_Neff_ss={r['delta_neff_ss']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  delta(Yp) [P1 - P0]:           {delta_yp:+.5f}  "
        f"(SM-reference Yp = {SM_YP_REFERENCE:.4f})",
        f"  delta(Neff) [P1 - P0]:         {delta_neff:+.4f}",
        f"  delta(D/H) [P1 - P0]:          {delta_dh:+.4f}",
        f"  delta(delta_Neff_ss) [P1 - P0]: {delta_dnss:+.4f}",
        "",
        f"  Stage F sprint 3 verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3_y_max_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3_y_max_probe.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "y_max_boltz": np.array([r["y_max_boltz"] for r in results]),
        "n_B": np.array([r["n_B"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "delta_Yp_p1_minus_p0": np.array(delta_yp),
        "delta_Neff_p1_minus_p0": np.array(delta_neff),
        "delta_DH_p1_minus_p0": np.array(delta_dh),
        "delta_delta_neff_ss_p1_minus_p0": np.array(delta_dnss),
        "SM_Yp_reference": np.array(SM_YP_REFERENCE),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
