"""Stage F sprint 3f: uniform-clamp variant probe.

Sprint 3e localised the Yp deficit to asymmetric per-flavor cure-clamp
firing under sterile mixing. Sprint 3f implements a new flag
qke_phaseB_clamp_mode with values "per_flavor" (default, sprint-19 behaviour)
and "uniform" (clamp fires on all active flavors unconditionally when the
cure flag is on, restoring nu-nubar symmetry on the high-y FD-tail
extrapolation).

This harness probes the uniform-clamp variant at Hannestad Point C, with
the physical-anchor target (qke_phaseB_clamp_anchor='T_nu_init',
1/T_nu_init=0.00952). Two runs:

  U0 - per_flavor (current sprint-19 default; matches F3-P0 baseline)
        Yp = 0.231, Neff = 1.83, delta_neff_ss = 0.095 (in HTT band)
  U1 - uniform (sprint 3f variant)
        Probes whether uniform clamp gives Yp moving in the literature-
        consensus direction (UP toward 0.247 +/- delta_sterile).

Decision tree:

| Outcome                                       | Verdict                                  | Action |
|----------------------------------------------|------------------------------------------|--------|
| Yp shifts UP (toward 0.247), gates pass     | UNIFORM CLAMP CURES Yp BUG               | Default-flip qke_phaseB_clamp_mode='uniform'; |
|                                              |                                          | re-run four-Hannestad-point scan.         |
| Yp shifts UP but gates fail                 | UNIFORM CLAMP HALF-FIX                   | Document Yp recovery + gate breakage;     |
|                                              |                                          | sprint 3g investigates secondary cause.   |
| Yp essentially unchanged                    | UNIFORM CLAMP DOES NOT CURE              | Pivot to alternate cure (e.g. polyfit on  |
|                                              |                                          | nu/nubar-averaged grid before clamp).     |
| Yp goes DOWN further                        | UNIFORM CLAMP MAKES IT WORSE             | Bug location wrong; re-investigate.       |

Wall-clock estimate: ~2.5 h sequential (75 min/run x 2).

Output:
  diag_stage_f3f_uniform_clamp_probe.npz  Per-run BBN observables and
                                           wall-clock.
  diag_stage_f3f_uniform_clamp_probe.out  Two-row comparison + verdict.
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


RUNS = [
    {"label": "U0 (per_flavor + T_nu_init anchor)", "mode": "per_flavor"},
    {"label": "U1 (uniform + T_nu_init anchor)",   "mode": "uniform"},
]


def _set_flags(run):
    """Cured Hannestad Point C config; vary qke_phaseB_clamp_mode."""
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
    PRyMini.qke_post_phaseB_clamp_flag = True
    # Sprint 3c physical anchor + sprint 3f mode (the probe parameter).
    PRyMini.qke_phaseB_clamp_anchor = "T_nu_init"
    PRyMini.qke_phaseB_clamp_mode = run["mode"]
    PRyMini.qke_phase0_flag = False
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = 12000
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


def _run_one(run):
    print(f"\n--- {run['label']} ---", flush=True)
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
        "mode": run["mode"],
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "sum_ss_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "wall_clock_s": float(dt_run),
    }


def _all_gates_pass(r):
    if not np.isfinite(r["delta_neff_ss"]):
        return False
    in_band = HTT_STERILE_BAND[0] <= r["delta_neff_ss"] <= HTT_STERILE_BAND[1]
    return (r["Yp"] <= ACTIVE_YP_MAX and r["Neff"] <= ACTIVE_NEFF_MAX
            and in_band)


def _verdict(u0, u1):
    delta_yp = u1["Yp"] - u0["Yp"]
    yp_recovery = (u1["Yp"] - u0["Yp"]) / (SM_YP_REFERENCE - u0["Yp"])
    gates_pass_u1 = _all_gates_pass(u1)
    yp_close_to_sm = abs(u1["Yp"] - SM_YP_REFERENCE) <= 0.01

    if gates_pass_u1 and yp_close_to_sm and delta_yp > 0:
        return ("UNIFORM CLAMP CURES Yp BUG",
                f"Uniform clamp shifts Yp from {u0['Yp']:.4f} to "
                f"{u1['Yp']:.4f} (recovery {yp_recovery:.1%}, sign now "
                f"correct), all gates in closure bands "
                f"(Neff={u1['Neff']:.3f}, "
                f"delta_neff_ss={u1['delta_neff_ss']:.4f}). "
                f"Default-flip qke_phaseB_clamp_mode='uniform' and "
                f"re-run four-Hannestad-point scan.")
    if delta_yp > 0 and not gates_pass_u1:
        broken = []
        if u1["Neff"] > ACTIVE_NEFF_MAX:
            broken.append(f"Neff={u1['Neff']:.3f} > 4.0")
        if u1["Yp"] > ACTIVE_YP_MAX:
            broken.append(f"Yp={u1['Yp']:.4f} > 0.255")
        if not (HTT_STERILE_BAND[0] <= u1["delta_neff_ss"]
                <= HTT_STERILE_BAND[1]):
            broken.append(f"delta_neff_ss={u1['delta_neff_ss']:.4f} not in "
                           f"HTT band")
        return ("UNIFORM CLAMP HALF-FIX",
                f"Uniform clamp shifts Yp from {u0['Yp']:.4f} to "
                f"{u1['Yp']:.4f} (recovery {yp_recovery:.1%}, sign now "
                f"correct), but gates broken: {' and '.join(broken)}. "
                f"Document and pivot to sprint 3g.")
    if abs(delta_yp) <= 0.001:
        return ("UNIFORM CLAMP DOES NOT CURE",
                f"Uniform clamp shifts Yp by only {delta_yp:+.5f}. "
                f"Pivot to alternate cure (e.g. polyfit on "
                f"nu-nubar-averaged grid before clamp).")
    if delta_yp < 0:
        return ("UNIFORM CLAMP MAKES IT WORSE",
                f"Uniform clamp shifts Yp from {u0['Yp']:.4f} to "
                f"{u1['Yp']:.4f} (deviation {delta_yp:+.5f}, going DOWN "
                f"from already-low baseline). Bug location may be wrong; "
                f"re-investigate.")
    return ("INDETERMINATE",
            f"Yp shift {delta_yp:+.5f}, recovery {yp_recovery:.1%}, "
            f"gates {'pass' if gates_pass_u1 else 'fail'}. Manual "
            f"review.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3f: uniform-clamp variant probe",
        "  Hannestad Point C (sin^2(2theta_24)=1e-4, dm^2=0.93 eV^2), "
        "n_B=12000, y_max=100, Ny=100",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Anchor: 'T_nu_init' (clamp target = 1/T_nu_init = 0.00952)",
        "  Mode: vary qke_phaseB_clamp_mode in {per_flavor, uniform}",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = [_run_one(run) for run in RUNS]
    t_total = time.time() - t_total_0
    u0, u1 = results

    delta_yp = u1["Yp"] - u0["Yp"]
    delta_neff = u1["Neff"] - u0["Neff"]
    delta_dnss = u1["delta_neff_ss"] - u0["delta_neff_ss"]

    verdict, verdict_detail = _verdict(u0, u1)

    summary = ["", "=" * 110,
               "Stage F sprint 3f uniform-clamp probe results",
               "=" * 110]
    for r in results:
        gates = "ALL-PASS" if _all_gates_pass(r) else "GATE-BREAK"
        summary.append(
            f"  [{gates:<10s}] {r['label']:42s}  "
            f"Yp={r['Yp']:.5f}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  delta_Neff_ss={r['delta_neff_ss']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  U1 - U0 deltas (per_flavor -> uniform):",
        f"    delta(Yp)            = {delta_yp:+.5f}  "
        f"(SM-reference Yp = {SM_YP_REFERENCE:.4f})",
        f"    delta(Neff)          = {delta_neff:+.4f}",
        f"    delta(delta_Neff_ss) = {delta_dnss:+.4f}",
        "",
        f"  Stage F sprint 3f verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3f_uniform_clamp_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3f_uniform_clamp_probe.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "modes": np.array([r["mode"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "delta_Yp_u1_minus_u0": np.array(delta_yp),
        "SM_Yp_reference": np.array(SM_YP_REFERENCE),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
