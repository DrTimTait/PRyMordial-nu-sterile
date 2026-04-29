"""Stage F sprint 3c: physical-anchor clamp probe.

Sprint 3 + 3b localised the Yp under-prediction at the cured production
config (y_max=100) to the cure flag's clamp target being y_max-dependent
(1/y_grid[-1]) rather than tied to a physical anchor. Sprint 3c adds an
opt-in flag PRyMini.qke_phaseB_clamp_anchor with values "y_grid" (default,
sprint-19 behaviour) and "T_nu_init" (clamps to 1/(T_start/MeV_to_Kelvin)
instead). This harness runs the cured Hannestad Point C config at y_max=100
twice — once with anchor="y_grid" (matching F3-P0) and once with
anchor="T_nu_init" — to measure whether the physical anchor recovers Yp
toward SM 0.247 while preserving HTT closure on Neff/delta_neff_ss.

Numerical context for this Hannestad run (T_start = 105 MeV, y_max = 100 MeV):
  1/y_grid[-1]                  = 1/99.5     ~ 0.01005
  1/(T_start/MeV_to_Kelvin)     = 1/105      ~ 0.00952
  ratio (T_nu_init / y_grid)    = 99.5/105   ~ 0.948 (5.2 percent shallower)

Decision tree at completion:

| Result                                  | Verdict                                | Action |
|----------------------------------------|----------------------------------------|--------|
| All 4 gates PASS (Yp toward 0.247)     | CLEAN CURE                             | Default-flip qke_phaseB_clamp_anchor to "T_nu_init"; |
|                                         |                                        | re-run four-Hannestad-point scan to confirm closure. |
| Yp recovers partially, gates still PASS | PARTIAL CURE                           | Document Yp residual; sprint 3d explores hypothesis 2/3.|
| Yp recovers but gates fail             | INDETERMINATE                          | Investigate clamp-anchor interaction with sterile.   |
| Yp essentially unchanged               | ANCHOR-INSENSITIVE                     | Pivot to hypotheses 2/3 (post-Phase-B re-thermalisation |
|                                         |                                        | or n->p QKE consumption); record current finding.    |

Output:
  diag_stage_f3c_clamp_anchor_probe.npz  Per-run results + comparison table.
  diag_stage_f3c_clamp_anchor_probe.out  Two-row table + verdict.
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

# Hannestad Point C parameters (matching gate-5 + F3 + F3b).
RUNS = [
    {
        "label": "C0 (anchor='y_grid', sprint-19 baseline)",
        "anchor": "y_grid",
    },
    {
        "label": "C1 (anchor='T_nu_init', sprint-3c probe)",
        "anchor": "T_nu_init",
    },
]


def _set_flags(run):
    """Sprint-2 closure-config defaults at production (y_max=100, Ny=100)."""
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
    # The sprint-3c knob.
    PRyMini.qke_phaseB_clamp_anchor = run["anchor"]


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
    print(f"\n--- {run['label']}: anchor={run['anchor']!r} ---", flush=True)
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
        "anchor": run["anchor"],
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


def _verdict(c0, c1):
    delta_yp = c1["Yp"] - c0["Yp"]
    yp_gap_c0 = SM_YP_REFERENCE - c0["Yp"]
    if abs(yp_gap_c0) > 1e-6:
        recovery = (c1["Yp"] - c0["Yp"]) / yp_gap_c0
    else:
        recovery = float("nan")
    gates_pass = _all_gates_pass(c1)
    yp_close_to_sm = abs(c1["Yp"] - SM_YP_REFERENCE) <= 0.01

    if gates_pass and yp_close_to_sm:
        return ("CLEAN CURE",
                f"anchor='T_nu_init' recovers {recovery:.1%} of Yp gap "
                f"(Yp {c0['Yp']:.4f} -> {c1['Yp']:.4f}; SM {SM_YP_REFERENCE}) "
                f"with all gates in band (Neff={c1['Neff']:.3f}, "
                f"delta_neff_ss={c1['delta_neff_ss']:.4f}). Default-flip "
                f"qke_phaseB_clamp_anchor='T_nu_init' and re-run four-"
                f"Hannestad-point scan to confirm closure.")
    if gates_pass and recovery >= 0.10:
        return ("PARTIAL CURE",
                f"anchor='T_nu_init' recovers {recovery:.1%} of Yp gap "
                f"(Yp {c0['Yp']:.4f} -> {c1['Yp']:.4f}); gates remain in "
                f"band but Yp residual ~{abs(c1['Yp'] - SM_YP_REFERENCE):.4f}. "
                f"Document; sprint 3d explores hypothesis 2/3 for the rest.")
    if not gates_pass:
        broken = []
        if c1["Neff"] > ACTIVE_NEFF_MAX:
            broken.append(f"Neff={c1['Neff']:.3f} > 4.0")
        if c1["Yp"] > ACTIVE_YP_MAX:
            broken.append(f"Yp={c1['Yp']:.4f} > 0.255")
        if not (HTT_STERILE_BAND[0] <= c1["delta_neff_ss"] <= HTT_STERILE_BAND[1]):
            broken.append(f"delta_neff_ss={c1['delta_neff_ss']:.4f} not in HTT")
        return ("INDETERMINATE (gate-break)",
                f"anchor='T_nu_init' gives Yp shift {delta_yp:+.4f} but "
                f"breaks gate(s): {' and '.join(broken)}. Investigate "
                f"clamp-anchor interaction with sterile sector.")
    return ("ANCHOR-INSENSITIVE",
            f"anchor='T_nu_init' shift Yp by {delta_yp:+.5f} (recovery "
            f"{recovery:.1%}). Hypothesis 1c falsified; pivot to "
            f"hypotheses 2/3 (post-Phase-B re-thermalisation or n->p "
            f"QKE consumption).")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3c: physical-anchor clamp probe",
        "  Hannestad Point C (sin^2(2theta_24)=1e-4, dm^2=0.93 eV^2), "
        "n_B=12000, y_max=100, Ny=100",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Probing: does clamp anchor 'T_nu_init' (1/T_start_MeV=0.0095) "
        "recover Yp toward SM 0.247 vs anchor 'y_grid' (1/99.5=0.0101)?",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for run in RUNS:
        results.append(_run_one(run))
    t_total = time.time() - t_total_0
    c0, c1 = results

    verdict, verdict_detail = _verdict(c0, c1)

    summary = ["", "=" * 110,
               "Stage F sprint 3c clamp-anchor probe results",
               "=" * 110]
    for r in results:
        gates = "ALL-PASS" if _all_gates_pass(r) else "GATE-BREAK"
        summary.append(
            f"  [{gates:<10s}] {r['label']:50s}  "
            f"Yp={r['Yp']:.5f}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  delta_Neff_ss={r['delta_neff_ss']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    delta_yp = c1["Yp"] - c0["Yp"]
    summary += [
        "",
        f"  C1 - C0 deltas (anchor 'y_grid' -> 'T_nu_init'):",
        f"    delta(Yp)            = {delta_yp:+.5f}  "
        f"(SM-reference Yp = {SM_YP_REFERENCE:.4f})",
        f"    delta(Neff)          = {c1['Neff'] - c0['Neff']:+.4f}",
        f"    delta(D/H)           = {c1['DH'] - c0['DH']:+.4f}",
        f"    delta(delta_Neff_ss) = "
        f"{c1['delta_neff_ss'] - c0['delta_neff_ss']:+.4f}",
        "",
        f"  Stage F sprint 3c verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3c_clamp_anchor_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3c_clamp_anchor_probe.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "anchors": np.array([r["anchor"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "delta_Yp_c1_minus_c0": np.array(delta_yp),
        "SM_Yp_reference": np.array(SM_YP_REFERENCE),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
