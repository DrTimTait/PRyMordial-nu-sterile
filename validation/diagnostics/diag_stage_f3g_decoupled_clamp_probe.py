"""Stage F sprint 3g: decoupled-clamp probe.

Sprint 3f showed that uniform-clamp mode cures Yp sign+magnitude but
over-drives sterile production via cure feedback into the Phase-B
inner iteration (delta_neff_ss out of HTT band by ~80 percent at Hannestad
Point C). Sprint 3g implements qke_phaseB_clamp_uniform_at_end: when True,
uniform mode is applied ONLY on the final end-of-Phase-B
update_thermo_distributions call; inner Phase-B calls keep
qke_phaseB_clamp_mode (default per_flavor). The end-of-Phase-B call is
distinguished by passing a non-None a_of_T_func.

This harness probes whether the decoupling preserves the F3f Yp recovery
(Yp ~ 0.254) AND restores the HTT-band sterile closure (delta_neff_ss in
[0.02, 0.10]) at Hannestad Point C.

Two runs:
  D0 - per_flavor + T_nu_init anchor; uniform_at_end=False
        (matches F3f-U0 baseline)
  D1 - per_flavor + T_nu_init anchor; uniform_at_end=True
        (the sprint 3g variant)

Decision tree:

| D1 outcome                                | Verdict                                | Action |
|-------------------------------------------|----------------------------------------|--------|
| Yp ~ 0.254 (F3f-U1 magnitude) AND        | DECOUPLED CURE WORKS                   | Default-flip                            |
|  delta_neff_ss in [0.02, 0.10]           |                                        | qke_phaseB_clamp_uniform_at_end=True;   |
|                                           |                                        | re-run four-Hannestad-point scan.       |
| Yp shifts up but delta_neff_ss still     | DECOUPLING INSUFFICIENT                | Sprint 3h: investigate alternative      |
|  out of band                              |                                        | sterile feedback path.                  |
| Yp essentially unchanged from D0          | DECOUPLED CURE FAILS                   | The full uniform-clamp feedback         |
|                                           |                                        | through Phase B is what was driving Yp; |
|                                           |                                        | re-think the cure design.               |

Wall-clock: ~2.5 h sequential.

Output:
  diag_stage_f3g_decoupled_clamp_probe.npz  Per-run BBN observables.
  diag_stage_f3g_decoupled_clamp_probe.out  Two-row + verdict.
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
    {"label": "D0 (per_flavor mode, uniform_at_end=False)",
     "uniform_at_end": False},
    {"label": "D1 (per_flavor mode, uniform_at_end=True)",
     "uniform_at_end": True},
]


def _set_flags(run):
    """Cured Hannestad Point C config; vary qke_phaseB_clamp_uniform_at_end."""
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
    PRyMini.qke_phaseB_clamp_anchor = "T_nu_init"
    PRyMini.qke_phaseB_clamp_mode = "per_flavor"
    # The probe parameter.
    PRyMini.qke_phaseB_clamp_uniform_at_end = bool(run["uniform_at_end"])
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
        "uniform_at_end": run["uniform_at_end"],
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


def _verdict(d0, d1):
    delta_yp = d1["Yp"] - d0["Yp"]
    yp_close_to_F3f = abs(d1["Yp"] - 0.254) <= 0.005  # F3f-U1 was 0.25356
    gates_pass_d1 = _all_gates_pass(d1)

    if gates_pass_d1 and yp_close_to_F3f:
        return ("DECOUPLED CURE WORKS",
                f"D1 gives Yp={d1['Yp']:.4f} (target ~0.254 from F3f-U1) "
                f"AND delta_neff_ss={d1['delta_neff_ss']:.4f} in HTT band "
                f"[0.02, 0.10] AND Neff={d1['Neff']:.3f} <= 4.0. "
                f"Default-flip qke_phaseB_clamp_uniform_at_end=True; "
                f"re-run four-Hannestad-point scan to confirm closure.")
    if delta_yp > 0.005 and not gates_pass_d1:
        broken = []
        if d1["Neff"] > ACTIVE_NEFF_MAX:
            broken.append(f"Neff={d1['Neff']:.3f} > 4.0")
        if d1["Yp"] > ACTIVE_YP_MAX:
            broken.append(f"Yp={d1['Yp']:.4f} > 0.255")
        if not (HTT_STERILE_BAND[0] <= d1["delta_neff_ss"]
                <= HTT_STERILE_BAND[1]):
            broken.append(f"delta_neff_ss={d1['delta_neff_ss']:.4f} not in "
                           f"HTT band")
        return ("DECOUPLING INSUFFICIENT",
                f"D1 gives Yp={d1['Yp']:.4f} (shifted up by {delta_yp:+.4f}) "
                f"but gates fail: {' and '.join(broken)}. Sprint 3h needed.")
    if abs(delta_yp) <= 0.001:
        return ("DECOUPLED CURE FAILS",
                f"D1 gives Yp={d1['Yp']:.4f}, shift only {delta_yp:+.5f}. "
                f"The full uniform-clamp feedback through Phase B was what "
                f"drove the F3f-U1 Yp recovery. Re-think the cure design.")
    return ("INDETERMINATE",
            f"D1 Yp={d1['Yp']:.4f} (delta {delta_yp:+.5f}); "
            f"gates {'pass' if gates_pass_d1 else 'fail'}. "
            f"Manual review.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3g: decoupled-clamp probe",
        "  Hannestad Point C (sin^2(2theta_24)=1e-4, dm^2=0.93 eV^2), "
        "n_B=12000, y_max=100, Ny=100",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Anchor: 'T_nu_init', Mode: 'per_flavor'",
        "  Probing qke_phaseB_clamp_uniform_at_end in {False, True}",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = [_run_one(run) for run in RUNS]
    t_total = time.time() - t_total_0
    d0, d1 = results

    delta_yp = d1["Yp"] - d0["Yp"]
    delta_neff = d1["Neff"] - d0["Neff"]
    delta_dnss = d1["delta_neff_ss"] - d0["delta_neff_ss"]

    verdict, verdict_detail = _verdict(d0, d1)

    summary = ["", "=" * 110,
               "Stage F sprint 3g decoupled-clamp probe results",
               "=" * 110]
    for r in results:
        gates = "ALL-PASS" if _all_gates_pass(r) else "GATE-BREAK"
        summary.append(
            f"  [{gates:<10s}] {r['label']:50s}  "
            f"Yp={r['Yp']:.5f}  Neff={r['Neff']:.4f}  "
            f"D/H={r['DH']:.4f}  delta_Neff_ss={r['delta_neff_ss']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  D1 - D0 deltas (uniform_at_end False -> True):",
        f"    delta(Yp)            = {delta_yp:+.5f}  "
        f"(SM-reference Yp = {SM_YP_REFERENCE:.4f})",
        f"    delta(Neff)          = {delta_neff:+.4f}",
        f"    delta(delta_Neff_ss) = {delta_dnss:+.4f}",
        "",
        f"  F3f-U1 reference (full-uniform): Yp=0.25356, Neff=3.367, "
        f"delta_Neff_ss=0.1809 (out of band)",
        f"  Targets for D1: Yp ~ 0.254 AND delta_Neff_ss in [0.02, 0.10]",
        "",
        f"  Stage F sprint 3g verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3g_decoupled_clamp_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3g_decoupled_clamp_probe.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "uniform_at_end": np.array([r["uniform_at_end"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "delta_Yp_d1_minus_d0": np.array(delta_yp),
        "SM_Yp_reference": np.array(SM_YP_REFERENCE),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
