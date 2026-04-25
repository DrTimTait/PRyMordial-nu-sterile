"""Stage E.2 sprint 12 — fast-iteration sub-step localisation probe.

Sibling of diag_phase0_pointC_fallback.py with three differences:
  * qke_phase0_substep_diag_flag = True (substep instrumentation on).
  * n_B_override reduced from 10000 to 200 so Phase B finishes in ~30 s
    (we only need Phase 0 to complete; Phase B physics is irrelevant to
    the istep 906->1035 step-function localisation).
  * Writes the substep tables to a separate output file so the gate-6
    canonical fallback diagnostic stays unchanged.

Use this for fast iteration during sprint-12 Phase C / Phase D. Once
the localised fix lands, re-run the canonical gate-6 probe
(diag_phase0_pointC_fallback.py) for the success-criterion verdict.
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

_OUT_SUBSTEP = os.path.join(
    _WT, "validation/diagnostics/diag_phase0_pointC_substep.out")


def _base_flags():
    """Same Point-C config as gate 6, with substep flag on and a fast
    Phase-B truncation."""
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
    PRyMini.massive_electron_flag = False
    # Phase B truncated: only Phase 0 matters for substep localisation.
    PRyMini.n_B_override = 200
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.n_B_phase0_override = 2500
    PRyMini.qke_phase0_diag_flag = True
    PRyMini.qke_expm_fallback_near_degeneracy = True
    PRyMini.qke_expm_fallback_eps_cross = 1.0e-3
    PRyMini.qke_phase0_substep_diag_flag = True
    PRyMini.qke_phase0_substep_y_target = 0.5
    # Sprint-12 5a fix: H-iteration in the corrector. Toggle via env var
    # PRYM_ITERATE_H so we can compare with/without inside the same harness.
    PRyMini.qke_etdrk2_iterate_h_flag = (
        os.environ.get("PRYM_ITERATE_H", "1") not in ("0", "", "false", "False"))


def _run():
    _base_flags()
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
    return c, res, dt


def _step_function_signature(values):
    """Detect the largest single-step relative jump in a 1D series.
    Returns (max_rel_jump, idx). 1e-30 floor on the denominator."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0, -1
    base = np.abs(arr[:-1])
    base = np.where(base > 1e-30, base, 1e-30)
    jumps = np.abs(arr[1:] - arr[:-1]) / base
    i = int(np.argmax(jumps))
    return float(jumps[i]), i


def _emit_tables(c, res, dt):
    boltz = c._boltz_dm_solver
    sub_hist = boltz._phase0_substep_hist
    rho_ss_hist = c._phase0_rho_ss_history

    if not sub_hist:
        print("FAIL: substep history is empty.", flush=True)
        sys.exit(2)

    # rho_ss summary across the full Phase-0 window.
    steps_p0 = np.asarray([e[0] for e in rho_ss_hist], dtype=int)
    Tg_p0 = np.asarray([e[2] for e in rho_ss_hist])
    max_per_step = np.asarray([float(np.max(np.abs(e[3]))) for e in rho_ss_hist])
    max_nubar = np.asarray([float(np.max(np.abs(e[3][1]))) for e in rho_ss_hist])
    sum_phase0_exit = float(np.sum(rho_ss_hist[-1][3]))

    win_lo, win_hi = 881, 1081
    win = [r for r in sub_hist if win_lo <= r["step"] <= win_hi]
    by_step = {}
    for r in win:
        by_step.setdefault(r["step"], {})[r["label"]] = r

    lines = [
        "=" * 132,
        "Stage E.2 sprint 12 — fast sub-step localisation, Hannestad Point C",
        f"Runtime: {dt:.0f} s.  Neff={res[0]:.5f}  Yp={res[4]:.5f}.",
        f"Phase-0 history: {len(rho_ss_hist)} steps.  "
        f"Sigma rho_ss(Phase-0 exit) = {sum_phase0_exit:.4f}.",
        f"Substep rows in window istep+1 in [{win_lo}, {win_hi}]: {len(win)}",
        "=" * 132,
    ]
    if win:
        lines.append(f"y_target = {PRyMini.qke_phase0_substep_y_target}, "
                     f"y_actual (closest grid mode) = {win[0]['y']:.6f}")
        lines.append("")

    # Phase-0 max|rho_ss| history within and around the window.
    lines.append("-" * 132)
    lines.append(
        "Phase-0 max|rho_ss(istep)| (sparse sample inside the window):")
    lines.append(f"  {'istep':>6s}  {'Tg [MeV]':>9s}  "
                 f"{'max|rho_ss|':>13s}  {'max nubar':>13s}")
    pick = list(range(880, 1081, 10))
    for istep in pick:
        if 0 <= istep < len(steps_p0):
            lines.append(
                f"  {steps_p0[istep]:6d}  {Tg_p0[istep]:9.4f}  "
                f"{max_per_step[istep]:13.4e}  {max_nubar[istep]:13.4e}")
    lines.append("")

    # ---------------- 5a table — V_nunu / H feedback signature -------------
    lines.append("-" * 132)
    lines.append(
        "5a probe: rho_nu - rho_nubar (active 3x3 trace + Frobenius) and "
        "H gap (H_aa - H_ss) for alpha=1, both sectors. (after_predictor only)")
    lines.append(
        f"  {'step':>6s}  {'Tg [MeV]':>9s}  "
        f"{'tr(d_act)':>11s}  {'|d_act|':>11s}  "
        f"{'gap1_nu':>11s}  {'gap1_nb':>11s}  "
        f"{'|H_13_nu|':>11s}  {'|H_13_nb|':>11s}")
    for step in sorted(by_step.keys()):
        r = by_step[step].get("after_predictor")
        if r is None:
            continue
        d = r["rho_diff_active"]
        tr_d = float(np.real(np.trace(d)))
        fro_d = float(np.linalg.norm(d))
        gap1_nu = float(r["H_diag_a_0"][1] - r["H_ss_0"])
        gap1_nb = float(r["H_diag_a_1"][1] - r["H_ss_1"])
        H13_nu = float(np.abs(r["H_as_0"][1]))
        H13_nb = float(np.abs(r["H_as_1"][1]))
        lines.append(
            f"  {step:6d}  {r['Tg']:9.4f}  "
            f"{tr_d:+11.4e}  {fro_d:11.4e}  "
            f"{gap1_nu:+11.4e}  {gap1_nb:+11.4e}  "
            f"{H13_nu:11.4e}  {H13_nb:11.4e}")
    lines.append("")

    # ---------------- 5b table — N_gain D*rho cancellation ----------------
    lines.append("-" * 132)
    lines.append(
        "5b probe: N_gain vs N_full at (alpha=1, sterile, y) for sector "
        "nubar (s=1); predictor and corrector deltas at the same entry.")
    lines.append(
        f"  {'step':>6s}  {'label':>14s}  {'|rho_13_nb|':>12s}  "
        f"{'|N_gain_nb|':>12s}  {'|N_full_nb|':>12s}  "
        f"{'|D*rho_nb|':>12s}  {'|pred_d_nb|':>12s}  "
        f"{'|corr_d_nb|':>12s}")
    for step in sorted(by_step.keys()):
        for lab in ("after_predictor", "after_corrector"):
            r = by_step[step].get(lab)
            if r is None or "N_gain_13_1" not in r:
                continue
            rho13 = abs(complex(r["rho_13_1"]))
            ng13 = abs(complex(r["N_gain_13_1"]))
            nf13 = abs(complex(r["N_full_13_1"]))
            d_off = float(r.get("D_off_13", 0.0))
            drho13 = d_off * rho13
            pd13 = abs(complex(r.get("pred_delta_13_1", 0.0)))
            cd13 = abs(complex(r.get("corr_delta_13_1", 0.0)))
            lines.append(
                f"  {step:6d}  {lab:>14s}  {rho13:12.4e}  "
                f"{ng13:12.4e}  {nf13:12.4e}  "
                f"{drho13:12.4e}  {pd13:12.4e}  {cd13:12.4e}")
    lines.append("")

    # ---------------- 5c table — phi_half threshold ------------------------
    lines.append("-" * 132)
    lines.append(
        "5c probe: z_h per channel and Taylor-branch flag at the y-target "
        "mode (after_half1; substep-invariant).")
    lines.append(
        f"  {'step':>6s}  {'Tg [MeV]':>9s}  "
        f"{'z_h[0]':>12s}  {'z_h[1]':>12s}  {'z_h[2]':>12s}  "
        f"{'tay[0]':>7s}  {'tay[1]':>7s}  {'tay[2]':>7s}")
    for step in sorted(by_step.keys()):
        r = by_step[step].get("after_half1")
        if r is None or "z_h" not in r:
            continue
        z = r["z_h"]
        t = r["taylor_branch"]
        lines.append(
            f"  {step:6d}  {r['Tg']:9.4f}  "
            f"{float(z[0]):12.4e}  {float(z[1]):12.4e}  "
            f"{float(z[2]):12.4e}  "
            f"{str(bool(t[0])):>7s}  {str(bool(t[1])):>7s}  "
            f"{str(bool(t[2])):>7s}")
    lines.append("")

    # ---------------- top-level summary ------------------------------------
    lines.append("-" * 132)
    lines.append("Localisation summary — single-step relative jumps:")
    steps_sorted = sorted(by_step.keys())
    series = {
        "|rho_diff_act| Fro": [],
        "gap1 nu (H_11 - H_ss)": [],
        "gap1 nubar (H_11 - H_ss)": [],
        "|N_gain_13| nubar": [],
        "|N_full_13| nubar": [],
        "|pred delta_13| nubar": [],
        "|corr delta_13| nubar": [],
        "z_h[1]": [],
    }
    for st in steps_sorted:
        r = by_step[st].get("after_predictor")
        if r is None:
            continue
        series["|rho_diff_act| Fro"].append(
            float(np.linalg.norm(r["rho_diff_active"])))
        series["gap1 nu (H_11 - H_ss)"].append(
            float(r["H_diag_a_0"][1] - r["H_ss_0"]))
        series["gap1 nubar (H_11 - H_ss)"].append(
            float(r["H_diag_a_1"][1] - r["H_ss_1"]))
        series["|N_gain_13| nubar"].append(
            abs(complex(r.get("N_gain_13_1", 0.0))))
        series["|N_full_13| nubar"].append(
            abs(complex(r.get("N_full_13_1", 0.0))))
        series["|pred delta_13| nubar"].append(
            abs(complex(r.get("pred_delta_13_1", 0.0))))
        rc = by_step[st].get("after_corrector")
        if rc is not None:
            series["|corr delta_13| nubar"].append(
                abs(complex(rc.get("corr_delta_13_1", 0.0))))
        rh = by_step[st].get("after_half1")
        if rh is not None and "z_h" in rh:
            series["z_h[1]"].append(float(rh["z_h"][1]))

    for name, vals in series.items():
        j, i = _step_function_signature(vals)
        # Map index back to absolute step (steps_sorted may be sparse).
        abs_step = steps_sorted[i] if 0 <= i < len(steps_sorted) else -1
        lines.append(
            f"  {name:>26s}: max single-step relative jump = {j:8.2e} "
            f"between step {abs_step} and {abs_step+1} "
            f"(of {len(vals)} samples)")
    lines.append("")
    lines.append(
        "Heuristic — the sub-suspect with the largest jump localised to the "
        "istep 906->1035 window is the active mechanism. A relative jump "
        "much greater than 1.0 in a SINGLE step is the step-function signature.")

    text = "\n".join(lines)
    with open(_OUT_SUBSTEP, "w") as fh:
        fh.write(text + "\n")
    print(text, flush=True)
    print(f"\nSubstep output written to {_OUT_SUBSTEP}", flush=True)


if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("Stage E.2 sprint 12 fast sub-step localisation (Phase-B truncated)",
          flush=True)
    print("=" * 100, flush=True)
    c, res, dt = _run()
    _emit_tables(c, res, dt)
