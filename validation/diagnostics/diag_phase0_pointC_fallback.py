"""Stage E.2 sprint 11 diagnostic: Phase-0 Point-C run with the
narrow-mixing eigendecomposition fallback (qke_expm_fallback_near_degeneracy)
turned on. Same config as diag_phase0_pointC.py (sprint-10 baseline) so the
two output files can be diffed side by side.

Sprint-10 baseline at flag=False produced the step-function jump
ρ_ss(istep=906)=0.1771 → ρ_ss(istep=1035)=0.4145 in the y=0.5 ν̄ sector,
with Σρ_ss at Phase-0 exit = 0.666. Sprint-11 success criterion: with
qke_expm_fallback_near_degeneracy=True and qke_expm_fallback_eps_cross=1e-3,
the step-function jump must vanish and Σρ_ss at Phase-0 exit must drop
below 0.1.
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

_OUT_TXT = os.path.join(_WT, "validation/diagnostics/diag_phase0_pointC_fallback.out")


def _base_flags():
    """Point C config, matching gate 9 (diag_hannestad_proj_w30_nB10k.py),
    plus the sprint-11 eigendecomposition fallback flag."""
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
    PRyMini.n_B_override = 10000
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0  # Point C: sin^2 2theta = 1e-4
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
    # Sprint-10 Phase-0 driver, same config as gate 9.
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.n_B_phase0_override = 2500
    # Sprint-10 post-landing probe: per-step rho_ss(y) history.
    PRyMini.qke_phase0_diag_flag = True
    # Sprint-11: eigendecomposition fallback for narrow-mixing MSW collapse.
    PRyMini.qke_expm_fallback_near_degeneracy = True
    PRyMini.qke_expm_fallback_eps_cross = 1.0e-3
    # Sprint-12: sub-step instrumentation for Phase-0 step-function localisation.
    PRyMini.qke_phase0_substep_diag_flag = True
    PRyMini.qke_phase0_substep_y_target = 0.5
    # Sprint-12 5a fix: H-iteration in the ETDRK2 corrector smooths the
    # V_nunu non-linear feedback that drives the istep 906→1035 step-function
    # in ρ_ss(y=0.5, ν̄). Required for the gate-6 success criterion
    # (Σρ_ss(Phase-0 exit) < 0.05). Toggle off via env var PRYM_ITERATE_H=0
    # to reproduce the sprint-11 baseline (step-function preserved).
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


if __name__ == "__main__":
    header = [
        "=" * 104,
        "Stage E.2 sprint 11 diagnostic: Point-C Phase-0 with eigendecomp fallback",
        "sin^2 2theta = 1e-4, dm^2 = 0.93, w30 projection, n_B=10000,",
        "Phase 0 on, qke_expm_fallback_near_degeneracy=True, eps_cross=1e-3.",
        "=" * 104,
    ]
    for line in header:
        print(line, flush=True)

    c, res, dt = _run()
    hist = getattr(c, "_phase0_rho_ss_history", None)

    if hist is None or len(hist) == 0:
        print("FAIL: Phase-0 rho_ss history is empty; diagnostic flag did not take.",
              flush=True)
        sys.exit(2)

    n_rec = len(hist)
    Ny = hist[0][3].shape[1]
    steps = np.asarray([e[0] for e in hist], dtype=int)
    a_arr = np.asarray([e[1] for e in hist])
    Tg_arr = np.asarray([e[2] for e in hist])
    max_abs_per_step = np.asarray(
        [float(np.max(np.abs(e[3]))) for e in hist])
    max_abs_nu = np.asarray(
        [float(np.max(np.abs(e[3][0]))) for e in hist])
    max_abs_nubar = np.asarray(
        [float(np.max(np.abs(e[3][1]))) for e in hist])
    sum_abs_per_step = np.asarray(
        [float(np.sum(np.abs(e[3]))) for e in hist])

    thresholds = [1.0e-10, 1.0e-6, 1.0e-3, 1.0e-1]
    crossings = {}
    for thr in thresholds:
        mask = max_abs_per_step > thr
        if np.any(mask):
            idx = int(np.argmax(mask))
            crossings[thr] = (int(steps[idx]), float(a_arr[idx]),
                              float(Tg_arr[idx]), float(max_abs_per_step[idx]))
        else:
            crossings[thr] = None

    final_rho_ss = hist[-1][3]
    y_grid = c._boltz_dm_solver.y_grid if hasattr(c, "_boltz_dm_solver") \
        and c._boltz_dm_solver is not None else np.arange(Ny, dtype=float)

    per_y_max = np.max(np.abs(final_rho_ss), axis=0)
    top_idx = np.argsort(per_y_max)[::-1][:10]

    n_print = 20
    pick = np.round(np.linspace(0, n_rec - 1, n_print)).astype(int)

    sum_phase0_exit = float(np.sum(final_rho_ss))

    # Sprint-11 fallback counters surfaced from the boltzmann solver.
    boltz = getattr(c, "_boltz_dm_solver", None)
    eig_count = getattr(boltz, "_expm_fallback_eig_count", 0) if boltz is not None else 0
    kappa_high_count = (getattr(boltz, "_expm_fallback_kappa_high_count", 0)
                        if boltz is not None else 0)

    lines = list(header)
    lines.append(f"Runtime: {dt:.0f} s.  Phase-0 history: {n_rec} steps.  "
                 f"Neff={res[0]:.5f}  Yp={res[4]:.5f}  D/H={res[5]:.4f}.")
    lines.append(f"Σρ_ss at Phase-0 exit = {sum_phase0_exit:.4f}  "
                 f"(sprint-10 baseline = 0.6656; target < 0.1).")
    lines.append(f"Fallback dispatch counts: "
                 f"eigendecomp={eig_count}, kappa_guard_revert={kappa_high_count}.")
    lines.append("")
    lines.append("-" * 104)
    lines.append("Threshold crossings for max|rho_ss(y)| over the Phase-0 window:")
    lines.append(f"  {'threshold':>12s}  {'istep':>7s}  {'a':>10s}  "
                 f"{'Tg [MeV]':>11s}  {'max|rho_ss|':>12s}")
    for thr in thresholds:
        c_entry = crossings[thr]
        if c_entry is None:
            lines.append(f"  {thr:12.0e}  {'never':>7s}  {'--':>10s}  "
                         f"{'--':>11s}  {'--':>12s}")
        else:
            lines.append(f"  {thr:12.0e}  {c_entry[0]:7d}  "
                         f"{c_entry[1]:10.4e}  {c_entry[2]:11.4f}  "
                         f"{c_entry[3]:12.4e}")
    lines.append("")
    lines.append("-" * 104)
    lines.append("max|rho_ss(y)| history (20 equally-spaced Phase-0 steps):")
    lines.append(f"  {'istep':>7s}  {'a':>10s}  {'Tg [MeV]':>11s}  "
                 f"{'max |rho_ss|':>13s}  {'max nu':>12s}  "
                 f"{'max nubar':>12s}  {'Σ|rho_ss|':>12s}")
    for p in pick:
        lines.append(f"  {int(steps[p]):7d}  {a_arr[p]:10.4e}  "
                     f"{Tg_arr[p]:11.4f}  {max_abs_per_step[p]:13.4e}  "
                     f"{max_abs_nu[p]:12.4e}  {max_abs_nubar[p]:12.4e}  "
                     f"{sum_abs_per_step[p]:12.4e}")
    lines.append("")
    lines.append("-" * 104)
    lines.append("Top-10 y-modes by final |rho_ss| at Phase-0 exit:")
    lines.append(f"  {'rank':>5s}  {'y':>8s}  {'rho_ss (nu)':>14s}  "
                 f"{'rho_ss (nubar)':>16s}  {'|max|':>12s}")
    for rank, iy in enumerate(top_idx, 1):
        lines.append(f"  {rank:5d}  {float(y_grid[iy]):8.3f}  "
                     f"{float(final_rho_ss[0, iy]):+14.4e}  "
                     f"{float(final_rho_ss[1, iy]):+16.4e}  "
                     f"{float(per_y_max[iy]):12.4e}")
    lines.append("")
    lines.append("-" * 104)
    peak_idx = int(np.argmax(max_abs_per_step))
    peak_Tg = float(Tg_arr[peak_idx])
    lines.append("Probe summary:")
    lines.append(f"  Peak max|rho_ss| during Phase 0: {max_abs_per_step[peak_idx]:.4e}")
    lines.append(f"  Peak step / Tg: istep={int(steps[peak_idx])} at Tg={peak_Tg:.3f} MeV")
    lines.append(f"  nu vs nubar asymmetry at Phase-0 exit: "
                 f"max_nu={max_abs_nu[-1]:.4e}, max_nubar={max_abs_nubar[-1]:.4e}")
    lines.append(f"  Σρ_ss(Phase-0 exit) ratio to sprint-10 baseline: "
                 f"{sum_phase0_exit / 0.6656:.4f}  "
                 f"(<0.15 = sprint-11 success — 6.6× reduction or better)")
    lines.append("")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(_OUT_TXT, "w") as fh:
        fh.write(text + "\n")

    # ----------------------------------------------------------------------
    # Sprint-12 sub-step instrumentation output. Independent of the existing
    # output above; writes to a separate file so the gate-6 success criterion
    # (this script's primary output) is unaffected.
    # ----------------------------------------------------------------------
    _OUT_SUBSTEP = os.path.join(
        _WT, "validation/diagnostics/diag_phase0_pointC_substep.out")
    boltz = getattr(c, "_boltz_dm_solver", None)
    sub_hist = getattr(boltz, "_phase0_substep_hist", None) if boltz is not None else None
    if sub_hist is None or len(sub_hist) == 0:
        print("Sprint-12 substep history is empty (flag may not be wired); "
              "skipping substep output.", flush=True)
    else:
        # Each evolve_step_ode_etdrk2 call emits 4 rows. Solver-side step
        # index is 1-based and is incremented at the top of each call,
        # whereas the Phase-0 caller's istep is 0-based — so step = istep + 1
        # on the Phase-0 segment. Filter to istep window [880, 1080] =>
        # solver step in [881, 1081].
        win_lo, win_hi = 881, 1081
        win = [r for r in sub_hist if win_lo <= r["step"] <= win_hi]
        # Group by step for compact emission (4 labels per step).
        by_step = {}
        for r in win:
            by_step.setdefault(r["step"], {})[r["label"]] = r

        sub_lines = [
            "=" * 132,
            "Sprint-12 sub-step localisation probe — Hannestad Point C, Phase 0",
            f"y_target = {PRyMini.qke_phase0_substep_y_target}, window istep+1 in "
            f"[{win_lo}, {win_hi}], substep rows: {len(win)}",
            "=" * 132,
        ]
        if win:
            y_actual = win[0]["y"]
            sub_lines.append(f"y_actual (closest grid mode) = {y_actual:.6f}")
            sub_lines.append("")

        # ---------------- 5a table — V_nunu / H feedback signature -----------
        sub_lines.append("-" * 132)
        sub_lines.append(
            "5a probe: rho_nu - rho_nubar (active 3x3 trace + (1,3)) and "
            "H gap (H_aa - H_ss) for alpha=1, both sectors.")
        sub_lines.append(
            f"  {'step':>6s}  {'label':>14s}  {'Tg [MeV]':>9s}  "
            f"{'tr(d_act)':>11s}  {'|d_act|':>11s}  "
            f"{'gap1_nu':>11s}  {'gap1_nb':>11s}  "
            f"{'|H_13_nu|':>11s}  {'|H_13_nb|':>11s}")
        for step in sorted(by_step.keys()):
            for lab in ("after_half1", "after_predictor",
                        "after_corrector", "after_half2"):
                r = by_step[step].get(lab)
                if r is None:
                    continue
                d = r["rho_diff_active"]
                tr_d = float(np.real(np.trace(d)))
                fro_d = float(np.linalg.norm(d))
                if "H_diag_a_0" in r:
                    gap1_nu = float(r["H_diag_a_0"][1] - r["H_ss_0"])
                    gap1_nb = float(r["H_diag_a_1"][1] - r["H_ss_1"])
                    H13_nu = float(np.abs(r["H_as_0"][1]))
                    H13_nb = float(np.abs(r["H_as_1"][1]))
                else:
                    gap1_nu = gap1_nb = H13_nu = H13_nb = float("nan")
                sub_lines.append(
                    f"  {step:6d}  {lab:>14s}  {r['Tg']:9.4f}  "
                    f"{tr_d:+11.4e}  {fro_d:11.4e}  "
                    f"{gap1_nu:+11.4e}  {gap1_nb:+11.4e}  "
                    f"{H13_nu:11.4e}  {H13_nb:11.4e}")
        sub_lines.append("")

        # ---------------- 5b table — N_gain D*rho cancellation --------------
        sub_lines.append("-" * 132)
        sub_lines.append(
            "5b probe: N_gain vs N_full at (alpha=1, sterile, y) for sector "
            "nubar (s=1); predictor and corrector deltas at the same entry.")
        sub_lines.append(
            f"  {'step':>6s}  {'label':>14s}  {'|rho_13_nb|':>12s}  "
            f"{'|N_gain_nb|':>12s}  {'|N_full_nb|':>12s}  "
            f"{'|D*rho_nb|':>12s}  {'|pred_d_nb|':>12s}  "
            f"{'|corr_d_nb|':>12s}")
        for step in sorted(by_step.keys()):
            for lab in ("after_half1", "after_predictor",
                        "after_corrector", "after_half2"):
                r = by_step[step].get(lab)
                if r is None:
                    continue
                rho13 = abs(complex(r.get("rho_13_1", 0.0)))
                ng13 = abs(complex(r.get("N_gain_13_1", 0.0)))
                nf13 = abs(complex(r.get("N_full_13_1", 0.0)))
                d_off = float(r.get("D_off_13", 0.0))
                drho13 = d_off * rho13
                pd13 = abs(complex(r.get("pred_delta_13_1", 0.0)))
                cd13 = abs(complex(r.get("corr_delta_13_1", 0.0)))
                # Skip rows with no 5b payload to keep table tight.
                if "N_gain_13_1" not in r and lab not in (
                        "after_predictor", "after_corrector"):
                    continue
                sub_lines.append(
                    f"  {step:6d}  {lab:>14s}  {rho13:12.4e}  "
                    f"{ng13:12.4e}  {nf13:12.4e}  "
                    f"{drho13:12.4e}  {pd13:12.4e}  {cd13:12.4e}")
        sub_lines.append("")

        # ---------------- 5c table — phi_half threshold ---------------------
        sub_lines.append("-" * 132)
        sub_lines.append(
            "5c probe: z_h per channel and Taylor-branch flag at the y-target "
            "mode (recorded once per step — substep-invariant).")
        sub_lines.append(
            f"  {'step':>6s}  {'label':>14s}  {'Tg [MeV]':>9s}  "
            f"{'z_h[0]':>12s}  {'z_h[1]':>12s}  {'z_h[2]':>12s}  "
            f"{'tay[0]':>7s}  {'tay[1]':>7s}  {'tay[2]':>7s}")
        for step in sorted(by_step.keys()):
            for lab in ("after_half1", "after_half2"):
                r = by_step[step].get(lab)
                if r is None or "z_h" not in r:
                    continue
                z = r["z_h"]
                t = r["taylor_branch"]
                sub_lines.append(
                    f"  {step:6d}  {lab:>14s}  {r['Tg']:9.4f}  "
                    f"{float(z[0]):12.4e}  {float(z[1]):12.4e}  "
                    f"{float(z[2]):12.4e}  "
                    f"{str(bool(t[0])):>7s}  {str(bool(t[1])):>7s}  "
                    f"{str(bool(t[2])):>7s}")
        sub_lines.append("")

        # ---------------- top-level summary ---------------------------------
        sub_lines.append("-" * 132)
        sub_lines.append("Localisation summary (rough heuristics):")

        def _step_function_signature(values, eps_jump=2.0):
            """Detect a step-function in a 1D series: max relative jump in a
            single window step. Returns (max_rel_jump, idx_of_jump). """
            arr = np.asarray(values, dtype=float)
            if arr.size < 2:
                return 0.0, -1
            base = np.abs(arr[:-1])
            base = np.where(base > 1e-30, base, 1e-30)
            jumps = np.abs(arr[1:] - arr[:-1]) / base
            i = int(np.argmax(jumps))
            return float(jumps[i]), i

        # Build per-step series at after_predictor (V_nunu / H proxy is most
        # informative there since H is built from rho_n).
        steps_sorted = sorted(by_step.keys())
        gap1_nu_series, gap1_nb_series = [], []
        rho_diff_fro = []
        ng13_series, nf13_series, pd13_series, cd13_series = [], [], [], []
        zh1_series = []
        for st in steps_sorted:
            r = by_step[st].get("after_predictor")
            if r is None:
                continue
            gap1_nu_series.append(float(r["H_diag_a_0"][1] - r["H_ss_0"]))
            gap1_nb_series.append(float(r["H_diag_a_1"][1] - r["H_ss_1"]))
            rho_diff_fro.append(float(np.linalg.norm(r["rho_diff_active"])))
            ng13_series.append(abs(complex(r.get("N_gain_13_1", 0.0))))
            nf13_series.append(abs(complex(r.get("N_full_13_1", 0.0))))
            pd13_series.append(abs(complex(r.get("pred_delta_13_1", 0.0))))
        for st in steps_sorted:
            r = by_step[st].get("after_corrector")
            if r is None:
                continue
            cd13_series.append(abs(complex(r.get("corr_delta_13_1", 0.0))))
        for st in steps_sorted:
            r = by_step[st].get("after_half1")
            if r is None or "z_h" not in r:
                continue
            zh1_series.append(float(r["z_h"][1]))

        def _fmt(name, vals):
            j, i = _step_function_signature(vals)
            sub_lines.append(
                f"  {name:>22s}: max relative jump = {j:8.2e} "
                f"at index {i:4d} of {len(vals):4d}")

        _fmt("|rho_diff_act| Fro", rho_diff_fro)
        _fmt("gap1 nu (H_11 - H_ss)", gap1_nu_series)
        _fmt("gap1 nubar (H_11 - H_ss)", gap1_nb_series)
        _fmt("|N_gain_13| nubar", ng13_series)
        _fmt("|N_full_13| nubar", nf13_series)
        _fmt("|pred delta_13| nubar", pd13_series)
        _fmt("|corr delta_13| nubar", cd13_series)
        _fmt("z_h[1] (electron chan)", zh1_series)
        sub_lines.append("")
        sub_lines.append(
            "Heuristic: a relative jump >> 1 in a SINGLE step (consecutive "
            "indices) indicates a step-function in that quantity. Compare "
            "the 5a / 5b / 5c series above to identify which sub-suspect "
            "drives the istep 906->1035 ρ_ss step-function.")

        substep_text = "\n".join(sub_lines)
        with open(_OUT_SUBSTEP, "w") as fh:
            fh.write(substep_text + "\n")
        print(substep_text, flush=True)
        print(f"\nSprint-12 substep output written to "
              f"{_OUT_SUBSTEP}", flush=True)
