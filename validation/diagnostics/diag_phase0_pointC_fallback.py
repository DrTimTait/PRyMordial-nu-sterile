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
