"""Stage E.2 sprint 19 active-sector pathology probe.

Sprint 18 closed the sterile sector under
`(qke_phase0_flag=False, qke_damping_formula='symmetric',
qke_v_nunu_active_only=True)` at production n_B=12000:
δNeff_ss = 0.0542 ∈ HTT 2012 band [0.02, 0.10]. But the same run
produced Neff=417.19, Yp=0.36186 — the active sector explodes. At
reduced n_B=3500 the same flags give Neff=3.91, Yp=0.249 (healthy).

This harness instruments both n_B configurations with the
sprint-19 `qke_active_probe_flag` (added to PRyM_init.py near
line 89; hooks into `_run_qke_segment` in PRyM_main.py and
appends per-outer-step y³ moments per flavor to
`PRyMclass._active_probe_history`). It runs the two
configurations in sequence, captures the full trajectory plus the
end-of-Phase-B per-y rho_νe profile, then aligns the two
trajectories by Tg to localise where the active sector diverges.

Output:
* `diag_sprint19_active_sector_probe.npz` — full trajectories +
  end-state arrays for both runs.
* `diag_sprint19_active_sector_probe.out` — human-readable summary
  with the divergence-window verdict (Early/Mid/End Phase B and
  the dominant flavor).

Cost: ~95 min wall-clock (sprint-18 measured Run A 73 min, Run B
24 min). Sequential to avoid NumPy threading interference.
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


# Sprint-18 reference numbers (sterile-sector closure context)
REF_PRODUCTION = {"n_B": 12000, "sum_raw": 4.9547, "delta_neff_ss": 0.0542,
                  "Neff": 417.1947, "Yp": 0.36186}
REF_REDUCED = {"n_B": 3500, "sum_raw": 4.4416, "delta_neff_ss": 0.0304,
               "Neff": 3.9090, "Yp": 0.24850}

# Window thresholds for the verdict, per sprint-19 brief lines 127-145.
T_EARLY_LO = 50.0    # > this is "Early Phase B"
T_MID_LO = 10.0      # [10, 50] is "Mid Phase B"  (sprint-12 region)
                     # (0.005, 10) is "End Phase B"


def _set_flags(n_B):
    """Sprint-19 closure config (Run C + no-Phase-0 + symmetric damping)
    with the active-sector probe enabled. Mirrors
    diag_sprint18_no_phase0_production.py except for n_B and the new flag.
    """
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
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(n_B)
    PRyMini.n_B_phase0_override = 0
    PRyMini.qke_active_probe_flag = True


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


def _run_one(n_B, label):
    """Execute one PRyMresults() run with the active probe on, returning
    a dict of the captured trajectory plus end-state observables."""
    print(f"\n--- {label}: n_B = {n_B} ---", flush=True)
    _set_flags(n_B)
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
    sum_ss = 0.0
    delta_neff_ss = None
    rho_e_final = None
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)
        # Save end-of-Phase-B per-y rho_νe (both sectors); shape (2, Ny).
        rho_e_final = np.array(c._boltz_rho_final[:, 0, :], dtype=float)

    # Per-step active-sector trajectory: list of
    # (istep, t, a, Tg, sigma, m3_e, m3_mu, m3_tau, m3_s) tuples.
    hist = getattr(c, "_active_probe_history", None) or []
    if hist:
        hist_arr = np.array(hist, dtype=float)  # shape (n_steps, 9)
    else:
        hist_arr = np.zeros((0, 9), dtype=float)

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  sum_ss(raw)={sum_ss:.4f}  "
          f"δNeff_ss={delta_neff_ss}  steps={hist_arr.shape[0]}  "
          f"({dt_run:.0f}s)", flush=True)

    return {
        "n_B": int(n_B),
        "label": label,
        "Neff": Neff,
        "Yp": Yp,
        "sum_raw": sum_ss,
        "delta_neff_ss": float(delta_neff_ss) if delta_neff_ss is not None else float("nan"),
        "wall_clock_s": float(dt_run),
        "history": hist_arr,
        "rho_e_final": rho_e_final,
    }


def _classify_window(Tg):
    if Tg > T_EARLY_LO:
        return "Early Phase B (T > 50 MeV)"
    if Tg >= T_MID_LO:
        return "Mid Phase B (T ∈ [10, 50] MeV; sprint-12 region)"
    return "End Phase B (T < 10 MeV)"


def _diff_trajectories(prod, red):
    """Tg-align prod vs red and find first descending Tg at which any
    flavor's m3 deviates by more than 5%. Returns (verdict_window,
    divergence_Tg, dominant_flavor, max_dev_at_div, flavor_devs_at_end).
    """
    from scipy.interpolate import interp1d
    h_p = prod["history"]
    h_r = red["history"]
    if h_p.shape[0] == 0 or h_r.shape[0] == 0:
        return ("UNKNOWN (empty trajectory)", float("nan"), "?",
                float("nan"), {})

    # Columns: 0=istep 1=t 2=a 3=Tg 4=sigma 5=m3_e 6=m3_mu 7=m3_tau 8=m3_s
    Tg_p = h_p[:, 3]
    Tg_r = h_r[:, 3]

    # Sort by Tg ascending so interp1d is well-defined.
    sort_r = np.argsort(Tg_r)
    Tg_r_s = Tg_r[sort_r]
    interp_e = interp1d(Tg_r_s, h_r[sort_r, 5], bounds_error=False,
                         fill_value="extrapolate")
    interp_mu = interp1d(Tg_r_s, h_r[sort_r, 6], bounds_error=False,
                          fill_value="extrapolate")
    interp_tau = interp1d(Tg_r_s, h_r[sort_r, 7], bounds_error=False,
                           fill_value="extrapolate")

    # Production trajectory: descending Tg by construction (Phase B from
    # 100 down to 0.005). Walk from highest Tg down; first deviation > 5%
    # is the divergence point.
    # Defensive sort: restrict to descending Tg for the verdict walk.
    order_desc = np.argsort(-Tg_p)
    Tg_walk = Tg_p[order_desc]
    m3_e_p = h_p[order_desc, 5]
    m3_mu_p = h_p[order_desc, 6]
    m3_tau_p = h_p[order_desc, 7]

    threshold = 0.05  # 5%
    div_Tg = None
    dominant = None
    max_dev_at_div = 0.0
    for i in range(len(Tg_walk)):
        Tg_i = float(Tg_walk[i])
        # Skip points outside the reduced trajectory's Tg range to avoid
        # extrapolation noise at the very first step.
        if Tg_i > Tg_r_s[-1] or Tg_i < Tg_r_s[0]:
            continue
        m3_e_r_i = max(float(interp_e(Tg_i)), 1e-30)
        m3_mu_r_i = max(float(interp_mu(Tg_i)), 1e-30)
        m3_tau_r_i = max(float(interp_tau(Tg_i)), 1e-30)
        d_e = abs(m3_e_p[i] - m3_e_r_i) / m3_e_r_i
        d_mu = abs(m3_mu_p[i] - m3_mu_r_i) / m3_mu_r_i
        d_tau = abs(m3_tau_p[i] - m3_tau_r_i) / m3_tau_r_i
        d_max = max(d_e, d_mu, d_tau)
        if d_max > threshold:
            div_Tg = Tg_i
            max_dev_at_div = d_max
            if d_e == d_max:
                dominant = "ν_e"
            elif d_mu == d_max:
                dominant = "ν_μ"
            else:
                dominant = "ν_τ"
            break

    # Ratios at end-of-trajectory (lowest Tg in production trajectory)
    Tg_end = float(Tg_walk[-1]) if len(Tg_walk) > 0 else float("nan")
    end_devs = {}
    if not np.isnan(Tg_end):
        Tg_end_clipped = min(max(Tg_end, Tg_r_s[0]), Tg_r_s[-1])
        m3_e_r_end = max(float(interp_e(Tg_end_clipped)), 1e-30)
        m3_mu_r_end = max(float(interp_mu(Tg_end_clipped)), 1e-30)
        m3_tau_r_end = max(float(interp_tau(Tg_end_clipped)), 1e-30)
        end_devs = {
            "ν_e": (m3_e_p[-1] / m3_e_r_end - 1.0),
            "ν_μ": (m3_mu_p[-1] / m3_mu_r_end - 1.0),
            "ν_τ": (m3_tau_p[-1] / m3_tau_r_end - 1.0),
        }

    if div_Tg is None:
        return ("Below 5% threshold across all Tg (trajectories track within tolerance)",
                float("nan"), "—", 0.0, end_devs)

    return (_classify_window(div_Tg), div_Tg, dominant,
            max_dev_at_div, end_devs)


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 19 active-sector pathology probe",
        "  Config: damping='symmetric', active_only=True, qke_phase0_flag=False",
        "  Phase B from T=100 MeV to T=0.005 MeV",
        "  Sprint-18 sterile-sector closure references:",
        f"    Production n_B={REF_PRODUCTION['n_B']}: "
        f"sum_raw={REF_PRODUCTION['sum_raw']:.4f}  "
        f"δNeff_ss={REF_PRODUCTION['delta_neff_ss']:.4f}  "
        f"Neff={REF_PRODUCTION['Neff']:.4f}  Yp={REF_PRODUCTION['Yp']:.5f}",
        f"    Reduced n_B={REF_REDUCED['n_B']}: "
        f"sum_raw={REF_REDUCED['sum_raw']:.4f}  "
        f"δNeff_ss={REF_REDUCED['delta_neff_ss']:.4f}  "
        f"Neff={REF_REDUCED['Neff']:.4f}  Yp={REF_REDUCED['Yp']:.5f}",
        "  Active-sector pathology: Neff=417, Yp=0.36 at n_B=12000 — investigate.",
        "  HTT 2012 reference band for sterile sector: δNeff_ss in [0.02, 0.10]",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    prod = _run_one(REF_PRODUCTION["n_B"], "Run A (production)")
    red = _run_one(REF_REDUCED["n_B"], "Run B (reduced reference)")
    t_total = time.time() - t_total_0

    verdict_window, div_Tg, dominant, max_dev, end_devs = _diff_trajectories(prod, red)

    summary = [
        "",
        "=" * 110,
        "Sprint 19 active-sector probe result",
        "=" * 110,
        f"  Run A (n_B={prod['n_B']}, production):  "
        f"Neff={prod['Neff']:.4f}  Yp={prod['Yp']:.5f}  "
        f"sum_raw={prod['sum_raw']:.4f}  δNeff_ss={prod['delta_neff_ss']:.4f}  "
        f"steps={prod['history'].shape[0]}  ({prod['wall_clock_s']:.0f}s)",
        f"  Run B (n_B={red['n_B']}, reduced reference):  "
        f"Neff={red['Neff']:.4f}  Yp={red['Yp']:.5f}  "
        f"sum_raw={red['sum_raw']:.4f}  δNeff_ss={red['delta_neff_ss']:.4f}  "
        f"steps={red['history'].shape[0]}  ({red['wall_clock_s']:.0f}s)",
        "",
        "  Tg-aligned divergence walk (5% threshold on m3_α per flavor):",
        f"    Verdict window: {verdict_window}",
        f"    Divergence Tg: {div_Tg:.4f} MeV" if not np.isnan(div_Tg)
            else "    Divergence Tg: (no divergence detected at 5% threshold)",
        f"    Dominant flavor: {dominant}",
        f"    Max deviation at divergence: {max_dev*100.0:.2f}%",
        "",
        "  End-of-Phase-B m3_α relative deviation (production / reduced − 1):",
    ]
    for k, v in (end_devs or {}).items():
        summary.append(f"    {k}: {v*100.0:+.2f}%")

    summary += [
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_active_sector_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_active_sector_probe.npz")
    np.savez_compressed(
        out_npz,
        # Trajectory column layout: istep, t, a, Tg, sigma, m3_e, m3_mu, m3_tau, m3_s
        history_columns=np.array(["istep", "t", "a", "Tg", "sigma",
                                    "m3_e", "m3_mu", "m3_tau", "m3_s"]),
        production_n_B=np.int64(prod["n_B"]),
        production_history=prod["history"],
        production_rho_e_final=(prod["rho_e_final"]
                                 if prod["rho_e_final"] is not None
                                 else np.zeros((0,))),
        production_Neff=np.float64(prod["Neff"]),
        production_Yp=np.float64(prod["Yp"]),
        production_sum_raw=np.float64(prod["sum_raw"]),
        production_delta_neff_ss=np.float64(prod["delta_neff_ss"]),
        production_wall_clock_s=np.float64(prod["wall_clock_s"]),
        reduced_n_B=np.int64(red["n_B"]),
        reduced_history=red["history"],
        reduced_rho_e_final=(red["rho_e_final"]
                              if red["rho_e_final"] is not None
                              else np.zeros((0,))),
        reduced_Neff=np.float64(red["Neff"]),
        reduced_Yp=np.float64(red["Yp"]),
        reduced_sum_raw=np.float64(red["sum_raw"]),
        reduced_delta_neff_ss=np.float64(red["delta_neff_ss"]),
        reduced_wall_clock_s=np.float64(red["wall_clock_s"]),
        verdict_window=np.array(verdict_window),
        divergence_Tg_MeV=np.float64(div_Tg),
        dominant_flavor=np.array(dominant),
        max_dev_at_divergence=np.float64(max_dev),
    )
    print(f"\nSaved trajectory dump: {out_npz}", flush=True)
    print(f"Saved summary: {out_txt}", flush=True)
