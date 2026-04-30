"""Stage F sprint 3e: sterile-on vs sterile-off end-of-Phase-B trace.

Sprint 3d localised the Yp deficit to active-sterile mixing handling, with
the literature confirming the deficit has the wrong sign (sterile mixing
should INCREASE Yp, not decrease it). Sprint 3e instruments the
end-of-Phase-B raw f_alpha grids for two paired runs at the cured Hannestad
production config:

  S0 - sterile_flag=False (3-flavor SM)
        baseline; should give Yp=0.247 (verified by F3d).
  S1 - sterile_flag=True at Hannestad Point C (sin^2(2theta)=1e-4,
        dm^2=0.93 eV^2)
        gives Yp=0.231 (verified by F3-P0).

Both runs enable qke_post_phaseB_trace_flag so that the raw f_alpha y-grid
arrays land on PRyMthermo._post_phaseB_trace_grids. The post-run analysis
diffs the active grids (f_nue, f_nuebar, f_numu, ...) between the two
runs and computes the polyfit tail slope _tail_b at the cure site.

The radiation-budget "missing energy" inventory:
  S0 Neff = 2.954 (deviation -0.090 from SM 3.044; QED-table issue, F3f)
  S1 Neff = 1.834 + sterile delta_Neff_ss = 0.095 = 1.929 total
  Difference = S0 - (S1 + sterile) = 2.954 - 1.929 = 1.025 species
  This 1+ species "missing" energy is the load-bearing signal.

Decision tree at completion:

| Finding                                        | Verdict                                    | Action |
|-----------------------------------------------|--------------------------------------------|--------|
| Active grids match between S0 and S1 within  | Sterile mixing leaves active grids        | Bug is in cure-flag interaction with     |
|  numerical noise; only sterile differs        | invariant; bug is in cure clamp           | sterile presence / n_flavor=4 code path. |
| Active grids strongly depleted in S1, polyfit | Active depletion drives the missing       | Trace where the depleted energy went;    |
|  slope shifts                                 | energy via cure clamp                     | likely missing diagonal collision        |
|                                                |                                            | restoration.                             |
| Sterile grid carries the full deficit         | Energy is conserved in raw grids          | Bug is in rho_3nu / weak-rate quadrature |
|  (sum_ss raw matches active deficit)          | (good); bug is downstream consumer        | of the sterile contribution.             |

Wall-clock estimate: ~2.5 h sequential at production grid (75 min/run x 2).

Output:
  diag_stage_f3e_sterile_trace.npz  Per-run raw f_alpha grids (e/ebar/mu/
                                     mubar/tau/taubar plus s/sbar for S1),
                                     polyfit slopes, key derived metrics.
  diag_stage_f3e_sterile_trace.out  Diagnostic table + verdict.
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
SM_NEFF_REFERENCE = 3.044


RUNS = [
    {"label": "S0 (sterile_flag=False, 3-flavor SM)",   "sterile": False},
    {"label": "S1 (sterile_flag=True, Hannestad Point C)", "sterile": True},
]


def _set_flags(run):
    """Cured Hannestad-style production config; vary only sterile_flag."""
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
    PRyMini.sterile_flag = bool(run["sterile"])
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = (np.arcsin(np.sqrt(1.0e-4)) / 2.0
                        if run["sterile"] else 0.0)
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phaseB_clamp_anchor = "y_grid"
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
    # CRUCIAL for this probe.
    PRyMini.qke_post_phaseB_trace_flag = True
    PRyMini.y_max_boltz = 100.0
    PRyMini.Ny_boltz = 100


def _polyfit_tail_b(f_grid, y_grid, f_min=1.0e-12, n_points=10):
    """Reproduces _make_f_callable's polyfit_b on the last n_points
    in-range tail values."""
    tail_y, tail_l = [], []
    for i in range(len(y_grid) - 1, -1, -1):
        fi = f_grid[i]
        if fi > f_min and fi < 1.0 - f_min:
            tail_y.append(y_grid[i])
            tail_l.append(np.log(1.0/fi - 1.0))
            if len(tail_y) >= n_points:
                break
    if len(tail_y) < 2:
        return float("nan")
    coeffs = np.polyfit(np.array(tail_y), np.array(tail_l), 1)
    return float(coeffs[0])  # _tail_b


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

    # Pull the trace data stashed by update_thermo_distributions.
    grids = getattr(PRyMthermo, "_post_phaseB_trace_grids", None) or {}
    meta = getattr(PRyMthermo, "_post_phaseB_trace_meta", None) or {}
    y_grid = np.array(grids.get("y_grid", []), dtype=float)

    # Compute polyfit slopes for each flavor.
    polyfits = {}
    for flav in ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                 "f_nutau", "f_nutaubar", "f_nus", "f_nusbar"):
        if flav in grids and len(y_grid) > 0:
            polyfits[flav] = _polyfit_tail_b(grids[flav], y_grid)
        else:
            polyfits[flav] = float("nan")

    # Per-flavor energy-density-like sums (raw + y^3 moment).
    sums_raw = {}
    sums_y3 = {}
    if len(y_grid) > 0:
        y3 = y_grid ** 3
        for flav in ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                     "f_nutau", "f_nutaubar", "f_nus", "f_nusbar"):
            arr = grids.get(flav)
            if arr is not None:
                sums_raw[flav] = float(np.sum(arr))
                sums_y3[flav] = float(np.sum(arr * y3))
            else:
                sums_raw[flav] = float("nan")
                sums_y3[flav] = float("nan")

    print(f"    Neff={Neff:.5f}  Yp={Yp:.5f}  D/H={DH:.5f}  ({dt_run:.0f}s)",
          flush=True)
    print(f"    polyfit _tail_b (1/MeV): nue={polyfits['f_nue']:.5f} "
          f"nuebar={polyfits['f_nuebar']:.5f} numu={polyfits['f_numu']:.5f} "
          f"nutau={polyfits['f_nutau']:.5f} "
          f"nus={polyfits['f_nus']:.5f}", flush=True)

    return {
        "label": run["label"],
        "sterile_on": bool(run["sterile"]),
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "wall_clock_s": float(dt_run),
        "y_grid": y_grid,
        "grids": grids,  # raw f_alpha y-grids
        "polyfit_tail_b": polyfits,
        "sums_raw": sums_raw,
        "sums_y3": sums_y3,
        "meta": meta,
    }


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3e: sterile-on vs sterile-off end-of-Phase-B trace",
        "  Cured config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  Production grid: y_max=100, Ny=100, n_B=12000, T_boltz_start=100, "
        "T_start=105 MeV",
        "  qke_post_phaseB_trace_flag=True so end-of-Phase-B raw f_alpha "
        "grids are stashed",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = [_run_one(run) for run in RUNS]
    t_total = time.time() - t_total_0
    s0, s1 = results

    # Per-flavor summary table.
    table = ["", "=" * 110, "End-of-Phase-B per-flavor metrics", "=" * 110]
    fmt_hdr = (f"{'flavor':<12s} {'_tail_b S0':>12s} {'_tail_b S1':>12s}"
               f" {'sum_raw S0':>12s} {'sum_raw S1':>12s}"
               f" {'sum_y3 S0':>12s} {'sum_y3 S1':>12s}"
               f" {'(S1-S0)/S0 sum_y3':>20s}")
    table.append(fmt_hdr)
    table.append("-" * len(fmt_hdr))
    for flav in ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                 "f_nutau", "f_nutaubar", "f_nus", "f_nusbar"):
        tb0 = s0["polyfit_tail_b"].get(flav, float("nan"))
        tb1 = s1["polyfit_tail_b"].get(flav, float("nan"))
        sr0 = s0["sums_raw"].get(flav, float("nan"))
        sr1 = s1["sums_raw"].get(flav, float("nan"))
        sy0 = s0["sums_y3"].get(flav, float("nan"))
        sy1 = s1["sums_y3"].get(flav, float("nan"))
        if np.isfinite(sy0) and abs(sy0) > 1e-12:
            rel = (sy1 - sy0) / sy0
        else:
            rel = float("nan")
        table.append(f"{flav:<12s} {tb0:>12.5f} {tb1:>12.5f}"
                     f" {sr0:>12.3f} {sr1:>12.3f}"
                     f" {sy0:>12.3e} {sy1:>12.3e}"
                     f" {rel:>+20.4f}")

    # Active total (sum over the 6 active flavors).
    active_y3_S0 = sum(s0["sums_y3"].get(f, 0.0) for f in
                       ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                        "f_nutau", "f_nutaubar"))
    active_y3_S1 = sum(s1["sums_y3"].get(f, 0.0) for f in
                       ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                        "f_nutau", "f_nutaubar"))
    sterile_y3_S1 = (s1["sums_y3"].get("f_nus", 0.0)
                     + s1["sums_y3"].get("f_nusbar", 0.0))
    total_y3_S0 = active_y3_S0  # no sterile in S0
    total_y3_S1 = active_y3_S1 + sterile_y3_S1

    table += [
        "",
        f"Active total y3 sum: S0={active_y3_S0:.4e}  S1={active_y3_S1:.4e}",
        f"Sterile total y3 sum: S0=0  S1={sterile_y3_S1:.4e}",
        f"Grand total y3 sum:  S0={total_y3_S0:.4e}  S1={total_y3_S1:.4e}",
        f"Conservation deviation: (S1 - S0) / S0 = "
        f"{(total_y3_S1 - total_y3_S0) / total_y3_S0:+.4%}",
        "",
        f"BBN observables: S0 Yp={s0['Yp']:.5f} Neff={s0['Neff']:.4f}  "
        f"|  S1 Yp={s1['Yp']:.5f} Neff={s1['Neff']:.4f}",
        f"BBN deficit:     dYp = {s1['Yp'] - s0['Yp']:+.5f} "
        f"(literature predicts ~+0.005)  dNeff = "
        f"{s1['Neff'] - s0['Neff']:+.4f}",
    ]
    text = "\n".join(table)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3e_sterile_trace.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
        fh.write(f"\nTotal wall-clock: {t_total:.0f}s "
                 f"({t_total/60:.1f} min)\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3e_sterile_trace.npz")
    payload = {
        "y_grid_S0": s0["y_grid"],
        "y_grid_S1": s1["y_grid"],
        "Yp_S0": np.array(s0["Yp"]),
        "Yp_S1": np.array(s1["Yp"]),
        "Neff_S0": np.array(s0["Neff"]),
        "Neff_S1": np.array(s1["Neff"]),
        "DH_S0": np.array(s0["DH"]),
        "DH_S1": np.array(s1["DH"]),
        "wall_clock_S0_s": np.array(s0["wall_clock_s"]),
        "wall_clock_S1_s": np.array(s1["wall_clock_s"]),
        "active_y3_S0": np.array(active_y3_S0),
        "active_y3_S1": np.array(active_y3_S1),
        "sterile_y3_S1": np.array(sterile_y3_S1),
        "total_y3_S0": np.array(total_y3_S0),
        "total_y3_S1": np.array(total_y3_S1),
    }
    for flav in ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                 "f_nutau", "f_nutaubar", "f_nus", "f_nusbar"):
        if flav in s0["grids"]:
            payload[f"S0_{flav}"] = np.array(s0["grids"][flav])
        if flav in s1["grids"]:
            payload[f"S1_{flav}"] = np.array(s1["grids"][flav])
        payload[f"polyfit_tail_b_S0_{flav}"] = np.array(
            s0["polyfit_tail_b"].get(flav, float("nan")))
        payload[f"polyfit_tail_b_S1_{flav}"] = np.array(
            s1["polyfit_tail_b"].get(flav, float("nan")))
        payload[f"sum_raw_S0_{flav}"] = np.array(
            s0["sums_raw"].get(flav, float("nan")))
        payload[f"sum_raw_S1_{flav}"] = np.array(
            s1["sums_raw"].get(flav, float("nan")))
        payload[f"sum_y3_S0_{flav}"] = np.array(
            s0["sums_y3"].get(flav, float("nan")))
        payload[f"sum_y3_S1_{flav}"] = np.array(
            s1["sums_y3"].get(flav, float("nan")))
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
