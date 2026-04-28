"""Stage E.2 sprint 19 part 2: post-Phase-B pipeline trace harness.

Sprint 19 part 1 (`diag_sprint19_active_sector_probe.py`) exonerated the QKE
Phase-B driver: end-of-Phase-B m3_α agree to 1.77% / 0.25% / 0.50% (ν_e / ν_μ /
ν_τ) between n_B=12000 and n_B=3500. Yet downstream Neff differs by 107×
(417 vs 3.91) and Yp by 47% (0.36 vs 0.249) at the higher n_B. The active-
sector pathology lives in the post-Phase-B pipeline.

This harness uses the new `qke_post_phaseB_trace_flag` to capture, for both
n_B settings:

* The raw f_α(y) grid arrays delivered to update_thermo_distributions
  (stashed on PRyMthermo._post_phaseB_trace_grids).
* The constructed f_α(p, Tg) callables resampled on a fixed p_grid (50
  logspace points covering [1e-3, 30] · Tg_B[-1]).
* The Phase-C (t_C, Tg_C) trajectory.
* The per-flavor rho_nu_from_f integrand (p_nodes, x³ f) at Tg_C[-1] —
  the Tg fed into N_eff.

It then walks the chain f_grids → f_samples → t_C/Tg_C → integrand_at_Tg_final
and reports the first quantity at which the two runs diverge by more than
the per-quantity tolerance. That identifies which of the three cure design
candidates (interpolator / Phase-C / quadrature) is responsible for the
107× Neff blow-up.

Output:
* `diag_sprint19_post_phaseB_trace.npz` — full per-run trace arrays plus
  end-state observables.
* `diag_sprint19_post_phaseB_trace.out` — human-readable verdict.

Cost: ~95 min wall-clock (production ~73 min + reduced ~24 min, sequential).
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


REF_PRODUCTION = {"n_B": 12000, "Neff": 417.1947, "Yp": 0.36186,
                  "delta_neff_ss": 0.0542}
REF_REDUCED = {"n_B": 3500, "Neff": 3.9090, "Yp": 0.24850,
               "delta_neff_ss": 0.0304}

ACTIVE_FLAVORS = ("f_nue_general", "f_nuebar_general",
                  "f_numu_general", "f_numubar_general",
                  "f_nutau_general", "f_nutaubar_general")

# Tolerances per pipeline stage. f-grids and f-samples should track to ~5%
# (matching the active-sector probe's 1.77% end-of-Phase-B agreement on ν_e).
# Phase-C Tg trajectory should track tightly (<1%) since it's a 1-variable
# ODE fed by interpolators. The integrand at Tg_final must explain a 107×
# Neff blow-up; even 1% dispersion in any flavor's integrand is a signal.
TOL_F_GRID = 0.05
TOL_F_SAMPLE = 0.05
TOL_TG_C = 0.01
TOL_INTEGRAND = 0.01


def _set_flags(n_B):
    """Sprint-19 closure config + part-2 trace flag enabled."""
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
    PRyMini.qke_active_probe_flag = False
    PRyMini.qke_post_phaseB_trace_flag = True


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
    delta_neff_ss = None
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)

    # Pull trace data from PRyMthermo module attrs and PRyMclass instance.
    grids = getattr(PRyMthermo, "_post_phaseB_trace_grids", None) or {}
    callables = getattr(PRyMthermo, "_post_phaseB_trace_callables", None) or {}
    meta = getattr(PRyMthermo, "_post_phaseB_trace_meta", None) or {}
    trace = getattr(c, "_post_phaseB_trace", None) or {}

    # Resample callables on a fixed p_grid in physical units of MeV. Use
    # log-spacing in [1e-3, 30] · Tg_trace so the same x = p/Tg range is
    # covered for both runs.
    Tg_trace = float(trace.get("Tg_trace", 0.0))
    if Tg_trace > 0.0:
        p_grid = np.logspace(np.log10(1.0e-3 * Tg_trace),
                             np.log10(30.0 * Tg_trace), 50)
        f_samples = {}
        for fname in ACTIVE_FLAVORS:
            f_call = callables.get(fname)
            if f_call is None:
                continue
            f_samples[fname] = np.array(f_call(p_grid, Tg_trace), dtype=float)
    else:
        p_grid = np.zeros((0,))
        f_samples = {}

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  sum_ss(raw)={sum_ss:.4f}  "
          f"δNeff_ss={delta_neff_ss}  ({dt_run:.0f}s)", flush=True)

    return {
        "n_B": int(n_B),
        "label": label,
        "Neff": Neff,
        "Yp": Yp,
        "sum_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "wall_clock_s": float(dt_run),
        "grids": grids,
        "callables_p_grid": p_grid,
        "callables_samples": f_samples,
        "trace": trace,
        "meta": meta,
    }


def _max_rel_dev(arr_p, arr_r, eps=1e-30):
    """Max |a/b - 1| over matching indices, ignoring near-zero baselines."""
    arr_p = np.asarray(arr_p, dtype=float)
    arr_r = np.asarray(arr_r, dtype=float)
    if arr_p.shape != arr_r.shape:
        return float("inf")
    denom = np.maximum(np.abs(arr_r), eps)
    rel = np.abs(arr_p - arr_r) / denom
    # Suppress noise where both values are essentially zero.
    mask = np.abs(arr_r) > 1e-15
    if not np.any(mask):
        return 0.0
    return float(np.max(rel[mask]))


def _verdict(prod, red):
    """Walk the post-Phase-B chain and report which stage diverges first."""
    findings = []  # (stage, max_dev, tolerance, payload)

    # Stage 1: raw f_α grids.
    grids_p = prod["grids"]
    grids_r = red["grids"]
    if grids_p and grids_r and len(grids_p.get("y_grid", [])) > 0:
        # The two runs use different Ny by construction (12000-driven vs
        # 3500-driven outer-step counts produce different y-grid resolutions
        # only if Ny depends on n_B; otherwise Ny is fixed and the grids
        # match). Match by linearly interpolating the reduced f onto the
        # production y-grid (or vice versa).
        from scipy.interpolate import interp1d
        y_p = grids_p["y_grid"]
        y_r = grids_r["y_grid"]
        for f_name in ("f_nue", "f_nuebar", "f_numu", "f_numubar",
                       "f_nutau", "f_nutaubar"):
            f_p = grids_p.get(f_name)
            f_r = grids_r.get(f_name)
            if f_p is None or f_r is None:
                continue
            if y_p.shape != y_r.shape:
                f_r_aligned = interp1d(y_r, f_r, bounds_error=False,
                                        fill_value=(f_r[0], f_r[-1]),
                                        kind="linear")(y_p)
            else:
                f_r_aligned = f_r
            dev = _max_rel_dev(f_p, f_r_aligned)
            findings.append(("f_grid:" + f_name, dev, TOL_F_GRID, None))

    # Stage 2: callable samples on the fixed p_grid.
    if (len(prod["callables_p_grid"]) > 0
            and len(red["callables_p_grid"]) > 0
            and prod["callables_samples"] and red["callables_samples"]):
        from scipy.interpolate import interp1d
        # The two p_grids differ because each is anchored at the run's own
        # Tg_trace. Project red samples onto the production p_grid via
        # log-linear interpolation in p.
        p_p = prod["callables_p_grid"]
        p_r = red["callables_p_grid"]
        for fname in ACTIVE_FLAVORS:
            samp_p = prod["callables_samples"].get(fname)
            samp_r = red["callables_samples"].get(fname)
            if samp_p is None or samp_r is None:
                continue
            if p_p.shape != p_r.shape or not np.allclose(p_p, p_r):
                samp_r_aligned = interp1d(np.log(p_r), samp_r,
                                           bounds_error=False,
                                           fill_value=(samp_r[0], samp_r[-1]),
                                           kind="linear")(np.log(p_p))
            else:
                samp_r_aligned = samp_r
            dev = _max_rel_dev(samp_p, samp_r_aligned)
            findings.append(("f_sample:" + fname, dev, TOL_F_SAMPLE, None))

    # Stage 3: Phase-C Tg_C trajectory at matched t (interpolate red onto
    # production t-grid).
    trace_p = prod["trace"]
    trace_r = red["trace"]
    if (trace_p and trace_r
            and len(trace_p.get("Tg_C", [])) > 0
            and len(trace_r.get("Tg_C", [])) > 0):
        from scipy.interpolate import interp1d
        t_C_p = trace_p["t_C"]
        Tg_C_p = trace_p["Tg_C"]
        t_C_r = trace_r["t_C"]
        Tg_C_r = trace_r["Tg_C"]
        # solve_ivp may emit different numbers of samples; align by t.
        Tg_C_r_aligned = interp1d(t_C_r, Tg_C_r, bounds_error=False,
                                    fill_value=(Tg_C_r[0], Tg_C_r[-1]),
                                    kind="linear")(t_C_p)
        dev = _max_rel_dev(Tg_C_p, Tg_C_r_aligned)
        findings.append(("Tg_C trajectory", dev, TOL_TG_C, None))

    # Stage 4: per-flavor integrand at Tg_C[-1].
    if (trace_p and trace_r
            and trace_p.get("integrand_snapshot")
            and trace_r.get("integrand_snapshot")):
        snap_p = trace_p["integrand_snapshot"]
        snap_r = trace_r["integrand_snapshot"]
        for fname in ACTIVE_FLAVORS:
            if fname not in snap_p or fname not in snap_r:
                continue
            integ_p = snap_p[fname]["integrand"]
            integ_r = snap_r[fname]["integrand"]
            # The p_nodes are anchored at each run's own Tg_trace, so the
            # node grids differ. Compare in x = p/Tg space — but p_nodes/Tg
            # is the same fixed Gauss-Legendre x grid for both runs (same
            # p_npoints_nu), so direct index-by-index comparison is valid
            # for the integrand magnitudes.
            dev = _max_rel_dev(integ_p, integ_r)
            findings.append(("integrand:" + fname, dev, TOL_INTEGRAND, None))
            # Also record the scalar rho per flavor for the .out body.
            rho_p = snap_p[fname]["rho_scalar"]
            rho_r = snap_r[fname]["rho_scalar"]
            findings.append(("rho_scalar:" + fname,
                             abs(rho_p / max(rho_r, 1e-30) - 1.0),
                             TOL_INTEGRAND, (rho_p, rho_r)))

    # Identify first stage in pipeline order to exceed its tolerance.
    pipeline_order = [
        ("f_grid:", TOL_F_GRID, "Stage 1: raw f_α grids"),
        ("f_sample:", TOL_F_SAMPLE, "Stage 2: callable samples on p_grid"),
        ("Tg_C", TOL_TG_C, "Stage 3: Phase-C Tg_C trajectory"),
        ("integrand:", TOL_INTEGRAND, "Stage 4: per-flavor integrand at Tg_final"),
    ]
    first_violator = None
    first_violator_label = None
    for prefix, tol, stage_label in pipeline_order:
        for name, dev, tol_n, payload in findings:
            if name.startswith(prefix) and dev > tol_n:
                first_violator = (name, dev, tol_n)
                first_violator_label = stage_label
                break
        if first_violator is not None:
            break

    return findings, first_violator, first_violator_label


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 19 part 2: post-Phase-B pipeline trace",
        "  Config: damping='symmetric', active_only=True, qke_phase0_flag=False",
        "  Phase B from T=100 MeV to T=0.005 MeV",
        "  Sprint-19 part 1 references (post-driver, pre-cure):",
        f"    Production n_B={REF_PRODUCTION['n_B']}: "
        f"Neff={REF_PRODUCTION['Neff']:.4f}  Yp={REF_PRODUCTION['Yp']:.5f}  "
        f"δNeff_ss={REF_PRODUCTION['delta_neff_ss']:.4f}",
        f"    Reduced n_B={REF_REDUCED['n_B']}: "
        f"Neff={REF_REDUCED['Neff']:.4f}  Yp={REF_REDUCED['Yp']:.5f}  "
        f"δNeff_ss={REF_REDUCED['delta_neff_ss']:.4f}",
        "  Trace pipeline: f_grids → f_samples → Tg_C trajectory → "
        "integrand at Tg_final",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    prod = _run_one(REF_PRODUCTION["n_B"], "Run A (production)")
    red = _run_one(REF_REDUCED["n_B"], "Run B (reduced reference)")
    t_total = time.time() - t_total_0

    findings, first_violator, first_violator_label = _verdict(prod, red)

    summary = [
        "",
        "=" * 110,
        "Sprint 19 part 2 post-Phase-B trace result",
        "=" * 110,
        f"  Run A (n_B={prod['n_B']}, production):  "
        f"Neff={prod['Neff']:.4f}  Yp={prod['Yp']:.5f}  "
        f"δNeff_ss={prod['delta_neff_ss']:.4f}  "
        f"({prod['wall_clock_s']:.0f}s)",
        f"  Run B (n_B={red['n_B']}, reduced):  "
        f"Neff={red['Neff']:.4f}  Yp={red['Yp']:.5f}  "
        f"δNeff_ss={red['delta_neff_ss']:.4f}  "
        f"({red['wall_clock_s']:.0f}s)",
        "",
        f"  Tg_trace (Tg_C[-1]):  prod={prod['trace'].get('Tg_trace', 0.0):.6e} MeV  "
        f"red={red['trace'].get('Tg_trace', 0.0):.6e} MeV",
        f"  a_B_end:  prod={prod['trace'].get('a_B_end', 0.0):.6e}  "
        f"red={red['trace'].get('a_B_end', 0.0):.6e}",
        "",
        "  Per-stage max relative deviation (production vs reduced):",
    ]
    for name, dev, tol, payload in findings:
        marker = "  ⚠" if dev > tol else "   "
        if name.startswith("rho_scalar:"):
            rho_p, rho_r = payload
            summary.append(f"  {marker}{name:32s}  rho_p={rho_p:.4e}  "
                           f"rho_r={rho_r:.4e}  Δ={dev*100.0:+.2f}%  "
                           f"(tol {tol*100.0:.1f}%)")
        else:
            summary.append(f"  {marker}{name:32s}  max_dev={dev*100.0:.4f}%  "
                           f"(tol {tol*100.0:.1f}%)")

    summary.append("")
    if first_violator is not None:
        name, dev, tol = first_violator
        summary.append(f"  VERDICT: first divergence at {first_violator_label}")
        summary.append(f"           {name}: {dev*100.0:.4f}% > tol {tol*100.0:.1f}%")
        summary.append("")
        if first_violator_label.startswith("Stage 1"):
            summary.append("  CURE PATTERN (cure_design §4.2): tighten interpolator "
                           "construction in update_thermo_distributions behind a "
                           "qke_post_phaseB_clamp_flag.")
        elif first_violator_label.startswith("Stage 2"):
            summary.append("  CURE PATTERN (cure_design §4.2): the f_α(p) callables "
                           "diverge despite ~equal raw grids — the y → p mapping or "
                           "interpolation tail is the bug.")
        elif first_violator_label.startswith("Stage 3"):
            summary.append("  CURE PATTERN (cure_design §4.3): tighten Phase-C "
                           "solve_ivp tolerances behind qke_phaseC_strict_tol_flag.")
        else:
            summary.append("  CURE PATTERN (cure_design §4.4): bump p_npoints_nu / "
                           "refine the GL quadrature grid for the rho_nu_from_f "
                           "integrand at low Tg.")
    else:
        summary.append("  VERDICT: no stage exceeded its tolerance. Re-examine "
                       "tolerances or check that the trace flag actually fired.")

    summary.append("")
    summary.append(f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)")
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_post_phaseB_trace.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    # Pack everything into the npz. Trace dicts get flattened.
    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_post_phaseB_trace.npz")

    def _pack(side):
        d = {
            f"{side}_n_B": np.int64(prod_or_red[side]["n_B"]),
            f"{side}_Neff": np.float64(prod_or_red[side]["Neff"]),
            f"{side}_Yp": np.float64(prod_or_red[side]["Yp"]),
            f"{side}_delta_neff_ss": np.float64(prod_or_red[side]["delta_neff_ss"]),
            f"{side}_wall_clock_s": np.float64(prod_or_red[side]["wall_clock_s"]),
            f"{side}_p_grid": np.array(prod_or_red[side]["callables_p_grid"], dtype=float),
        }
        for fn, arr in (prod_or_red[side]["callables_samples"] or {}).items():
            d[f"{side}_sample_{fn}"] = np.array(arr, dtype=float)
        grids = prod_or_red[side]["grids"] or {}
        for k, v in grids.items():
            if isinstance(v, np.ndarray):
                d[f"{side}_grid_{k}"] = v
            else:
                d[f"{side}_grid_{k}"] = np.float64(v)
        trace = prod_or_red[side]["trace"] or {}
        if "t_C" in trace:
            d[f"{side}_t_C"] = trace["t_C"]
        if "Tg_C" in trace:
            d[f"{side}_Tg_C"] = trace["Tg_C"]
        if "Tg_trace" in trace:
            d[f"{side}_Tg_trace"] = np.float64(trace["Tg_trace"])
        if "Tg_B_end" in trace:
            d[f"{side}_Tg_B_end"] = np.float64(trace["Tg_B_end"])
        if "a_B_end" in trace:
            d[f"{side}_a_B_end"] = np.float64(trace["a_B_end"])
        snap = trace.get("integrand_snapshot", {})
        for fn, sub in snap.items():
            d[f"{side}_integrand_{fn}_p_nodes"] = sub["p_nodes"]
            d[f"{side}_integrand_{fn}_f_vals"] = sub["f_vals"]
            d[f"{side}_integrand_{fn}_integrand"] = sub["integrand"]
            d[f"{side}_integrand_{fn}_weights"] = sub["weights"]
            d[f"{side}_integrand_{fn}_rho_scalar"] = np.float64(sub["rho_scalar"])
        return d

    prod_or_red = {"production": prod, "reduced": red}
    payload = {}
    payload.update(_pack("production"))
    payload.update(_pack("reduced"))
    payload["findings_names"] = np.array([n for n, _, _, _ in findings])
    payload["findings_devs"] = np.array([d for _, d, _, _ in findings],
                                          dtype=float)
    payload["findings_tols"] = np.array([t for _, _, t, _ in findings],
                                          dtype=float)
    if first_violator is not None:
        payload["verdict_first_violator"] = np.array(first_violator[0])
        payload["verdict_first_violator_dev"] = np.float64(first_violator[1])
        payload["verdict_first_violator_label"] = np.array(first_violator_label)
    else:
        payload["verdict_first_violator"] = np.array("(none)")
        payload["verdict_first_violator_dev"] = np.float64(0.0)
        payload["verdict_first_violator_label"] = np.array("(no divergence above tol)")

    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved trace dump: {out_npz}", flush=True)
    print(f"Saved summary: {out_txt}", flush=True)
