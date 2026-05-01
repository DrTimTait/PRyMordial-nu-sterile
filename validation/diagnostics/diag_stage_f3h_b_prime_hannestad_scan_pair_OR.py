"""Stage F sprint 3h-b' (prime): four-Hannestad-point scan under pair-OR per_flavor clamp.

Sprint 3h-b (avg-pair-symm) regressed the active sector at mid/narrow
mixing because the avg-of-slopes firing rule is strictly LAXER than
per_flavor:
  pair-avg < target  ⟹  at least one member < target  (firing condition)
but NOT vice versa — so pair-avg under-fires whenever individual slopes
straddle the target. At Point A both members drift below target uniformly
(strong mixing → both QKE-deformed) so avg < target → fires every step
→ Neff=2.99 (PASS). At Point B/C asymmetric drift gives avg > target
intermittently → under-fires → tails approach pre-cure pathology →
Neff blows up (4.37 at B, 8.84 at C).

Sprint 3h-b' tests the strict-symmetric reading: use min(b1, b2) for
the firing test, equivalent to logical OR — fire on both members
whenever EITHER member's individual slope is below target. This
strictly ADDS firings relative to per_flavor (every per_flavor fire
on ν_e now also clamps ν̄_e), preserving per_flavor's coverage AND
restoring nu-nubar tail symmetry (which kills the Pauli-blocking
imbalance that drove sprint-3e's sign-flipped Yp shift).

Hypothesis: 3h-b' lands 3/4 PASS (matching gate-5) AND right-sign
dYp_sterile at all four points. If true, this is the cleanest cure
shipped to date — strictly better than gate-5 (right-sign Yp) and
strictly better than 3h-a/3h-b (3/4 vs 1/4 coverage).

Reference walls:
  Mode               A         B         C         NH
  ---------------    -------   -------   -------   -------
  gate-5 per_flavor  4315 s    4262 s    4916 s    4896 s
  3h-a uniform       4498 s    5649 s    5169 s    4645 s
  3h-b avg-pair      4452 s    4473 s    4849 s    5123 s

Sequential ~5 h wall-clock at production n_B=12000.

Output:
  diag_stage_f3h_b_prime_hannestad_scan_pair_OR.npz  Per-point results.
  diag_stage_f3h_b_prime_hannestad_scan_pair_OR.out  Closure decision +
                                                       trade-off analysis.
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
SBBN_NEFF_REFERENCE = 3.044
ACTIVE_NEFF_MAX = 4.0
ACTIVE_YP_MAX = 0.255

# Saviano 2013 +-0.005 expected sterile-induced Yp shift bracket.
LIT_YP_SHIFT_LO = 0.001
LIT_YP_SHIFT_HI = 0.012

# F3d sterile-OFF SBBN baseline (from gate-5 reference).
SBBN_YP_BASELINE = 0.24717


# Four Hannestad benchmark points (matching gate-5 / 3h-a / 3h-b scans).
POINTS = [
    {
        "label": "A (strong mixing)",
        "sin2_2theta_24": 0.1,
        "Dm2_41": 0.93,
        "expected_delta_neff_ss": 1.0,
        "pass_band": (0.9, 1.1),
        "n_B": 12000,
    },
    {
        "label": "B (mid mixing)",
        "sin2_2theta_24": 2.26e-3,
        "Dm2_41": 0.93,
        "expected_delta_neff_ss": 0.5,
        "pass_band": (0.3, 0.7),
        "n_B": 12000,
    },
    {
        "label": "C (narrow mixing)",
        "sin2_2theta_24": 1e-4,
        "Dm2_41": 0.93,
        "expected_delta_neff_ss": 0.03,
        "pass_band": (0.02, 0.10),
        "n_B": 12000,
    },
    {
        # HTT 2012 page 8 (§3.2) explicit: "(δm²_s, sin²2θ_s) =
        # (0.9 eV², 0.089). For that point δNeff = 1 in both
        # hierarchies, i.e. complete thermalization occurs."
        # The earlier 0.55 / [0.4, 0.7] entry was a citation
        # error (sprint 3h-d audit); HTT's 0.55 is at a
        # different IH-resonance illustration point on page 10.
        "label": "Global-fit (NH)",
        "sin2_2theta_24": 0.089,
        "Dm2_41": 0.9,
        "expected_delta_neff_ss": 1.0,
        "pass_band": (0.9, 1.1),
        "n_B": 12000,
    },
]


# Reference values: gate-5 sprint-19 (per_flavor + y_grid).
GATE5_PRIOR = {
    "A (strong mixing)":   {"delta_neff_ss": 0.9151, "Yp": 0.2386,
                              "Neff": 2.271},
    "B (mid mixing)":      {"delta_neff_ss": 0.6454, "Yp": 0.2400,
                              "Neff": 2.250},
    "C (narrow mixing)":   {"delta_neff_ss": 0.0947, "Yp": 0.2386,
                              "Neff": 2.271},
    "Global-fit (NH)":     {"delta_neff_ss": 0.9445, "Yp": 0.2429,
                              "Neff": 2.359},
}

# Reference values: sprint 3h-a (full-uniform).
F3HA_PRIOR = {
    "A (strong mixing)":   {"delta_neff_ss": 0.9150, "Yp": 0.2509,
                              "Neff": 3.279},
    "B (mid mixing)":      {"delta_neff_ss": 0.2173, "Yp": 0.2524,
                              "Neff": 3.445},
    "C (narrow mixing)":   {"delta_neff_ss": 0.1809, "Yp": 0.2536,
                              "Neff": 3.367},
    "Global-fit (NH)":     {"delta_neff_ss": 0.9448, "Yp": 0.2509,
                              "Neff": 3.257},
}

# Reference values: sprint 3h-b (avg-pair-symm).
F3HB_PRIOR = {
    "A (strong mixing)":   {"delta_neff_ss": 0.9431, "Yp": 0.2486,
                              "Neff": 2.987},
    "B (mid mixing)":      {"delta_neff_ss": 0.4040, "Yp": 0.2385,
                              "Neff": 4.367},
    "C (narrow mixing)":   {"delta_neff_ss": 0.0273, "Yp": 0.2319,
                              "Neff": 8.837},
    "Global-fit (NH)":     {"delta_neff_ss": 0.8995, "Yp": 0.2413,
                              "Neff": 2.555},
}


def _set_flags(point):
    """3h-b' cure: closure-config + per_flavor + T_nu_init + pair_symmetric=True + rule='min'."""
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
    PRyMini.Dm2_41 = float(point["Dm2_41"])
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(float(point["sin2_2theta_24"]))) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # Closure config + 3h-b' cure.
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_post_phaseB_clamp_flag = True
    PRyMini.qke_phaseB_clamp_anchor = "T_nu_init"
    PRyMini.qke_phaseB_clamp_mode = "per_flavor"
    PRyMini.qke_phaseB_clamp_uniform_at_end = False
    PRyMini.qke_phaseB_clamp_pair_symmetric = True
    PRyMini.qke_phaseB_clamp_pair_symmetric_rule = "min"  # the 3h-b' cure
    PRyMini.qke_phase0_flag = False
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(point["n_B"])
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


def _run_one(point):
    print(f"\n--- {point['label']}: sin^2(2theta)={point['sin2_2theta_24']:.3e}, "
          f"dm^2={point['Dm2_41']} eV^2, n_B={point['n_B']} ---", flush=True)
    _set_flags(point)
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

    band_lo, band_hi = point["pass_band"]
    in_band = (delta_neff_ss is not None
               and band_lo <= delta_neff_ss <= band_hi)
    active_healthy = (Neff <= ACTIVE_NEFF_MAX and Yp <= ACTIVE_YP_MAX)
    pass_full = bool(in_band and active_healthy)

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  D/H={DH:.4f}  "
          f"sum_ss(raw)={sum_ss:.4f}  delta_Neff_ss={delta_neff_ss}  "
          f"in_band={in_band}  active_healthy={active_healthy}  "
          f"({dt_run:.0f}s)", flush=True)

    return {
        "label": point["label"],
        "sin2_2theta_24": point["sin2_2theta_24"],
        "Dm2_41": point["Dm2_41"],
        "n_B": int(point["n_B"]),
        "Neff": Neff,
        "Yp": Yp,
        "DH": DH,
        "sum_ss_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "expected_delta_neff_ss": float(point["expected_delta_neff_ss"]),
        "band_lo": float(band_lo),
        "band_hi": float(band_hi),
        "in_band": bool(in_band),
        "active_healthy": bool(active_healthy),
        "pass_full": pass_full,
        "wall_clock_s": float(dt_run),
    }


def _stage_E2_verdict(results):
    n_total = len(results)
    n_pass = sum(1 for r in results if r["pass_full"])
    if n_pass == n_total:
        return ("FULL Stage E.2 closure",
                "All four points in band; active sector healthy under "
                "3h-b' (pair-OR per_flavor clamp).")
    elif n_pass == n_total - 1:
        out = next(r for r in results if not r["pass_full"])
        return (f"SUBSTANTIAL Stage E.2 closure (one outlier: "
                f"{out['label']})",
                f"3/4 points pass; outlier delta_neff_ss="
                f"{out['delta_neff_ss']:.4f} not in "
                f"[{out['band_lo']:.2f}, {out['band_hi']:.2f}].")
    else:
        return (f"INCOMPLETE Stage E.2 closure ({n_pass}/{n_total} points)",
                "Less coverage than gate-5 sprint-19 (3/4); "
                "pair-OR mode does not preserve coverage.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3h-b': four-Hannestad-point scan under pair-OR per_flavor clamp",
        "  Closure config: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True",
        "  3h-b' cure: anchor='T_nu_init', mode='per_flavor', "
        "pair_symmetric=True, rule='min'",
        "  Production grid: y_max=100, Ny=100, n_B=12000, T_boltz_start=100, "
        "T_start=105 MeV",
        "  References: gate-5 (per_flavor) 3/4 PASS Yp wrong-sign; "
        "3h-a (uniform) 1/4 PASS Yp right-sign; "
        "3h-b (avg-pair) 1/4 PASS active-sector regression at mid/narrow",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for point in POINTS:
        results.append(_run_one(point))
    t_total = time.time() - t_total_0

    verdict, verdict_detail = _stage_E2_verdict(results)

    summary = ["", "=" * 110,
               "Stage F sprint 3h-b' four-Hannestad-point scan under pair-OR mode",
               "=" * 110]
    for r in results:
        marker = "PASS" if r["pass_full"] else "FAIL"
        summary.append(
            f"  [{marker}] {r['label']:38s}  "
            f"delta_Neff_ss={r['delta_neff_ss']:.4f} "
            f"(expected {r['expected_delta_neff_ss']:.2f}, band "
            f"[{r['band_lo']:.2f}, {r['band_hi']:.2f}])  "
            f"Yp={r['Yp']:.4f}  Neff={r['Neff']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        "Comparison vs gate-5 (per_flavor) / 3h-a (uniform) / 3h-b (avg-pair):",
        f"  {'point':<38s} {'g5 dNss':>9s} {'3ha dNss':>9s} {'3hb dNss':>9s} "
        f"{'3hbp dNss':>10s} {'g5 Yp':>8s} {'3ha Yp':>8s} {'3hb Yp':>8s} "
        f"{'3hbp Yp':>9s} {'3hbp dYp_st':>13s}",
    ]
    for r in results:
        prior5 = GATE5_PRIOR.get(r["label"], {})
        priorA = F3HA_PRIOR.get(r["label"], {})
        priorB = F3HB_PRIOR.get(r["label"], {})
        prior5_dnss = prior5.get("delta_neff_ss", float("nan"))
        priorA_dnss = priorA.get("delta_neff_ss", float("nan"))
        priorB_dnss = priorB.get("delta_neff_ss", float("nan"))
        prior5_yp = prior5.get("Yp", float("nan"))
        priorA_yp = priorA.get("Yp", float("nan"))
        priorB_yp = priorB.get("Yp", float("nan"))
        dyp_sterile = r["Yp"] - SBBN_YP_BASELINE
        summary.append(
            f"  {r['label']:<38s} {prior5_dnss:>9.4f} {priorA_dnss:>9.4f} "
            f"{priorB_dnss:>9.4f} {r['delta_neff_ss']:>10.4f} "
            f"{prior5_yp:>8.4f} {priorA_yp:>8.4f} {priorB_yp:>8.4f} "
            f"{r['Yp']:>9.4f} {dyp_sterile:>+13.5f}")
    summary += [
        "",
        f"  Stage E.2 verdict (under 3h-b' cure): {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Sterile-induced Yp shifts vs SBBN (Saviano 2013 expected "
        f"+0.001 to +0.012):",
    ]
    for r in results:
        dyp = r["Yp"] - SBBN_YP_BASELINE
        sign_match = "POSITIVE (literature-consistent)" if dyp > 0 else (
                      "ZERO" if abs(dyp) < 0.001 else
                      "NEGATIVE (literature-inconsistent)")
        summary.append(f"    {r['label']:<38s}  delta_Yp = {dyp:+.5f}  "
                        f"({sign_match})")
    summary += [
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3h_b_prime_hannestad_scan_pair_OR.npz")
    payload = {
        "labels": np.array([r["label"] for r in results]),
        "sin2_2theta_24": np.array([r["sin2_2theta_24"] for r in results]),
        "Dm2_41": np.array([r["Dm2_41"] for r in results]),
        "n_B": np.array([r["n_B"] for r in results]),
        "Neff": np.array([r["Neff"] for r in results]),
        "Yp": np.array([r["Yp"] for r in results]),
        "DH": np.array([r["DH"] for r in results]),
        "sum_ss_raw": np.array([r["sum_ss_raw"] for r in results]),
        "delta_neff_ss": np.array([r["delta_neff_ss"] for r in results]),
        "expected_delta_neff_ss": np.array([r["expected_delta_neff_ss"] for r in results]),
        "band_lo": np.array([r["band_lo"] for r in results]),
        "band_hi": np.array([r["band_hi"] for r in results]),
        "in_band": np.array([r["in_band"] for r in results]),
        "active_healthy": np.array([r["active_healthy"] for r in results]),
        "pass_full": np.array([r["pass_full"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
