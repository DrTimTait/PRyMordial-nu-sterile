"""Stage E.2 sprint 19 part 2: four-Hannestad-point benchmark scan.

After sprint-19-part-2 cure is implemented (qke_post_phaseB_*_flag set in
_set_flags below), this harness extends the HTT 2012 reproduction beyond
sprint 18's single Point C to all four Hannestad benchmark points:

| Point      | sin²2θ_24  | δm² (eV²) | HTT 2012 expected δNeff_ss | Pass band       |
|------------|-----------|-----------|----------------------------|-----------------|
| A          | 0.1       | 0.93      | ~1.0                       | [0.9, 1.1]      |
| B          | 2.26e-3   | 0.93      | ~0.5                       | [0.3, 0.7]      |
| C          | 1e-4      | 0.93      | ~0.03 (already in band)    | [0.02, 0.10]    |
| Global-fit | 0.089     | 0.9       | =1 (HTT page 8, both NH/IH) | [0.9, 1.1]      |

All under cured closure config at production n_B. Sequential ~35 min each
plus overhead → ~2.5 h. Targets: each point's δNeff_ss must land inside
its pass band AND the active sector must remain healthy (Neff ≤ 4.0,
Yp ≤ 0.255).

Output:
* `diag_sprint19_hannestad_scan.npz` — per-point Neff, Yp, δNeff_ss,
  Σρ_ss(raw), wall-clock.
* `diag_sprint19_hannestad_scan.out` — pass/fail verdict per point and
  the overall Stage E.2 closure decision (full / substantial / incomplete).
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


# Sprint-19-part-2 cure (per STAGE_E2_SPRINT19_CURE_DESIGN.md §8.2):
# extend the BoltzmannSolver FD-tail-extrapolation fallback to also fire
# when polyfit's _tail_b is below the FD-equivalent floor 1/y_grid[-1].
CURE_FLAGS = [("qke_post_phaseB_clamp_flag", True)]


# Four Hannestad benchmark points from HTT 2012 Fig. 2 / Table.
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
        "label": "C (narrow mixing — sprint 18 in-band)",
        "sin2_2theta_24": 1e-4,
        "Dm2_41": 0.93,
        "expected_delta_neff_ss": 0.03,
        "pass_band": (0.02, 0.10),
        "n_B": 12000,
    },
    {
        # HTT 2012 page 8 (§3.2) explicit: δNeff = 1 at this point
        # under L=0 NH (and L=0 IH). The 0.55 / [0.4, 0.7] cited
        # in earlier sprints was a citation error — see
        # doc/STAGE_F_SPRINT3HD_HANNESTAD_AUDIT.md. The NPZ data
        # from earlier scans is still valid; only the verdict
        # cell was misclassified.
        "label": "Global-fit (NH)",
        "sin2_2theta_24": 0.089,
        "Dm2_41": 0.9,
        "expected_delta_neff_ss": 1.0,
        "pass_band": (0.9, 1.1),
        "n_B": 12000,
    },
]

# Active-sector health checks (cured config must keep these near-SM at every point).
ACTIVE_NEFF_MAX = 4.0
ACTIVE_YP_MAX = 0.255


def _set_flags(point):
    """Sprint-19 closure config + cure flag(s)."""
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
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(point["n_B"])
    PRyMini.n_B_phase0_override = 0
    PRyMini.qke_active_probe_flag = False
    PRyMini.qke_post_phaseB_trace_flag = False
    # Cure flag(s) — set per sprint-19-part-2 verdict.
    for flag_name, flag_value in CURE_FLAGS:
        setattr(PRyMini, flag_name, flag_value)


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
    print(f"\n--- Point {point['label']}: "
          f"sin²2θ={point['sin2_2theta_24']:.3e}, "
          f"δm²={point['Dm2_41']} eV², n_B={point['n_B']} ---", flush=True)
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
          f"sum_ss(raw)={sum_ss:.4f}  δNeff_ss={delta_neff_ss}  "
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
        return "FULL Stage E.2 closure", "All four points in band; active sector healthy."
    elif n_pass == n_total - 1:
        out = next(r for r in results if not r["pass_full"])
        return ("SUBSTANTIAL Stage E.2 closure (one outlier)",
                f"3/4 points pass; outlier: {out['label']}")
    else:
        return ("INCOMPLETE Stage E.2 closure",
                f"{n_pass}/{n_total} points pass; sprint-20 investigation needed.")


if __name__ == "__main__":
    if not CURE_FLAGS:
        print("WARNING: CURE_FLAGS is empty. This scan will run with the "
              "sprint-18 closure config WITHOUT any sprint-19-part-2 cure. "
              "Expect production-n_B active-sector pathology.\n", flush=True)

    header = [
        "=" * 110,
        "Stage E.2 sprint 19 part 2: four-Hannestad-point benchmark scan",
        f"  Cure flags: {CURE_FLAGS or '(none — uncured config)'}",
        "  Closure config: damping='symmetric', active_only=True, "
        "qke_phase0_flag=False, n_B=12000",
        "  Phase B from T=100 MeV to T=0.005 MeV",
        "  Active health gates per point: Neff ≤ 4.0, Yp ≤ 0.255",
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

    summary = ["", "=" * 110, "Sprint 19 part 2 four-point Hannestad scan results",
               "=" * 110]
    for r in results:
        marker = "PASS" if r["pass_full"] else "FAIL"
        summary.append(
            f"  [{marker}] {r['label']:38s}  "
            f"δNeff_ss={r['delta_neff_ss']:.4f} "
            f"(expected {r['expected_delta_neff_ss']:.2f}, band [{r['band_lo']:.2f}, "
            f"{r['band_hi']:.2f}])  "
            f"Neff={r['Neff']:.3f}  Yp={r['Yp']:.4f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  Stage E.2 verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_hannestad_scan.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_hannestad_scan.npz")
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
