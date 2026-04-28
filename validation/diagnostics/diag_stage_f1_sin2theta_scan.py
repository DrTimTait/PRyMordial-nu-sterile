"""Stage F sprint 1: sin²2θ_24 saturation-curve scan at fixed δm² = 0.93 eV².

Stage E.2 closed substantially with 3/4 Hannestad benchmark points in HTT 2012
bands (sprint-19-part-2). The outlier — global-fit (NH) at sin²2θ_24 = 0.089,
δm² = 0.9 eV² — overshoots its expected δNeff_ss = 0.55 by ~70% (lands at 0.944,
near Point A's 0.915 at sin²2θ = 0.1). Stage F sprint 1 tests whether the project's
L=0 NH non-resonant QKE saturates at the strong-mixing plateau for any
sin²2θ_24 ≳ 0.05 — i.e. whether the QKE driver fails to differentiate 0.089
from 0.1 the way HTT 2012 does — by mapping δNeff_ss(sin²2θ_24) directly at
fixed δm² = 0.93 eV² over four scan values.

| Scan point | sin²2θ_24  | δm² (eV²) | role                                |
|------------|-----------|-----------|-------------------------------------|
| s1         | 0.01      | 0.93      | sub-saturation regime               |
| s2         | 0.05      | 0.93      | proposed saturation onset           |
| s3         | 0.1       | 0.93      | matches gate-5 Point A (cross-check)|
| s4         | 0.5       | 0.93      | extreme strong mixing               |

All under cured closure config at production n_B=12000. Sequential ~75 min
each plus overhead → ~5 h total. Active health gates kept from gate 4
(Neff ≤ 4.0, Yp ≤ 0.255 — confirms cure remains effective across the scan).

Saturation verdict logic: compute ratios δNeff_ss[i+1] / δNeff_ss[i] for the
upper-three intervals (s1→s2, s2→s3, s3→s4). Saturation CONFIRMED if the
top two ratios (s2→s3 and s3→s4) ∈ [0.95, 1.05]. Saturation FALSIFIED if any
of the upper ratios > 1.10. Otherwise PARTIAL.

Output:
* `diag_stage_f1_sin2theta_scan.npz` — per-point Neff, Yp, δNeff_ss,
  Σρ_ss(raw), wall-clock; saturation ratios + verdict.
* `diag_stage_f1_sin2theta_scan.out` — saturation table + verdict + Stage F
  sprint 1b direction recommendation.
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
# Confirmed sin²2θ-agnostic by Stage F sprint 1 pre-flight; the cure fires
# on a polyfit numerical artefact, not on mixing strength.
CURE_FLAGS = [("qke_post_phaseB_clamp_flag", True)]


# Saturation-curve scan points (fixed δm² = 0.93 eV², varying sin²2θ_24).
POINTS = [
    {
        "label": "s1 (sub-saturation, sin²2θ=0.01)",
        "sin2_2theta_24": 0.01,
        "Dm2_41": 0.93,
        "n_B": 12000,
    },
    {
        "label": "s2 (proposed onset, sin²2θ=0.05)",
        "sin2_2theta_24": 0.05,
        "Dm2_41": 0.93,
        "n_B": 12000,
    },
    {
        "label": "s3 (gate-5 Point A cross-check, sin²2θ=0.1)",
        "sin2_2theta_24": 0.1,
        "Dm2_41": 0.93,
        "n_B": 12000,
    },
    {
        "label": "s4 (extreme strong mixing, sin²2θ=0.5)",
        "sin2_2theta_24": 0.5,
        "Dm2_41": 0.93,
        "n_B": 12000,
    },
]

# Active-sector health checks (cure must keep these near-SM at every point).
ACTIVE_NEFF_MAX = 4.0
ACTIVE_YP_MAX = 0.255

# Saturation-ratio bands.
SAT_FLAT_LO = 0.95
SAT_FLAT_HI = 1.05
SAT_RISE_HI = 1.10


def _set_flags(point):
    """Sprint-19 closure config + cure flag(s) — identical to gate-5 scan."""
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

    active_healthy = (Neff <= ACTIVE_NEFF_MAX and Yp <= ACTIVE_YP_MAX)

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  D/H={DH:.4f}  "
          f"sum_ss(raw)={sum_ss:.4f}  δNeff_ss={delta_neff_ss}  "
          f"active_healthy={active_healthy}  "
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
        "active_healthy": bool(active_healthy),
        "wall_clock_s": float(dt_run),
    }


def _saturation_verdict(results):
    """Compute upper-interval saturation ratios and emit the verdict."""
    if any(np.isnan(r["delta_neff_ss"]) or r["delta_neff_ss"] <= 0.0
           for r in results):
        return ("INDETERMINATE",
                "One or more points returned non-finite δNeff_ss.",
                [float("nan")] * (len(results) - 1))

    deltas = [r["delta_neff_ss"] for r in results]
    ratios = [deltas[i + 1] / deltas[i] for i in range(len(deltas) - 1)]
    # Upper-interval ratios are the s2→s3 and s3→s4 transitions.
    upper = ratios[1:]  # index 1 = s2→s3, index 2 = s3→s4

    if all(SAT_FLAT_LO <= r <= SAT_FLAT_HI for r in upper):
        verdict = "SATURATION CONFIRMED"
        detail = (
            f"Upper ratios {upper} both in [{SAT_FLAT_LO}, {SAT_FLAT_HI}]: "
            f"δNeff_ss is flat across sin²2θ ∈ [0.05, 0.5]. "
            f"Stage F sprint 1b should pursue lepton-asymmetry seeding "
            f"or resonance-aware Phase-0 to break the plateau "
            f"(L=0 NH non-resonant QKE is missing the physics that "
            f"differentiates 0.089 from 0.1).")
    elif any(r > SAT_RISE_HI for r in upper):
        verdict = "SATURATION FALSIFIED"
        detail = (
            f"Upper ratios {upper} include rise > {SAT_RISE_HI}: "
            f"δNeff_ss continues to grow with sin²2θ across the upper "
            f"scan range. Strong-mixing-plateau hypothesis is wrong. "
            f"Stage F sprint 1b should re-open Suspect 8 (HTT 2012 target "
            f"band incompatible with V_nunu setup) or re-investigate the "
            f"cure's interaction with strong-mixing dynamics.")
    else:
        verdict = "PARTIAL SATURATION"
        detail = (
            f"Upper ratios {upper} are between flat and rise bands. "
            f"Stage F sprint 1b should run a finer-grained scan to "
            f"localise the saturation onset.")
    return verdict, detail, ratios


if __name__ == "__main__":
    if not CURE_FLAGS:
        print("WARNING: CURE_FLAGS is empty. This scan will run with the "
              "sprint-18 closure config WITHOUT any sprint-19-part-2 cure. "
              "Expect production-n_B active-sector pathology.\n", flush=True)

    header = [
        "=" * 110,
        "Stage F sprint 1: sin²2θ_24 saturation-curve scan at fixed δm²=0.93 eV²",
        f"  Cure flags: {CURE_FLAGS or '(none — uncured config)'}",
        "  Closure config: damping='symmetric', active_only=True, "
        "qke_phase0_flag=False, n_B=12000",
        "  Phase B from T=100 MeV to T=0.005 MeV",
        "  Active health gates per point: Neff ≤ 4.0, Yp ≤ 0.255",
        "  Probing the strong-mixing-plateau hypothesis from sprint-19-part-2 §8.5",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    results = []
    for point in POINTS:
        results.append(_run_one(point))
    t_total = time.time() - t_total_0

    verdict, verdict_detail, ratios = _saturation_verdict(results)

    summary = ["", "=" * 110,
               "Stage F sprint 1 sin²2θ saturation-curve results",
               "=" * 110]
    for r in results:
        marker = "OK" if r["active_healthy"] else "ACTIVE-UNHEALTHY"
        summary.append(
            f"  [{marker:<16s}] {r['label']:42s}  "
            f"δNeff_ss={r['delta_neff_ss']:.4f}  "
            f"Neff={r['Neff']:.3f}  Yp={r['Yp']:.4f}  "
            f"sum_ss(raw)={r['sum_ss_raw']:.3f}  "
            f"({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        "  Saturation ratios (δNeff_ss[i+1] / δNeff_ss[i]):",
    ]
    for i, ratio in enumerate(ratios):
        a = results[i]["sin2_2theta_24"]
        b = results[i + 1]["sin2_2theta_24"]
        summary.append(f"    sin²2θ {a:.3g} → {b:.3g}: ratio = {ratio:.4f}")
    summary += [
        "",
        f"  Stage F sprint 1 verdict: {verdict}",
        f"    {verdict_detail}",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f1_sin2theta_scan.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f1_sin2theta_scan.npz")
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
        "active_healthy": np.array([r["active_healthy"] for r in results]),
        "wall_clock_s": np.array([r["wall_clock_s"] for r in results]),
        "saturation_ratios": np.array(ratios),
        "verdict": np.array(verdict),
        "verdict_detail": np.array(verdict_detail),
    }
    np.savez_compressed(out_npz, **payload)
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
