"""Stage E.2 sprint 18 joint axis bracket: V_nunu legacy + HTT damping.

Single ETDRK2 run at reduced n_B = 2500 + n_B_phase0 = 1000 (apples-to-
apples with sprint-17 Run A baseline Σρ_ss(raw) = 14.5122). The
configuration is the JOINT of sprint-17 Run B (V_nunu axis flip) and
Run C (damping axis flip):

    qke_v_nunu_active_only = False       (sprint-5 legacy V_nunu)
    qke_damping_formula = "symmetric"    (HTT 2012 Eq. 2.15-16)

This is the most-HTT-2012-consistent configuration we can produce with
existing flags. Sprint 17 single-axis brackets gave:

    Run A (project default; True / mirizzi):  δNeff_ss = 0.629  (6× above band)
    Run B (V_nunu flip;     False / mirizzi): δNeff_ss = 0.963  (worse)
    Run C (damping flip;    True / symmetric): δNeff_ss = 0.331  (3.3× above band)

Decision tree at completion (per sprint-18 brief Phase 1):

    δNeff_ss in [0.02, 0.10]   → Stage E.2 closes with multi-axis cure;
                                  document HTT-matching config in PRyM_init.py
                                  and propose default flips for Hannestad
                                  benchmarks.
    δNeff_ss in [0.10, 0.30]   → close to band, not in band; investigate
                                  Ny grid axis before sprint-19 escalation.
    δNeff_ss > 0.30            → axes do not compose toward the band;
                                  sprint-19 escalation: structural look
                                  at V_nunu live-integral vs HTT closed
                                  form V_1.
    δNeff_ss < 0.02            → over-shoots; isolate via single-axis
                                  re-confirmation.

Total expected wall-clock: ~25 min (one ETDRK2 run at reduced n_B).
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


NEFF_3X3_ETDRK2 = 3.00034              # SM 3-flavor baseline
SUM_SS_GATE_5B = 14.5122               # ETDRK2 reduced-n_B Hannestad Point C baseline
HANNESTAD_BAND_LO = 0.02               # δNeff lower edge from HTT 2012 Fig. 2 top blue curve
HANNESTAD_BAND_HI = 0.10               # δNeff upper edge

# Sprint-17 single-axis bracket reference values (for delta computation)
SPRINT17_RUN_A = {"sum_raw": 14.5122, "delta_neff_ss": 0.6287, "Neff": 9.8534, "Yp": 0.31284}
SPRINT17_RUN_B = {"sum_raw": 27.8882, "delta_neff_ss": 0.9629, "Neff": 3.0897, "Yp": 0.24953}
SPRINT17_RUN_C = {"sum_raw": 7.1729,  "delta_neff_ss": 0.3312, "Neff": 4.1040, "Yp": 0.26131}


def _set_flags():
    """Hannestad Point C config at reduced n_B; ETDRK2 driver; JOINT axis configuration."""
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
    PRyMini.n_B_override = 2500
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # JOINT: HTT-matching damping + sprint-5 legacy V_nunu projection
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = False
    # Standard scales (no diagnostic detuning)
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    # Phase-0 + Phase-B segment chain
    PRyMini.T_boltz_start = 30.0
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 1000


def _delta_neff_ss_from_rho(rho_final):
    """Convert the final density matrix to δN_eff_ss (Hannestad units).

    δN_eff_ss = (Σ_sectors Σ_y y³ ρ_ss(y, s)) / (Σ_sectors Σ_y y³ ρ_νe(y, s))

    Numerator: sterile (flavor index 3) energy moment, summed across both
    sectors. Denominator: active ν_e (flavor index 0) energy moment under
    the SAME (y_grid, dy) discretisation (≈ one-thermal-species energy).
    Ratio cancels T_ν,com⁴ exactly and gives Hannestad's convention.
    See doc/STAGE_E2_SPRINT17_LITERATURE.md §1.3 for derivation.
    """
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


def _format_run(label, sum_raw, delta_neff_ss, Neff, Yp, wall_s=None):
    de_str = f"{delta_neff_ss:.4f}" if delta_neff_ss is not None else "n/a"
    wall_str = f"({wall_s:.0f}s)" if wall_s is not None else ""
    return (
        f"  {label:<46s}  "
        f"Σρ_ss(raw)={sum_raw:.4f}  "
        f"δNeff_ss={de_str}  "
        f"Neff={Neff:.4f}  "
        f"Yp={Yp:.5f}  "
        f"{wall_str}"
    ).rstrip()


def _verdict(delta_neff_ss, Neff, Yp, sum_raw):
    lines = [""]
    if delta_neff_ss is None:
        lines.append("  VERDICT: δNeff_ss could not be computed; harness output is incomplete.")
        return lines

    if delta_neff_ss < HANNESTAD_BAND_LO:
        lines.append(
            f"  VERDICT (δNeff_ss = {delta_neff_ss:.4f} < {HANNESTAD_BAND_LO}): joint configuration "
            "OVERSHOOTS the Hannestad band toward zero. Stage E.2 ambiguous; one or "
            "both axes are over-correcting. Run single-axis re-confirmation passes "
            "to isolate the dominant cure direction.")
    elif HANNESTAD_BAND_LO <= delta_neff_ss <= HANNESTAD_BAND_HI:
        lines.append(
            f"  VERDICT (δNeff_ss = {delta_neff_ss:.4f} in [{HANNESTAD_BAND_LO}, {HANNESTAD_BAND_HI}]): "
            "**SUSPECT 8 CURED BY JOINT CONFIGURATION**. The HTT 2012-matching joint "
            "(active_only=False, damping='symmetric') reproduces the published δNeff "
            "band for sin²(2θ_24)=1e-4, δm²=0.93 eV², L=0, NH. Stage E.2 closes here. "
            "Recommended actions:")
        lines.append(
            "    1. Re-run all four Hannestad benchmark points (A: sin²2θ=0.1; B: 2.26e-3; "
            "C: 1e-4; plus the global-fit point at sin²2θ=0.089, δm²=0.9 eV²) under "
            "the joint configuration to confirm the band is reproduced across the "
            "parameter scan.")
        lines.append(
            "    2. Document the HTT-2012 vs project-default rationale in "
            "PRyM_init.py (line 165-200) and propose default flips for "
            "Hannestad-style benchmarks (Stage F task).")
        lines.append(
            "    3. Investigate the V_nunu paradox (Run B near-SM Neff/Yp) at the "
            "structural level — the active-only projection's mechanism is unclear.")
    elif HANNESTAD_BAND_HI < delta_neff_ss <= 0.30:
        lines.append(
            f"  VERDICT (δNeff_ss = {delta_neff_ss:.4f} in ({HANNESTAD_BAND_HI}, 0.30]): "
            "joint configuration brings the gap to within 3× of the band — "
            "much closer than any single-axis configuration but not in band. "
            "Sprint 19 should investigate the momentum-grid axis (Ny=200 or 300, "
            "Kainulainen-Sorri non-uniform mapping per HTT 2012 Eq. 3.3) before "
            "declaring Suspect 8 unresolved. Phase-0/Phase-B handoff diagnostic "
            "(sprint-18 Phase 2) remains relevant.")
    else:
        lines.append(
            f"  VERDICT (δNeff_ss = {delta_neff_ss:.4f} > 0.30): joint configuration does NOT "
            "compose the single-axis effects toward the Hannestad band. Either "
            "V_nunu has a non-linear interaction with damping that we have not "
            "understood, or there is a third structural axis (likely the Phase-0/"
            "Phase-B handoff or the live-integral V_nunu vs HTT's closed form V_1) "
            "that is the actual divergence.")
        lines.append(
            "    Sprint-19 escalation: implement HTT 2012's closed-form thermal "
            "V_1 (Eq. 2.10) as a third option for qke_v_nunu_form alongside "
            "live-integral active-only and live-integral full. This is a "
            "structural change, not a flag flip — ~1 day of implementation.")

    # Composition diagnostic — does the joint result look like A+ΔB+ΔC, A·(B/A)·(C/A), or non-linear?
    A = SPRINT17_RUN_A["delta_neff_ss"]
    B = SPRINT17_RUN_B["delta_neff_ss"]
    C = SPRINT17_RUN_C["delta_neff_ss"]
    pred_lin = A + (B - A) + (C - A)
    pred_mul = A * (B / A) * (C / A)
    lines.append("")
    lines.append(
        f"  Composition check (sprint-17 single-axis baselines: A={A:.3f}, B={B:.3f}, C={C:.3f}):")
    lines.append(
        f"    Linear-superposition prediction: A + (B-A) + (C-A) = {pred_lin:.3f}")
    lines.append(
        f"    Multiplicative prediction:       A · (B/A) · (C/A) = {pred_mul:.3f}")
    lines.append(
        f"    Observed joint:                                      {delta_neff_ss:.3f}")
    lin_err = abs(delta_neff_ss - pred_lin) / max(abs(pred_lin), 1e-9)
    mul_err = abs(delta_neff_ss - pred_mul) / max(abs(pred_mul), 1e-9)
    if lin_err < 0.10:
        lines.append(
            f"    Linear superposition matches within 10% (relative error {lin_err*100:.1f}%); "
            "axes appear to compose linearly.")
    elif mul_err < 0.10:
        lines.append(
            f"    Multiplicative composition matches within 10% (relative error {mul_err*100:.1f}%); "
            "axes appear to compose multiplicatively.")
    else:
        lines.append(
            f"    Neither linear nor multiplicative composition matches within 10% "
            f"(lin err {lin_err*100:.1f}%, mul err {mul_err*100:.1f}%); axes interact non-linearly. "
            "The joint case is genuinely beyond extrapolation from single-axis bracket data.")

    return lines


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 18 joint axis bracket: V_nunu legacy + HTT damping",
        f"  Hannestad Point C: sin²(2θ_24)=1e-4, δm²_41=0.93 eV², L=0, NH",
        f"  Reduced n_B = 2500 + n_B_phase0 = 1000",
        f"  Joint configuration: qke_v_nunu_active_only=False, qke_damping_formula='symmetric'",
        f"  HTT 2012 reference band: δNeff in [{HANNESTAD_BAND_LO}, {HANNESTAD_BAND_HI}]",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    print("", flush=True)
    print("  Sprint-17 single-axis baselines for context:", flush=True)
    print(_format_run("A: active_only=True,  damping='mirizzi'  (default)",
                      SPRINT17_RUN_A["sum_raw"], SPRINT17_RUN_A["delta_neff_ss"],
                      SPRINT17_RUN_A["Neff"], SPRINT17_RUN_A["Yp"]), flush=True)
    print(_format_run("B: active_only=False, damping='mirizzi'  (V_nunu flip)",
                      SPRINT17_RUN_B["sum_raw"], SPRINT17_RUN_B["delta_neff_ss"],
                      SPRINT17_RUN_B["Neff"], SPRINT17_RUN_B["Yp"]), flush=True)
    print(_format_run("C: active_only=True,  damping='symmetric' (HTT)",
                      SPRINT17_RUN_C["sum_raw"], SPRINT17_RUN_C["delta_neff_ss"],
                      SPRINT17_RUN_C["Neff"], SPRINT17_RUN_C["Yp"]), flush=True)

    _set_flags()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    print("", flush=True)
    print("--- Joint run: active_only=False, damping='symmetric' (HTT) ---", flush=True)
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt_run = time.time() - t0

    Neff = float(res[0])
    Yp = float(res[4])
    DoH = float(res[5])
    sum_ss = 0.0
    delta_neff_ss = None
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)

    summary = [
        "",
        "=" * 110,
        "Sprint 18 joint axis bracket result",
        "=" * 110,
        _format_run(
            "JOINT: active_only=False, damping='symmetric' (HTT)",
            sum_ss, delta_neff_ss, Neff, Yp, dt_run),
    ]
    summary.extend(_verdict(delta_neff_ss, Neff, Yp, sum_ss))
    summary.append("")
    summary.append(f"  Total wall-clock: {dt_run:.0f}s ({dt_run/60:.1f} min)")
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint18_joint_axis.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write("\n  Sprint-17 single-axis baselines for context:\n")
        fh.write(_format_run("A: active_only=True,  damping='mirizzi'  (default)",
                             SPRINT17_RUN_A["sum_raw"], SPRINT17_RUN_A["delta_neff_ss"],
                             SPRINT17_RUN_A["Neff"], SPRINT17_RUN_A["Yp"]) + "\n")
        fh.write(_format_run("B: active_only=False, damping='mirizzi'  (V_nunu flip)",
                             SPRINT17_RUN_B["sum_raw"], SPRINT17_RUN_B["delta_neff_ss"],
                             SPRINT17_RUN_B["Neff"], SPRINT17_RUN_B["Yp"]) + "\n")
        fh.write(_format_run("C: active_only=True,  damping='symmetric' (HTT)",
                             SPRINT17_RUN_C["sum_raw"], SPRINT17_RUN_C["delta_neff_ss"],
                             SPRINT17_RUN_C["Neff"], SPRINT17_RUN_C["Yp"]) + "\n")
        fh.write(text + "\n")
