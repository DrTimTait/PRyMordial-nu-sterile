"""Stage E.2 sprint 17 axis bracket: V_nunu projection + damping kernel.

Three ETDRK2 runs at reduced n_B = 2500 + n_B_phase0 = 1000 (apples-to-
apples with sprint-16 gate 5b's baseline Σ_raw_ss = 14.512). All three
runs are otherwise the Hannestad Point C config (sin²(2θ_24)=1e-4,
δm²_41=0.93 eV², L=0):

  A: qke_v_nunu_active_only=True, qke_damping_formula="mirizzi"
     (existing project default for Hannestad runs; expected ≈ 14.5
     reproducing gate 5b)

  B: qke_v_nunu_active_only=False, qke_damping_formula="mirizzi"
     (V_nunu axis flip — sprint-5 legacy projection; full 4×4 V_nunu
     matrix including sterile rows/cols)

  C: qke_v_nunu_active_only=True, qke_damping_formula="symmetric"
     (damping axis flip — matches HTT 2012 Eq. 2.15-2.16 directly,
     D = 0.5 Γ with Γ = C_a G_F² T⁵ E)

Each run reports both the raw `Σρ_ss = c._boltz_rho_final[:, 3, :].sum()`
(no measure factor — the legacy harness convention) and the
**δN_eff_ss** computed as the energy-weighted ratio of sterile to one
fully thermal active species (Hannestad-comparable units, see
doc/STAGE_E2_SPRINT17_LITERATURE.md §1.3 for the derivation).

Decision tree at completion:

  Run C δN_eff_ss in [0.02, 0.10]   → damping-kernel axis is the
                                      divergence point; flip default
                                      qke_damping_formula to "symmetric"
                                      to reproduce HTT 2012; close
                                      Stage E.2 with a single-axis cure.
  Run B δN_eff_ss in [0.02, 0.10]   → V_nunu projection is the
                                      divergence point; flip default
                                      qke_v_nunu_active_only to False;
                                      close Stage E.2.
  All three δN_eff_ss outside band  → neither axis cures the gap;
                                      escalate to mixing-angle / IC /
                                      resonance / momentum-grid axes
                                      in sprint 18.
  Run A != gate-5b baseline 14.5    → harness wiring bug; investigate
                                      before any structural verdict.

Total expected wall-clock: 3 × 21 min ≈ 63 min.
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


NEFF_3X3_ETDRK2 = 3.00034              # SM 3-flavor baseline (gate 3, sprint 16)
SUM_SS_GATE_5B = 14.5122               # ETDRK2 reduced-n_B baseline (gate 5b, sprint 16)
HANNESTAD_BAND_LO = 0.02               # δNeff lower edge from HTT 2012 Fig. 2 top blue curve
HANNESTAD_BAND_HI = 0.10               # δNeff upper edge (project-internal, see literature.md §1.3)


def _base_flags():
    """Hannestad Point C config at reduced n_B; ETDRK2 driver."""
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
    PRyMini.qke_damping_formula = "mirizzi"      # axis variable
    PRyMini.T_boltz_start = 30.0
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True        # axis variable
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 1000


def _delta_neff_ss_from_rho(rho_final):
    """Convert the final density matrix to δN_eff_ss (Hannestad units).

    δN_eff_ss = (Σ_sectors Σ_y y³ ρ_ss(y, s)) / (Σ_sectors Σ_y y³ ρ_νe(y, s))

    Numerator: the sterile (flavor index 3) energy moment, summed across
    both sectors (ν, ν̄).

    Denominator: the active ν_e (flavor index 0) energy moment under
    the SAME (y_grid, dy) discretisation. ν_e at end of Phase B is
    approximately a thermal Fermi-Dirac distribution `f_FD(y / T_ν,com)`,
    so its y³-moment is `≈ 2 · T_ν,com⁴ · 7π⁴/120` per fully-thermal
    species (both sectors). Taking this ratio cancels T_ν,com⁴ exactly
    and gives δN_eff_ss = (sterile energy density) / (one species
    fully-thermalised) — Hannestad's convention.

    Caveats: the active ν_e at end of Phase B is heated above its
    canonical post-decoupling temperature by e+e- annihilation
    (Neff_SM ≈ 3.043, not 3.0); using ν_e as the denominator under-
    counts δN_eff_ss by ~1.4%. Acceptable for a sprint-17 axis bracket
    where the discriminating ratio is order-of-magnitude.
    """
    if rho_final is None or rho_final.shape[1] < 4:
        return None
    Ny = rho_final.shape[2]
    # Reproduce the linear y-grid used by DensityMatrixSolver:
    #   y_grid = linspace(dy/2, y_max - dy/2, Ny), dy = y_max / Ny
    y_max = PRyMini.y_max_boltz
    dy = y_max / Ny
    y_grid = np.linspace(dy / 2.0, y_max - dy / 2.0, Ny)
    y3 = y_grid ** 3

    rho_ss = rho_final[:, 3, :]                # (n_sectors=2, Ny)
    rho_e = rho_final[:, 0, :]                 # (n_sectors=2, Ny)

    m3_ss = float((rho_ss * y3[None, :]).sum())
    m3_e = float((rho_e * y3[None, :]).sum())
    if m3_e <= 0.0:
        return None
    return m3_ss / m3_e


def _run_one(label, override_fn):
    """Run one Hannestad Point C job at reduced n_B with given overrides.

    Reloads PRyM_thermo / PRyM_boltzmann / PRyM_main between calls so
    that flag changes propagate (each module reads PRyMini at import
    time for some pieces).
    """
    _base_flags()
    override_fn()

    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    print(f"\n--- Run {label} ---", flush=True)
    print(f"  qke_v_nunu_active_only = {PRyMini.qke_v_nunu_active_only}", flush=True)
    print(f"  qke_damping_formula    = {PRyMini.qke_damping_formula!r}", flush=True)
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

    return {
        "label": label,
        "v_nunu_active_only": PRyMini.qke_v_nunu_active_only,
        "damping_formula": PRyMini.qke_damping_formula,
        "Neff": Neff,
        "delta_Neff_total": Neff - NEFF_3X3_ETDRK2,
        "delta_Neff_ss": delta_neff_ss,
        "Yp": Yp,
        "DoH": DoH,
        "sum_rho_ss_raw": sum_ss,
        "wall_s": dt_run,
    }


def _override_A():
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_formula = "mirizzi"


def _override_B():
    PRyMini.qke_v_nunu_active_only = False
    PRyMini.qke_damping_formula = "mirizzi"


def _override_C():
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_formula = "symmetric"


def _format_result(r):
    de = r["delta_Neff_ss"]
    de_str = f"{de:.4f}" if de is not None else "n/a"
    return (
        f"  Run {r['label']:<54s}  "
        f"v_nunu_active_only={str(r['v_nunu_active_only']):<5s}  "
        f"damp={r['damping_formula']:<10s}  "
        f"Neff={r['Neff']:.4f}  "
        f"Σρ_ss(raw)={r['sum_rho_ss_raw']:.4f}  "
        f"δNeff_ss={de_str}  "
        f"Yp={r['Yp']:.5f}  "
        f"({r['wall_s']:.0f}s)"
    )


def _verdict(results):
    by_label = {r["label"][0]: r for r in results}  # 'A', 'B', 'C'
    A, B, C = by_label.get("A"), by_label.get("B"), by_label.get("C")
    lines = [""]

    if A is not None:
        gap_A = abs(A["sum_rho_ss_raw"] - SUM_SS_GATE_5B)
        if gap_A > 0.5:
            lines.append(
                f"  WARNING: Run A Σρ_ss(raw) = {A['sum_rho_ss_raw']:.3f} differs from "
                f"sprint-16 gate-5b baseline {SUM_SS_GATE_5B:.3f} by {gap_A:.3f}.")
            lines.append(
                "  This indicates a harness wiring bug; debug before drawing axis-bracket conclusions.")

    def _in_band(d):
        return d is not None and HANNESTAD_BAND_LO <= d <= HANNESTAD_BAND_HI

    in_A = _in_band(A["delta_Neff_ss"]) if A else False
    in_B = _in_band(B["delta_Neff_ss"]) if B else False
    in_C = _in_band(C["delta_Neff_ss"]) if C else False

    lines.append(
        f"  Hannestad band [{HANNESTAD_BAND_LO}, {HANNESTAD_BAND_HI}] (HTT 2012 Fig. 2 top blue curve):")
    lines.append(f"    Run A (project default) δNeff_ss in band: {in_A}")
    lines.append(f"    Run B (V_nunu axis flip) δNeff_ss in band: {in_B}")
    lines.append(f"    Run C (damping = HTT 2012) δNeff_ss in band: {in_C}")
    lines.append("")

    if in_C and not in_A:
        lines.append(
            "  VERDICT: damping-kernel axis is the divergence point. HTT 2012's "
            "symmetric Stodolsky form (D = 0.5 Γ, Γ = C_a G_F² T⁵ E) drives δNeff_ss "
            "into the [0.02, 0.10] band; our default Mirizzi+2012 form (post-E.1 "
            "calibration) does not. **Suspect 8 confirmed with single-axis cure.** "
            "Recommend default-flipping qke_damping_formula to 'symmetric' for "
            "Hannestad-style benchmarks; document the Mirizzi vs Stodolsky tradeoff "
            "in PRyM_init.py:165-170. Stage E.2 closes here.")
    elif in_B and not in_A:
        lines.append(
            "  VERDICT: V_nunu projection axis is the divergence point. The "
            "sprint-5 legacy 'full 4×4 V_nunu' (active_only=False) drives δNeff_ss "
            "into the [0.02, 0.10] band; the sprint-5 fix (active_only=True) does "
            "not. **Suspect 8 confirmed with single-axis cure**, and the sprint-5 "
            "fix is implicated as the source of the divergence. Investigate the "
            "Z-exchange projection convention before default-flipping anything.")
    elif in_C and in_B:
        lines.append(
            "  VERDICT: both axes are sensitive in the right direction; either "
            "alone reaches the band. Single-axis cure ambiguous. Run a fourth "
            "configuration (active_only=False, damping='symmetric') to disambiguate. "
            "Defer to sprint 18.")
    elif (B["delta_Neff_ss"] is not None and A["delta_Neff_ss"] is not None and
          abs(B["delta_Neff_ss"] - A["delta_Neff_ss"]) / max(A["delta_Neff_ss"], 1e-9) > 0.05 and
          C["delta_Neff_ss"] is not None and
          abs(C["delta_Neff_ss"] - A["delta_Neff_ss"]) / max(A["delta_Neff_ss"], 1e-9) > 0.05):
        lines.append(
            "  VERDICT: both axes are sensitive (>5% on δNeff_ss) but neither "
            "alone reaches the Hannestad band. Joint configuration may be needed; "
            "defer to sprint 18 with the joint (active_only=False, damping="
            "'symmetric') run as the natural fourth bracket point.")
    elif (B["delta_Neff_ss"] is not None and A["delta_Neff_ss"] is not None and
          abs(B["delta_Neff_ss"] - A["delta_Neff_ss"]) / max(A["delta_Neff_ss"], 1e-9) < 0.05 and
          C["delta_Neff_ss"] is not None and
          abs(C["delta_Neff_ss"] - A["delta_Neff_ss"]) / max(A["delta_Neff_ss"], 1e-9) < 0.05):
        lines.append(
            "  VERDICT: neither V_nunu nor damping-kernel axis moves δNeff_ss by "
            "more than 5%. Both candidate axes are RULED OUT. Escalate to "
            "mixing-angle convention, momentum-grid resolution, or "
            "Phase-B/Phase-0 handoff timing in sprint 18.")
    else:
        lines.append(
            "  VERDICT: mixed sensitivity result. Inspect the per-run δNeff_ss "
            "values above and decide axis priorities for sprint 18 manually.")

    return lines


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 17 axis bracket: V_nunu projection + damping kernel",
        f"  Hannestad Point C: sin²(2θ_24)=1e-4, δm²_41=0.93 eV², L=0, NH",
        f"  Reduced n_B = 2500 + n_B_phase0 = 1000 (sprint-16 gate-5b baseline Σρ_ss(raw)={SUM_SS_GATE_5B:.3f})",
        f"  HTT 2012 reference: δNeff ≈ 0.02-0.03 (Fig. 2 top blue curve, T = 1 MeV)",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    runs = [
        ("A: active_only=True,  damping='mirizzi'  (project default)",   _override_A),
        ("B: active_only=False, damping='mirizzi'  (V_nunu axis flip)",  _override_B),
        ("C: active_only=True,  damping='symmetric' (HTT 2012)",         _override_C),
    ]

    results = []
    t_total = time.time()
    for label, override_fn in runs:
        r = _run_one(label, override_fn)
        results.append(r)
        print(_format_result(r), flush=True)
    dt_total = time.time() - t_total

    summary = [
        "",
        "=" * 110,
        "Sprint 17 axis bracket summary",
        "=" * 110,
    ]
    for r in results:
        summary.append(_format_result(r))
    summary.extend(_verdict(results))
    summary.append("")
    summary.append(f"Total wall-clock: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint17_axis_v_nunu.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
