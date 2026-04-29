"""Stage F sprint 3d: sterile-OFF cured-config Yp baseline.

Sprint 3c localised the Yp under-prediction to the QKE+BBN pipeline
downstream of Phase B (anchor change reduces sterile by 4x but Yp barely
moves). The literature review (Saviano 2013, Kirilova 2003/2024, HTT 2012,
Hannestad 2013) found that every published QKE+BBN study with active-
sterile mixing predicts Yp goes UP relative to SBBN, never down. Our
Yp = 0.231 at Point C is off by ~0.032 (sign + magnitude) from the
literature consensus.

This sprint runs a SINGLE diagnostic: the cured config at the production
grid (y_max=100, Ny=100, n_B=12000) but with sterile mixing OFF
(sterile_flag=False, theta_24=0). The cure flags stay ON. The 3-flavor SM
QKE evolution should produce Yp = 0.247 ± 0.001 (the standard SBBN value)
if the cured pipeline is consistent.

Decision tree:
  Yp = 0.247 ± 0.001:  bug is in active-sterile mixing handling itself
                       (sprint 3e investigates: weak-rate consumer of
                       distorted f_α, or sterile contribution
                       double-counting).
  Yp = 0.231 ± 0.001:  bug is in the active-only QKE -> BBN pipeline
                       (predates sterile coupling). Sprint 3e investigates
                       the SM-with-QKE -> n/p freeze-out interface
                       (likely the FD-tail extrapolation feeding back
                       into rho_3nu(low Tg) -> dTtotdt -> a(T) -> the
                       weak-rate quadrature).
  Anything else:       New finding; document and decide.
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
SM_NEFF_REFERENCE = 3.044  # Bennett+2021
SM_DH_REFERENCE = 2.467e-5  # rough SBBN at PDG eta_b


def _set_flags():
    """3-flavor SM cured config at production grid."""
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
    # STERILE OFF — the diagnostic.
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # Closure-config defaults (sprint 2 production); kept ON to match the
    # config under which the Point C run gives Yp = 0.231.
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
    PRyMini.qke_post_phaseB_trace_flag = False
    PRyMini.y_max_boltz = 100.0
    PRyMini.Ny_boltz = 100


def _verdict(Yp, Neff, DH):
    yp_dev = Yp - SM_YP_REFERENCE
    if abs(yp_dev) <= 0.001:
        return ("STERILE-OFF YP CONSISTENT WITH SBBN",
                f"Yp = {Yp:.5f} matches SM-SBBN {SM_YP_REFERENCE} within "
                f"+-0.001. The +6.5 percent deficit at sterile-on Point C "
                f"is downstream of the active-only pipeline; the bug lives "
                f"in active-sterile mixing handling. Sprint 3e diagnostic "
                f"targets: weak-rate consumer of distorted f_alpha, or "
                f"sterile double-counting in rho_rad / spectrum integrals.")
    if abs(yp_dev + 0.016) <= 0.005:
        return ("STERILE-OFF YP REPRODUCES THE DEFICIT",
                f"Yp = {Yp:.5f} (deviation {yp_dev:+.5f}) matches the "
                f"sterile-on Point C deficit (-0.016) within +-0.005. The "
                f"bug is UPSTREAM of sterile mixing — it lives in the "
                f"active-only QKE -> BBN pipeline (most likely the "
                f"FD-tail-extrapolation feeding back into rho_3nu(low Tg) "
                f"-> dTtotdt -> a(T) -> the weak-rate quadrature). Sprint "
                f"3e should diagnose by toggling qke_post_phaseB_clamp_flag "
                f"= False and re-running this same SM-cured-config "
                f"baseline; the deficit should disappear.")
    return ("UNEXPECTED",
            f"Yp = {Yp:.5f} (deviation {yp_dev:+.5f}). Neither matches "
            f"SM-SBBN nor reproduces the sterile-on deficit. Manual review.")


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage F sprint 3d: sterile-OFF cured-config Yp baseline",
        "  3-flavor SM (sterile_flag=False, all theta=0)",
        "  Cure flags: damping='symmetric', v_nunu_active_only=True, "
        "phase0=False, post-Phase-B clamp=True (anchor=y_grid)",
        "  Production grid: y_max=100, Ny=100, n_B=12000",
        f"  Reference: SM-SBBN Yp = {SM_YP_REFERENCE} (Bennett+2021 "
        f"Neff = {SM_NEFF_REFERENCE})",
        "  Probing: does the cured pipeline produce Yp=0.247 in the "
        "sterile-OFF limit?",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    print("\n--- Sterile-OFF SM cured-config run (y_max=100, Ny=100, "
          "n_B=12000) ---", flush=True)
    _set_flags()
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

    print(f"    Neff={Neff:.5f}  Yp={Yp:.5f}  D/H={DH:.5f}  "
          f"({dt_run:.0f}s)", flush=True)

    verdict, verdict_detail = _verdict(Yp, Neff, DH)

    summary = ["", "=" * 110,
               "Stage F sprint 3d sterile-OFF baseline result",
               "=" * 110,
               f"  Neff = {Neff:.5f} (SM reference {SM_NEFF_REFERENCE}, "
               f"deviation {Neff - SM_NEFF_REFERENCE:+.5f})",
               f"  Yp   = {Yp:.5f} (SM reference {SM_YP_REFERENCE}, "
               f"deviation {Yp - SM_YP_REFERENCE:+.5f})",
               f"  D/H  = {DH:.5f}",
               "",
               f"  Stage F sprint 3d verdict: {verdict}",
               f"    {verdict_detail}",
               "",
               f"  Wall-clock: {dt_run:.0f}s ({dt_run/60:.1f} min)"]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3d_sterile_off_yp_baseline.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_stage_f3d_sterile_off_yp_baseline.npz")
    np.savez_compressed(out_npz,
        Neff=np.array(Neff), Yp=np.array(Yp), DH=np.array(DH),
        SM_Yp_reference=np.array(SM_YP_REFERENCE),
        SM_Neff_reference=np.array(SM_NEFF_REFERENCE),
        wall_clock_s=np.array(dt_run),
        verdict=np.array(verdict),
        verdict_detail=np.array(verdict_detail))
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
