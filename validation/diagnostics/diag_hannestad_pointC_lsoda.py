"""Stage E.2 sprint 14: LSODA driver, Hannestad Point C only.

Trimmed version of diag_hannestad_proj_w30_nB10k_lsoda.py that runs ONLY
Point C (sin²2θ=1e-4, δm²=0.93). Point C is the decisive case for the
Suspect 7 vs Suspect 8 decision tree:
  ΔNeff ∈ [0.02, 0.10] → Suspect 7 (Strang split is the bottleneck)
  ΔNeff ≈ 5.52        → Suspect 8 (Hannestad target wrong)

Skips the 3×3 reference run (uses the ETDRK2 baseline Neff_3x3 = 3.00034
from diag_hannestad_proj_w30_nB10k.out as the SM proxy; LSODA at 3×3 is
expected to match ETDRK2 to <1e-4 per gate-3 sanity).

Total expected wall-clock: ~30-90 minutes (ETDRK2's Point C alone took
~10 min; LSODA's adaptive multistep can be 3-9× slower in stiff regimes).
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

# ETDRK2 baseline 3×3 Neff (from diag_hannestad_proj_w30_nB10k.out, line 17)
NEFF_3X3_ETDRK2 = 3.00034


def _base_flags():
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
    PRyMini.qke_lsoda_driver_flag = True
    # Sprint-14: tolerances loosened from the production defaults (1e-6 / 1e-10)
    # to keep wall-clock under gate-6 budget through the narrow-mixing
    # resonance. Decision-tree band separation (ΔNeff ~0.04 vs ~5.52, factor
    # of ~100) is unambiguous at this accuracy.
    PRyMini.qke_lsoda_rtol = 1.0e-4
    PRyMini.qke_lsoda_atol = 1.0e-8
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = 10000
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 30.0
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_phase0_flag = True
    PRyMini.T_phase0_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_phase0_override = 2500


if __name__ == "__main__":
    header = [
        "=" * 104,
        "Stage E.2 sprint 14 Point C only: LSODA driver, sin²2θ=1e-4, δm²=0.93",
        "Decision: ΔNeff ∈ [0.02, 0.10] → Suspect 7; ΔNeff ≈ 5.52 → Suspect 8.",
        f"ETDRK2 baseline (n_B=10000): Neff(C)=8.52219, ΔNeff=5.5219, Σρ_ss=22.225",
        "=" * 104,
    ]
    for line in header:
        print(line, flush=True)

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
    dt_run = time.time() - t0

    Neff = res[0]
    Yp = res[4]
    DoH = res[5]
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())

    dNeff = Neff - NEFF_3X3_ETDRK2
    band = (0.02, 0.10)
    in_band = band[0] <= dNeff <= band[1]

    summary = [
        "",
        f"  Point C  Neff={Neff:.5f}  ΔNeff={dNeff:.4f}  Yp={Yp:.5f}  "
        f"D/H={DoH:.4f}  Σρ_ss={sum_ss:.3f}  ({dt_run:.0f}s)",
        f"  Hannestad band: [{band[0]:.2f}, {band[1]:.2f}]   verdict: "
        f"{'IN BAND (Suspect 7 confirmed)' if in_band else 'MISS'}",
        "",
    ]
    if not in_band:
        if abs(dNeff - 5.52) < 0.5:
            summary.append("  Verdict: ΔNeff matches ETDRK2 within 0.5 → "
                           "Suspect 8 confirmed (Hannestad target wrong).")
        else:
            summary.append(f"  Verdict: ΔNeff = {dNeff:.4f} doesn't match either "
                           "Hannestad band or ETDRK2 baseline. Investigate.")

    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_hannestad_pointC_lsoda.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
