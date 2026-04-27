"""Stage E.2 sprint 16 gate 5: ETDRK4, Hannestad Point C only.

Sibling of diag_hannestad_pointC_lsoda.py. Runs ONLY Hannestad Point C
(sin²(2θ)=1e-4, δm²=0.93 eV²) under the Krogstad ETDRK4 driver and
records the Σρ_ss verdict.

Decision tree (from sprint 16 brief):
  Σρ_ss ∈ [0.02, 0.10]  → Suspect 7 confirmed (ETDRK2 order-2 truncation
                          was over-pumping the resonance).
  Σρ_ss ≈ 22 (matches    → Suspect 8 confirmed (Hannestad target wrong /
            ETDRK2)         V_nunu setup mismatch).
  Crash / wall-clock     → ETDRK4 design issue. Cross-check on closed-form
                          test problem; verify augmented-matrix expm and
                          Krogstad coefficient signs.

Total expected wall-clock: ~30-90 minutes (ETDRK4 ~3× ETDRK2's 27 min Point C run).
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

# ETDRK2 baseline 3×3 Neff (from diag_hannestad_proj_w30_nB10k.out)
NEFF_3X3_ETDRK2 = 3.00034
# ETDRK2 baseline 4×4 Point C (from diag_hannestad_pointC_lsoda.py header)
NEFF_4X4_ETDRK2_POINTC = 8.52219
SUM_RHO_SS_ETDRK2_POINTC = 22.225


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
    PRyMini.qke_etdrk4_flag = True
    PRyMini.qke_ode_etdrk2_flag = False
    PRyMini.qke_lsoda_driver_flag = False
    PRyMini.massive_electron_flag = False
    # Sprint-16 first-pass: reduced n_B for tractable wall-clock. Production
    # n_B=10000 + n_B_phase0=2500 ≈ 8 hours under ETDRK4 (~2.3 s/step at
    # 4-flavor sterile from gate 4); reduced n_B=2500 + n_B_phase0=1000
    # gives a Σρ_ss verdict in ~2 hours which is enough to discriminate
    # Suspect 7 (Σρ_ss → band [0.02, 0.10]) from Suspect 8 (Σρ_ss ≈ 22).
    # Convergence at full n_B is a follow-up if the first-pass verdict
    # is ambiguous (Σρ_ss between band and saturation).
    PRyMini.n_B_override = 2500
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
    PRyMini.n_B_phase0_override = 1000


if __name__ == "__main__":
    header = [
        "=" * 104,
        "Stage E.2 sprint 16 Point C: ETDRK4 driver, sin²(2θ)=1e-4, δm²=0.93 eV²",
        "Decision: Σρ_ss ∈ [0.02, 0.10] → Suspect 7; Σρ_ss ≈ 22 → Suspect 8.",
        f"ETDRK2 baseline (n_B=10000): Neff(C)={NEFF_4X4_ETDRK2_POINTC}, "
        f"Σρ_ss={SUM_RHO_SS_ETDRK2_POINTC}",
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
    in_band_dN = band[0] <= dNeff <= band[1]
    in_band_ss = band[0] <= sum_ss <= band[1]
    matches_etdrk2 = abs(sum_ss - SUM_RHO_SS_ETDRK2_POINTC) < 0.5

    summary = [
        "",
        f"  Point C  Neff={Neff:.5f}  ΔNeff={dNeff:.4f}  Yp={Yp:.5f}  "
        f"D/H={DoH:.4f}  Σρ_ss={sum_ss:.4f}  ({dt_run:.0f}s)",
        f"  Hannestad Σρ_ss band: [{band[0]:.2f}, {band[1]:.2f}]",
        "",
    ]

    if in_band_ss:
        summary.append(
            "  VERDICT: Σρ_ss IN BAND  →  Suspect 7 confirmed "
            "(ETDRK2 order-2 truncation was over-pumping the resonance).")
    elif matches_etdrk2:
        summary.append(
            "  VERDICT: Σρ_ss ≈ ETDRK2 baseline  →  Suspect 8 confirmed "
            "(Hannestad target wrong / V_nunu setup mismatch).")
    else:
        summary.append(
            f"  VERDICT: inconclusive — Σρ_ss={sum_ss:.3f} matches neither "
            "Hannestad band nor ETDRK2 saturation. Investigate.")

    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_hannestad_pointC_etdrk4.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
