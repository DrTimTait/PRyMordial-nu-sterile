"""Stage E.2 sprint 16 gate 5b: ETDRK2 baseline at the same reduced
n_B as the ETDRK4 gate-5 run, for apples-to-apples comparison.

Gate 5 (ETDRK4 at n_B=2500 + n_B_phase0=1000) gave Σρ_ss=15.8052,
which is inconclusive: lower than ETDRK2 production-n_B saturation
(22.225 at n_B=10000) but well above the Hannestad band [0.02, 0.10].
The reduction of Σρ_ss from 22 to 16 could be either:
  (a) ETDRK4's higher-order corrector capturing dynamics ETDRK2 misses
      (partial Suspect 7 confirmation), OR
  (b) Reduced n_B under-resolving the resonance and giving a different
      Σρ_ss for n_B reasons unrelated to corrector order.

Running ETDRK2 at the SAME reduced n_B disambiguates:
  * ETDRK2 ≈ 16 → n_B is the controlling variable; ETDRK4 vs ETDRK2
    agree at this resolution → Suspect 7 not confirmed.
  * ETDRK2 ≈ 22 → corrector order is the controlling variable; ETDRK4
    drives Σρ_ss down by 30% just from order-4 vs order-2 → Suspect 7
    partially confirmed (driving toward band but not reaching it; would
    need higher-order method or Magnus integrator).
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
        "Stage E.2 sprint 16 gate 5b: ETDRK2 at SAME reduced n_B as gate 5 (apples-to-apples)",
        f"  ETDRK4 result (gate 5): Σρ_ss=15.8052, Neff=11.32125, Yp=0.33569 at n_B=2500+phase0=1000",
        f"  ETDRK2 production: Σρ_ss=22.225, Neff=8.52219, Yp=0.27209 at n_B=10000+phase0=2500",
        "Disambiguation: ETDRK2-here ≈ 22 → order matters (Suspect 7 partial); ETDRK2-here ≈ 16 → n_B does",
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

    summary = [
        "",
        f"  ETDRK2 @ reduced n_B: Neff={Neff:.5f}  ΔNeff={dNeff:.4f}  Yp={Yp:.5f}  "
        f"D/H={DoH:.4f}  Σρ_ss={sum_ss:.4f}  ({dt_run:.0f}s)",
        "",
    ]

    # Disambiguation logic
    delta_vs_etdrk4 = abs(sum_ss - 15.8052)
    matches_etdrk4 = delta_vs_etdrk4 < 1.0
    matches_prod_etdrk2 = abs(sum_ss - 22.225) < 1.0

    if matches_etdrk4:
        summary.append(
            "  VERDICT: ETDRK2-here ≈ ETDRK4-here  →  Σρ_ss is n_B-controlled, "
            "not corrector-order-controlled. Order-4 correction does not move "
            "the resonance saturation. Suspect 7 NOT confirmed at this resolution.")
    elif matches_prod_etdrk2:
        summary.append(
            "  VERDICT: ETDRK2-here ≈ ETDRK2-production (22.2)  →  n_B is the "
            "non-controlling variable, ETDRK4's drop to 15.8 is from corrector "
            "order. Suspect 7 PARTIALLY CONFIRMED (order-4 reduces Σρ_ss by "
            "~30% from saturation; would need higher-order or Magnus integrator "
            "to reach Hannestad band).")
    else:
        summary.append(
            f"  VERDICT: ETDRK2-here = {sum_ss:.3f} matches neither ETDRK4-here "
            f"({15.8052:.3f}) nor ETDRK2-production ({22.225:.3f}). Both n_B "
            "and corrector order matter; results not separable from this run alone.")

    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_hannestad_pointC_etdrk2_n3500.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")
