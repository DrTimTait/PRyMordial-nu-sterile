"""Stage E.2 sprint-4: damping-magnitude sensitivity test.

Following sprint-3 (y-grid ruled out) the agent survey of FortEPiaNO
vs PRyMordial identified a possible ~2-3x absolute-normalisation
difference in the pair damping rate D_{a,b}. Rather than faithfully
re-implementing FortEPiaNO's coefficient machinery (which uses a
dimensionally awkward x = a*m_e / z = T/m_e / y = p/m_e scheme that
is error-prone to translate), this diagnostic uses a scalar multiplier
on PRyMordial's existing Mirizzi damping and measures how ΔNeff
responds.

Mechanism: PRyMini.qke_damping_scale multiplies every entry of
DensityMatrixSolver._compute_D_pair_matrix. Default 1.0 reproduces
sprint-2's ΔNeff=+0.8453 at Point C bit-identically.

Regime diagnosis:
  - If ΔNeff tracks the scale linearly (0.3x -> 0.3x ΔNeff, 3x ->
    3x ΔNeff), we are in the damping-dominated Dodelson-Widrow linear
    regime and PRyMordial's damping magnitude matters. Fortepiano
    coefficients would move ΔNeff by the corresponding ratio.
  - If ΔNeff is insensitive to the scale (stays ~0.85 across all
    three scales), PRyMordial is in a different regime
    (coherent-oscillation-saturated, or some numerical artefact)
    and damping magnitude is NOT the bug. Then the 10x gap is
    elsewhere -- likely in the Hamiltonian or the Strang-split
    time-evolution.
  - Partial sensitivity (0.3x -> 0.5x ΔNeff, 3x -> 1.5x ΔNeff)
    indicates the linear-D regime is only part of the story.

Point C config copied from diag_qke_window.py: sin^2 2theta_24=1e-4,
Dm2_41=0.93 eV^2, PMNS on, Mirizzi, default 5 MeV window.
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


def _point_c_flags():
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
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = None
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 5.0
    PRyMini.T_start = 10.0 * PRyMini.MeV_to_Kelvin


def _configure_3x3():
    _point_c_flags()
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0


def _configure_point_c():
    _point_c_flags()
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0


def _run(label, configure, damp_scale):
    configure()
    PRyMini.qke_damping_scale = float(damp_scale)
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final")
            and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
    Neff = res[0]
    print(f"  {label:60s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss, dt


if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint-4: damping-magnitude sensitivity (PRyMini.qke_damping_scale)")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi, window=5 MeV.")
    print("Sprint-2 baseline (scale=1.0): Neff=3.88605, dNeff=+0.8453, sum_rho_ss=5.08.")
    print("=" * 96)

    Neff_3x3, _, _ = _run("3x3 reference (no sterile, scale=1.0)", _configure_3x3, 1.0)
    print()

    results = []
    for scale in [0.3, 1.0, 3.0]:
        label = f"Point C, qke_damping_scale = {scale}"
        Neff, ss, dt = _run(label, _configure_point_c, scale)
        dNeff = Neff - Neff_3x3
        results.append((scale, Neff, ss, dNeff, dt))
        print(f"     dNeff = {dNeff:+.4f}")
        print()

    print("=" * 96)
    print(f"{'scale':>8s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s} {'runtime(s)':>12s}")
    print("-" * 96)
    for scale, Neff, ss, dNeff, dt in results:
        print(f"{scale:>8.2f} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f} {dt:>12.0f}")
    print("=" * 96)
    print()
    print("Regime diagnosis:")
    print("  - dNeff linear in scale (0.3x -> 0.3x dNeff)  -> damping-dominated, D magnitude matters.")
    print("  - dNeff constant across scales                -> NOT damping-dominated. Bug elsewhere.")
    print("  - partial trend                              -> mixed regime.")
