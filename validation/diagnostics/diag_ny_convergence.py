"""Stage E.2 sprint-3: y-grid Ny convergence test (hypothesis C).

doc/STAGE_E2_BRIEF.md hypothesis C: "y-integration discretization
artifact. PRyMordial uses Ny_boltz = 100 points on a linear grid from
y_max_boltz = 100 MeV to y ~ 0. At small y (IR), numerical aliasing
between the oscillation phase H*dt/E ~ 1/y and the damping rate
D ~ y could leave a residual sterile population that escapes the
clamp. Gariazzo uses Gauss-Laguerre quadrature specifically to avoid
IR pile-up."

Sprints 1 and 2 falsified hypothesis A (active-sterile gain asymmetry)
and closed hypothesis B (window mismatch: engineering fixes landed but
dNeff non-monotone across 5/30/60 MeV, so B is not a clean bug). C is
the last of the three E.2 structural candidates. Per the brief it is
"lowest likelihood" since the default Ny=100 regression suite would
already be showing convergence failures if this were dominant; but
cheap to rule out.

Diagnostic: Hannestad Point C (sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93,
PMNS on, Mirizzi damping) at default window (5 MeV), sweep
Ny_boltz in {50, 100, 200}. Compare sterile dNeff and sum_rho_ss
across Ny. Also run 3x3 reference at Ny in {100, 200} to bracket
reference drift.

Decision rules:
  - dNeff stable within 1e-2 across 50/100/200  -> hypothesis C
    ruled out; the Ny=100 default is fine.
  - monotone drift in dNeff at > 5e-2 level     -> C is real;
    escalate to log-spaced / Gauss-Laguerre y-grid redesign.

Compute budget: collision integrals and D-kernel tables scale as
O(Ny_coll^3) with Ny_coll ~ Ny/2 (y_coll_max=50 MeV, y_max=100 MeV),
so each step scales ~Ny^3. Ny=50 run ~3 min; Ny=100 ~15 min;
Ny=200 ~100-120 min. Plus 3x3 at Ny=200 ~90 min. Total ~4 h.
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
    # Default 5 MeV window; sprint-2 auto-scale is a no-op at defaults.
    PRyMini.T_boltz_start = 5.0
    PRyMini.T_start = 10.0 * PRyMini.MeV_to_Kelvin


def _configure_3x3(Ny):
    _point_c_flags()
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.Ny_boltz = int(Ny)


def _configure_point_c(Ny):
    _point_c_flags()
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.Ny_boltz = int(Ny)


def _run(label, configure, Ny):
    configure(Ny)
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
    print(f"  {label:50s} Neff={Neff:.5f}  Yp={res[4]:.5f}  "
          f"D/H={res[5]:.4f}  sum_rho_ss={sum_ss:.3f}  ({dt:.0f}s)",
          flush=True)
    return Neff, sum_ss, dt


if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint-3: y-grid Ny convergence (hypothesis C)")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi, window=5 MeV.")
    print("Sprint-2 Ny=100 baseline: Neff=3.88605, dNeff=+0.8453, sum_rho_ss=5.08.")
    print("=" * 96)

    # 3x3 references (bracket the reference drift with Ny).
    print("\n-- 3x3 references (no sterile) --")
    Neff_3x3 = {}
    for Ny in [100, 200]:
        label = f"3x3 reference, Ny={Ny}"
        Neff, _, _ = _run(label, _configure_3x3, Ny)
        Neff_3x3[Ny] = Neff

    # Sterile Point C at each Ny.
    print("\n-- Hannestad Point C --")
    results = []
    for Ny in [50, 100, 200]:
        label = f"Point C, Ny={Ny}"
        Neff, ss, dt = _run(label, _configure_point_c, Ny)
        # Matched-Ny 3x3 reference: Ny=50 reuses Ny=100 ref (3x3 is Ny-insensitive in this regime);
        # Ny=100 uses Ny=100 ref; Ny=200 uses Ny=200 ref.
        ref_Ny = 200 if Ny == 200 else 100
        dNeff = Neff - Neff_3x3[ref_Ny]
        results.append((Ny, Neff, ss, dNeff, ref_Ny, dt))
        print(f"     dNeff = {dNeff:+.4f}  (3x3 ref at Ny={ref_Ny}: {Neff_3x3[ref_Ny]:.5f})")
        print()

    print("=" * 96)
    print(f"{'Ny':>6s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s} {'ref_Ny':>8s} {'runtime(s)':>12s}")
    print("-" * 96)
    for Ny, Neff, ss, dNeff, ref_Ny, dt in results:
        print(f"{Ny:>6d} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f} {ref_Ny:>8d} {dt:>12.0f}")
    print("=" * 96)
    print()
    print("Decision:")
    print("  - dNeff stable within 1e-2 across 50/100/200 -> hypothesis C ruled out.")
    print("  - monotone drift > 5e-2 in dNeff             -> C real; consider log-y / GL grid.")
