"""Stage D.5 attempt: SF dt-convergence under a naive predictor-corrector.

This script is kept as the honest documentation of why the naive
predictor-corrector does NOT work for Shi-Fuller, and why Stage D.5 was
ultimately shipped as a documentary finding rather than a code change.

What was attempted: a two-stage scheme where each evolve_step_ode call
runs (1) a predictor Strang step with H built from ρ_n, then
(2) a corrector Strang step starting from ρ_n with H built from ρ_mid =
0.5·(ρ_n + ρ_pred). Hypothesis: the ~O(dt) behaviour seen at default
in Stage D.4 would be restored to O(dt²) because H would track the
rapidly-varying asymmetry through the MSW resonance.

Observed result (SF scenario, sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2):

    n_B = 2400   Neff = 3.95428   (+1.3e-3 vs D.4 single-stage — improving)
    n_B = 4800   Neff = 3.96442   (+2.7e-3 vs D.4 single-stage — improving)
    n_B = 9600   Neff = 3.84467   (-121e-3 vs D.4 single-stage — UNSTABLE)

Turning-point analysis: near the MSW resonance, ω_αs = H_αα − H_ss
sweeps through zero. Between the predictor (which uses ω(ρ_n)) and the
corrector (which uses ω(ρ_mid)), the sign of ω can flip. The unitary
halves exp(-iω·dt/2) then rotate in OPPOSITE senses across the two
stages, and cumulative interference corrupts the state. The effect
accumulates with step count, hence destabilization at small dt.

This is a known failure mode of naive H-averaging predictor-corrector
schemes at turning points. The proper fix handles the commutator -i[H,·]
and the collision operator in the instantaneous H-eigenbasis, where the
relevant phases are tied to each eigenmode and the collision piece
becomes block-diagonal. That upgrade is the open follow-up Stage D.6
(proper ETDRK2 with eigenbasis collision) documented in doc/ROADMAP.md.

To rerun this script you would need to restore the qke_ode_corrector_flag
machinery that was added to PRyM_init.py and PRyM_boltzmann.py during
the D.5 attempt and removed after the diagnostic.

Runtime at the time of the attempt: ~30 minutes wall (3 runs, each ~2×
the D.4 baseline).
"""
import os
import sys
import time
import importlib

import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

import PRyM.PRyM_init as PRyMini


def _configure():
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
    PRyMini.qke_ode_corrector_flag = True   # Stage D.5 enabled
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = np.arcsin(np.sqrt(1.0e-3)) / 2.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 5.0e-2


def _run(label, n_B_override):
    _configure()
    PRyMini.n_B_override = n_B_override
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0
    sum_ss = 0.0
    n_asym = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())
        if (hasattr(c, "_boltz_solver") and c._boltz_solver is not None
                and hasattr(c._boltz_solver, "y_grid")):
            y = np.asarray(c._boltz_solver.y_grid)
            n_asym = float((y**2 * (rho[0, 0] - rho[1, 0])).sum())
    print(f"{label:32s} Neff={res[0]:.5f}  Σρ_ss={sum_ss:.3f}  "
          f"n_ξe={n_asym:+.3e}  ({dt:.0f}s)", flush=True)
    return res[0], sum_ss, n_asym


print("=" * 80)
print("Stage D.5 validation: SF convergence under predictor-corrector")
print("sin²(2θ_14)=1e-3, Δm²_41=1 eV², ξ_νe=5e-2, qke_ode_corrector_flag=True")
print("=" * 80)

results = []
for n_B in (2400, 4800, 9600):
    results.append((n_B, *_run(f"PC  ODE SF (n_B={n_B})", n_B)))

PRyMini.n_B_override = None

Neffs = [r[1] for r in results]
print()
print(f"Convergence metrics:")
print("-" * 80)
for (n_B, Neff, ss, na) in results:
    print(f"  n_B={n_B:5d}  Neff={Neff:.5f}  Σρ_ss={ss:.3f}  n_ξe={na:+.3e}")

dN_12 = Neffs[1] - Neffs[0]
dN_24 = Neffs[2] - Neffs[1]
print()
print(f"  ΔNeff (2400→4800) = {dN_12*1e3:+.3f} × 10⁻³")
print(f"  ΔNeff (4800→9600) = {dN_24*1e3:+.3f} × 10⁻³")
if abs(dN_12) > 1e-30:
    ratio = dN_24 / dN_12
    print(f"  Ratio = {ratio:+.3f}   "
          f"(≈ 0.25 for converged O(dt²), ≈ 0.5 for O(dt))")

rich = (4 * Neffs[2] - Neffs[1]) / 3.0
print()
print(f"  Richardson extrapolation (4800/9600): Neff_∞ ≈ {rich:.5f}")
print(f"  D.4 baseline (single-stage Strang):    Neff(n_B=2400) = 3.95297")
print(f"  D.4 extrapolated:                      Neff(∞)        = 3.96692")
print(f"  Gap D.5-at-default − D.5-extrapolated: {Neffs[0]-rich:+.5f}")
