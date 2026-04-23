"""Stage E.2 sprint 8: 2-level (mu, s) energy/number conservation under
damping-only dynamics.

In a closed 2-state block under unitary Hamiltonian H plus off-diagonal
damping D (no diagonal back-reaction), number and energy are exactly
conserved:

  d/dt (rho_aa + rho_ss) = -i*([H,rho]_aa + [H,rho]_ss) - D * [0] = 0

because the diagonal of a commutator always has vanishing trace piece
on diagonals in a 2-level system with only off-diagonal coupling in H.

This diagnostic verifies that PRyMordial's _build_L_list produces a
2-level L (for the (mu, s) block at a single y-mode) that conserves
number and energy under direct expm evolution over 1000 steps. Any
drift indicates a bug in L construction itself, independent of the
ETDRK2 driver's step composition.

Two checks:
  (A) Analytic 2-level L (hand-built from H, D via the same helpers):
      |dN/N|, |dE/E| < 1e-8 after 1000 steps (pure expm).
  (B) PRyMordial 4x4 L restricted to the (mu, s) 2x2 sub-block:
      the sub-block L_(mu,s) should equal the analytic L_2 up to the
      3x3 active-block mixing. In a damping-only, synthetic-IC scenario
      (rho[:, 0, 2] = 0 everywhere, rho_ee = rho_tau = 0), the (mu, s)
      block should decouple and also conserve N+E. |dN/N|, |dE/E| < 1e-8.

Tolerance 1e-8 absorbs float64 roundoff of 1000 expm@vec multiplies
(~10^3 * eps_mach with modest condition-number amplification); anything
fundamentally broken (e.g. a missing conservation term) produces drift
orders of magnitude larger than this threshold.

If (A) passes and (B) fails, there's a coupling-in-L bug between the
(mu, s) block and the rest of the 4x4. If both pass, L is correct and
any energy anomaly observed in evolve_step_ode_etdrk2 at Point A comes
from the driver's step composition (half-diag / predictor / corrector /
clip), not from L itself.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini


def _flags():
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True
    PRyMini.sterile_flag = True
    PRyMini.Dm2_21 = 0.0
    PRyMini.Dm2_31 = 0.0
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_12 = 0.0
    PRyMini.theta_13 = 0.0
    PRyMini.theta_23 = 0.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.delta_CP = 0.0
    PRyMini.delta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-1)) / 2.0  # Point A sized coupling
    PRyMini.eta0b = 0.0
    PRyMini.qke_damping_formula = "mirizzi"


def _run():
    _flags()
    import PRyM.PRyM_boltzmann as PRyMboltz
    solver = PRyMboltz.DensityMatrixSolver()
    rho_all = solver.initial_conditions(10.0, 1.0)
    # Zero all non-(mu, s) diagonals so the 4x4 behaves as a 2-level in
    # the mu-s block (active-active couplings vanish when rho_ee=rho_tau=0).
    rho_all[:, 0, :] = 0.0
    rho_all[:, 2, :] = 0.0

    y_idx = 10  # representative mid-y mode
    a, Tg = 1.0, 10.0  # MeV (T), dimensionless (a)
    L_list, _N_gain, _I_total = solver._build_L_list(rho_all, a, Tg)
    H = solver._build_H_list(rho_all, a, Tg)[0][y_idx]

    # ----- Analytic 2-level setup -----
    H_aa = H[1, 1]
    H_ss = H[3, 3]
    H_as = H[1, 3]
    T_eV = Tg * 1.0e6
    E_eV = solver.y_grid[y_idx] / a * 1.0e6
    D_off = solver._compute_D_pair_matrix(T_eV, np.array([E_eV]), units="eV")
    D_pair = D_off[1, 3, 0]  # (mu, s)

    I2 = np.eye(2, dtype=complex)
    H2 = np.array([[H_aa, H_as], [np.conj(H_as), H_ss]], dtype=complex)
    L_comm = -1j * (np.kron(H2, I2) - np.kron(I2, H2.T))
    L_damp = np.diag([0.0, -D_pair, -D_pair, 0.0]).astype(complex)
    L_2 = L_comm + L_damp

    rho_aa_init = rho_all[0, 1, y_idx]
    vec2_0 = np.array([rho_aa_init, 0.0, 0.0, 0.0], dtype=complex)

    # ----- PRyMordial 4x4 L at the same y-mode -----
    L_16 = L_list[0][y_idx]
    vec16_0 = solver._to_mat(rho_all[0])[y_idx].reshape(16)

    # ----- Evolve for n_steps -----
    dt_nat = 4.0e-4 * solver._eV_to_secm1  # same step as diag_2level_damped
    n_steps = 1000
    # Precompute step propagators (constant L).
    P_2 = expm(L_2 * dt_nat)
    P_16 = expm(L_16 * dt_nat)

    # Trajectories.
    N_2 = np.empty(n_steps + 1)
    E_2 = np.empty(n_steps + 1)
    C_2 = np.empty(n_steps + 1)
    N_16 = np.empty(n_steps + 1)
    E_16 = np.empty(n_steps + 1)
    C_16 = np.empty(n_steps + 1)

    y = float(solver.y_grid[y_idx])
    vec2 = vec2_0.copy()
    vec16 = vec16_0.copy()
    for i in range(n_steps + 1):
        r2 = vec2.reshape(2, 2)
        r16 = vec16.reshape(4, 4)
        N_2[i] = float(np.real(r2[0, 0] + r2[1, 1]))
        E_2[i] = y * N_2[i]
        C_2[i] = float(np.abs(r2[0, 1]))
        N_16[i] = float(np.real(r16[1, 1] + r16[3, 3]))
        E_16[i] = y * N_16[i]
        C_16[i] = float(np.abs(r16[1, 3]))
        if i < n_steps:
            vec2 = P_2 @ vec2
            vec16 = P_16 @ vec16

    return {
        "D_pair_eV": float(D_pair),
        "H_as_eV": complex(H_as),
        "H_aa_minus_ss_eV": float((H_aa - H_ss).real),
        "dt_nat": float(dt_nat),
        "n_steps": int(n_steps),
        "N_2": N_2,
        "E_2": E_2,
        "C_2": C_2,
        "N_16": N_16,
        "E_16": E_16,
        "C_16": C_16,
    }


def _fmt(x):
    return f"{x:+.3e}"


def main():
    res = _run()
    N_2 = res["N_2"]
    E_2 = res["E_2"]
    C_2 = res["C_2"]
    N_16 = res["N_16"]
    E_16 = res["E_16"]
    C_16 = res["C_16"]
    n = res["n_steps"]

    dN_2 = (N_2[-1] - N_2[0]) / N_2[0]
    dE_2 = (E_2[-1] - E_2[0]) / E_2[0]
    dN_16 = (N_16[-1] - N_16[0]) / N_16[0]
    dE_16 = (E_16[-1] - E_16[0]) / E_16[0]
    # Coherence decay at the analytic rate.
    D = res["D_pair_eV"]
    dt_nat = res["dt_nat"]
    C_expected = C_2[0] * np.exp(-D * dt_nat * n) if C_2[0] > 0 else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append("Stage E.2 sprint 8 unit test: 2-level (mu, s) N+E conservation")
    lines.append("=" * 80)
    lines.append(f"D_pair        = {res['D_pair_eV']:.3e} eV")
    lines.append(f"H_aa - H_ss   = {res['H_aa_minus_ss_eV']:.3e} eV")
    lines.append(f"|H_as|        = {abs(res['H_as_eV']):.3e} eV")
    lines.append(f"dt_nat        = {dt_nat:.3e} (1/eV)")
    lines.append(f"n_steps       = {n}")
    lines.append("")
    lines.append("(A) Analytic 2-level L:")
    lines.append(f"    N(0)       = {N_2[0]:.6e}")
    lines.append(f"    N({n})      = {N_2[-1]:.6e}")
    lines.append(f"    dN/N(0)    = {_fmt(dN_2)}       (expect |..| < 1e-8)")
    lines.append(f"    E(0)       = {E_2[0]:.6e}")
    lines.append(f"    E({n})      = {E_2[-1]:.6e}")
    lines.append(f"    dE/E(0)    = {_fmt(dE_2)}       (expect |..| < 1e-8)")
    lines.append(f"    C(0)       = {C_2[0]:.6e}")
    lines.append(f"    C({n})      = {C_2[-1]:.6e}")
    if C_2[0] > 0:
        lines.append(f"    C_expected = {C_expected:.6e}  (from pure exponential decay)")
    lines.append("")
    lines.append("(B) PRyMordial 4x4 L, (mu, s) sub-block extraction:")
    lines.append(f"    N(0)       = {N_16[0]:.6e}")
    lines.append(f"    N({n})      = {N_16[-1]:.6e}")
    lines.append(f"    dN/N(0)    = {_fmt(dN_16)}       (expect |..| < 1e-8)")
    lines.append(f"    E(0)       = {E_16[0]:.6e}")
    lines.append(f"    E({n})      = {E_16[-1]:.6e}")
    lines.append(f"    dE/E(0)    = {_fmt(dE_16)}       (expect |..| < 1e-8)")
    lines.append(f"    C(0)       = {C_16[0]:.6e}")
    lines.append(f"    C({n})      = {C_16[-1]:.6e}")
    lines.append("")

    tol = 1.0e-8
    pass_A = (abs(dN_2) < tol) and (abs(dE_2) < tol)
    pass_B = (abs(dN_16) < tol) and (abs(dE_16) < tol)
    if pass_A and pass_B:
        verdict = "PASS: 2-level L conserves N and E under damping-only dynamics (both paths)."
    elif pass_A and not pass_B:
        verdict = ("WARN: analytic 2-level OK but PRyMordial 4x4 (mu, s) sub-block "
                   "leaks. Indicates coupling between (mu, s) block and other "
                   "blocks in L, or a construction bug.")
    elif not pass_A and not pass_B:
        verdict = "FAIL: both paths leak. L construction is wrong."
    else:
        verdict = "WARN: analytic 2-level leaks — tolerance too tight or test broken?"
    lines.append("Verdict: " + verdict)
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)
    out_txt = os.path.join(_WT, "validation/diagnostics/diag_2level_damped_energy.out")
    with open(out_txt, "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
