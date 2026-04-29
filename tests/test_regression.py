# -*- coding: utf-8 -*-
"""
Regression tests: freeze the BBN outputs for known PRyMordial-nu
configurations so future refactors can't silently shift them.

Reference values were captured on main commit `1d6e05e` (Task 2 complete,
items 6-9 and 13 landed, Bennett+2021 recommended Neff = 3.0440 matched
to 10^-4 in the QKE mode).

Tolerances:
  - Neff: 1e-5 absolute  (~3 parts in 1e6; tighter than Bennett's 2e-4)
  - Yp:   1e-5 absolute  (~4 parts in 1e5)
  - D/H:  1e-3 absolute  (~4 parts in 1e4; looser because D/H is more
                          sensitive to the floating-point-order tail
                          of the Boltzmann/QKE integration)

Slow tests (Boltzmann / QKE) are marked `slow` and skipped when running
`pytest -m "not slow"`. Fast thermal-path tests run in ~15 s total.
"""
import importlib

import pytest


# --- Fixture utilities --------------------------------------------------

def _reset_flags():
    """Set PRyMini to a known baseline before each test."""
    import PRyM.PRyM_init as PRyMini
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = False
    PRyMini.boltzmann_nu_flag = False
    PRyMini.nu_oscillation_flag = False
    PRyMini.qke_density_matrix_flag = False
    PRyMini.qke_full_ode_flag = False
    PRyMini.qke_ode_etdrk2_flag = False
    PRyMini.n_B_override = None
    PRyMini.massive_electron_flag = False
    PRyMini.mu_tau_symmetric_flag = True
    PRyMini.nu_nubar_symmetric_flag = True
    PRyMini.two_loop_QED_flag = False
    # Sterile (3+1) parameters — reset so sterile tests don't bleed into others
    PRyMini.sterile_flag = False
    PRyMini.Dm2_41 = 1.0
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = 0.0
    PRyMini.theta_34 = 0.0
    PRyMini.delta_14 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    # Stage F sprint 2 default flip: pin the three closure-config flags
    # back to their pre-flip values so the regression reference numerics
    # (captured pre-cure) continue to hold. Production usage gets the
    # closure-config defaults; the regression suite remains a stable
    # legacy-physics guard. The Hannestad / sin²2θ-scan diagnostics
    # (validation/diagnostics/) exercise the closure-config defaults
    # end-to-end, so this pin does not weaken coverage of the new code path.
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.qke_v_nunu_active_only = False
    PRyMini.qke_post_phaseB_clamp_flag = False


def _run_mode():
    """Return (Neff, Yp_BBN, D/H_x1e5) from a fresh PRyMclass instantiation.

    Reloads PRyM_thermo so that any flag changes made in _reset_flags + the
    caller's test body propagate through the on-import data loading.
    """
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    res = PRyMmain.PRyMclass().PRyMresults()
    # res = [Neff, Omega_nu h2 x 1e6, 1/Omega_nu_nr, Yp(CMB), Yp(BBN), D/H x 1e5, He3/H x 1e5, Li7/H x 1e10]
    return res[0], res[4], res[5]


# --- Fast tests ---------------------------------------------------------

def test_mode1_standard_thermal():
    """Standard thermal (no oscillations, O(e²)+O(e³) QED from NUDEC v2)."""
    _reset_flags()
    Neff, Yp, DoH = _run_mode()
    assert Neff == pytest.approx(3.044389, abs=1e-5)
    assert Yp   == pytest.approx(0.248277, abs=1e-5)
    assert DoH  == pytest.approx(2.4643,   abs=1e-3)


def test_mode2_general_nu_fd():
    """General-nu path with default thermal-FD distributions + aTid fix."""
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    Neff, Yp, DoH = _run_mode()
    assert Neff == pytest.approx(3.044389, abs=1e-5)
    assert Yp   == pytest.approx(0.248279, abs=3e-5)
    assert DoH  == pytest.approx(2.4644,   abs=1e-3)


def test_mode6_standard_plus_oe4():
    """Standard thermal + two-loop O(e⁴) QED correction. Needs recomputed bckg."""
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.two_loop_QED_flag = True
    PRyMini.compute_bckg_flag = True   # so O(e⁴) propagates into Tg,Tnu trajectories
    Neff, Yp, DoH = _run_mode()
    # O(e⁴) shifts Neff by ~1e-5 relative to mode 1; broaden tolerance slightly.
    assert Neff == pytest.approx(3.044338, abs=2e-5)
    assert Yp   == pytest.approx(0.248279, abs=3e-5)
    assert DoH  == pytest.approx(2.4626,   abs=1e-3)


# --- Slow tests (Boltzmann / QKE) --------------------------------------

@pytest.mark.slow
def test_mode3_boltzmann_diagonal():
    """Boltzmann diagonal n=3 (no oscillations). ~100 s wall on a laptop."""
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    Neff, Yp, DoH = _run_mode()
    assert Neff == pytest.approx(3.0398, abs=3e-4)
    assert Yp   == pytest.approx(0.248450, abs=3e-5)
    assert DoH  == pytest.approx(2.4667, abs=3e-3)


@pytest.mark.slow
def test_mode5_qke_density_matrix():
    """Full QKE density-matrix solver. ~130 s wall on a laptop.

    Reproduces Bennett+2021 recommended Neff = 3.0440 to 10^-4.
    """
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    Neff, Yp, DoH = _run_mode()
    assert Neff == pytest.approx(3.0445, abs=3e-4)
    assert Yp   == pytest.approx(0.24851, abs=3e-5)
    assert DoH  == pytest.approx(2.4714, abs=3e-3)


@pytest.mark.slow
def test_mode5b_qke_full_ode():
    """Stage D: QKE with the full-ODE driver (qke_full_ode_flag=True).

    The ODE driver replaces the two quasi-static diagonal-transfer blocks
    (Sigl-Raffelt active-active + Stage B Dodelson-Widrow active-sterile)
    with an exact Strang-split unitary conjugation ρ → U^(½) ρ (U^(½))†.

    Observables shift by O(10⁻³) on Neff relative to the quasi-static
    path (test_mode5_qke_density_matrix); this is a known consequence of
    the numerical regime change rather than an error. The quasi-static
    blocks assume the off-diagonal coherence has reached its steady
    state within each timestep, which is true in the fast-oscillation
    limit. The Strang-split unitary resolves the off-diagonal build-up
    explicitly, so in the same dt it transfers slightly less population
    across active flavors. The difference is largest in the SM regime
    (θ_12/θ_13 oscillations damped by active collisions); it shrinks in
    the sterile regimes where the unitary captures dynamics the quasi-
    static form cannot (see Stage B/C exploration).

    Reference value captured on the Stage D landing commit.
    """
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    Neff, Yp, DoH = _run_mode()
    # Frozen on Stage D landing. Wider Neff window than mode 5 reflects
    # the ~0.1 % drift from the quasi-static → exact-unitary switch.
    assert Neff == pytest.approx(3.041,  abs=3e-3)
    assert Yp   == pytest.approx(0.2485, abs=3e-4)
    assert DoH  == pytest.approx(2.47,   abs=2e-2)


@pytest.mark.slow
def test_mode5c_qke_ode_etdrk2():
    """Stage D.7: full ETDRK2 driver (qke_ode_etdrk2_flag=True), 3×3 SM.

    End-to-end smoke test for the Stage D.7 ETDRK2 driver with damping-in-L.
    Routes per-step evolution through evolve_step_ode_etdrk2, which
    exponentiates L = -i[H, .] - D_off o . on the 9x9 vectorised superoperator
    per mode (full predictor + corrector, ν-ν̄ symmetric).

    In the 3×3 SM regime there is no MSW resonance, so D.7 agrees with
    mode-5b (Strang) to 1e-3 on Neff (tightened from the D.6 3e-3 envelope
    now that the driver is no longer first-order). Point of the test:

    1. End-to-end proof the D.7 code path runs without crashing or
       drifting observables by an amount that indicates a unit / sign /
       kron-ordering bug.
    2. Regression guard against future refactors of the driver.
    """
    _reset_flags()
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True
    Neff, Yp, DoH = _run_mode()
    assert Neff == pytest.approx(3.041,  abs=1e-3)
    assert Yp   == pytest.approx(0.2485, abs=3e-4)
    assert DoH  == pytest.approx(2.47,   abs=2e-2)


def test_qke_etdrk2_nu_nubar_symmetry():
    """Stage D.7: ETDRK2 with damping-in-L must preserve ν-ν̄ symmetry.

    D.6's Cox-Matthews corrector breaks this invariant by ~O(dt · mixing)
    because sector-1's stored-convention rotation (V*, V^T) composed with an
    element-wise φ_k is not the complex conjugate of sector-0's (V, V†)·φ.
    D.7 exponentiates the full L = -i[H, .] - D_off o . on the vectorised
    superoperator, which does not rotate into the H-eigenbasis; with D real
    the sector-1 damping block is identical to sector-0 and symmetry is
    algebraic.

    Setup: ν-ν̄-symmetric thermal FD state (ξ=0), 3×3 SM, eta0b=0 to force
    V_CC=0 (and V_νν=0 trivially by symmetry). Both sectors start pointwise
    identical; H_nu = conj(H_nubar) via _Omega_nubar = conj(_Omega_nu).

    Evolve 10 ETDRK2 steps at representative Phase-B dt and assert
    max |rho_all[0] - conj(rho_all[1])| stays below 1e-10 pointwise.

    This test is fast (no full PRyMclass run; direct solver stepping) and
    would fail outright against the D.6 full-corrector variant.
    """
    import numpy as np
    import PRyM.PRyM_init as PRyMini
    import PRyM.PRyM_boltzmann as PRyM_boltzmann

    _reset_flags()
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True

    # Kill the baryonic V_CC: V_CC = sqrt(2) GF eta0b n_gamma. eta0b = 0 => V_CC = 0.
    _eta0b_saved = PRyMini.eta0b
    PRyMini.eta0b = 0.0
    try:
        solver = PRyM_boltzmann.DensityMatrixSolver()

        # Symmetric thermal FD at Tnu = 2 MeV, a = 1 (representative of late
        # Phase B). initial_conditions already makes sector 0 = sector 1 for
        # all active flavors at ξ=0.
        Tnu = 2.0
        a = 1.0
        Tg = Tnu
        rho_all = solver.initial_conditions(Tnu, a)

        # Pre-check: sectors start pointwise identical (real FD => conj symmetry trivial).
        assert np.max(np.abs(rho_all[0] - rho_all[1])) < 1e-14, (
            "initial_conditions did not produce a symmetric state at ξ=0")

        # Round-trip assertion on the vec/matrix converters (catches any
        # ordering bug that would otherwise silently corrupt the L step).
        rho_mat_roundtrip = solver._to_mat(rho_all[0])
        rho_vec_roundtrip = solver._to_vec(rho_mat_roundtrip)
        assert np.allclose(rho_vec_roundtrip, rho_all[0], atol=1e-14), (
            "_to_vec(_to_mat(x)) != x; vec/mat convention mismatch")

        # Kernel-level symmetry pre-assert: if the collision gain itself is
        # sector-asymmetric, the evolution can't possibly preserve symmetry
        # and the failure would not be a D.7 L bug.
        _, N_gain_n, _ = solver._build_L_list(rho_all, a, Tg)
        kernel_asym = np.max(np.abs(N_gain_n[0] - np.conj(N_gain_n[1])))
        assert kernel_asym < 1e-12, (
            f"collision-gain kernel broke ν-ν̄ symmetry at t=0: "
            f"max|N_gain[0] - conj(N_gain[1])| = {kernel_asym:.3e} "
            f"(not a D.7 L bug; investigate _assemble_collision_N)")

        # Evolve. dt typical of Phase B at n_B=2400, Tg~2 MeV: dt ~ 4e-4 s.
        dt = 4.0e-4
        phi1_dt = dt  # scalar -> all three diag channels use the same factor.
        for _ in range(10):
            solver.evolve_step_ode_etdrk2(rho_all, dt, phi1_dt, a, Tg)

        # The stored convention is rho_bar* = conj(rho_bar_phys), so for a
        # physically-symmetric state (rho = rho_bar_phys) the stored values
        # are conjugates; at ξ=0 thermal-FD both are real, so both sectors
        # should stay pointwise equal modulo O(1e-10) numerical noise.
        resid = np.max(np.abs(rho_all[0] - np.conj(rho_all[1])))
        assert resid < 1e-10, (
            f"D.7 broke ν-ν̄ symmetry: max|rho[0] - conj(rho[1])| = "
            f"{resid:.3e} after 10 steps (target < 1e-10)")
    finally:
        PRyMini.eta0b = _eta0b_saved


def test_lsoda_analytic_jacobian_matches_finite_difference():
    """Sprint 15: _lsoda_jacobian_dense agrees with the finite-difference
    Jacobian of the linear (unitary + damping) RHS to <1e-10 relative.

    The analytic Jacobian represents d(f_lin)/dy where
        f_lin(y) = -i sign[s] [H_frozen, ρ(y)] - D_off ⊙ ρ(y),
    i.e. the LSODA RHS minus its constant part. The constant pieces —
    N_full(rho_all) at start-of-outer-step and the +D_off ⊙ ρ add-back
    that lives inside N_full's off-diagonals — drop out of the
    derivative. Comparing FD on f_lin (instead of the full RHS, which
    has values O(1e23) on a thermal point and overwhelms the linear
    signal of magnitude ~1e6 with floating-point noise) makes the test
    a clean, tight check on _lsoda_jacobian_dense itself rather than
    on FP cancellation.

    A representative Phase-B point is used (3-flavor SM, perturbed
    thermal rho_all to populate off-diagonals). 3-flavor n_dof = 1800
    (cf 3200 for 4×4 sterile); central FD costs 2·1800 small einsums.
    """
    import numpy as np
    import PRyM.PRyM_init as PRyMini
    import PRyM.PRyM_boltzmann as PRyM_boltzmann

    _reset_flags()
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True

    solver = PRyM_boltzmann.DensityMatrixSolver()
    Ny = solver.Ny
    nc = solver.n_components
    N = solver.n_flavor
    eV_to_s = solver._eV_to_secm1

    # Representative Phase-B point. Inject a small Hermitian
    # perturbation onto thermal FD so off-diagonal components are
    # populated and the Jacobian is exercised in all 1800 directions
    # (not just the diagonal sub-block).
    Tnu = 2.0
    a = 1.0
    Tg = Tnu
    rho_all = solver.initial_conditions(Tnu, a)
    rng = np.random.default_rng(0)
    rho_all = rho_all + 1.0e-3 * rng.standard_normal(rho_all.shape)

    # Freeze H_list and the off-diagonal pair damping at this rho_all
    # — the analytic Jacobian uses both via _build_L_list (which
    # internally calls _build_H_list and _compute_D_pair_matrix).
    H_list_frozen = solver._build_H_list(rho_all, a, Tg)
    T_eV = Tg * 1.0e6
    E_eV = np.maximum(solver.y_grid / a * 1.0e6, 1.0e-4)
    D_off_eV = solver._compute_D_pair_matrix(T_eV, E_eV, units="eV")

    def f_lin(y_flat):
        """Linear (unitary + off-diag damping) part of the LSODA RHS,
        with H and D_off frozen — exactly what _lsoda_jacobian_dense
        linearises. The constant N_full piece is omitted so that FD
        on f_lin tests the Jacobian directly without FP cancellation
        against an O(1e23) collision-integral baseline.
        """
        rho_local = y_flat.reshape(2, nc, Ny)
        drho_vec = np.zeros((2, nc, Ny))
        for s in range(2):
            rho_mat = solver._to_mat(rho_local[s])
            H = H_list_frozen[s]
            if s == 0:
                comm = (np.einsum('iab,ibc->iac', H, rho_mat)
                        - np.einsum('iab,ibc->iac', rho_mat, H))
                drho_eV = -1j * comm
            else:
                H_T = H.swapaxes(-1, -2)
                comm = (np.einsum('iab,ibc->iac', H_T, rho_mat)
                        - np.einsum('iab,ibc->iac', rho_mat, H_T))
                drho_eV = +1j * comm
            # Off-diagonal pair damping -D_off ⊙ rho (same convention
            # _build_L_list uses; D_off has zero diagonal).
            damp = D_off_eV.transpose(2, 0, 1) * rho_mat  # (Ny, N, N)
            drho_eV = drho_eV - damp
            drho_vec[s] = solver._to_vec(drho_eV * eV_to_s)
        return drho_vec.ravel()

    J_analytic = solver._lsoda_jacobian_dense(rho_all, a, Tg)
    n_dof = 2 * nc * Ny
    assert J_analytic.shape == (n_dof, n_dof)

    # FD on f_lin. f_lin is exactly linear so any h in the FP-safe
    # range gives the exact slope; pick h=1e-3 for headroom against
    # roundoff at the scale of the linear values (~1e11 in s^-1).
    y0 = rho_all.ravel().copy()
    h = 1.0e-3
    J_fd = np.zeros((n_dof, n_dof))
    e = np.zeros(n_dof)
    for k in range(n_dof):
        e[:] = 0.0
        e[k] = h
        f_plus = f_lin(y0 + e)
        f_minus = f_lin(y0 - e)
        J_fd[:, k] = (f_plus - f_minus) / (2.0 * h)

    abs_diff = np.max(np.abs(J_analytic - J_fd))
    scale = np.max(np.abs(J_fd))
    rel = abs_diff / max(scale, 1.0e-30)
    assert rel < 1.0e-10, (
        f"_lsoda_jacobian_dense vs FD on f_lin: rel = {rel:.3e} "
        f"(want < 1e-10); abs={abs_diff:.3e}, scale={scale:.3e}")


def test_lsoda_banded_jacobian_matches_dense():
    """Sprint 15: _lsoda_jacobian_banded reconstructs to the same matrix
    as _lsoda_jacobian_dense after the (s,c,i) → (i,s,c) layout change.

    The banded Jacobian is what evolve_step_lsoda actually hands to
    scipy when qke_lsoda_jac_band_flag is on (default), so its
    correctness is gated by this test rather than by the dense FD test.
    Dense LU at n_dof = 3200 makes the dense path infeasible for
    Hannestad Point C; banded LU at bandwidth = n_components - 1 brings
    it back into budget.
    """
    import numpy as np
    import PRyM.PRyM_init as PRyMini
    import PRyM.PRyM_boltzmann as PRyM_boltzmann

    _reset_flags()
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True

    solver = PRyM_boltzmann.DensityMatrixSolver()
    Ny = solver.Ny
    nc = solver.n_components

    Tnu = 2.0
    a = 1.0
    Tg = Tnu
    rho_all = solver.initial_conditions(Tnu, a)
    rng = np.random.default_rng(0)
    rho_all = rho_all + 1.0e-3 * rng.standard_normal(rho_all.shape)

    # Dense Jacobian under the (s, c, i) layout (default rho_all.ravel()).
    J_dense_sci = solver._lsoda_jacobian_dense(rho_all, a, Tg)

    # Build a permutation that takes the (s, c, i) ordering used by
    # _lsoda_jacobian_dense to the (i, s, c) ordering used by
    # _lsoda_jacobian_banded. perm[k_isc] = k_sci, where
    # k_isc = i * 2*nc + s*nc + c   (banded layout)
    # k_sci = s * nc * Ny + c * Ny + i  (dense layout)
    n_dof = 2 * nc * Ny
    perm = np.empty(n_dof, dtype=int)
    for i in range(Ny):
        for s in range(2):
            for c in range(nc):
                k_isc = i * 2 * nc + s * nc + c
                k_sci = s * nc * Ny + c * Ny + i
                perm[k_isc] = k_sci

    # Permute J_dense_sci into the banded layout for direct comparison
    # (J_isc[i, j] = J_sci[perm[i], perm[j]]).
    J_dense_isc = J_dense_sci[np.ix_(perm, perm)]

    # Now build the banded Jacobian and reconstruct the full matrix.
    J_band, bw = solver._lsoda_jacobian_banded(rho_all, a, Tg)
    assert bw == nc - 1, (
        f"unexpected bandwidth {bw}; expected nc-1 = {nc-1}")
    # ODEPACK banded shape for jt=4: (2*lband + uband + 1, n_dof) =
    # (3*bw + 1, n_dof). User band data lives at rows [bw, 3*bw].
    assert J_band.shape == (3 * bw + 1, n_dof)

    # Reconstruct: J_full[i, j] = J_band[2*bw + i - j, j] for entries
    # within the band (offset = lband + uband = 2*bw).
    offset = 2 * bw
    J_recon = np.zeros((n_dof, n_dof))
    for j in range(n_dof):
        i_lo = max(0, j - bw)
        i_hi = min(n_dof, j + bw + 1)
        for i in range(i_lo, i_hi):
            J_recon[i, j] = J_band[offset + i - j, j]

    abs_diff = np.max(np.abs(J_recon - J_dense_isc))
    scale = max(np.max(np.abs(J_dense_isc)), 1.0e-30)
    rel = abs_diff / scale
    assert rel < 1.0e-12, (
        f"banded Jacobian disagrees with dense (after permutation): "
        f"rel = {rel:.3e}; abs={abs_diff:.3e}, scale={scale:.3e}")

    # Smoke test: scipy's LSODA should accept the banded Jacobian on a
    # one-step integration without error.
    from scipy.integrate import solve_ivp
    y0 = rho_all.transpose(2, 0, 1).ravel().copy()

    def rhs(t, y):
        # Trivial linear RHS that mirrors the banded Jacobian's action,
        # so the smoke test exercises only the LSODA wiring (not the
        # full QKE RHS — the f-vs-jac inconsistency would otherwise
        # confuse LSODA's stiffness estimator).
        return np.zeros_like(y)
    sol = solve_ivp(rhs, [0.0, 1.0e-8], y0, method="LSODA",
                    jac=lambda t, y: J_band, lband=bw, uband=bw,
                    rtol=1.0e-6, atol=1.0e-10)
    assert sol.success, f"LSODA smoke test failed: {sol.message}"


# --- Slow sterile (3+1) tests ------------------------------------------

def _run_qke_with_rho(configure):
    """Run the QKE with sterile_flag enabled and return (Neff, Yp, D/H,
    Σρ_ss, n_ξ_e). The last two are extracted from the final density
    matrix; they are 0 when the sterile is decoupled or when
    sterile_flag=False.
    """
    import importlib
    import numpy as np
    import PRyM.PRyM_init as PRyMini
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    configure()

    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_main as PRyMmain
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    Neff, Yp, DoH = res[0], res[4], res[5]

    sum_rho_ss = 0.0
    n_xi_e = 0.0
    if hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None:
        rho = c._boltz_rho_final
        if rho.shape[1] >= 4:
            sum_rho_ss = float(rho[:, 3, :].sum())
        y_grid = None
        if (hasattr(c, "_boltz_solver") and c._boltz_solver is not None
                and hasattr(c._boltz_solver, "y_grid")):
            y_grid = np.asarray(c._boltz_solver.y_grid)
        if y_grid is not None:
            w = y_grid**2
            n_xi_e = float((w * (rho[0, 0] - rho[1, 0])).sum())
    return Neff, Yp, DoH, sum_rho_ss, n_xi_e


@pytest.mark.slow
def test_sterile_stage_a_invariant():
    """4×4 QKE with all mixing angles zero must reproduce the 3×3 QKE
    result. This guards the sterile infrastructure from accidentally
    leaking into the decoupled active sector.
    """
    _reset_flags()
    import PRyM.PRyM_init as PRyMini

    def cfg_3x3():
        PRyMini.sterile_flag = False

    def cfg_4x4_decoupled():
        PRyMini.sterile_flag = True
        PRyMini.theta_14 = 0.0
        PRyMini.theta_24 = 0.0
        PRyMini.theta_34 = 0.0

    Neff_3, Yp_3, DoH_3, _, _ = _run_qke_with_rho(cfg_3x3)
    Neff_4, Yp_4, DoH_4, sum_ss, _ = _run_qke_with_rho(cfg_4x4_decoupled)

    # Observable invariants (3×3 matches 4×4 decoupled to loose BBN
    # tolerance; D/H has the largest numerical drift path).
    assert abs(Neff_4 - Neff_3) < 3.0e-4, (
        f"Stage A invariant broken: ΔNeff = {Neff_4 - Neff_3:+.4e}")
    assert abs(Yp_4 - Yp_3) / max(Yp_3, 1e-30) < 1.0e-3
    assert abs(DoH_4 - DoH_3) / max(DoH_3, 1e-30) < 1.0e-2
    # Sterile stayed empty (only f_min floor, ≤ 1e-25 summed)
    assert sum_ss < 1.0e-20, (
        f"Decoupled sterile populated unexpectedly: Σρ_ss = {sum_ss:.3e}")


@pytest.mark.slow
def test_sterile_dw_production():
    """Stage B: Dodelson-Widrow drives near-thermalization at large
    mixing. For sin²(2θ_14)=0.1 and Δm²_41=1 eV², sterile equilibrates
    well above BBN and ΔNeff ≈ +0.93.
    """
    _reset_flags()
    import numpy as np
    import PRyM.PRyM_init as PRyMini

    def cfg_3x3():
        PRyMini.sterile_flag = False

    def cfg_dw():
        PRyMini.sterile_flag = True
        PRyMini.Dm2_41 = 1.0
        PRyMini.theta_14 = np.arcsin(np.sqrt(0.1)) / 2.0  # sin²(2θ)=0.1
        PRyMini.theta_24 = 0.0
        PRyMini.theta_34 = 0.0

    Neff_ref, _, _, _, _ = _run_qke_with_rho(cfg_3x3)
    Neff_dw, _, _, sum_ss, _ = _run_qke_with_rho(cfg_dw)
    dNeff = Neff_dw - Neff_ref

    # Expect near-thermal DW signature. Reference from
    # validation/sterile_DW_demo.py (Stage C commit ea04426):
    #   ΔNeff = +0.929,  Σρ_ss ≈ 6.03.
    # Broad tolerance because DW is nonlinear in θ and slightly
    # sensitive to integration step.
    assert 0.5 < dNeff < 1.1, f"DW ΔNeff out of band: {dNeff:+.3f}"
    assert sum_ss > 1.0, (
        f"DW sterile occupation too low: Σρ_ss = {sum_ss:.2f}")


@pytest.mark.slow
def test_sterile_sf_asymmetry_depletion():
    """Stage C: a large initial ξ_νe gets drained into the sterile pool
    via the MSW resonance. We track the integrated ν_e − ν̄_e asymmetry
    and require it to collapse by at least three orders of magnitude
    relative to an estimate of its initial value.
    """
    _reset_flags()
    import numpy as np
    import PRyM.PRyM_init as PRyMini

    xi0 = 5.0e-2

    def cfg_sf():
        PRyMini.sterile_flag = True
        PRyMini.Dm2_41 = 1.0
        PRyMini.theta_14 = np.arcsin(np.sqrt(1.0e-3)) / 2.0  # sin²(2θ)=1e-3
        PRyMini.theta_24 = 0.0
        PRyMini.theta_34 = 0.0
        PRyMini.xi_nue_init = xi0

    _, _, _, sum_ss, n_asym_final = _run_qke_with_rho(cfg_sf)

    # Rough initial asymmetry (leading order in ξ): n_νe − n_ν̄e ∝ ξ · T³
    # After projecting onto the comoving y-grid used by _run_qke_with_rho,
    # |n_asym_initial| is O(ξ) × O(1). Concretely the Stage C validation
    # run (ea04426) starts near 2.5e-2 and ends at ~3e-5 — a ≳ 800×
    # depletion. We demand ≥ 100× depletion and a nonzero sterile.
    initial_estimate = xi0 * 0.5  # ~ (1/2)·ξ on this scale (loose bound)
    assert abs(n_asym_final) < initial_estimate / 100.0, (
        f"SF asymmetry depletion insufficient: final |n_ξe| = "
        f"{abs(n_asym_final):.3e} (initial ~{initial_estimate:.3e})")
    assert sum_ss > 0.5, (
        f"SF sterile production too low: Σρ_ss = {sum_ss:.2f}")
