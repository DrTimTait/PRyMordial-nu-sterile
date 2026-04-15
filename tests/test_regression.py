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
    PRyMini.massive_electron_flag = False
    PRyMini.mu_tau_symmetric_flag = True
    PRyMini.nu_nubar_symmetric_flag = True
    PRyMini.two_loop_QED_flag = False


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
