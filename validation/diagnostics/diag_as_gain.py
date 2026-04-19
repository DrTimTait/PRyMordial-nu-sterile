"""Stage E.2(a) diagnostic: test hypothesis A (active-sterile gain asymmetry).

Background. Stage E.1 landed Mirizzi+2012 pair-specific off-diagonal damping
(_compute_D_pair_matrix helper; qke_damping_formula default "mirizzi") but
the small-mixing DW literature gap did NOT close. At Hannestad Point C
(sin^2 2theta_24=1e-4, Dm2_41=0.93) with PMNS on PRyMordial gives dNeff ~ 0.858;
with PMNS off it gives dNeff ~ 0.29 vs Hannestad's 0.04.

Hypothesis A (per doc/STAGE_E2_BRIEF.md). In _assemble_collision_N
(PRyM/PRyM_boltzmann.py lines 4469-4473) the loop sets
    S_gain_si = gain_active[2*p_idx] + 1j*gain_active[2*p_idx+1]   for p_idx < 3 (active-active)
    S_gain_si = 0.0                                                 for p_idx >= 3 (active-sterile)
Gariazzo+2019 Eq. A.16 uses the damping approximation for ALL off-diagonals.
PRyMordial's asymmetric treatment may be inflating active-active coherence,
which then leaks into active-sterile coherence via the Hamiltonian and is
dumped into rho_ss by damping.

Important scope note. With PMNS OFF (theta_12 = theta_13 = theta_23 = 0)
active-active coherences start at zero and have NO Hamiltonian driving term;
they stay at zero throughout evolution. _offdiag_collision_gain is linear in
the input rho_offdiag, so it returns zero identically, and variants (i), (ii),
(iii) are numerically equivalent. A previous PMNS-off run of this script
confirmed all three variants produce dNeff = +0.2928 to 4 decimals. That run
therefore tells us hypothesis A is unobservable in the PMNS-off regime but
does NOT rule it out: the residual 0.29 - 0.04 = 0.25 over-production is
structural to the 2-level mu-s subsystem and unrelated to active-active gain.

Where hypothesis A CAN contribute is in the 0.858 - 0.29 = 0.57 extra
over-production induced by turning PMNS on. That is the PMNS-on run this
script performs by default.

Variants (PMNS on, Hannestad Point C, 4-flavor, Mirizzi damping):
  (i)   baseline       -- no patch. Expect dNeff ~ 0.858.
  (ii)  zero_all_gain  -- _offdiag_collision_gain patched to return zeros,
                          so all off-diagonals get pure damping. If this
                          drops dNeff toward ~0.29, active-active gain is
                          what inflates coherence in the PMNS-on regime.
  (iii) sym_as_gain    -- active-sterile pairs receive the active-active
                          gain of a matching active-active pair (pair 3
                          <- pair 0, pair 4 <- pair 1, pair 5 <- pair 2).
                          Tests whether adding active-sterile gain (rather
                          than removing active-active gain) is the fix.

Run:
    python validation/diagnostics/diag_as_gain.py
Runtime ~70 min (4 BBN solves at n_B default).

Decision tree (post-run):
  - (ii) drops to ~0.29 or below -> active-active gain inflates sterile
                                    production through PMNS-mixing. Land
                                    fix as qke_offdiag_gain_mode flag.
  - (iii) drops (only)           -> need matched active-sterile gain; defer
                                    physics-motivated coupling to follow-on.
  - neither drops                -> hypothesis A falsified at PMNS-on too.
                                    Hand off to hypothesis B (evolution window).
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


def _point_c_flags(pmns_off=False):
    """Shared flag setup for Hannestad Point C.

    pmns_off=True forces theta_12 = theta_13 = theta_23 = 0 (Hannestad 1+1
    equivalent). Default (pmns_off=False) uses PRyMini's PDG PMNS defaults --
    this is the regime where hypothesis A has observational leverage.
    """
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
    if pmns_off:
        PRyMini.theta_12 = 0.0
        PRyMini.theta_13 = 0.0
        PRyMini.theta_23 = 0.0
    # Mirizzi default is the committed E.1 baseline; do not flip.
    PRyMini.qke_damping_formula = "mirizzi"


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


# Gain stash for variant (iii): _offdiag_collision_gain writes into this list
# (one entry per sector per _assemble_collision_N invocation); the wrapped
# _assemble_collision_N reads from it and adds symmetric active-sterile gain.
_gain_stash = {"buffer": []}


def _run(label, configure, variant):
    """Run one BBN solve with a specified diagnostic variant.

    variant in {"baseline", "zero_all_gain", "sym_as_gain"}.
    Returns (Neff, sum_rho_ss).
    """
    configure()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    # Reload the boltzmann module so monkey-patches always start from a
    # clean baseline and do not compound across variants.
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)

    # Apply variant-specific patches AFTER reload, BEFORE PRyMmain builds
    # the solver (PRyMmain.PRyMclass constructs DensityMatrixSolver at run
    # time and resolves _offdiag_collision_gain via module globals).
    original_gain = PRyMboltz._offdiag_collision_gain
    original_gain_massive = PRyMboltz._offdiag_collision_gain_massive
    original_assemble = PRyMboltz.DensityMatrixSolver._assemble_collision_N

    if variant == "baseline":
        pass  # no patch
    elif variant == "zero_all_gain":
        def _zero_gain(rho_offdiag, f_all, y_grid, quad_w, a, Tg,
                       GF2_prefactor, c_emu_scat, c_mutau_scat,
                       fnu_emu_scat_val, fnu_mutau_scat_val,
                       B_spectator_e_idx, tail_params,
                       D_k0, D_k2, Ny_coll):
            return np.zeros((6, len(y_grid)))

        def _zero_gain_massive(rho_offdiag, f_all, y_grid, quad_w, a, Tg,
                               GF2_prefactor, c_emu_scat, c_mutau_scat,
                               B_spectator_e_idx, tail_params,
                               D_k0, D_k2, Ny_coll, me):
            return np.zeros((6, len(y_grid)))

        PRyMboltz._offdiag_collision_gain = _zero_gain
        PRyMboltz._offdiag_collision_gain_massive = _zero_gain_massive

    elif variant == "sym_as_gain":
        # Stash the gain each call; the wrapped _assemble_collision_N below
        # reads the stash and adds symmetric active-sterile contributions.
        def _stashing_gain(*args, **kwargs):
            out = original_gain(*args, **kwargs)
            _gain_stash["buffer"].append(np.asarray(out).copy())
            return out

        def _stashing_gain_massive(*args, **kwargs):
            out = original_gain_massive(*args, **kwargs)
            _gain_stash["buffer"].append(np.asarray(out).copy())
            return out

        PRyMboltz._offdiag_collision_gain = _stashing_gain
        PRyMboltz._offdiag_collision_gain_massive = _stashing_gain_massive

        def _wrapped_assemble(self, rho_all, a, Tg):
            _gain_stash["buffer"] = []
            N_list, I_total = original_assemble(self, rho_all, a, Tg)
            # Buffer now holds [gain_sector0, gain_sector1], each (6, Ny).
            # Map active-sterile pair q in {0,1,2} to active-active pair q
            # (pair 3 "e-s" <- pair 0 "e-mu", etc.). Simple symmetric rule.
            inv_rate = 1.0 / self._eV_to_secm1
            if self.n_flavor != 4:
                return N_list, I_total
            if len(_gain_stash["buffer"]) != 2:
                # Defensive: unexpected call pattern; skip injection.
                return N_list, I_total
            for sector, gain_active in enumerate(_gain_stash["buffer"]):
                for q in range(3):
                    p_idx = 3 + q
                    alpha, beta = self._all_pair_flavors[p_idx]
                    S_gain_si = (gain_active[2 * q]
                                 + 1j * gain_active[2 * q + 1])
                    rhs_eV_extra = S_gain_si * inv_rate
                    N_list[sector][:, alpha, beta] += rhs_eV_extra
                    N_list[sector][:, beta, alpha] += np.conj(rhs_eV_extra)
            return N_list, I_total

        PRyMboltz.DensityMatrixSolver._assemble_collision_N = _wrapped_assemble

    else:
        raise ValueError(f"unknown variant {variant!r}")

    try:
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
    finally:
        # Restore originals so subsequent variants see a clean module.
        PRyMboltz._offdiag_collision_gain = original_gain
        PRyMboltz._offdiag_collision_gain_massive = original_gain_massive
        PRyMboltz.DensityMatrixSolver._assemble_collision_N = original_assemble
        _gain_stash["buffer"] = []

    return Neff, sum_ss


if __name__ == "__main__":
    print("=" * 88)
    print("Stage E.2(a) diagnostic: active-sterile gain asymmetry (hypothesis A)")
    print("Point C: sin^2 2theta_24 = 1e-4, Dm2_41 = 0.93 eV^2, PMNS ON, Mirizzi damping.")
    print("Hannestad target: dNeff ~ 0.04.  PMNS-on baseline (E.1): dNeff ~ 0.858.")
    print("PMNS-off prior run (separately recorded): all 3 variants = 0.2928 (null)")
    print("because active-active coherences are identically zero without PMNS driving.")
    print("=" * 88)

    Neff_3x3, _ = _run("3x3 reference (no sterile)", _configure_3x3, "baseline")
    print()

    print("--- Variant (i) baseline: no patch -------------------------------------")
    Neff_i, ss_i = _run("(i) baseline", _configure_point_c, "baseline")
    dNeff_i = Neff_i - Neff_3x3
    print(f"     dNeff = {dNeff_i:+.4f}")
    print()

    print("--- Variant (ii) zero all off-diagonal gain (Gariazzo A.16 uniform) ---")
    Neff_ii, ss_ii = _run("(ii) zero_all_gain", _configure_point_c, "zero_all_gain")
    dNeff_ii = Neff_ii - Neff_3x3
    print(f"     dNeff = {dNeff_ii:+.4f}")
    print()

    print("--- Variant (iii) symmetric active-sterile gain (pair q <- pair q) ---")
    Neff_iii, ss_iii = _run("(iii) sym_as_gain", _configure_point_c, "sym_as_gain")
    dNeff_iii = Neff_iii - Neff_3x3
    print(f"     dNeff = {dNeff_iii:+.4f}")
    print()

    print("=" * 88)
    print(f"{'variant':30s} {'dNeff':>10s} {'sum_rho_ss':>12s} {'vs Hannestad 0.04':>22s}")
    print("-" * 88)
    for name, dN, ss in [("(i) baseline", dNeff_i, ss_i),
                         ("(ii) zero_all_gain", dNeff_ii, ss_ii),
                         ("(iii) sym_as_gain", dNeff_iii, ss_iii)]:
        verdict = "CLOSE"       if abs(dN - 0.04) < 0.05 else \
                  "<0.1 (good)" if dN < 0.1            else \
                  "<0.3"        if dN < 0.3            else \
                  "OVERPRODUCED"
        print(f"{name:30s} {dN:+10.4f} {ss:12.3f} {verdict:>22s}")
    print("=" * 88)
    print()
    print("Decision:")
    print("  - If (ii) < 0.1 and (i) ~ 0.29: active-active gain is the culprit.")
    print("    Fix via new qke_offdiag_gain_mode flag ('damping_only').")
    print("  - If (iii) < 0.1 and (ii) does not drop: need matched active-sterile")
    print("    gain; defer to Stage E.2(a-ii) for physics-motivated coupling.")
    print("  - If neither drops: hypothesis A falsified; hand off to B (window).")
