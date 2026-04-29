# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.special import zeta

#########################
# Set working directory #
#########################
working_dir = os.getcwd()
if(working_dir == " "):
    print(" ")
    print("User has to properly set working directory in PRyM_init.py")
    print("Example: /Users/.../PRyMordial")
    print(" ")
    exit(0)
# Set flag to True for verbose mode
verbose_flag = False
# Set flag to True if Numba is installed
numba_flag = True # if True, speed up some integrations in PRyM_thermo.py with Numba
# Set flag to True if Numdifftools is installed
numdiff_flag = False # if True, numerical derivative in PRyM_main.py via Numdifftools

########################################
# Units for PRyMordial nuclear network #
########################################
# PDG = Particle Data Group, https://pdglive.lbl.gov/
Kelvin = 1. # temperature unit
second = 1. # [CGS unit]
cm = 1. # [CGS unit]
gram = 1. # [CGS unit]
erg = gram*cm**2/second # [CGS unit]
kB = 1.380649*1.e-16*erg/Kelvin # Boltzmann constant, PDG
clight = 2.99792458*1.e+10*cm/second # speed of light, PDG
hbar = 6.62607015/(2*np.pi)*1.e-27*erg*second # Planck constant, PDG
Mpc = 3.08567758149*1.e+24*cm
MeV = 1.602176634*1.e-6*erg
keV = 1.e-3*MeV
# Useful conversion factors
MeV_to_Kelvin = MeV/kB
MeV_to_secm1 = MeV/hbar
MeV_to_g = MeV/clight**2
MeV_to_cmm1 = MeV/(hbar*clight)
MeV4_to_gcmm3 = MeV_to_g*MeV_to_cmm1**3

####################
# Temperature eras #
####################
# High temperatures: T_start - T_weak
T_start = 10.*MeV_to_Kelvin # O(10^-2) [s]
T_weak = 1.*MeV_to_Kelvin # O(1) [s]
# Mid temperatures: T_weak - T_nucl
T_nucl = 0.1*MeV_to_Kelvin # O(10^2) [s]
# Low temperatures: T_nucl - T_end
T_end = 1.e-3*MeV_to_Kelvin # O(10^6) [s]
# Number of sampling points for thermodynamics background
n_sampling = 1200 # recommended for accuracy
# Optional override for the Phase B (Boltzmann/QKE) step count. When None,
# n_B defaults to max(2000, 2 * n_sampling). Setting a larger integer here
# refines the timestep dt = da/(a H) used by the density-matrix / Boltzmann
# integrators without perturbing Phase A or Phase C resolution. Intended
# mainly for dt-convergence diagnostics and Stage D refinement tests.
n_B_override = None
# Stage E.2 sprint 10: Phase-0 QKE driver that evolves the 4x4 active-sterile
# density matrix from T_phase0_start (~100 MeV) down to T_boltz_start before
# Phase B begins. Supplies a history-preserving rho_all(T_boltz_start) to
# Phase B, replacing the previous Phase-A thermal-FD cutoff IC. Targets the
# low-y over-thermalisation localised by sprint 9's MSW-passage diagnostic:
# modes whose MSW resonance lies at T_res >= T_boltz_start would otherwise
# arrive at Phase B already past crossing with rho_ss = 0, letting the
# driver overdrive them under large vacuum mixing. Default False preserves
# sprint-9 behaviour bit-identically; enabling the flag requires T_start
# to exceed T_phase0_start so Phase A can reach the Phase-0 entry point.
qke_phase0_flag = False
# Stage E.2 sprint 10: Phase-0 upper temperature [MeV]. Must lie above
# T_boltz_start; the validator refuses equal or inverted windows. Default
# 100 MeV chosen so V_thermal dominates the vacuum mass splitting across
# all practical y-modes (theta_m ~ 0), making the thermal-FD adiabatic-
# vacuum IC correct at the Phase-0 entry.
T_phase0_start = 100.0
# Stage E.2 sprint 10: optional Phase-0 step-count override. When None,
# validate_configuration auto-scales from the Phase-0 window per the
# heuristic documented in the sprint-10 ROADMAP entry.
n_B_phase0_override = None
# Stage E.2 sprint 10 post-landing probe: per-step Phase-0 rho_ss(y) history.
# When True, _run_qke_segment appends a (istep, a, Tg, rho_ss_slice.copy())
# tuple after each Phase-0 step to PRyMclass._phase0_rho_ss_history. Used by
# validation/diagnostics/diag_phase0_pointC.py to localise the Point-C
# narrow-mixing Phase-0 runaway. Default False: zero overhead, no allocation.
qke_phase0_diag_flag = False
# Stage E.2 sprint 19 active-sector probe: per-outer-step active-flavor and
# sterile y³ moments across both Phase 0 and Phase B. When True,
# _run_qke_segment appends a tuple
# (istep, t, a, Tg, sigma, m3_e, m3_mu, m3_tau, m3_s) per outer step to
# PRyMclass._active_probe_history, where m3_α = Σ_sector (Σ_y y³ rho_αα(y, sector)) · dy
# is the comoving y³ moment per flavor (single value per step). Used by
# validation/diagnostics/diag_sprint19_active_sector_probe.py to localise the
# active-sector pathology that appears at production n_B for the no-Phase-0
# closure config (Neff=417, Yp=0.36 at n_B=12000 vs Neff=3.91, Yp=0.249 at
# n_B=3500). Default False: zero overhead, no allocation.
qke_active_probe_flag = False
# Stage E.2 sprint 19 part 2 post-Phase-B trace: capture the post-Phase-B
# pipeline state to localise the active-sector pathology to one of three
# downstream paths (interpolator construction in update_thermo_distributions /
# Phase-C solve_ivp / rho_nu_from_f quadrature). Sprint-19 part 1 already
# exonerated the QKE driver itself (end-of-Phase-B m3_α agree to ~1.77% across
# n_B=12000 vs 3500), but downstream Neff differs by 107x. When True,
# update_thermo_distributions stashes the raw f_α grids and the constructed
# callables on PRyMthermo._post_phaseB_trace_grids /
# PRyMthermo._post_phaseB_trace_callables; PRyMclass attaches
# _post_phaseB_trace = {t_C, Tg_C, p_nodes_alpha, integrand_alpha} after
# Phase C / N_eff readout. Used by
# validation/diagnostics/diag_sprint19_post_phaseB_trace.py. Default False:
# zero overhead, no allocation.
qke_post_phaseB_trace_flag = False
# Stage E.2 sprint 19 part 2 cure: tighten the FD-tail extrapolation in
# BoltzmannSolver._make_f_callable / make_f_callable. The default code uses
# polyfit on the last 10 in-range grid points to estimate the tail decay
# rate _tail_b, falling back to 1/y_grid[-1] only when _tail_b <= 0. For
# QKE-evolved distributions the last-10 tail is dominated by initial-condition
# FD plateau (f ~ 0.27 at y ~ 99 for Tstart_MeV = 105) with small fluctuations,
# so polyfit returns a tiny but POSITIVE _tail_b ~ 1e-3 (vs the physical
# FD slope ~1/Tstart_MeV ~ 0.01). The shallow tail then survives the y → ~4200
# extrapolation needed by rho_nu_from_f at low Tg, contaminating rho_3nu by
# ~100x and inflating Neff to 417 at production n_B (sprint-18 closure config).
# When True, the fallback condition is extended to _tail_b < 1/y_grid[-1],
# clamping the polyfit to at least the physical FD-equivalent decay rate.
# Sprint-19 part 2 trace data (validation/diagnostics/diag_sprint19_post_phaseB_trace)
# verifies: cure brings Neff at end-of-BBN from 417 → 2.88 (production) and
# 3.91 → 2.40 (reduced n_B=3500), preserving the sterile-sector closure
# δNeff_ss in HTT 2012 band [0.02, 0.10].
# Stage F sprint 2 default flip: True. The cure was originally introduced
# default-False to preserve bit-identical pre-cure behaviour. Stage F
# sprint 2 verifies the cure does not regress any other code path
# (Mode 1/2/6 thermal, Mode 3 Boltzmann, Mode 5 QKE; see ROADMAP Stage F
# sprint 2 record), so the closure-config value becomes the production
# default. To recover pre-cure behaviour set this back to False.
qke_post_phaseB_clamp_flag = True
# Range in time for sampling of thermodynamics background
t_end = 1.e+7 # [s], chosen as 10 x O(t(T_end))

#######################################
# Flags for background thermodynamics #
#######################################
# Set flag to True for incomplete decoupling effects in a(T)
aTid_flag = True
# Set flag to True to include O(e^4) two-loop QED plasma corrections
# (Escudero, Jackson, Laine, Sandner 2025, arXiv:2511.04747; NUDEC_BSM v2).
# When True, the plasma pressure P, dP/dT, d^2P/dT^2 pick up the p^(4) term
# on top of the baseline O(e^2) + O(e^3) contributions. Expected shifts:
# delta Neff ~ +1e-4, delta Yp/Yp ~ +1e-4. Tables live in PRyMrates/thermo/
# (regenerate via PRyMrates/thermo/build_QED_e4.py).
two_loop_QED_flag = False
# Enable an approximate NLO electroweak correction to nu-e rates.
# When True, the nu-e collision prefactor GF^2 is scaled by
# nlo_weak_rate_scale. This is a single-factor placeholder for the
# full one-loop EW corrections (vertex + box + photon emission) computed
# by Akita & Yamaguchi 2020 (arXiv:2005.07047) and discussed in
# Bennett+2021 Table 5. The literature reports delta Neff ~ -6e-5 to
# -8e-5 for this class of corrections — small. Zero overhead when False.
nlo_weak_flag = False
# Multiplicative scale for nu-e rates when nlo_weak_flag=True. Default
# 1.003 corresponds to a representative ~0.3% enhancement consistent
# with the Sirlin-like NC radiative factor rho_NC ~ 1.003 (see e.g.
# Bardin & Passarino, "The Standard Model in the Making"). Full per-
# process corrections require the scale_scat / scale_ann plumbing
# already present in _collision_integral_nu_e; this scalar gives the
# leading effect in one line. Users who care about the ~5e-5 precision
# floor should consult Akita & Yamaguchi 2020 for the exact structure.
nlo_weak_rate_scale = 1.003
# Set flag to compute background
compute_bckg_flag = True
# Set flag to True to save background thermodynamics, if recomputed
save_bckg_flag = False
# Set flag to True for general (non-thermal) neutrino distribution functions
# When True, the user supplies f_nue(p,Tg), f_nuebar(p,Tg), f_numu(p,Tg), f_numubar(p,Tg)
# via the PRyMclass constructor. The code computes rho_nu, P_nu, weak rates, and collision
# terms by numerically integrating over these distributions instead of using analytic T^4 formulas.
general_nu_flag = False
# Quadrature for numerical integration over general distributions
# Uses dimensionless variable x = p/Tg with GL quadrature on [0, x_max_nu]
# so the grid automatically scales with temperature.
x_max_nu = 30. # dimensionless, upper limit p/Tg for quadrature
p_npoints_nu = 200 # number of Gauss-Legendre quadrature points
# Set flag to True to evolve neutrino distributions via internal Boltzmann solver
# When True, the code solves the unintegrated Boltzmann equation on a comoving momentum
# grid, computing SM collision integrals from first principles. Implies general_nu_flag = True.
boltzmann_nu_flag = False
# Full 3×3 density matrix QKE solver for neutrino oscillations.
# Tracks off-diagonal flavor coherences to recover Neff = 3.044.
# Implies boltzmann_nu_flag = True and general_nu_flag = True.
qke_density_matrix_flag = False
# Stage D: full-QKE ODE-driver variant of the density matrix solver.
# When True (and qke_density_matrix_flag is True), per-step evolution uses an
# ETD1-style scheme: exact unitary conjugation ρ → U ρ U† via per-mode
# diagonalization of H, replacing the two quasi-static approximations used
# by the default evolve_step (Sigl-Raffelt active-active diagonal relaxation
# and the Stage B Dodelson-Widrow active-sterile transfer). Expected impact
# on SM observables is ≤ 10⁻³ on Neff; main payoff is architectural — cleaner
# BSM matter-potential composition and no need to maintain the two quasi-
# static approximations when extending the code. Zero overhead when False.
qke_full_ode_flag = False
# Stage D.6: ETDRK2 entry point for the full-QKE ODE driver. When True (and
# qke_full_ode_flag is also True), per-step evolution uses ETDRK2 with the
# collision operator rotated into the instantaneous H-eigenbasis. The
# eigenbasis transform gives each eigenpair (k, l) its own phi_1 / phi_2
# factor, so MSW turning points (where the in-medium mixing angle sweeps
# through pi/4 faster than dt) are handled without the O(dt) degradation
# the flavor-basis Strang split exhibits (Stage D.4 finding). Default
# False; zero impact on all existing code paths.
qke_ode_etdrk2_flag = False
# Stage E.1: off-diagonal pair damping formula.
#   "symmetric": legacy D_{alpha,beta} = 0.5 * (Gamma_alpha + Gamma_beta).
#     Physically wrong for active-sterile pairs (gives D_{alpha,s} = 0.5*Gamma_alpha
#     because C_D[sterile]=0); kept for reproducing pre-E.1 results.
#   "mirizzi":   Mirizzi+2012 Eq. 28, pair-specific,
#     D_{alpha,beta} = 0.5*G_F^2*T^4*E*[(g_alpha^s - g_beta^s)^2 + (g_alpha^a + g_beta^a)^2]
#     with g^s = sqrt(C_D), g^a = sqrt(C_A).
#   "gariazzo":  Gariazzo+2019 App. A.17-A.20, sin^2(theta_W)-specific (fallback form).
# Stage F sprint 2 default flip: 'symmetric'. The sprint-17 → sprint-18
# axis bracket showed 'symmetric' is the closure-config choice for the
# Hannestad-style sterile-sector cure (sprint-18 part 2 + sprint-19
# substantial closure). 'mirizzi' is the more physically motivated
# pair-specific form, kept available as an opt-in.
qke_damping_formula = "symmetric"
# Stage E.2 sprint 4: global multiplier on the pair-damping rate D_{a,b}.
# Default 1.0 (no effect). Non-unit values multiply every off-diagonal
# damping entry returned by DensityMatrixSolver._compute_D_pair_matrix.
# Intended only as a diagnostic knob for sensitivity tests against
# FortEPiaNO-scale references; do NOT set != 1.0 in production runs.
qke_damping_scale = 1.0
# Stage E.2 sprint 5: Hamiltonian-potential diagnostic knobs. Each
# multiplies the respective potential wherever it is ADDED to H in
# _build_H_list (and the two adjacent in-place H-builders). Default 1.0
# preserves all existing behaviour. Non-unit values (and especially 0.0)
# are purely diagnostic; do NOT set != 1.0 in production runs.
#   qke_v_nc_scale:      Notzold-Raffelt NC thermal term on active diagonals
#                        (8√2·G_F·ρ_e·E/(3·m_Z²)).
#   qke_v_thermal_scale: Notzold-Raffelt CC thermal term on ν_e only
#                        (8√2·G_F·ρ_e·E/(3·m_W²)).
#   qke_v_nunu_scale:    ν-ν self-interaction matrix (Pantaleone-Sigl-Raffelt
#                        √2·G_F·∫y²(ρ-ρ̄)dy/(2π²·a³)).
qke_v_nc_scale = 1.0
qke_v_thermal_scale = 1.0
qke_v_nunu_scale = 1.0
# Stage E.2 sprint 5: restrict the V_nunu self-interaction matrix to the
# active-flavor block. Physical: the Z-exchange vertex couples only to
# SU(2)_L doublet neutrinos; the sterile (singlet) has zero NC charge, so
# V_nunu[s, *] = V_nunu[*, s] = V_nunu[s, s] = 0 identically. When True,
# the sterile row and column of V_nunu_eV are zeroed before being added
# to H in _build_H_list.
# Stage F sprint 2 default flip: True. Sprint-18 part 2 + sprint-19
# substantial closure verified this as a physical fix (zero NC charge
# on the sterile is exact, not a model approximation), forming part of
# the Stage E.2 closure config. To recover pre-sprint-5 behaviour set
# this back to False.
qke_v_nunu_active_only = True
# Stage E.2 sprint 8: per-step energy-accounting diagnostic inside
# evolve_step_ode_etdrk2. When True, the Strang-split driver appends a
# row per time-step to self._energy_hist containing, for each active-
# sterile pair (alpha, s) in {(0,3), (1,3), (2,3)} and each sector, the
# number-weighted integral N_alphas = dy*sum_y y^2*(rho_aa+rho_ss) and
# the energy-weighted integral E_alphas = dy*sum_y y^3*(rho_aa+rho_ss)
# (uniform midpoint rule on the full Ny grid), at three sub-step
# positions (pre-step, post-corrector, post-clip). Sprint-8 diagnostic
# hook for Suspect 1 (active-sterile energy-balance anomaly). Default
# False: zero branches taken, zero measurable effect on runtime,
# bit-identical output.
qke_energy_diag_flag = False
# When qke_energy_diag_flag is True and this path is non-None, the
# accumulated energy history is saved as a .npz file at Phase-B exit.
qke_energy_diag_path = None
# Stage E.2 sprint 9: per-step per-y-mode MSW-passage instrumentation on
# evolve_step_ode_etdrk2. When True, the driver appends a row per time-
# step to self._msw_hist containing, for a single active-sterile pair
# (alpha, s=3) selected by qke_msw_diag_pair_idx and both sectors, the
# per-y Hamiltonian entries (H_aa, H_ss, H_as) in eV and the diagonal
# populations (rho_aa, rho_ss) at two sub-step positions (pre-step,
# post-clip). Sprint-9 diagnostic hook for Suspect 2 (MSW-passage
# adiabatic over-pumping in the ETDRK2 eigenbasis). Default False:
# zero branches taken, zero measurable effect on runtime, bit-identical
# output.
qke_msw_diag_flag = False
# When qke_msw_diag_flag is True and this path is non-None, the
# accumulated MSW-passage history is saved as a .npz file at Phase-B
# exit.
qke_msw_diag_path = None
# Pair index in self._all_pair_flavors for the channel to instrument;
# 4 = (alpha=1=numu, s=3), the Hannestad Point-A (theta_24) channel.
qke_msw_diag_pair_idx = 4
# Stage E.2 sprint 11: per-mode eigendecomposition fallback inside
# _etdrk2_expm_phi for y-modes near MSW eigenvalue collapse, where the
# Al-Mohy & Higham augmented-matrix expm produces a step-function
# branch transition (sprint-10 Point-C signature: ν̄ y=0.5 jump
# at Tg≈62 MeV, sin^2(2θ)=1e-4). When True, modes whose active-sterile
# Hamiltonian-diagonal gap |H_αα − H_ss| falls below
# qke_expm_fallback_eps_cross times the largest commutator gap on that
# mode are propagated via direct eigendecomposition of L; modes that
# pass the gate use the existing Al-Mohy path. Default False:
# zero branches taken, bit-identical to sprint-10 output.
qke_expm_fallback_near_degeneracy = False
# Relative threshold for the MSW gate (see qke_expm_fallback_near_degeneracy).
# Smaller values keep more modes on the Al-Mohy path.
qke_expm_fallback_eps_cross = 1.0e-3
# Stage E.2 sprint 12: per-step sub-step instrumentation inside
# evolve_step_ode_etdrk2 to localise the Phase-0 step-function in
# rho_ss(y=0.5, nu-bar) at istep 906->1035 (Hannestad Point C with
# qke_expm_fallback_near_degeneracy on, Phase-0 unchanged from sprint 10).
# When True, the Strang-split driver appends one row per Strang substep
# (after_half1, after_predictor, after_corrector, after_half2) to
# self._phase0_substep_hist at the y-grid index closest to
# qke_phase0_substep_y_target. Each row captures: H diagonals + active-
# sterile coupling at both sectors (5a probe); (rho_nu - rho_nubar) at the
# target y-mode (5a probe); N_gain pre/post D*rho add-back (5b probe);
# predictor and corrector deltas (5b probe); and z_h per channel (5c
# probe). Default False: zero branches taken, bit-identical output.
qke_phase0_substep_diag_flag = False
# y-mode under investigation for the Phase-0 sub-step probe. The
# instrumentation selects the grid index whose y-value is closest to this
# target.
qke_phase0_substep_y_target = 0.5
# Stage E.2 sprint 12: opt-in H-iteration in the ETDRK2 corrector to
# smooth V_nunu non-linear feedback at narrow mixing. When True (and the
# sprint-11 fallback flag is also True), the Strang-split driver
# re-evaluates H(rho_star), L(rho_star), and Phi2(L(rho_star)) between the
# predictor and corrector, replacing the cached Phi2(L(rho_n)) used by the
# corrector update. Sprint-12 sub-suspect 5a fix: the V_nunu(ρ - ρ̄)
# integral grows non-linearly across the istep 906→1035 window at
# Hannestad Point C, shifting H_α,α relative to H_ss; without iterating H,
# the corrector's frozen-coefficient Phi2 kick over-pumps rho_star into
# the resonance, producing the step-function in ρ_ss(y=0.5, ν̄). Cost when
# on: one extra _build_H_list + _build_L_list + _etdrk2_expm_phi per
# step. Default False: bit-identical to sprint-11.
qke_etdrk2_iterate_h_flag = False
# Stage E.2 sprint 14: opt-in parallel LSODA reference driver. When True
# (and qke_full_ode_flag is True), the QKE per-outer-step dispatcher routes
# through DensityMatrixSolver.evolve_step_lsoda — a scipy.integrate.solve_ivp
# call over [0, dt] with method='LSODA' that integrates the unsplit ODE
# dρ/dt = -i[H, ρ] + N_collision. Reuses _build_H_list and
# _assemble_collision_N at the kernel level so physics matches ETDRK2; the
# only difference is the time integrator (adaptive multistep vs Strang split).
# Built as a falsifier for Suspect 7 (Strang-split incompatibility with
# narrow-mixing resonance crossings) and Suspect 8 (Hannestad target wrong
# for self-consistent V_nunu). Default False: ETDRK2 path stays as production.
qke_lsoda_driver_flag = False
# Tolerances for solve_ivp under qke_lsoda_driver_flag. Defaults chosen to
# match ETDRK2 corrector accuracy on Phase-0 / Phase-B trajectories.
qke_lsoda_rtol = 1.0e-6
qke_lsoda_atol = 1.0e-10
# Sprint 15: analytic Jacobian for the LSODA driver. Cuts the FD-Jacobian
# rebuild cost (3200 RHS evals/rebuild at 4×4 sterile, Ny=100) to one
# matrix evaluation per outer step — the FD path was the wall-clock
# bottleneck that killed sprint-14 gate 5. The Jacobian uses H frozen at
# start-of-outer-step (consistent with the Strang/ETDRK2 freezing
# convention) and drops the collision-integral derivative (smooth across
# the outer dt). The RHS itself stays exact. Disable to fall back to FD
# Jacobian for cross-validation against the sprint-14 baseline.
qke_lsoda_analytic_jac_flag = True
# Sprint 15 fallback: use a banded Jacobian and a (mode, sector, component)
# state-vector layout so LSODA's per-step LU drops from O(n_dof³) (which
# made the dense-Jacobian path 7-10× slower than ETDRK2 at 3×3 SM) to
# O(n_dof · bandwidth²). With H frozen at start-of-outer-step the unitary
# + damping Jacobian has no cross-sector coupling, so bandwidth =
# n_components - 1 (8 at 3-flavor, 15 at 4-flavor sterile). Effective only
# when qke_lsoda_analytic_jac_flag is also True.
qke_lsoda_jac_band_flag = True
# Sprint 15 option (c): segment-only LSODA dispatch. When both Tg_max
# and Tg_min are positive AND qke_lsoda_driver_flag is on, the dispatcher
# in PRyM_main._run_qke_segment uses LSODA only when Tg_mid is inside
# (Tg_min, Tg_max] and falls back to ETDRK2 outside that window. Default
# 0 / 0 disables the window, preserving sprint-14 behaviour: LSODA on for
# every outer step. The window option exists because LSODA is ~10× slower
# than ETDRK2 in benign regimes (RHS-bound on collision integrals); using
# it only across the resonance crossing where ETDRK2's Strang split is
# under suspicion (Suspect 7) restores ETDRK2's bulk wall-clock.
qke_lsoda_window_Tg_max_MeV = 0.0
qke_lsoda_window_Tg_min_MeV = 0.0
# Stage E.2 sprint 16: opt-in Krogstad ETDRK4 driver. When True (and
# qke_full_ode_flag is True), the QKE per-outer-step dispatcher routes
# through DensityMatrixSolver.evolve_step_ode_etdrk4 — a 4-stage
# exponential time-differencing corrector built on the same
# _build_L_list / _assemble_collision_N kernels as ETDRK2 but with
# phi_0..phi_3 functions, a half-step + full-step augmented expm cache,
# and the literature dt·phi_k convention. Cost ~3× ETDRK2 wall-clock.
# Built as a falsifier for Suspect 7 (ETDRK2 order-2 truncation
# over-pumping the resonance crossing): if Hannestad Point C Σρ_ss
# collapses toward the [0.02, 0.10] band when the corrector order is
# upgraded from 2 to 4, the order-2 truncation was the source. If
# Σρ_ss stays ≈ 22, Suspect 8 wins (Hannestad target incompatible
# with self-consistent V_nunu setup). ETDRK4 takes precedence over
# qke_lsoda_driver_flag and qke_ode_etdrk2_flag in the dispatcher
# when set. Default False: production path stays on ETDRK2.
qke_etdrk4_flag = False
# Comoving momentum grid for Boltzmann evolution (y = p*a)
y_max_boltz = 100.0 # MeV, maximum comoving momentum
Ny_boltz = 100 # number of evenly-spaced grid points
# Temperature range for Boltzmann evolution (neutrinos thermal above, frozen below)
T_boltz_start = 5.0 # MeV, neutrino decoupling onset
# T_boltz_end must lie below the nuclear-network active window (D burn ~0.07 MeV,
# synthesis ~0.05-0.03 MeV, quench ~0.02 MeV). Freezing f inside that window
# creates a Phase B/C handoff discontinuity that biases Yp_BBN by ~0.17% and
# scatters D/H and Li7/H at the 0.1-0.5% level. T_boltz_end = 0.005 MeV puts
# the handoff well after all nuclear activity and brings Yp_BBN under 0.01%
# vs legacy. Integrated collision contribution below 0.1 MeV is negligible,
# so this change costs ~nothing on Neff.
T_boltz_end = 0.005  # MeV
# Collision integral momentum cutoff (y_coll_max < y_max avoids spurious D-kernel artifacts)
y_coll_max_boltz = 50.0  # MeV, collision integral summation limit
# Use massive electron kinematics in nu-e collision integrals.
# When True, D-kernels are computed on-the-fly with E=sqrt(p^2+me^2)
# instead of pre-computed massless tables + fnu correction factors.
# Slower (~3x for nu-e part) but eliminates momentum-averaging bias.
massive_electron_flag = True
# Include time-averaged neutrino flavor oscillations
nu_oscillation_flag = True
# Oscillation method:
#   'collision_mixing' — Sabti/FortEPiaNO: mix collision integrals with
#       time-averaged PMNS transition probabilities (Sabti Eq. 3.18).
#       No freezeout; continuously redistributes e/mu collision rates.
#   'relaxation' — Sigl-Raffelt quasi-static density-matrix relaxation.
#       Operator-split exponential decay toward flavor-averaged f.
#       Freezes out at T ~ 0.5 MeV; requires coll_scale >> 1.
nu_oscillation_method = 'collision_mixing'
# If False, evolve numu, numubar, nutau, nutaubar as four independent species
# rather than collapsing onto a single numu_eff (default True). Required for
# BSM scenarios that break mu-tau symmetry: L_mu-L_tau gauge bosons, flavor-
# specific NSI, flavor-specific DM decay/annihilation, lepton asymmetry.
# Weak rates (n<->p) only see nu_e so Yp/D/H are unaffected in SM; the flag
# opens the door for physics that shifts Neff and the relic nu abundance.
# Stage 1 (QKE path): full 6-species native to DensityMatrixSolver.
# Stage 2 (diagonal Boltzmann, n=4): mu/tau distinct, pcle+antipcle aggregated
#     per flavor. Uses n=3's D-kernel averaging (valid when f_nu = f_nubar).
# Stage 3 (diagonal Boltzmann, n=6): full nu/nubar per flavor with explicit
#     D_k2 (same-sign) vs D_k0 (opposite-sign) kinematics. Needed for
#     lepton-asymmetric scenarios.
mu_tau_symmetric_flag = True
# nu/nu-bar symmetry within each flavor for the diagonal BoltzmannSolver.
# When False (and mu_tau_symmetric_flag=False) the solver tracks all 6
# distributions independently with proper kinematic kernels. QKE path is
# always full 6-species regardless of this flag.
nu_nubar_symmetric_flag = True
# Collision integral calibration scale. Applied to GF2_prefactor so that it
# uniformly scales both nu-nu and nu-e collision integrals.
# At coll_scale=1.0 (no fudge), the diagonal density matrix approximation
# yields Neff ~ 3.030, about 0.4% below the full QKE result of 3.044
# (FortEPiaNO, Bennett+ 2021). The deficit is from missing off-diagonal
# coherences in the density matrix, not from grid resolution or time-stepping
# (verified: Ny=100,150,200 give identical Neff; Heun vs FE negligible).
coll_scale = 1.0
# Neutrino oscillation parameters (PDG 2024, normal ordering)
# Mass-squared differences [eV^2]
Dm2_21 = 7.53e-5   # solar: Δm²₂₁
Dm2_31 = 2.453e-3  # atmospheric: Δm²₃₁ (normal ordering, >0)
# Mixing angles [radians]
theta_12 = np.arcsin(np.sqrt(0.307))  # sin²θ₁₂ = 0.307
theta_13 = np.arcsin(np.sqrt(0.0220)) # sin²θ₁₃ = 0.0220
theta_23 = np.arcsin(np.sqrt(0.546))  # sin²θ₂₃ = 0.546
# CP-violating phase (drops out in damped-oscillation limit)
delta_CP = 1.36 * np.pi  # δ_CP ≈ 245° (PDG 2024 best fit)
# 3+1 sterile neutrino extension.
# When sterile_flag=True, the DensityMatrixSolver upgrades from 3×3 to 4×4,
# adding an extra mass eigenstate (Δm²₄₁) and three new mixing angles
# (θ₁₄, θ₂₄, θ₃₄) plus one new CP phase (δ₁₄). The sterile sector has
# NO SM gauge coupling: zero collision rate, zero matter potential. Production
# occurs via oscillation-mediated decoherence (Dodelson-Widrow) or MSW
# resonance driven by a lepton asymmetry (Shi-Fuller).
# Zero overhead when False (all 3×3 code paths unchanged).
sterile_flag = False
# Sterile mass-squared splitting [eV²]. User-settable for eV-scale
# (short-baseline anomalies, warm DM) or keV-scale (X-ray line).
Dm2_41 = 1.0
# Active-sterile mixing angles [radians] in the standard 3+1 parameterization:
#   U₄ₓ₄ = R₃₄(θ₃₄) × R₂₄(θ₂₄, δ₂₄) × R₁₄(θ₁₄) × [U_PMNS₃ₓ₃ ⊕ 1]
# Recovers the 3-flavor PMNS when θ₁₄ = θ₂₄ = θ₃₄ = 0.
theta_14 = 0.0
theta_24 = 0.0
theta_34 = 0.0
delta_14 = 0.0   # CP phase for active-sterile sector
# Initial lepton chemical potentials ξ_α = μ_α / T_ν at the start of the
# Boltzmann phase. Stage C (Shi-Fuller): a nonzero ξ_νe seeds an asymmetric
# ν_e / ν̄_e distribution, f(y) = 1/(exp(y/T ∓ ξ) + 1), which generates a
# matter potential that drives the active↔sterile MSW resonance. ξ_μ, ξ_τ
# are usually taken zero but are exposed for completeness. Only honored
# when sterile_flag=True; ignored otherwise. A user-supplied f_initial
# callable (through PRyMclass) still takes precedence over these values.
xi_nue_init   = 0.0
xi_numu_init  = 0.0
xi_nutau_init = 0.0
# Set flag to True for some new species with temperature T_NP
NP_thermo_flag = False
# Set the initial temperature of the NP species via relation TNP_start = xi_NP*T_start
xi_NP = 1.
# Set flag to True for new species in thermal equilibrium with neutrinos
NP_nu_flag = False
# Set flag to True for new species in thermal equilibrium with plasma, i.e. photons and e+-
NP_e_flag = False

#################################
# Flags for n <--> p weak rates #
#################################
# Set flag to True to re-compute the weak rates
compute_nTOp_flag = True # if True, re-compute bulk of weak-rate effects from scratch
sampling_nTOp = 50  # recommended for accuracy (number of points used for each era: HT,MT,LT)
# Set flag to True to compute the weak rates in Born approximation
nTOpBorn_flag = False # if True, faster evaluation of n <--> p  rates (at the expense of precision)
# Set flag to True to re-compute thermal radiative corrections in n <--> p  rates
# Computationally intensive, yield subpermil effects on D/H and Li7/H (even smaller on helium)
compute_nTOp_thermal_flag = False # recommended: effects already stored, will need vegas otherwise
sampling_nTOp_thermal = 50 # recommended for accuracy (number of points from T_start to T_end)
# Set flag to True to use the neutron lifetime as standard normalization for the weak rates
tau_n_flag = True
# Set flag to True to save bulk of weak-rate effects, if re-computed
save_nTOp_flag = False
# Number of parallel cores for weak rate computation (0 = auto-detect, 1 = serial)
n_cores_nTOp = 0
# Set flag to True to save tiny thermal effects on weak rates, if re-computed
save_nTOp_thermal_flag = False
# Set flag to True for NP modification of weak rates in units of standard n <--> p Born rates
NP_nTOp_flag = False
# % shift in terms of Born rates
NP_delta_nTOp = 0.

###########################
# Flags for nuclear rates #
###########################
# Set flag to True to use NACRE II database for 12 key nuclear reactions
nacreii_flag = False # as default code adopts PRIMAT compilation for nuclear rates
if(nacreii_flag):
    rates_dir = "key_nacreii_rates/"
else:
    rates_dir = "key_primat_rates/"
# Set flag to True to restrict to 12 nuclear reactions (OK for YP and D/H, not for Li7/H)
smallnet_flag = False
# Set flag to True for NP modification of key nuclear rates in units of standard ones
NP_nuclear_flag = False
# Set flag to True to speed up ODE computation with Julia
julia_flag = False # if True, requires Julia dependencies

##############################################################
# Fundamental constants and particle masses in natural units #
##############################################################
# EW sector: Most precise measurements adopted
alphaem = 1./137.035999084 # fine structure constant, PDG
GF = 1.1663787*10**-5*1.e-6 # Fermi coupling constant in [MeV-2], PDG
mZ = 91.1876*1.e+3 # Z mass in [MeV], PDG
# Tree-level relations for EW couplings
sW2 = 0.5*(1.-np.sqrt(1.-2.*np.sqrt(2.)*np.pi*alphaem/(GF*mZ**2)))
geL, geR, gmuL, gmuR = 1./2.+sW2,sW2,-1./2.+sW2,sW2
# Fermion masses
me = 0.51099895 # electron mass in [MeV], PDG
mn = 939.56542052 # neutron mass in [MeV], PDG
mp = 938.27208816 # proton mass in [MeV], PDG
# Gravitational constant and mass
GN = 6.70883*1.e-39*1.e-6 # Newton constant in [MeV-2]
Mpl = 1./np.sqrt(GN) # Planck mass in [MeV], PDG

#################################################
# Additional parameters for n <--> p weak rates #
#################################################
gA = 1.2756 # Axial current constant of structure of nucleons, PDG (gV = 1)
kappa_p = 2.79284734463-1. # anomalous magnetic moment of the proton, PDG
kappa_n = -1.91304273 # anomalous magnetic moment of the neutron, PDG
deltakappa = kappa_p-kappa_n # weak magnetism constant, arXiv:1212.0332
radproton = 0.8409*1.e-13*cm # proton charge radius in [cm], PDG
tau_n = 878.4*second # neutron lifetime [s], PDG 23 w/ 8 best measurements
Vud = 0.9738 # UTfit prediction, table I in 2212.03894

#######################
# Cosmological inputs #
#######################
# CMB
T0CMB = 2.7255*Kelvin # photon temperature today in K, PDG
s0bar = 4.*np.pi**2/45.
s0CMB = s0bar*(T0CMB/MeV_to_Kelvin)**3 # photon entropy today in MeV^3
n0CMB = (2.*zeta(3))/(np.pi**2)*(T0CMB/MeV_to_Kelvin)**3 # photon density today in [MeV^3]
# Baryons
HubbleOverh = 100*(1.e+5*cm*MeV_to_cmm1)/(second*MeV_to_secm1)/(Mpc*MeV_to_cmm1) # [MeV/h]
rhocOverh2 = 3./(8.*np.pi*GN)*HubbleOverh**2 # [MeV^4/h]
ma = 931.494061 # unit of atomic mass (u.m.a.) in [MeV]
He4Overma = 4.0026032541 # He4 mass in u.m.a.
HOverma = 1.00782503223 # H mass in u.m.a.
# Educated guess on chemical composition at end of BBN for averaged baryon mass
percentHe = 24.7/100. # percentage of helium
percentH = 1.-percentHe # percentage of hydrogen
mB = (percentH*HOverma+percentHe*He4Overma/4.)*ma # averaged baryon mass in MeV
maOvermB = ma/mB # offset between CMB baryonic density and nucleonic BBN one (0.5% effect)
Omegabh2 = 0.02230 # baryon abundance by Planck [A.A. 652 (2021)] (no BBN prior)
Omegabh2_to_eta0b = (rhocOverh2/n0CMB)/(ma/maOvermB)
eta0b = Omegabh2_to_eta0b*Omegabh2 # baryon-to-photon ratio
munuOverTnu = 0. # neutrino chemical potential in units of neutrino temperature
# extra-relativistic degrees of freedom as a shift of neutrino total energy density
DeltaNeff = 0. # must be > -3 to not entirely eat up the SM neutrino contribution

###############################
# Compilation from NUBASE2020 #
###############################
# List of species implemented in terms of [A-Z,Z] composition, A nuclear number, Z atomic number
Nuclides = {"n":[1,0],"p":[0,1],"d":[1,1],"t":[2,1],"He3":[1,2],"a":[2,2],"He6":[4,2],"Li6":[3,3],"Li7":[4,3],"Be7":[3,4],"Li8":[5,3],"B8":[3,5]}
# Nuclei binding energies in MeV
NuclExcessMass = {"n":8071.3171,"p":7288.9706,"d":13135.722,"t":14949.81,"He3":14931.218,"a":2424.9156,"He6":17592.10,"Li6":14086.8789, "Li7":14907.105,"Be7":15769.,"Li8":20945.80,"Be8":4941.67,"B8":22921.6}
# Nuclei spins
NuclSpin = {"n":1./2.,"p":1./2.,"d":1.,"t": 1./2.,"He3":1./2.,"a":0.,"He6":0.,"Li6":1.,"Li7":3./2.,"Be7":3./2.,"Li8":2.,"Be8":0.,"B8":2.}

# Initialization of weights for an MCMC analysis such that median value are adopted for the nuclear rates
num_reactions = 63
p_npdg,p_dpHe3g,p_ddHe3n,p_ddtp,p_tpag,p_tdan,p_taLi7g,p_He3ntp,p_He3dap,p_He3aBe7g,p_Be7nLi7p,p_Li7paa,p_Li7paag,p_Be7naa,p_Be7daap, p_daLi6g,p_Li6pBe7g,p_Li6pHe3a, p_B8naap, p_Li6He3aap, p_Li6taan, p_Li6tLi8p, p_Li7He3Li6a, p_Li8He3Li7a, p_Be7tLi6a, p_B8tBe7a, p_B8nLi6He3, p_B8nBe7d, p_Li6tLi7d, p_Li6He3Be7d, p_Li7He3aad, p_Li8He3aat, p_Be7taad, p_Be7tLi7He3, p_B8dBe7He3, p_B8taaHe3, p_Be7He3ppaa, p_ddag, p_He3He3app, p_Be7pB8g, p_Li7daan, p_dntg, p_ttann, p_He3nag, p_He3tad, p_He3tanp, p_Li7taan, p_Li7He3aanp, p_Li8dLi7t, p_Be7taanp, p_Be7He3aapp, p_Li6nta, p_He3tLi6g, p_anpLi6g, p_Li6nLi7g, p_Li6dLi7p, p_Li6dBe7n, p_Li7nLi8g, p_Li7dLi8p, p_Li8paan, p_annHe6g, p_ppndp, p_Li7taann = np.zeros(num_reactions)
# % shift in terms of median value of all nuclear rates implemented
NP_delta_npdg,NP_delta_dpHe3g,NP_delta_ddHe3n,NP_delta_ddtp,NP_delta_tpag,NP_delta_tdan,NP_delta_taLi7g,NP_delta_He3ntp,NP_delta_He3dap,NP_delta_He3aBe7g, NP_delta_Be7nLi7p,NP_delta_Li7paa,NP_delta_Li7paag,NP_delta_Be7naa,NP_delta_Be7daap, NP_delta_daLi6g,NP_delta_Li6pBe7g,NP_delta_Li6pHe3a, NP_delta_B8naap, NP_delta_Li6He3aap, NP_delta_Li6taan, NP_delta_Li6tLi8p, NP_delta_Li7He3Li6a, NP_delta_Li8He3Li7a, NP_delta_Be7tLi6a, NP_delta_B8tBe7a, NP_delta_B8nLi6He3, NP_delta_B8nBe7d, NP_delta_Li6tLi7d, NP_delta_Li6He3Be7d, NP_delta_Li7He3aad, NP_delta_Li8He3aat, NP_delta_Be7taad, NP_delta_Be7tLi7He3, NP_delta_B8dBe7He3, NP_delta_B8taaHe3, NP_delta_Be7He3ppaa, NP_delta_ddag, NP_delta_He3He3app, NP_delta_Be7pB8g, NP_delta_Li7daan, NP_delta_dntg, NP_delta_ttann, NP_delta_He3nag, NP_delta_He3tad, NP_delta_He3tanp, NP_delta_Li7taan, NP_delta_Li7He3aanp, NP_delta_Li8dLi7t, NP_delta_Be7taanp, NP_delta_Be7He3aapp, NP_delta_Li6nta, NP_delta_He3tLi6g, NP_delta_anpLi6g, NP_delta_Li6nLi7g, NP_delta_Li6dLi7p, NP_delta_Li6dBe7n, NP_delta_Li7nLi8g, NP_delta_Li7dLi8p, NP_delta_Li8paan, NP_delta_annHe6g, NP_delta_ppndp, NP_delta_Li7taann = np.zeros(num_reactions)

#################
# Nuclear rates #
#################
# Credit for dataset to PRIMAT and NACRE II:
# Physics Reports, 04, (2018) 005 [arXiv:1801.08023]
# MNRAS 2021 [arXiv:2011.11320]
# Nuclear Physics A 918 (2013) 61–169
# http://www.astro.ulb.ac.be/nacreii
# Below, 12 or 63 tabulated nuclear rates in log space, T9 = = np.logspace(-3,1,500)
####################################
# 12 fundamental nuclear reactions #
####################################
# alpha_R,beta_R,gamma_R = coefficients for inverse reaction obtained via detailed balance
# np -> dg
alpha_npdg,beta_npdg,gamma_npdg = 4.71614e+09,1.5,-25.815
npdg_T9,npdg_median,npdg_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"npdg.txt",unpack = True)
# dp -> He3g
alpha_dpHe3g,beta_dpHe3g,gamma_dpHe3g = 1.6335e+10,1.5,-63.7491
dpHe3g_T9,dpHe3g_median,dpHe3g_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"dpHe3g.txt",unpack = True)
# dd -> He3n
alpha_ddHe3n,beta_ddHe3n,gamma_ddHe3n = 1.73183e+00,0.,-37.9341
ddHe3n_T9,ddHe3n_median,ddHe3n_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"ddHe3n.txt",unpack = True)
# dd -> tp
alpha_ddtp,beta_ddtp,gamma_ddtp = 1.73492e+00,0.,-46.7971
ddtp_T9,ddtp_median,ddtp_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"ddtp.txt",unpack = True)
# tp -> ag
alpha_tpag,beta_tpag,gamma_tpag = 2.61058e+10,1.5,-229.93
tpag_T9,tpag_median,tpag_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"tpag.txt",unpack = True)
# td -> an
alpha_tdan,beta_tdan,gamma_tdan = 5.5369e+00,0.,-204.1236
tdan_T9,tdan_median,tdan_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"tdan.txt",unpack = True)
# ta -> Li7g
alpha_taLi7g,beta_taLi7g,gamma_taLi7g = 1.1133e+10,1.5,-28.6355
taLi7g_T9,taLi7g_median,taLi7g_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"taLi7g.txt",unpack = True)
# He3n -> tp
alpha_He3ntp,beta_He3ntp,gamma_He3ntp = 1.00178e+00,0.0,-8.8630
He3ntp_T9,He3ntp_median,He3ntp_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"He3ntp.txt",unpack = True)
# He3d -> ap
alpha_He3dap,beta_He3dap,gamma_He3dap = 5.5438e+00,0.0,-212.987
He3dap_T9,He3dap_median,He3dap_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"He3dap.txt",unpack = True)
# He3a -> Be7g
alpha_He3aBe7g,beta_He3aBe7g,gamma_He3aBe7g = 1.11289e+10,1.5,-18.4179
He3aBe7g_T9,He3aBe7g_median,He3aBe7g_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"He3aBe7g.txt",unpack = True)
# Be7n -> Li7p
alpha_Be7nLi7p,beta_Be7nLi7p,gamma_Be7nLi7p = 1.00215,0.,-19.0806
Be7nLi7p_T9,Be7nLi7p_median,Be7nLi7p_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"Be7nLi7p.txt",unpack = True)
# Li7p -> aa
alpha_Li7paa,beta_Li7paa,gamma_Li7paa = 4.6898,0.,-201.295
Li7paa_T9,Li7paa_median,Li7paa_expsigma = np.loadtxt(working_dir+"/PRyMrates/nuclear/"+rates_dir+"Li7paa.txt",unpack = True)
###############################
# Reloading key nuclear rates #
###############################
def ReloadKeyRates():
    global working_dir,rates_dir,rates_dir
    if(nacreii_flag):
        rates_dir = "key_nacreii_rates/"
    else:
        rates_dir = "key_primat_rates/"
    dir_key_rates = working_dir+"/PRyMrates/nuclear/"+rates_dir
    global npdg_T9,npdg_median,npdg_expsigma
    npdg_T9,npdg_median,npdg_expsigma = np.loadtxt(dir_key_rates+"npdg.txt",unpack = True)
    global dpHe3g_T9,dpHe3g_median,dpHe3g_expsigma
    dpHe3g_T9,dpHe3g_median,dpHe3g_expsigma = np.loadtxt(dir_key_rates+"dpHe3g.txt",unpack = True)
    global ddHe3n_T9,ddHe3n_median,ddHe3n_expsigma
    ddHe3n_T9,ddHe3n_median,ddHe3n_expsigma = np.loadtxt(dir_key_rates+"ddHe3n.txt",unpack = True)
    global ddtp_T9,ddtp_median,ddtp_expsigma
    ddtp_T9,ddtp_median,ddtp_expsigma = np.loadtxt(dir_key_rates+"ddtp.txt",unpack = True)
    global tpag_T9,tpag_median,tpag_expsigma
    tpag_T9,tpag_median,tpag_expsigma = np.loadtxt(dir_key_rates+"tpag.txt",unpack = True)
    global tdan_T9,tdan_median,tdan_expsigma
    tdan_T9,tdan_median,tdan_expsigma = np.loadtxt(dir_key_rates+"tdan.txt",unpack = True)
    global taLi7g_T9,taLi7g_median,taLi7g_expsigma
    taLi7g_T9,taLi7g_median,taLi7g_expsigma = np.loadtxt(dir_key_rates+"taLi7g.txt",unpack = True)
    global He3ntp_T9,He3ntp_median,He3ntp_expsigma
    He3ntp_T9,He3ntp_median,He3ntp_expsigma = np.loadtxt(dir_key_rates+"He3ntp.txt",unpack = True)
    global He3dap_T9,He3dap_median,He3dap_expsigma
    He3dap_T9,He3dap_median,He3dap_expsigma = np.loadtxt(dir_key_rates+"He3dap.txt",unpack = True)
    global He3aBe7g_T9,He3aBe7g_median,He3aBe7g_expsigma
    He3aBe7g_T9,He3aBe7g_median,He3aBe7g_expsigma = np.loadtxt(dir_key_rates+"He3aBe7g.txt",unpack = True)
    global Be7nLi7p_T9,Be7nLi7p_median,Be7nLi7p_expsigma
    Be7nLi7p_T9,Be7nLi7p_median,Be7nLi7p_expsigma = np.loadtxt(dir_key_rates+"Be7nLi7p.txt",unpack = True)
    global Li7paa_T9,Li7paa_median,Li7paa_expsigma
    Li7paa_T9,Li7paa_median,Li7paa_expsigma = np.loadtxt(dir_key_rates+"Li7paa.txt",unpack = True)
#####################################################
# Extra nuclear reactions implemented (63 in total) #
#####################################################
# Directory for additional nuclear rates
dir_other_rates = working_dir+"/PRyMrates/nuclear/other_nucl_rates/"
# Li7p -> aag
alpha_Li7paag,beta_Li7paag,gamma_Li7paag = 4.6898,0.,-201.295
Li7paag_T9,Li7paag_median,Li7paag_expsigma = np.loadtxt(dir_other_rates+"Li7paag.txt",unpack = True)
# Be7n -> aa
alpha_Be7naa,beta_Be7naa,gamma_Be7naa = 4.6982,0.,-220.3871
Be7naa_T9,Be7naa_median,Be7naa_expsigma = np.loadtxt(dir_other_rates+"Be7naa.txt",unpack = True)
# Be7d -> aap
alpha_Be7daap,beta_Be7daap,gamma_Be7daap = 9.9579*1.e-10,-1.5,-194.5722
Be7daap_T9,Be7daap_median,Be7daap_expsigma = np.loadtxt(dir_other_rates+"Be7daap.txt",unpack = True)
# da -> Li6g
alpha_daLi6g,beta_daLi6g,gamma_daLi6g = 1.53053*1.e+10,1.5,-17.1023
daLi6g_T9,daLi6g_median,daLi6g_expsigma = np.loadtxt(dir_other_rates+"daLi6g.txt",unpack = True)
# Li6p -> Be7g
alpha_Li6pBe7g,beta_Li6pBe7g,gamma_Li6pBe7g = 1.18778*1.e+10,1.5,-65.0648
Li6pBe7g_T9,Li6pBe7g_median,Li6pBe7g_expsigma = np.loadtxt(dir_other_rates+"Li6pBe7g.txt",unpack = True)
# Li6p -> He3a
alpha_Li6pHe3a,beta_Li6pHe3a,gamma_Li6pHe3a = 1.06729,0.,-46.6469
Li6pHe3a_T9,Li6pHe3a_median,Li6pHe3a_expsigma = np.loadtxt(dir_other_rates+"Li6pHe3a.txt",unpack = True)
# B8n -> aap
alpha_B8naap,beta_B8naap,gamma_B8naap = 3.6007*10**-10,-1.5,-218.7915
B8naap_T9,B8naap_median,B8naap_expsigma = np.loadtxt(dir_other_rates+"B8naap.txt",unpack = True)
# Li6He3 -> aap
alpha_Li6He3aap,beta_Li6He3aap,gamma_Li6He3aap = 7.2413*10**-10,-1.5,-195.8748
Li6He3aap_T9,Li6He3aap_median,Li6He3aap_expsigma = np.loadtxt(dir_other_rates+"Li6He3aap.txt",unpack = True)
# Li6t -> aan
alpha_Li6taan,beta_Li6taan,gamma_Li6taan = 7.2333*10**-10,-1.5,-187.0131
Li6taan_T9,Li6taan_median,Li6taan_expsigma = np.loadtxt(dir_other_rates+"Li6taan.txt",unpack = True)
# Li6t -> Li8p
alpha_Li6tLi8p,beta_Li6tLi8p,gamma_Li6tLi8p = 2.0167,0.,-9.306
Li6tLi8p_T9,Li6tLi8p_median,Li6tLi8p_expsigma = np.loadtxt(dir_other_rates+"Li6tLi8p.txt",unpack = True)
# Li7He3 -> Li6a
alpha_Li7He3Li6a,beta_Li7He3Li6a,gamma_Li7He3Li6a = 2.1972,0.,-154.6607
Li7He3Li6a_T9,Li7He3Li6a_median,Li7He3Li6a_expsigma = np.loadtxt(dir_other_rates+"Li7He3Li6a.txt",unpack = True)
# Li8He3 -> Li7a
alpha_Li8He3Li7a,beta_Li8He3Li7a,gamma_Li8He3Li7a = 1.9994,0.,-215.2055
Li8He3Li7a_T9,Li8He3Li7a_median,Li8He3Li7a_expsigma = np.loadtxt(dir_other_rates+"Li8He3Li7a.txt",unpack = True)
# Be7t -> Li6a
alpha_Be7tLi6a,beta_Be7tLi6a,gamma_Be7tLi6a = 2.1977,0.,-164.8783
Be7tLi6a_T9,Be7tLi6a_median,Be7tLi6a_expsigma = np.loadtxt(dir_other_rates+"Be7tLi6a.txt",unpack = True)
# B8t -> Be7a
alpha_B8tBe7a,beta_B8tBe7a,gamma_B8tBe7a = 1.9999,0.,-228.3344
B8tBe7a_T9,B8tBe7a_median,B8tBe7a_expsigma = np.loadtxt(dir_other_rates+"B8tBe7a.txt",unpack = True)
# B8n -> Li6He3
alpha_B8nLi6He3,beta_B8nLi6He3,gamma_B8nLi6He3 = 0.49669,0.,-22.9167
B8nLi6He3_T9,B8nLi6He3_median,B8nLi6He3_expsigma = np.loadtxt(dir_other_rates+"B8nLi6He3.txt",unpack = True)
# B8n -> Be7d
alpha_B8nBe7d,beta_B8nBe7d,gamma_B8nBe7d = 0.36119,0.,-24.2194
B8nBe7d_T9,B8nBe7d_median,B8nBe7d_expsigma = np.loadtxt(dir_other_rates+"B8nBe7d.txt",unpack = True)
# Li6t -> Li7d
alpha_Li6tLi7d,beta_Li6tLi7d,gamma_Li6tLi7d = 0.72734,0.,-11.5332
Li6tLi7d_T9,Li6tLi7d_median,Li6tLi7d_expsigma = np.loadtxt(dir_other_rates+"Li6tLi7d.txt",unpack = True)
# Li6He3 -> Be7d
alpha_Li6He3Be7d,beta_Li6He3Be7d,gamma_Li6He3Be7d = 0.72719,0.,-1.3157
Li6He3Be7d_T9,Li6He3Be7d_median,Li6He3Be7d_expsigma = np.loadtxt(dir_other_rates+"Li6He3Be7d.txt",unpack = True)
# Li7He3 -> aad
alpha_Li7He3aad,beta_Li7He3aad,gamma_Li7He3aad = 2.8700*10**-10,-1.5,-137.5575
Li7He3aad_T9,Li7He3aad_median,Li7He3aad_expsigma = np.loadtxt(dir_other_rates+"Li7He3aad.txt",unpack = True)
# Li8He3 -> aat
alpha_Li8He3aat,beta_Li8He3aat,gamma_Li8He3aat = 3.5907*10**-10,-1.5,-186.5821
Li8He3aat_T9,Li8He3aat_median,Li8He3aat_expsigma = np.loadtxt(dir_other_rates+"Li8He3aat.txt",unpack = True)
# Be7t -> aad
alpha_Be7taad,beta_Be7taad,gamma_Be7taad = 2.8706*10**-10,-1.5,-147.7751
Be7taad_T9,Be7taad_median,Be7taad_expsigma = np.loadtxt(dir_other_rates+"Be7taad.txt",unpack = True)
# Be7t -> aad
alpha_Be7tLi7He3,beta_Be7tLi7He3,gamma_Be7tLi7He3 = 1.0002,0.,-10.2176
Be7tLi7He3_T9,Be7tLi7He3_median,Be7tLi7He3_expsigma = np.loadtxt(dir_other_rates+"Be7tLi7He3.txt",unpack = True)
# B8d -> Be7He3
alpha_B8dBe7He3,beta_B8dBe7He3,gamma_B8dBe7He3 = 1.2514,0,-62.1535
B8dBe7He3_T9,B8dBe7He3_median,B8dBe7He3_expsigma = np.loadtxt(dir_other_rates+"B8dBe7He3.txt",unpack = True)
# B8t -> aaHe3
alpha_B8taaHe3,beta_B8taaHe3,gamma_B8taaHe3 = 3.5922*10**-10,-1.5,-209.9285
B8taaHe3_T9,B8taaHe3_median,B8taaHe3_expsigma = np.loadtxt(dir_other_rates+"B8taaHe3.txt",unpack = True)
# Be7He3p -> paa
alpha_Be7He3ppaa,beta_Be7He3ppaa,gamma_Be7He3ppaa = 1.2201*10**-19,-3.,-130.8113
Be7He3ppaa_T9,Be7He3ppaa_median,Be7He3ppaa_expsigma = np.loadtxt(dir_other_rates+"Be7He3ppaa.txt",unpack = True)
# dd -> ag
alpha_ddag,beta_ddag,gamma_ddag = 4.5310*10**10,1.5,-276.7271
ddag_T9,ddag_median,ddag_expsigma = np.loadtxt(dir_other_rates+"ddag.txt",unpack = True)
# He3He3 -> app
alpha_He3He3app,beta_He3He3app,gamma_He3He3app = 3.3915*10**-10,-1.5,-149.2290
He3He3app_T9,He3He3app_median,He3He3app_expsigma = np.loadtxt(dir_other_rates+"He3He3app.txt",unpack = True)
# Be7p -> B8g
alpha_Be7pB8g,beta_Be7pB8g,gamma_Be7pB8g = 1.3063*10**10,1.5,-1.5825
Be7pB8g_T9,Be7pB8g_median,Be7pB8g_expsigma = np.loadtxt(dir_other_rates+"Be7pB8g.txt",unpack = True)
# Li7d -> aan
alpha_Li7daan,beta_Li7daan,gamma_Li7daan = 9.9435*10**-10,-1.5,-175.4916
Li7daan_T9,Li7daan_median,Li7daan_expsigma = np.loadtxt(dir_other_rates+"Li7daan.txt",unpack = True)
# dn -> tg
alpha_dntg,beta_dntg,gamma_dntg = 1.6364262*10**10,1.5,-72.612132
dntg_T9,dntg_median,dntg_expsigma = np.loadtxt(dir_other_rates+"dntg.txt",unpack = True)
# tt -> ann
alpha_ttann,beta_ttann,gamma_ttann = 3.3826187*10**-10,-1.5,-131.50322
ttann_T9,ttann_median,ttann_expsigma = np.loadtxt(dir_other_rates+"ttann.txt",unpack = True)
# He3n -> ag
alpha_He3nag,beta_He3nag,gamma_He3nag = 2.6152351*10**10,1.5,-238.79338
He3nag_T9,He3nag_median,He3nag_expsigma = np.loadtxt(dir_other_rates+"He3nag.txt",unpack = True)
# He3t -> ad
alpha_He3tad,beta_He3tad,gamma_He3tad = 1.5981381,0.,-166.18124
He3tad_T9,He3tad_median,He3tad_expsigma = np.loadtxt(dir_other_rates+"He3tad.txt",unpack = True)
# He3t -> anp
alpha_He3tanp,beta_He3tanp,gamma_He3tanp = 3.3886566*10**-10,-1.5,-140.36623
He3tanp_T9,He3tanp_median,He3tanp_expsigma = np.loadtxt(dir_other_rates+"He3tanp.txt",unpack = True)
# Li7t -> aan
alpha_Li7taan,beta_Li7taan,gamma_Li7taan = 1.2153497*10**-19,-3.,-102.86767
Li7taan_T9,Li7taan_median,Li7taan_expsigma = np.loadtxt(dir_other_rates+"Li7taan.txt",unpack = True)
# Li7He3 -> aanp
alpha_Li7He3aanp,beta_Li7He3aanp,gamma_Li7He3aanp = 6.0875952*10**-20,-3.,-111.73068
Li7He3aanp_T9,Li7He3aanp_median,Li7He3aanp_expsigma = np.loadtxt(dir_other_rates+"Li7He3aanp.txt",unpack = True)
# Li8d -> Li7t
alpha_Li8dLi7t,beta_Li8dLi7t,gamma_Li8dLi7t = 1.2509926,0.,-49.02453
Li8dLi7t_T9,Li8dLi7t_median,Li8dLi7t_expsigma = np.loadtxt(dir_other_rates+"Li8dLi7t.txt",unpack = True)
# Be7t -> aanp
alpha_Be7taanp,beta_Be7taanp,gamma_Be7taanp = 6.0898077*10**-20,-3.,-121.9483
Be7taanp_T9,Be7taanp_median,Be7taanp_expsigma = np.loadtxt(dir_other_rates+"Be7taanp.txt",unpack = True)
# Be7He3 -> aapp
alpha_Be7He3aapp,beta_Be7He3aapp,gamma_Be7He3aapp = 1.2201356*10**-19,-3.,-130.81131
Be7He3aapp_T9,Be7He3aapp_median,Be7He3aapp_expsigma = np.loadtxt(dir_other_rates+"Be7He3aapp.txt",unpack = True)
# Li6n -> ta
alpha_Li6nta,beta_Li6nta,gamma_Li6nta = 1.0691921,0.,-55.509875
Li6nta_T9,Li6nta_median,Li6nta_expsigma = np.loadtxt(dir_other_rates+"Li6nta.txt",unpack = True)
# He3t -> Li6g
alpha_He3tLi6g,beta_He3tLi6g,gamma_He3tLi6g = 2.4459918*10**10,1.5,-183.2835
He3tLi6g_T9,He3tLi6g_median,He3tLi6g_expsigma = np.loadtxt(dir_other_rates+"He3tLi6g.txt",unpack = True)
# an -> pLi6g
alpha_anpLi6g,beta_anpLi6g,gamma_anpLi6g = 7.2181753*10**19,3.,-42.917276
anpLi6g_T9,anpLi6g_median,anpLi6g_expsigma = np.loadtxt(dir_other_rates+"anpLi6g.txt",unpack = True)
# Li6n -> Li7g
alpha_Li6nLi7g,beta_Li6nLi7g,gamma_Li6nLi7g = 1.1903305*10**10,1.5,-84.145424
Li6nLi7g_T9,Li6nLi7g_median,Li6nLi7g_expsigma = np.loadtxt(dir_other_rates+"Li6nLi7g.txt",unpack = True)
# Li6d -> Li7p
alpha_Li6dLi7p,beta_Li6dLi7p,gamma_Li6dLi7p = 2.5239503,0.,-58.330405
Li6dLi7p_T9,Li6dLi7p_median,Li6dLi7p_expsigma = np.loadtxt(dir_other_rates+"Li6dLi7p.txt",unpack = True)
# Li6d -> Be7n
alpha_Li6dBe7n,beta_Li6dBe7n,gamma_Li6dBe7n = 2.5185377,0.,-39.249773
Li6dBe7n_T9,Li6dBe7n_median,Li6dBe7n_expsigma = np.loadtxt(dir_other_rates+"Li6dBe7n.txt",unpack = True)
# Li7n -> Li8g
alpha_Li7nLi8g,beta_Li7nLi8g,gamma_Li7nLi8g = 1.3081022*10**10,1.5,-23.587602
Li7nLi8g_T9,Li7nLi8g_median,Li7nLi8g_expsigma = np.loadtxt(dir_other_rates+"Li7nLi8g.txt",unpack = True)
# Li7d -> Li8p
alpha_Li7dLi8p,beta_Li7dLi8p,gamma_Li7dLi8p = 2.7736709,0.,2.2274166
Li7dLi8p_T9,Li7dLi8p_median,Li7dLi8p_expsigma = np.loadtxt(dir_other_rates+"Li7dLi8p.txt",unpack = True)
# Li8p -> aan
alpha_Li8paan,beta_Li8paan,gamma_Li8paan = 3.5851946*10**-10,-1.5,-177.70722
Li8paan_T9,Li8paan_median,Li8paan_expsigma = np.loadtxt(dir_other_rates+"Li8paan.txt",unpack = True)
# an -> nHe6g
alpha_annHe6g,beta_annHe6g,gamma_annHe6g = 1.0837999*10**20,3.,-11.319626
annHe6g_T9,annHe6g_median,annHe6g_expsigma = np.loadtxt(dir_other_rates+"annHe6g.txt",unpack = True)
# pp -> ndp
alpha_ppndp,beta_ppndp,gamma_ppndp = 2.3580703*10**9,1.5,-25.815019
ppndp_T9,ppndp_median,ppndp_expsigma = np.loadtxt(dir_other_rates+"ppndp.txt",unpack = True)
# Li7t -> aann
alpha_Li7taann,beta_Li7taann,gamma_Li7taann = 1.2153497*10**-19,-3.,-102.86767
Li7taann_T9,Li7taann_median,Li7taann_expsigma = np.loadtxt(dir_other_rates+"Li7taann.txt",unpack = True)

#########################
# Configuration validator #
#########################
# Called once from PRyMclass.__init__ to catch mis-configurations early.
# See doc/FLAG_AUDIT.md for the rationale; the guards here target the
# failure modes enumerated there.

_QKE_DAMPING_FORMULAS = ("symmetric", "mirizzi", "gariazzo")
_NU_OSCILLATION_METHODS = ("collision_mixing", "relaxation")


def validate_configuration():
    """Validate PRyMini flag combinations and warn on silent-null configs.

    Raises ValueError on string-typed flags with disallowed values (typos
    caught at construction rather than deep in the call stack). Emits
    warnings via `print` on combinations that parse legally but produce
    silently nonsensical or mis-scaled results. PRyMclass.__init__ also
    performs its own auto-enable cascade for the qke / boltzmann /
    general_nu chain; this function is complementary, covering the checks
    that auto-enable can't express.

    Also mutates `n_B_override` when the user leaves it at None and the
    Phase B temperature window is wider than the default (Stage E.2
    sprint 2). This is the one path in this function that modifies
    module state; the alternative (silently running Phase B with a
    too-coarse step grid at T >> 5 MeV) produced the internally
    inconsistent results documented in doc/STAGE_E2_SPRINT2_BRIEF.md.
    """
    import warnings as _warnings
    global n_B_override  # auto-scaled below when window extends beyond default
    global n_B_phase0_override  # sprint-10 auto-scale for the Phase-0 segment

    # 1. String-typed flags: reject typos immediately.
    if qke_damping_formula not in _QKE_DAMPING_FORMULAS:
        raise ValueError(
            f"PRyMini.qke_damping_formula = {qke_damping_formula!r} is not "
            f"a recognised value. Allowed: {_QKE_DAMPING_FORMULAS}."
        )
    if nu_oscillation_method not in _NU_OSCILLATION_METHODS:
        raise ValueError(
            f"PRyMini.nu_oscillation_method = {nu_oscillation_method!r} is not "
            f"a recognised value. Allowed: {_NU_OSCILLATION_METHODS}."
        )

    # 2. QKE driver chain: warn on upstream-disabled configs. Auto-enable
    #    already covers qke_density_matrix_flag -> boltzmann_nu_flag, but
    #    the Stage D ETD1 / ETDRK2 flags are not in that cascade.
    if qke_ode_etdrk2_flag and not qke_full_ode_flag:
        _warnings.warn(
            "PRyMini.qke_ode_etdrk2_flag=True requires qke_full_ode_flag=True; "
            "the ETDRK2 path will be inactive. Set both or neither.",
            stacklevel=2,
        )
    if qke_lsoda_driver_flag and not qke_full_ode_flag:
        _warnings.warn(
            "PRyMini.qke_lsoda_driver_flag=True requires qke_full_ode_flag=True; "
            "the LSODA reference driver will be inactive. Set both or neither.",
            stacklevel=2,
        )
    if qke_etdrk4_flag and not qke_full_ode_flag:
        _warnings.warn(
            "PRyMini.qke_etdrk4_flag=True requires qke_full_ode_flag=True; "
            "the ETDRK4 driver will be inactive. Set both or neither.",
            stacklevel=2,
        )
    if qke_full_ode_flag and not qke_density_matrix_flag:
        _warnings.warn(
            "PRyMini.qke_full_ode_flag=True requires qke_density_matrix_flag=True; "
            "the Stage D ODE driver will be inactive. Set both or neither.",
            stacklevel=2,
        )
    # Stage E.2 sprint 10: Phase-0 QKE driver preconditions.
    if qke_phase0_flag and not qke_full_ode_flag:
        _warnings.warn(
            "PRyMini.qke_phase0_flag=True requires qke_full_ode_flag=True; the "
            "Phase-0 segment reuses the Stage D ODE driver. Enable both or neither.",
            stacklevel=2,
        )
    if qke_phase0_flag and T_phase0_start <= T_boltz_start:
        _warnings.warn(
            f"PRyMini.qke_phase0_flag=True but T_phase0_start = {T_phase0_start} "
            f"MeV is not above T_boltz_start = {T_boltz_start} MeV; the Phase-0 "
            "window is empty and the segment will no-op.",
            stacklevel=2,
        )
    if qke_phase0_flag and T_start / MeV_to_Kelvin < T_phase0_start:
        _warnings.warn(
            f"PRyMini.qke_phase0_flag=True but T_start = {T_start/MeV_to_Kelvin:.3f} "
            f"MeV is below T_phase0_start = {T_phase0_start} MeV; Phase A cannot "
            "reach the Phase-0 entry point. Bump T_start above T_phase0_start.",
            stacklevel=2,
        )

    # 3. Silently-ignored knobs: warn when user-set but the gating flag is off.
    if not sterile_flag:
        for name, val in (("xi_nue_init", xi_nue_init),
                          ("xi_numu_init", xi_numu_init),
                          ("xi_nutau_init", xi_nutau_init)):
            if val != 0.0:
                _warnings.warn(
                    f"PRyMini.{name} = {val} is non-zero but sterile_flag=False; "
                    "the value is ignored. Set sterile_flag=True to seed an "
                    "active-sector lepton asymmetry.",
                    stacklevel=2,
                )
    if not nlo_weak_flag and nlo_weak_rate_scale != 1.003:
        _warnings.warn(
            f"PRyMini.nlo_weak_rate_scale = {nlo_weak_rate_scale} but "
            "nlo_weak_flag=False; the value is ignored. Set nlo_weak_flag=True "
            "to activate the scaled nu-e rates.",
            stacklevel=2,
        )

    # 4. Sterile-on but no active-sterile mixing: ρ_ss will remain at zero,
    #    so Neff is bit-identical to sterile_flag=False.
    if sterile_flag and (theta_14 == 0.0 and theta_24 == 0.0 and theta_34 == 0.0
                         and xi_nue_init == 0.0 and xi_numu_init == 0.0
                         and xi_nutau_init == 0.0):
        _warnings.warn(
            "PRyMini.sterile_flag=True but all active-sterile mixing angles "
            "(theta_14, theta_24, theta_34) and all xi_* are zero; the sterile "
            "sector is fully decoupled and rho_ss will stay at zero. "
            "Set at least one non-zero theta_α4 to enable Dodelson-Widrow, or "
            "set xi_nue_init to drive Shi-Fuller.",
            stacklevel=2,
        )

    # 5. QKE Phase B step-count auto-scale. The default n_B = max(2000, 2*
    #    n_sampling) at PRyM_main.py:317 is calibrated for T_boltz_start =
    #    5 MeV, T_boltz_end = 0.005 MeV (3.0 decades in T). Extending the
    #    window to higher T_boltz_start requires more steps to resolve the
    #    collision damping that dominates at high T; without scaling, Stage
    #    E.2 sprint 1 observed dNeff=0.05 with sum_rho_ss=26.17 at
    #    T_boltz_start=30 MeV (internally inconsistent) and Neff=13.66 at
    #    60 MeV (unphysical). The exponent-4 heuristic below was pinned
    #    from a sprint-2 n_B-convergence scan
    #    (validation/diagnostics/diag_nB_convergence.py) where sum_rho_ss
    #    stabilised around n_B ~ 5000 at T_boltz_start = 30 MeV. At the
    #    default window the scale factor is exactly 1.0, so regression-test
    #    results are bit-identical.
    if qke_full_ode_flag and n_B_override is None \
            and T_boltz_start > 0.0 and T_boltz_end > 0.0 \
            and T_boltz_start > T_boltz_end:
        decades = np.log10(T_boltz_start / T_boltz_end)
        decades_default = 3.0   # log10(5.0 / 0.005)
        scale = max(1.0, decades / decades_default)
        if scale > 1.0:
            n_B_override = int(2000 * scale**4)
            print(
                f"PRyMini.validate_configuration: T window "
                f"[{T_boltz_start}, {T_boltz_end}] MeV spans {decades:.2f} "
                f"decades (default {decades_default:.1f}); auto-scaling "
                f"n_B_override = {n_B_override} to preserve Phase B "
                "resolution. Set n_B_override explicitly to opt out."
            )

    # 5b. Stage E.2 sprint 10: Phase-0 QKE step-count auto-scale. Phase 0 is
    #     MSW-resonance-dominated, not collision-damping-dominated like Phase B,
    #     so the sprint-2 exponent-4 formula is not portable (at 0.52 decades
    #     the exponent-4 scaling collapses to ~0 steps). Anchor at 2500 steps
    #     for the reference 100->30 MeV window (decades_p0_default = 0.52);
    #     exponent-2 margin covers longer Phase-0 spans without overshooting.
    #     Clamped to [500, 20000]. At the reference window the scale is
    #     exactly 1.0, so the override picks the anchor 2500 verbatim.
    if qke_phase0_flag and n_B_phase0_override is None \
            and T_phase0_start > 0.0 and T_boltz_start > 0.0 \
            and T_phase0_start > T_boltz_start:
        decades_p0 = np.log10(T_phase0_start / T_boltz_start)
        decades_p0_default = 0.52   # log10(100.0 / 30.0) ~= 0.5229
        n_B_phase0_base = 2500
        scale_p0 = max(1.0, decades_p0 / decades_p0_default)
        n_B_phase0_override = int(max(500, min(20000,
                                                n_B_phase0_base * scale_p0**2)))
        print(
            f"PRyMini.validate_configuration: Phase-0 window "
            f"[{T_phase0_start}, {T_boltz_start}] MeV spans "
            f"{decades_p0:.2f} decades (anchor {decades_p0_default:.2f}); "
            f"auto-scaling n_B_phase0_override = {n_B_phase0_override} to "
            "resolve MSW crossings. Set n_B_phase0_override explicitly to opt out."
        )

    # 6. QED correction tables top out at T = 40 MeV. Above that top the
    #    baseline interpolators zero-clamp rather than linear-extrapolate
    #    (PRyM_thermo.py, _T_QED_TABLE_TOP). That is numerically safe but
    #    drops the O(alpha/pi) ~ 0.2% QED correction to the plasma
    #    pressure at T > 40 MeV. Warn so users running extended windows
    #    know to regenerate the tables (PRyMrates/thermo/QED_P_int.txt and
    #    siblings, NUDEC_BSM v2) if that precision matters.
    if qke_full_ode_flag and T_boltz_start > 40.0:
        _warnings.warn(
            f"PRyMini.T_boltz_start = {T_boltz_start} MeV exceeds the top "
            "of the baseline QED correction tables (40 MeV). Above 40 MeV "
            "the O(e^2)+O(e^3) plasma corrections zero-clamp in PRyM_thermo; "
            "fractional effect on Neff is O(alpha/pi) ~ 0.2%. Regenerate "
            "PRyMrates/thermo/QED_P_int.txt and siblings via NUDEC_BSM v2 "
            "to extend coverage if needed.",
            stacklevel=2,
        )

    # 7. Verbose-mode physics-config summary. Catches Category-3 silent-nulls
    #    by making the effective-physics config explicit at run start.
    if verbose_flag:
        _print_physics_config_summary()


def _print_physics_config_summary():
    """Print the active physics configuration at PRyMclass construction."""
    lines = ["PRyMini effective configuration:"]

    # Evolution driver
    if qke_density_matrix_flag:
        if qke_etdrk4_flag and qke_full_ode_flag:
            driver = "QKE density-matrix, Stage E.2 sprint 16 Krogstad ETDRK4 driver"
        elif qke_lsoda_driver_flag and qke_full_ode_flag:
            driver = "QKE density-matrix, Stage E.2 sprint 14 LSODA reference driver"
        elif qke_ode_etdrk2_flag and qke_full_ode_flag:
            driver = "QKE density-matrix, Stage D.7 ETDRK2 driver"
        elif qke_full_ode_flag:
            driver = "QKE density-matrix, Stage D ODE driver"
        else:
            driver = "QKE density-matrix, Strang/Sigl-Raffelt driver"
    elif boltzmann_nu_flag:
        driver = "diagonal Boltzmann (general_nu)"
    else:
        driver = "thermal (analytic, no general_nu)"
    lines.append(f"  driver: {driver}")

    # PMNS status
    pmns_off = (theta_12 == 0.0 and theta_13 == 0.0 and theta_23 == 0.0)
    lines.append(f"  PMNS active-active mixing: {'OFF' if pmns_off else 'ON (PDG defaults or user-set)'}")

    # Sterile status
    if sterile_flag:
        mixing_on = (theta_14 != 0.0 or theta_24 != 0.0 or theta_34 != 0.0)
        lines.append(
            f"  sterile (3+1): ON, Dm2_41={Dm2_41}, mixing {'ON' if mixing_on else 'OFF'}"
        )
    else:
        lines.append("  sterile (3+1): OFF")

    # Damping formula / oscillation method
    lines.append(f"  qke_damping_formula: {qke_damping_formula!r}")
    lines.append(f"  nu_oscillation_method: {nu_oscillation_method!r}")

    # Temperature windows
    T_boltz_start_val = T_boltz_start
    T_boltz_end_val = T_boltz_end
    lines.append(
        f"  T_start = {T_start / MeV_to_Kelvin:.1f} MeV, "
        f"T_boltz_start = {T_boltz_start_val} MeV, "
        f"T_boltz_end = {T_boltz_end_val} MeV"
    )

    for line in lines:
        print(line)
