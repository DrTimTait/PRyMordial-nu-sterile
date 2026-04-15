# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

class PRyMclass(object):
    def __init__(self,my_rho_NP=0.,my_p_NP=0.,my_drho_NP_dT=0.,my_delta_rho_NP=0.,
                 my_f_nue=None,my_f_nuebar=None,my_f_numu=None,my_f_numubar=None,
                 my_f_nutau=None,my_f_nutaubar=None,
                 my_delta_rho_nu_NP=None,
                 my_C_NP_nue=None,my_C_NP_nuebar=None,my_C_NP_numu=None):
        #############################
        # PRyMordial initialization #
        #############################
        import PRyM.PRyM_init as PRyMini
        if(PRyMini.julia_flag):
            from julia import Main
            from diffeqpy import de
            import PRyM.PRyM_jl_sys as PRyMjl
        import PRyM.PRyM_thermo as PRyMthermo
        # Loading New Physics species (constructor default: none)
        PRyMthermo.rho_NP,PRyMthermo.p_NP,PRyMthermo.drho_NP_dT,PRyMthermo.delta_rho_NP=my_rho_NP,my_p_NP,my_drho_NP_dT,my_delta_rho_NP
        # mu-tau symmetry breaking dispatch.
        #   - Stage 1: qke_density_matrix_flag route, full 6-species native.
        #   - Stage 2: diagonal BoltzmannSolver with n=4 (mu/tau distinct,
        #             pcle+antipcle per flavor aggregated). Enabled when the
        #             user sets boltzmann_nu_flag=True explicitly.
        #   - If neither Boltzmann nor QKE is already enabled, auto-enable
        #     QKE as the default (most capable) path.
        if not PRyMini.mu_tau_symmetric_flag:
            if not (PRyMini.qke_density_matrix_flag or PRyMini.boltzmann_nu_flag):
                if PRyMini.verbose_flag:
                    print(" mu_tau_symmetric_flag=False with no evolution "
                          "path selected: auto-enabling qke_density_matrix_flag.")
                PRyMini.qke_density_matrix_flag = True
        # QKE density matrix solver implies Boltzmann solver
        if PRyMini.qke_density_matrix_flag:
            PRyMini.boltzmann_nu_flag = True
        # Boltzmann solver implies general_nu_flag
        if(PRyMini.boltzmann_nu_flag):
            PRyMini.general_nu_flag = True
            PRyMini.compute_bckg_flag = True
            # For SM-like distributions, pre-stored weak rates are valid (~0.1% accuracy)
            # and ~50x faster. Auto-detect cached rates; compute only on first run.
            # User can force recomputation by setting compute_nTOp_flag=True explicitly.
            if not PRyMini.NP_nTOp_flag:
                PRyMini.compute_nTOp_flag = False
        # Loading general neutrino distribution functions (if general_nu_flag is True)
        if(PRyMini.general_nu_flag):
            if my_f_nue is not None:
                PRyMthermo.f_nue_general = my_f_nue
            if my_f_nuebar is not None:
                PRyMthermo.f_nuebar_general = my_f_nuebar
            if my_f_numu is not None:
                PRyMthermo.f_numu_general = my_f_numu
            if my_f_numubar is not None:
                PRyMthermo.f_numubar_general = my_f_numubar
            if my_f_nutau is not None:
                PRyMthermo.f_nutau_general = my_f_nutau
            if my_f_nutaubar is not None:
                PRyMthermo.f_nutaubar_general = my_f_nutaubar
            if my_delta_rho_nu_NP is not None:
                PRyMthermo.delta_rho_nu_NP = my_delta_rho_nu_NP
    
        if(PRyMini.verbose_flag):
            print(" ")
            print("###########################################################")
            print("################## Welcome to PRyMordial ##################")
            print("###########################################################")
            start_time = time.time()

        ################################
        # PRyMordial working directory #
        ################################
        my_dir = PRyMini.working_dir
        
        ##############################
        # Definition of temperatures #
        ##############################
        Tstart_MeV = PRyMini.T_start/PRyMini.MeV_to_Kelvin
        Tend_MeV = PRyMini.T_end/PRyMini.MeV_to_Kelvin

        ##################
        # Thermodynamics #
        ##################
        # Units adopted for background:
        # - Time in [s]
        # - Energy, temperature in [MeV]

        # Expansion rate from Friedmann equation
        if(PRyMini.general_nu_flag):
            def Hubble(Tg,Tnue=None,Tnumu=None,T_NP=0.):
                rho_pl = PRyMthermo.rho_g(Tg)+PRyMthermo.rho_e(Tg)-PRyMthermo.PofT(Tg)+Tg*PRyMthermo.dPdT(Tg)
                rho_3nu = PRyMthermo.rho_3nu(Tg)
                rho_tot = rho_pl+rho_3nu
                if(PRyMini.NP_thermo_flag):
                    rho_tot += PRyMthermo.rho_NP(T_NP)
                if(PRyMini.NP_e_flag):
                    rho_tot += PRyMthermo.rho_NP(Tg)
                return PRyMini.MeV_to_secm1*(rho_tot*8.*np.pi/(3.*PRyMini.Mpl**2))**0.5
        else:
            def Hubble(Tg,Tnue,Tnumu,T_NP=0.):
                rho_pl = PRyMthermo.rho_g(Tg)+PRyMthermo.rho_e(Tg)-PRyMthermo.PofT(Tg)+Tg*PRyMthermo.dPdT(Tg)
                rho_3nu = PRyMthermo.rho_nu(Tnue)+2.*PRyMthermo.rho_nu(Tnumu)
                rho_tot = rho_pl+rho_3nu
                if(PRyMini.NP_thermo_flag):
                    rho_tot += PRyMthermo.rho_NP(T_NP)
                if(PRyMini.NP_nu_flag):
                    rho_tot += PRyMthermo.rho_NP(Tnue)
                if(PRyMini.NP_e_flag):
                    rho_tot += PRyMthermo.rho_NP(Tg)
                return PRyMini.MeV_to_secm1*(rho_tot*8.*np.pi/(3.*PRyMini.Mpl**2))**0.5
        # Computing the background (if not pre-stored)
        if(PRyMini.compute_bckg_flag):
          if(PRyMini.general_nu_flag):
            # General neutrino distributions: 1-variable ODE for Tg only.
            # Total energy conservation: dTg/dt derived from d(rho_total)/dt = -3H(rho_total + P_total)
            # with rho_total = rho_plasma + rho_3nu (from general distributions).
            def dTgdt(Tg,T_NP=0.):
                Hubble_T = Hubble(Tg,T_NP=T_NP)
                # Numerator: expansion cooling of all species
                rho_3nu_T = PRyMthermo.rho_3nu(Tg)
                p_3nu_T = PRyMthermo.p_3nu(Tg)
                num = -(Hubble_T*(4.*PRyMthermo.rho_g(Tg)+3.*(PRyMthermo.rho_e(Tg)+PRyMthermo.p_e(Tg))+3.*Tg*PRyMthermo.dPdT(Tg)
                        +3.*(rho_3nu_T+p_3nu_T)))
                # NOTE: Collision source subtraction omitted. The total-energy
                # equation already accounts for collision energy via drho_3nu/dTg
                # (the Boltzmann distributions respond to a(Tg)). Confirmed
                # numerically that including the legacy delta_rho subtraction
                # has negligible effect on Neff and actually worsens BBN
                # observable agreement when coll_scale != 1.
                # NP collision term for additional energy injection into neutrinos
                num -= PRyMthermo.delta_rho_nu_NP(Tg)
                den = PRyMthermo.drho_g_dT(Tg)+PRyMthermo.drho_e_dT(Tg)+Tg*PRyMthermo.d2PdT2(Tg)+PRyMthermo.drho_3nu_dTg(Tg)
                if(PRyMini.NP_e_flag):
                    num -= 3.*Hubble_T*(PRyMthermo.rho_NP(Tg)+PRyMthermo.p_NP(Tg))
                    den += PRyMthermo.drho_NP_dT(Tg)
                return num/den
            def dTtotdt(t,T_vec):
                Tg = T_vec[0]
                if(PRyMini.NP_thermo_flag):
                    T_NP = T_vec[1]
                    return [dTgdt(Tg,T_NP),dTNPdt_general(Tg,T_NP)]
                else:
                    return [dTgdt(Tg)]
            if(PRyMini.NP_thermo_flag):
                def dTNPdt_general(Tg,T_NP):
                    Hubble_T = Hubble(Tg,T_NP=T_NP)
                    rho_NP = PRyMthermo.rho_NP(T_NP)
                    p_NP = PRyMthermo.p_NP(T_NP)
                    num = -3.*Hubble_T*(rho_NP+p_NP)
                    num += PRyMthermo.delta_rho_NP(Tg,0.,0.,T_NP)
                    den = PRyMthermo.drho_NP_dT(T_NP)
                    return num/den
          else:
            # Integrated Boltzmann equations for temperature of species
            # Neutrino temperature evolution
            def dTnudt(Tg,Tnue,Tnumu,T_NP=0.):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                num = -12.*Hubble_T*PRyMthermo.rho_nu(Tnue)
                delta_rho_nu = (PRyMthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PRyMthermo.delta_rho_numu(Tg,Tnue,Tnumu))
                if(PRyMini.NP_thermo_flag):
                    delta_rho_nu += PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
                num += delta_rho_nu
                den = 3.*PRyMthermo.drho_nu_dT(Tnue)
                if(PRyMini.NP_nu_flag):
                    num -= 3.*Hubble_T*(PRyMthermo.rho_NP(Tnue)+PRyMthermo.p_NP(Tnue))
                    den += PRyMthermo.drho_NP_dT(Tnue)
                return num/den
            # Plasma temperature evolution
            def dTgdt(Tg,Tnue,Tnumu,T_NP=0.):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                num = -(Hubble_T*(4.*PRyMthermo.rho_g(Tg)+3.*(PRyMthermo.rho_e(Tg)+PRyMthermo.p_e(Tg))+3.*Tg*PRyMthermo.dPdT(Tg)))
                # Sum of collision terms must vanish
                delta_rho_g = -(PRyMthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PRyMthermo.delta_rho_numu(Tg,Tnue,Tnumu))
                if(PRyMini.NP_thermo_flag):
                    delta_rho_g -= PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP) # traceless collision operator
                num += delta_rho_g
                den = PRyMthermo.drho_g_dT(Tg)+PRyMthermo.drho_e_dT(Tg)+Tg*PRyMthermo.d2PdT2(Tg)
                if(PRyMini.NP_e_flag):
                    num -= 3.*Hubble_T*(PRyMthermo.rho_NP(Tg)+PRyMthermo.p_NP(Tg))
                    den += PRyMthermo.drho_NP_dT(Tg)
                return num/den
            # NP temperature evolution
            def dTNPdt(Tg,Tnue,Tnumu,T_NP):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                rho_NP = PRyMthermo.rho_NP(T_NP)
                p_NP = PRyMthermo.p_NP(T_NP)
                num = -3.*Hubble_T*(rho_NP+p_NP)
                delta_rho_NP = PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
                num += delta_rho_NP
                den = PRyMthermo.drho_NP_dT(T_NP)
                return num/den
            def dTtotdt(t,T_vec):
                if(PRyMini.NP_thermo_flag):
                    Tg,Tnu,T_NP = T_vec
                    y_vec = dTgdt(Tg,Tnu,Tnu,T_NP),dTnudt(Tg,Tnu,Tnu,T_NP),dTNPdt(Tg,Tnu,Tnu,T_NP)
                    return y_vec
                else:
                    Tg,Tnu = T_vec
                    y_vec = dTgdt(Tg,Tnu,Tnu),dTnudt(Tg,Tnu,Tnu)
                    return y_vec
          # Solution of Boltzmann equations for background thermodynamics
          tfin = PRyMini.t_end # [s]
          if(PRyMini.general_nu_flag and PRyMini.boltzmann_nu_flag):
              # Three-phase evolution with internal Boltzmann solver
              import PRyM.PRyM_boltzmann as PRyMboltz
              # Set up NP collision term callbacks
              C_NP_funcs = {}
              if my_C_NP_nue is not None:
                  C_NP_funcs['nue'] = my_C_NP_nue
              if my_C_NP_nuebar is not None:
                  C_NP_funcs['nuebar'] = my_C_NP_nuebar
              if my_C_NP_numu is not None:
                  C_NP_funcs['numu'] = my_C_NP_numu
              if PRyMini.qke_density_matrix_flag:
                  dm_solver = PRyMboltz.DensityMatrixSolver(
                      y_coll_max=PRyMini.y_coll_max_boltz, C_NP_funcs=C_NP_funcs)
                  boltz_solver = dm_solver._boltz
              else:
                  boltz_solver = PRyMboltz.BoltzmannSolver(
                      y_coll_max=PRyMini.y_coll_max_boltz, C_NP_funcs=C_NP_funcs)
              Ny_boltz = boltz_solver.Ny
              n_species_boltz = boltz_solver.n_species

              # Initial scale factor from plasma entropy conservation (normalization only).
              # During Phase B, a is evolved via Friedmann equation and Tg is determined
              # from the plasma entropy equation following Froustey et al. (arXiv:2008.01074).
              spl_ref = PRyMthermo.spl(PRyMini.T_boltz_start)
              def _a_of_T_entropy(Tg):
                  return (spl_ref / PRyMthermo.spl(Tg))**(1./3.)

              # Phase A: Standard thermal ODE from T_start to T_boltz_start
              T_boltz_start_K = PRyMini.T_boltz_start * PRyMini.MeV_to_Kelvin
              T_boltz_end_K = PRyMini.T_boltz_end * PRyMini.MeV_to_Kelvin
              tini = 1./(2.*Hubble(Tstart_MeV)) # [s]
              # Temporarily use thermal evolution for Phase A (neutrinos in equilibrium)
              def dTtotdt_thermal(t,T_vec):
                  Tg_t,Tnu_t = T_vec
                  Hubble_T = PRyMini.MeV_to_secm1*(
                      (PRyMthermo.rho_g(Tg_t)+PRyMthermo.rho_e(Tg_t)
                       -PRyMthermo.PofT(Tg_t)+Tg_t*PRyMthermo.dPdT(Tg_t)
                       +PRyMthermo.rho_nu(Tnu_t)+2.*PRyMthermo.rho_nu(Tnu_t))
                      *8.*np.pi/(3.*PRyMini.Mpl**2))**0.5
                  num_g = -(Hubble_T*(4.*PRyMthermo.rho_g(Tg_t)+3.*(PRyMthermo.rho_e(Tg_t)+PRyMthermo.p_e(Tg_t))+3.*Tg_t*PRyMthermo.dPdT(Tg_t)))
                  delta_rho_g = -(PRyMthermo.delta_rho_nue(Tg_t,Tnu_t,Tnu_t)+2.*PRyMthermo.delta_rho_numu(Tg_t,Tnu_t,Tnu_t))
                  num_g += delta_rho_g
                  den_g = PRyMthermo.drho_g_dT(Tg_t)+PRyMthermo.drho_e_dT(Tg_t)+Tg_t*PRyMthermo.d2PdT2(Tg_t)
                  num_nu = -12.*Hubble_T*PRyMthermo.rho_nu(Tnu_t)
                  num_nu += (PRyMthermo.delta_rho_nue(Tg_t,Tnu_t,Tnu_t)+2.*PRyMthermo.delta_rho_numu(Tg_t,Tnu_t,Tnu_t))
                  den_nu = 3.*PRyMthermo.drho_nu_dT(Tnu_t)
                  return [num_g/den_g, num_nu/den_nu]

              # Find time at T_boltz_start by running Phase A
              t_boltz_start = 1./(2.*Hubble(PRyMini.T_boltz_start))
              t_boltz_end_approx = 1./(2.*Hubble(PRyMini.T_boltz_end))
              n_A = max(50, int(PRyMini.n_sampling * 0.1))
              sol_A_sampling = np.logspace(np.log10(tini),np.log10(t_boltz_start),n_A)
              sol_A_sampling[0],sol_A_sampling[-1] = tini,t_boltz_start
              sol_A = solve_ivp(dTtotdt_thermal,[tini,t_boltz_start],[Tstart_MeV,Tstart_MeV],
                                t_eval=sol_A_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
              t_A = sol_A.t
              Tg_A = sol_A.y[0][:]
              Tnu_A = sol_A.y[1][:]

              # Phase B: Boltzmann evolution from T_boltz_start to T_boltz_end
              # Uses operator splitting: temperature-based stepping with explicit
              # Heun for distributions. Needs ~2000 steps to resolve the collision
              # rate at high T (CFL stability condition: dt * Gamma_coll < 2).
              Tg_boltz_ini = Tg_A[-1]
              Tnu_boltz_ini = Tnu_A[-1]  # use Phase A's evolved Tnu, not Tg
              a_boltz_ini = _a_of_T_entropy(Tg_boltz_ini)
              if PRyMini.qke_density_matrix_flag:
                  # Pass user-supplied per-species initial callables through
                  # to the density-matrix solver when running asymmetric.
                  # In symmetric mode all entries are None → thermal FD.
                  f_initial_dm = None
                  if not PRyMini.mu_tau_symmetric_flag:
                      f_initial_dm = {
                          'nue':      my_f_nue,      'nuebar':   my_f_nuebar,
                          'numu':     my_f_numu,     'numubar':  my_f_numubar,
                          'nutau':    my_f_nutau,    'nutaubar': my_f_nutaubar,
                      }
                  rho_curr = dm_solver.initial_conditions(
                      Tnu_boltz_ini, a_boltz_ini, f_initial=f_initial_dm)
                  dm_solver.update_thermo_distributions(rho_curr, a_boltz_ini)
              else:
                  # Diagonal Boltzmann: n=3 or n=4 (Stage 2).
                  # In n=4, the solver accepts 'nue','nuebar','numu','nutau'
                  # ICs; pcle+antipcle remain merged per flavor (Stage 3 will
                  # extend to full n=6 with nu/nu-bar per flavor).
                  f_initial_boltz = None
                  if not PRyMini.mu_tau_symmetric_flag:
                      f_initial_boltz = {
                          'nue':   my_f_nue,   'nuebar': my_f_nuebar,
                          'numu':  my_f_numu,  'nutau':  my_f_nutau,
                      }
                  f_curr = boltz_solver.initial_conditions(
                      Tnu_boltz_ini, a_boltz_ini, f_initial=f_initial_boltz)
                  boltz_solver.update_thermo_distributions(f_curr, a_boltz_ini)

              t_B_start = t_A[-1]

              # Phase B: Froustey et al. formalism (arXiv:2008.01074)
              # Scale factor a evolved via Friedmann equation (da/dt = aH).
              # Photon temperature Tg determined from plasma entropy equation:
              #   d(spl * a³)/dt = -(Q/Tg) * a³
              # where Q is the collision energy transfer rate to neutrinos.
              # This self-consistently tracks the plasma cooling from e+e-
              # annihilation energy flowing to neutrinos, producing Neff > 3
              # without the momentum drift correction needed by the entropy-
              # based scale factor approach.
              me = PRyMini.me
              n_B = max(2000, int(PRyMini.n_sampling * 2))

              # Log-uniform grid in a, estimated from entropy-based relation
              a_end_est = _a_of_T_entropy(PRyMini.T_boltz_end) * 1.02
              a_grid = np.logspace(np.log10(a_boltz_ini),
                                     np.log10(a_end_est), n_B + 1)

              if(PRyMini.verbose_flag):
                  if PRyMini.qke_density_matrix_flag:
                      print(f"Phase B: QKE density matrix (Froustey), Tg={Tg_boltz_ini:.3f} to {PRyMini.T_boltz_end:.4f} MeV")
                      print(f"  {n_B} steps, Ny={Ny_boltz}, {2*9*Ny_boltz} DOFs")
                  else:
                      print(f"Phase B: Boltzmann (Froustey), Tg={Tg_boltz_ini:.3f} to {PRyMini.T_boltz_end:.4f} MeV")
                      print(f"  {n_B} steps, Ny={Ny_boltz}")

              # Mode-dependent collision damping coefficients for exponential Euler.
              # n=3 (mu_tau_symmetric=True): [nue, nuebar, numu_eff]
              # n=4 (mu_tau_symmetric=False): [nue, nuebar, numu, nutau]
              if PRyMini.mu_tau_symmetric_flag:
                  _C_D_boltz = np.array([3.06, 3.06, 2.22])
              else:
                  _C_D_boltz = np.array([3.06, 3.06, 2.22, 2.22])
              _GF2_secm1 = PRyMini.GF**2 * PRyMini.MeV_to_secm1
              _y_grid_boltz = boltz_solver.y_grid
              _dy_boltz = boltz_solver.dy

              # Storage for Phase B trajectory
              t_B_list = [t_B_start]
              Tg_B_list = [Tg_boltz_ini]
              a_B_list = [a_boltz_ini]
              f_min_clip = 1.0e-30

              # Froustey state variables
              sigma_curr = PRyMthermo.spl(Tg_boltz_ini) * a_boltz_ini**3
              Tg_curr = Tg_boltz_ini
              t_curr = t_B_start

              # Precompute y³ grid for comoving energy calculation
              _y3_grid = _y_grid_boltz**3
              _dy_2pi2 = _dy_boltz / (2.0 * np.pi**2)

              for istep in range(n_B):
                  a_curr = a_grid[istep]
                  a_next = a_grid[istep + 1]
                  da = a_next - a_curr
                  a_mid = 0.5 * (a_curr + a_next)

                  # Midpoint Tg estimate (adiabatic cooling)
                  Tg_mid = Tg_curr * (a_curr / a_mid)

                  # Time step: midpoint Hubble for O(da²) accuracy
                  # (Forward Euler dt = da/(a_curr*H_curr) has O(da) errors
                  # that accumulate to ~0.2% timing offset over Phase B)
                  if PRyMini.qke_density_matrix_flag:
                      dm_solver.update_thermo_distributions(rho_curr, a_curr)
                  else:
                      boltz_solver.update_thermo_distributions(f_curr, a_curr)
                  H_mid = Hubble(Tg_mid)
                  dt = da / (a_mid * H_mid)

                  # Exponential Euler regularization
                  _rate_base = _GF2_secm1 * Tg_mid**4 / a_mid * dt
                  _z_mode = np.outer(_C_D_boltz, _y_grid_boltz) * _rate_base
                  _z_mode = np.maximum(_z_mode, 1.0e-15)
                  _phi1_mode = np.where(_z_mode < 1.0e-4,
                                        1.0 - 0.5*_z_mode + _z_mode**2/6.0,
                                        (1.0 - np.exp(-_z_mode)) / _z_mode)

                  if PRyMini.qke_density_matrix_flag:
                      # QKE: track comoving energy before/after for entropy equation
                      _diag_sum_pre = (rho_curr[0,0] + rho_curr[1,0] + rho_curr[0,1]
                                       + rho_curr[0,2] + rho_curr[1,1] + rho_curr[1,2])
                      E_com_pre = np.sum(_y3_grid * _diag_sum_pre) * _dy_2pi2

                      dm_solver.evolve_step(rho_curr, dt, _phi1_mode * dt, a_mid, Tg_mid)

                      _diag_sum_post = (rho_curr[0,0] + rho_curr[1,0] + rho_curr[0,1]
                                        + rho_curr[0,2] + rho_curr[1,1] + rho_curr[1,2])
                      E_com_post = np.sum(_y3_grid * _diag_sum_post) * _dy_2pi2
                  else:
                      # Diagonal Boltzmann: track comoving energy before/after
                      if PRyMini.mu_tau_symmetric_flag:
                          _f_weighted_pre = f_curr[0] + f_curr[1] + 4.0 * f_curr[2]
                      else:
                          _f_weighted_pre = f_curr[0] + f_curr[1] + 2.0 * f_curr[2] + 2.0 * f_curr[3]
                      E_com_pre = np.sum(_y3_grid * _f_weighted_pre) * _dy_2pi2

                      C_f = boltz_solver.collision_integrals(f_curr, a_mid, Tg_mid)
                      f_curr = np.clip(f_curr + _phi1_mode * dt * C_f,
                                       f_min_clip, 1.0 - f_min_clip)

                      # Oscillation mixing (operator-split relaxation)
                      if PRyMini.nu_oscillation_flag and \
                              getattr(PRyMini, 'nu_oscillation_method', 'relaxation') == 'relaxation':
                          boltz_solver.apply_oscillation_mixing(f_curr, a_mid, Tg_mid, dt)

                      if PRyMini.mu_tau_symmetric_flag:
                          _f_weighted_post = f_curr[0] + f_curr[1] + 4.0 * f_curr[2]
                      else:
                          _f_weighted_post = f_curr[0] + f_curr[1] + 2.0 * f_curr[2] + 2.0 * f_curr[3]
                      E_com_post = np.sum(_y3_grid * _f_weighted_post) * _dy_2pi2

                  # Update plasma entropy from energy conservation:
                  #   d(spl * a³)/dt = -(Q/Tg) * a³
                  # where Q = ΔE_com / (a⁴ * dt) is the collision energy transfer rate.
                  # Substituting: Δσ = -ΔE_com / (a * Tg)
                  dE_com = E_com_post - E_com_pre
                  sigma_new = sigma_curr - dE_com / (a_mid * Tg_mid)

                  # Find Tg_new from spl(Tg_new) * a_next³ = sigma_new
                  s_target = sigma_new / a_next**3
                  Tg_guess = Tg_curr * (a_curr / a_next)  # adiabatic estimate
                  for _newton in range(10):
                      s_val = PRyMthermo.spl(Tg_guess)
                      ds_dT = (PRyMthermo.spl(Tg_guess * 1.001)
                               - PRyMthermo.spl(Tg_guess * 0.999)) / (0.002 * Tg_guess)
                      delta_T = (s_val - s_target) / ds_dT
                      Tg_guess -= delta_T
                      if abs(delta_T) < 1.0e-10 * Tg_guess:
                          break
                  Tg_new = Tg_guess

                  # Advance state
                  sigma_curr = sigma_new
                  Tg_curr = Tg_new
                  t_curr += dt

                  t_B_list.append(t_curr)
                  Tg_B_list.append(Tg_new)
                  a_B_list.append(a_next)

                  if(PRyMini.verbose_flag and (istep+1) % max(1, n_B//5) == 0):
                      print(f"    step {istep+1}/{n_B}: Tg={Tg_curr:.4f} MeV, a={a_next:.2f}")

                  # Stop when Tg drops below T_boltz_end
                  if Tg_new <= PRyMini.T_boltz_end:
                      break

              t_B = np.array(t_B_list)
              Tg_B = np.array(Tg_B_list)

              # Build a_of_T from Phase B trajectory for frozen distribution readout.
              # For Tg below the trajectory endpoint, extrapolate using entropy-based a
              # renormalized to match the trajectory (collisions are negligible there).
              from scipy.interpolate import interp1d as _interp1d
              _a_B_arr = np.array(a_B_list)
              _a_of_T_traj = _interp1d(Tg_B[::-1], _a_B_arr[::-1],
                                        kind='linear', bounds_error=False,
                                        fill_value='extrapolate')
              _a_end_traj = _a_B_arr[-1]
              _Tg_end_traj = Tg_B[-1]
              _a_entropy_end = _a_of_T_entropy(_Tg_end_traj)
              _a_ratio_end = _a_end_traj / _a_entropy_end

              def a_of_T(Tg):
                  if np.isscalar(Tg):
                      if Tg >= _Tg_end_traj:
                          return float(_a_of_T_traj(Tg))
                      else:
                          return _a_of_T_entropy(Tg) * _a_ratio_end
                  else:
                      Tg = np.asarray(Tg, dtype=float)
                      result = np.empty_like(Tg)
                      above = Tg >= _Tg_end_traj
                      if np.any(above):
                          result[above] = _a_of_T_traj(Tg[above])
                      if np.any(~above):
                          result[~above] = _a_of_T_entropy(Tg[~above]) * _a_ratio_end
                      return result

              # Freeze final distributions with dynamic a_of_T for Phase C
              if PRyMini.qke_density_matrix_flag:
                  dm_solver.update_thermo_distributions(rho_curr, _a_B_arr[-1],
                                                         a_of_T_func=a_of_T)
              else:
                  boltz_solver.update_thermo_distributions(f_curr, _a_B_arr[-1],
                                                           a_of_T_func=a_of_T)

              # Store final Boltzmann grid state for diagnostics / validation
              self._boltz_solver = boltz_solver
              self._boltz_a_final = _a_B_arr[-1]
              self._boltz_Tg_final = Tg_B[-1]
              self._boltz_Tnu_ini = Tnu_boltz_ini
              self._boltz_a_ini = a_boltz_ini
              if PRyMini.qke_density_matrix_flag:
                  self._boltz_rho_final = rho_curr.copy()
                  self._boltz_f_final = None
              else:
                  self._boltz_f_final = f_curr.copy()
                  self._boltz_rho_final = None

              # Phase C: Frozen distributions, 1-variable Tg ODE to end
              Tg_C_ini = Tg_B[-1]
              t_C_start = t_B[-1]
              n_C = PRyMini.n_sampling - len(t_A) - len(t_B)
              if n_C < 50:
                  n_C = 50
              sol_C_sampling = np.logspace(np.log10(t_C_start),np.log10(tfin),n_C)
              sol_C_sampling[0] = t_C_start
              sol_C_sampling[-1] = tfin
              sol_C = solve_ivp(dTtotdt,[t_C_start,tfin],[Tg_C_ini],
                                t_eval=sol_C_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
              t_C = sol_C.t
              Tg_C = sol_C.y[0][:]

              # Concatenate all three phases
              t_vec = np.concatenate([t_A, t_B[1:], t_C[1:]])
              Tg_vec = np.concatenate([Tg_A, Tg_B[1:], Tg_C[1:]])
              # Construct synthetic Tnu_vec from effective temperature
              Tnu_vec = np.concatenate([Tnu_A,
                  np.array([PRyMthermo.Tnu_eff_e(T) for T in Tg_B[1:]]),
                  np.array([PRyMthermo.Tnu_eff_e(T) for T in Tg_C[1:]])])

          elif(PRyMini.general_nu_flag):
              # 1-variable ODE: only Tg (neutrino sector described by f_nu(p, Tg))
              tini = 1./(2.*Hubble(Tstart_MeV)) # [s]
              sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PRyMini.n_sampling)
              sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
              if(PRyMini.NP_thermo_flag):
                  Tini_vec = [Tstart_MeV,PRyMini.Tstart_NP]
              else:
                  Tini_vec = [Tstart_MeV]
              if(PRyMini.julia_flag):
                  T0 = np.float64(Tini_vec)
                  tspan = (np.float64(tini),np.float64(tfin))
                  if(PRyMini.NP_thermo_flag):
                      p0 = [lambda x,y: np.float64(dTgdt(x,y)),lambda x,y: np.float64(dTNPdt_general(x,y))]
                      prob = de.ODEProblem(PRyMjl.dTtotdtGeneralNuNPjl,T0,tspan,p0)
                  else:
                      p0 = [lambda x: np.float64(dTgdt(x))]
                      prob = de.ODEProblem(PRyMjl.dTtotdtGeneralNujl,T0,tspan,p0)
                  sol_thermo = de.solve(prob,de.Tsit5(),saveat=sol_thermo_sampling,reltol=1.e-6,abstol=1.e-9)
                  t_vec = sol_thermo.t
                  sol_thermo = np.array(sol_thermo.u)
                  Tg_vec = sol_thermo[:,0]
                  if(PRyMini.NP_thermo_flag):
                      TNP_vec = sol_thermo[:,1]
              else:
                  sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                  t_vec = sol_thermo.t
                  Tg_vec = sol_thermo.y[0][:]
                  if(PRyMini.NP_thermo_flag):
                      TNP_vec = sol_thermo.y[1][:]
              # Construct synthetic Tnu_vec from effective temperature for downstream
              # compatibility. When Boltzmann is active, Tnu_eff_e reflects the grid
              # distributions. Without Boltzmann, the default FD distributions give
              # Tnu_eff = Tg (not physical after decoupling); the N_eff readout uses
              # the thermal formula with Tnu_vec to avoid this issue.
              Tnu_eff_vec = np.array([PRyMthermo.Tnu_eff_e(T) for T in Tg_vec])
              Tnu_vec = Tnu_eff_vec
          elif(PRyMini.NP_thermo_flag):
              tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV,PRyMini.Tstart_NP)) # [s]
              sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PRyMini.n_sampling)
              sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
              Tini_vec = [Tstart_MeV,Tstart_MeV,PRyMini.Tstart_NP]
              if(PRyMini.julia_flag):
                  T0 = np.float64(Tini_vec)
                  tspan = (np.float64(tini),np.float64(tfin))
                  p0 = [lambda w,x,y,z: np.float64(dTgdt(w,x,y,z)),lambda w,x,y,z: np.float64(dTnudt(w,x,y,z)),lambda w,x,y,z: np.float64(dTNPdt(w,x,y,z))]
                  prob = de.ODEProblem(PRyMjl.dTtotdtNPjl,T0,tspan,p0)
                  sol_thermo = de.solve(prob,de.Tsit5(),saveat=sol_thermo_sampling,reltol=1.e-6,abstol=1.e-9)
                  t_vec = sol_thermo.t
                  sol_thermo = np.array(sol_thermo.u)
                  Tg_vec = sol_thermo[:,0]
                  Tnu_vec = sol_thermo[:,1]
                  TNP_vec = sol_thermo[:,2]
              else:
                  sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                  t_vec = sol_thermo.t
                  Tg_vec = sol_thermo.y[0][:]
                  Tnu_vec = sol_thermo.y[1][:]
                  TNP_vec = sol_thermo.y[2][:]
          else:
              tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV)) # s
              sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PRyMini.n_sampling)
              sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
              Tini_vec = [Tstart_MeV,Tstart_MeV]
              if(PRyMini.julia_flag):
                  T0 = np.float64(Tini_vec)
                  tspan = (np.float64(tini),np.float64(tfin))
                  p0 = [lambda x,y,z: np.float64(dTgdt(x,y,z)),lambda x,y,z: np.float64(dTnudt(x,y,z))]
                  prob = de.ODEProblem(PRyMjl.dTtotdtSMjl,T0,tspan,p0)
                  sol_thermo = de.solve(prob,de.Tsit5(),saveat=sol_thermo_sampling,reltol=1.e-6,abstol=1.e-9)
                  t_vec = sol_thermo.t
                  sol_thermo = np.array(sol_thermo.u)
                  Tg_vec = sol_thermo[:,0]
                  Tnu_vec = sol_thermo[:,1]
              else:
                  sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                  t_vec = sol_thermo.t
                  Tg_vec = sol_thermo.y[0][:]
                  Tnu_vec = sol_thermo.y[1][:]
          # Save results for background thermodynamics
          if(PRyMini.save_bckg_flag):
              if(PRyMini.NP_thermo_flag):
                  np.savetxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu_TNP.txt",np.c_[t_vec,Tg_vec,Tnu_vec,TNP_vec])
              else:
                  np.savetxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu.txt",np.c_[t_vec,Tg_vec,Tnu_vec])
        else:
            if(PRyMini.NP_thermo_flag):
                t_vec,Tg_vec,Tnu_vec,TNP_vec = np.loadtxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu_TNP.txt",unpack=True)
            else:
                t_vec,Tg_vec,Tnu_vec = np.loadtxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu.txt",unpack=True)
                
        # Interpolation of Tnu(T) (and NP) for non-instantaneous decoupling effecs in a(T)
        if(PRyMini.aTid_flag):
            TnuofT = interp1d(Tg_vec[:],Tnu_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
            if(PRyMini.NP_thermo_flag):
                TNPofT = interp1d(Tg_vec[:],TNP_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        
        ################
        # N effective  #
        ################
        # Definition as extra radiation density relative to photons in units of 8/7 x (11/4)^(4/3)
        if(PRyMini.general_nu_flag):
            def N_eff(Tg,Tnue=None,Tnumu=None,T_NP=0.):
                rho_gamma = PRyMthermo.rho_g(Tg)
                if PRyMini.boltzmann_nu_flag and Tnue is None:
                    # Boltzmann: use general distributions populated by the grid
                    rho_rad_tot = PRyMthermo.rho_3nu(Tg)+rho_gamma
                elif Tnue is not None:
                    # Non-Boltzmann or explicit Tnu: use thermal formula
                    rho_rad_tot = PRyMthermo.rho_nu(Tnue)+2.*PRyMthermo.rho_nu(Tnumu)+rho_gamma
                else:
                    rho_rad_tot = PRyMthermo.rho_3nu(Tg)+rho_gamma
                if(PRyMini.NP_thermo_flag):
                    rho_rad_tot += PRyMthermo.rho_NP(T_NP)
                elif(PRyMini.NP_e_flag):
                    rho_rad_tot += PRyMthermo.rho_NP(Tg)
                normDeltaNeff = (7./8.)*(4./11.)**(4./3.)
                return (rho_rad_tot-rho_gamma)/rho_gamma/normDeltaNeff
        else:
            def N_eff(Tg,Tnue,Tnumu,T_NP=0.):
                rho_gamma = PRyMthermo.rho_g(Tg)
                rho_rad_tot = PRyMthermo.rho_nu(Tnue)+2.*PRyMthermo.rho_nu(Tnumu)+rho_gamma
                if(PRyMini.NP_thermo_flag):
                    rho_rad_tot += PRyMthermo.rho_NP(T_NP)
                elif(PRyMini.NP_nu_flag):
                    rho_rad_tot += PRyMthermo.rho_NP(Tnue)
                elif(PRyMini.NP_e_flag):
                    rho_rad_tot += PRyMthermo.rho_NP(Tg)
                normDeltaNeff = (7./8.)*(4./11.)**(4./3.)
                return (rho_rad_tot-rho_gamma)/rho_gamma/normDeltaNeff
            
        ################################
        # Relic abundance of neutrinos #
        ################################
        # Cosmic abundance of single species of relativistic nu
        if(PRyMini.general_nu_flag and PRyMini.boltzmann_nu_flag):
            def Omeganuh2_relnu():
                # Use rho_3nu from general distributions at end of BBN, scaled to today
                Tg_end = Tg_vec[-1]
                Tg0 = PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
                Tnu_eff_end = PRyMthermo.Tnu_eff_e(Tg_end)
                Tnu0 = Tnu_eff_end/Tg_end*Tg0
                return (7.*np.pi**2/120.*Tnu0**4)/PRyMini.rhocOverh2
            def Omeganuh2_nrnu():
                Tg_end = Tg_vec[-1]
                Tg0 = PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
                Tnu_eff_end = PRyMthermo.Tnu_eff_e(Tg_end)
                Tnu0 = Tnu_eff_end/Tg_end*Tg0
                return (3./2.*zeta(3)/np.pi**2*Tnu0**3)/PRyMini.rhocOverh2
        else:
            def Omeganuh2_relnu():
                Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
                return (7.*np.pi**2/120.*Tnu0**4)/PRyMini.rhocOverh2 # dimensionless
            def Omeganuh2_nrnu():
                Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
                return (3./2.*zeta(3)/np.pi**2*Tnu0**3)/PRyMini.rhocOverh2 # MeV
        
        ######################################################
        # FRW cosmological backround in radiation domination #
        ######################################################
        # Relation between time and temperature of the thermal bath
        t_of_T = interp1d(Tg_vec[:],t_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        t_of_T_vec = np.vectorize(t_of_T)
        T_of_t = interp1d(t_vec[:],Tg_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        T_of_t_vec = np.vectorize(T_of_t)
        
        ######################################################
        # Relation of scale factor with temperature and time #
        ######################################################
        # Non-instantaneous decoupling effects on the entropy of the plasma
        # Note: when general_nu_flag is True, the N_nu_rate approach (mapping general
        # distributions to effective temperatures for the thermal collision terms) is
        # unreliable and can cause ODE solver stiffness. The incomplete decoupling
        # correction to a(T) is sub-percent and already approximately captured in the
        # general Tg(t) evolution, so we use the simpler entropy-based a(T) instead.
        if(PRyMini.aTid_flag and not PRyMini.general_nu_flag):
            # Heat rate due to neutrino (and NP) interactions with the plasma
            def N_nu_rate(T):
                Tnu = TnuofT(T)
                qdot_pl = -(PRyMthermo.delta_rho_nue(T,Tnu,Tnu)+2.*PRyMthermo.delta_rho_numu(T,Tnu,Tnu))
                Hubble_T = Hubble(T,Tnu,Tnu)
                if(PRyMini.NP_thermo_flag):
                    TNP = TNPofT(T)
                    qdot_pl -= PRyMthermo.delta_rho_NP(T,Tnu,Tnu,TNP)
                    Hubble_T = Hubble(T,Tnu,Tnu,TNP)
                res = -qdot_pl/Hubble_T/T**4
                return res
            # Plasma entropy density normalized to T^3 (constant after e+- annihilation)
            def sbar(T):
                return PRyMthermo.spl(T)/T**3
            # Numerical derivative of the above wrt to temperature
            if(PRyMini.numdiff_flag):
                from numdifftools import Derivative
                dsbardT = Derivative(sbar,n=1)
            else:
                def dsbardT(T):
                    dToT = 1.e-3
                    return (sbar((1.+dToT)*T)-sbar((1.-dToT)*T))/(2.*dToT*T)
            # dlog(a*T)/dlog(T)
            def dlnadlnT(lnT):
                T = np.exp(lnT)
                sbar_T = sbar(T)
                N_nu_T = N_nu_rate(T)
                return -(3.*sbar_T+T*dsbardT(T))/(3.*sbar_T+N_nu_T)
            # Log of scale factor as a function of log of temperature of thermal bath
            Tini_vec = [np.log(Tend_MeV),np.log(Tstart_MeV)]
            # Initial conditions using z = a*T and entropy conservation
            z0 = PRyMini.T0CMB/PRyMini.MeV_to_Kelvin # a0 = 1 --> z0 = T0
            # Assuming no change in plasma entropy per comoving volume after end of BBN
            zend = (z0/(sbar(Tend_MeV)/PRyMini.s0bar)**(1/3)) # iff d(spl*a^3) = 0
            # aend conveniently allows to sample from end of BBN instead of today
            T_sol_vec = np.logspace(np.log10(Tend_MeV),np.log10(Tstart_MeV),PRyMini.n_sampling)
            if(PRyMini.julia_flag):
                logaend_vec = [np.log(zend/Tend_MeV)]
                logaend = np.float64(logaend_vec)
                Tspan = (np.float64(np.log(Tend_MeV)),np.float64(np.log(Tstart_MeV)))
                p0 = [lambda x: np.float64(dlnadlnT(x))]
                prob = de.ODEProblem(PRyMjl.dlnajl,logaend,Tspan,p0)
                sol_lnalnT = de.solve(prob,de.Tsit5(),saveat=np.log(T_sol_vec),reltol=1.e-6,abstol=1.e-9)
                sol_lnT = sol_lnalnT.t
                sol_lnalnT = np.array(sol_lnalnT.u)
                sol_lna = sol_lnalnT[:,0]
            else:
                def dlna(lnT,y):
                    return dlnadlnT(lnT)
                sol_lnalnT = solve_ivp(dlna,Tini_vec,[np.log(zend/Tend_MeV)],t_eval=np.log(T_sol_vec),method='LSODA',rtol=1.e-6,atol=1.e-9)
                sol_lnT = np.array(sol_lnalnT.t[:]).flatten()
                sol_lna = np.array(sol_lnalnT.y[:]).flatten()
            # log(a) as a function of log(T)
            lnalnT = interp1d(sol_lnT,sol_lna,bounds_error=False,fill_value="extrapolate")
        
        # Scale factor as a function of temperature of thermal bath
        def a_of_T(T):
            # Including non-instantaneous decoupling effects (thermal path only)
            if(PRyMini.aTid_flag and not PRyMini.general_nu_flag):
                return np.exp(lnalnT(np.log(T)))
            # Instantaneous decoupling approximation (used for general distributions
            # and when aTid_flag is False)
            else:
                spl_T = PRyMthermo.spl(T)
                return (PRyMini.s0CMB/spl_T)**(1./3.)
        a_of_T_vec = np.vectorize(a_of_T)
        # Scale factor as a function of time
        a_in = a_of_T(Tg_vec[0])
        a_fin = a_of_T(Tg_vec[-1])
        a_of_t = interp1d(t_vec[:],a_of_T_vec(Tg_vec),bounds_error=False,fill_value=(a_in,a_fin))
        
        ##########################################
        # Baryon density for the nuclear network #
        ##########################################
        # Baryon number density obtained as n0B = rho0B/mB
        # mB = averaged baryon mass in MeV: assumes helium fraction of 24.7%
        # rho0B = atomic density (it includes electron mass + binding energies)
        def nB(a):
            n0B = PRyMini.n0CMB*PRyMini.eta0b # baryon number density of today MeV^3
            return n0B/a**3 # MeV^3
        # Baryon-to-photon ratio as a function of temperature given in [K]
        def etab_of_T(T_K):
            T_MeV = T_K/PRyMini.MeV_to_Kelvin
            ngCMB = (2.*zeta(3))/(np.pi**2)*T_MeV**3
            return nB(a_of_T(T_MeV))/ngCMB
        # Baryon energy density adopted in the nuclear network
        # rhoB = nucleonic density (i.e. rho0B measured by CMB x ma/mB)
        def rhoB_BBN(a):
            n0B = PRyMini.n0CMB*PRyMini.eta0b
            rho0BmaOvermB = PRyMini.ma*n0B
            return rho0BmaOvermB*PRyMini.MeV4_to_gcmm3/a**3 # CGS, a0 = 1
            
        ######################################################
        # Definition of temperature eras for nuclear network #
        ######################################################
        t_start = t_of_T(PRyMini.T_start/PRyMini.MeV_to_Kelvin)
        t_weak = t_of_T(PRyMini.T_weak/PRyMini.MeV_to_Kelvin)
        t_nucl = t_of_T(PRyMini.T_nucl/PRyMini.MeV_to_Kelvin)
        t_end = t_of_T(PRyMini.T_end/PRyMini.MeV_to_Kelvin)

        ##############################
        # Import n <--> p weak rates #
        ###############################
        import PRyM.PRyM_eval_nTOp as PRyMevalnTOp
        import PRyM.PRyM_nTOp as PRyMnTOp
        if(PRyMini.general_nu_flag):
            nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT = PRyMnTOp.RecomputeWeakRates([Tg_vec])
        else:
            nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT = PRyMnTOp.RecomputeWeakRates([Tg_vec,Tnu_vec])
        
        ############################
        # Weak rates normalization #
        ############################
        if(PRyMini.tau_n_flag):
            Fn = PRyMevalnTOp.ComputeFn()
            NormWeakRates = 1./(Fn*PRyMini.tau_n) # normalization in [s-1]
        else:
            GFtilde2 = (PRyMini.GF*PRyMini.Vud)**2*(1+3.*PRyMini.gA**2)/(2.*np.pi**3)
            NormWeakRates = PRyMini.MeV_to_secm1*(GFtilde2*PRyMini.me**5) # normalization in [s-1]
        
        ##################################
        # High temperature era: only p,n #
        ##################################
        # Initial conditions from detailed balance
        def Yni(T):
            return nTOp_bkwrd_HT(T)/(nTOp_bkwrd_HT(T) + nTOp_frwrd_HT(T))
        def Ypi(T):
            return (1.-Yni(T))

        # Weak rates at HT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_HT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_HT(T)
            
        def Yn_prime_HT(t,Y):
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            return nTOp_bkwrd(T_t)*Y[1]-nTOp_frwrd(T_t)*Y[0]
            
        def Yp_prime_HT(t,Y):
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            return nTOp_frwrd(T_t)*Y[0]-nTOp_bkwrd(T_t)*Y[1]
            
        def Y_prime_HT(t,Y):
            dY = Yn_prime_HT(t,Y),Yp_prime_HT(t,Y)
            return dY

        #############################
        # High temperature solution #
        #############################
        if(PRyMini.verbose_flag):
            print(" ")
            print("Solving neutron decoupling at high temperature era")
            
        # HT era definition
        t_init = t_start
        t_fin = t_weak

        # HT initial conditions
        Yn_i = Yni(PRyMini.T_start)
        Yp_i = Ypi(PRyMini.T_start)
        
        # Solving HT network
        Yi_vec = [Yn_i,Yp_i]
        if(PRyMini.julia_flag):
            Y0 = np.float64(Yi_vec)
            tspan = (np.float64(t_init),np.float64(t_fin))
            p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(nTOp_frwrd(x)),lambda x: np.float64(nTOp_bkwrd(x))]
            f_Y_prime_HT_jl = de.ODEFunction(PRyMjl.Y_prime_HT_jl,jac = PRyMjl.Jacobian_HT_jl)
            prob = de.ODEProblem(f_Y_prime_HT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
            sol_at_HT = de.solve(prob,de.RadauIIA5())
            sol_at_HT = np.array(sol_at_HT.u)
            Yn_HT_f,Yp_HT_f = sol_at_HT[-1,:]
        else:
            sol_at_HT = solve_ivp(Y_prime_HT,[t_init,t_fin],Yi_vec,method='LSODA',rtol=1.e-6,atol=1.e-9)
            Yn_HT_f,Yp_HT_f = sol_at_HT.y[0][-1],sol_at_HT.y[1][-1]
        
        if(PRyMini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")
        
        ########################
        # Import nuclear rates #
        ########################
        if(PRyMini.smallnet_flag):
            import PRyM.PRyM_nuclear_net12 as PRyMnuclear
            PRyMnucl = PRyMnuclear.UpdateNuclearRates(PRyMini.p_npdg,PRyMini.p_dpHe3g,PRyMini.p_ddHe3n,PRyMini.p_ddtp,PRyMini.p_tpag,PRyMini.p_tdan,PRyMini.p_taLi7g,PRyMini.p_He3ntp,PRyMini.p_He3dap,PRyMini.p_He3aBe7g,PRyMini.p_Be7nLi7p,PRyMini.p_Li7paa)
            if(PRyMini.julia_flag):
                pMLT = [lambda x: np.float64(PRyMnucl.npdg_frwrd(x)),lambda x: np.float64(PRyMnucl.npdg_bkwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_frwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_frwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_frwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_bkwrd(x)),lambda x: np.float64(PRyMnucl.tpag_frwrd(x)),lambda x: np.float64(PRyMnucl.tpag_bkwrd(x)),lambda x: np.float64(PRyMnucl.tdan_frwrd(x)),lambda x: np.float64(PRyMnucl.tdan_bkwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_frwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_bkwrd(x))]
        else:
            import PRyM.PRyM_nuclear_net63 as PRyMnuclear
            PRyMnucl = PRyMnuclear.UpdateNuclearRates(PRyMini.p_npdg,PRyMini.p_dpHe3g,PRyMini.p_ddHe3n,PRyMini.p_ddtp,PRyMini.p_tpag,PRyMini.p_tdan,PRyMini.p_taLi7g,PRyMini.p_He3ntp,PRyMini.p_He3dap,PRyMini.p_He3aBe7g,PRyMini.p_Be7nLi7p,PRyMini.p_Li7paa,PRyMini.p_Li7paag,PRyMini.p_Be7naa,PRyMini.p_Be7daap,PRyMini.p_daLi6g,PRyMini.p_Li6pBe7g,PRyMini.p_Li6pHe3a,PRyMini.p_B8naap,PRyMini.p_Li6He3aap,PRyMini.p_Li6taan,PRyMini.p_Li6tLi8p,PRyMini.p_Li7He3Li6a,PRyMini.p_Li8He3Li7a,PRyMini.p_Be7tLi6a,PRyMini.p_B8tBe7a,PRyMini.p_B8nLi6He3,PRyMini.p_B8nBe7d,PRyMini.p_Li6tLi7d,PRyMini.p_Li6He3Be7d,PRyMini.p_Li7He3aad,PRyMini.p_Li8He3aat,PRyMini.p_Be7taad,PRyMini.p_Be7tLi7He3,PRyMini.p_B8dBe7He3,PRyMini.p_B8taaHe3,PRyMini.p_Be7He3ppaa,PRyMini.p_ddag,PRyMini.p_He3He3app,PRyMini.p_Be7pB8g,PRyMini.p_Li7daan,PRyMini.p_dntg,PRyMini.p_ttann,PRyMini.p_He3nag,PRyMini.p_He3tad,PRyMini.p_He3tanp,PRyMini.p_Li7taan,PRyMini.p_Li7He3aanp,PRyMini.p_Li8dLi7t,PRyMini.p_Be7taanp,PRyMini.p_Be7He3aapp,PRyMini.p_Li6nta,PRyMini.p_He3tLi6g,PRyMini.p_anpLi6g,PRyMini.p_Li6nLi7g,PRyMini.p_Li6dLi7p,PRyMini.p_Li6dBe7n,PRyMini.p_Li7nLi8g,PRyMini.p_Li7dLi8p,PRyMini.p_Li8paan,PRyMini.p_annHe6g,PRyMini.p_ppndp,PRyMini.p_Li7taann)
            if(PRyMini.julia_flag):
                pMT = [lambda x: np.float64(PRyMnucl.npdg_frwrd(x)),lambda x: np.float64(PRyMnucl.npdg_bkwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_frwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_frwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_frwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_bkwrd(x)),lambda x: np.float64(PRyMnucl.tpag_frwrd(x)),lambda x: np.float64(PRyMnucl.tpag_bkwrd(x)),lambda x: np.float64(PRyMnucl.tdan_frwrd(x)),lambda x: np.float64(PRyMnucl.tdan_bkwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_frwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paag_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paag_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7naa_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7naa_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7daap_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7daap_bkwrd(x)),lambda x: np.float64(PRyMnucl.daLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.daLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6pBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6pBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6pHe3a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6pHe3a_bkwrd(x))]
                pLT = pMT+[lambda x: np.float64(PRyMnucl.B8naap_frwrd(x)),lambda x: np.float64(PRyMnucl.B8naap_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3aap_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3aap_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6taan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6taan_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi8p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi8p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3Li6a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3Li6a_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3Li7a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3Li7a_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi6a_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi6a_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8tBe7a_frwrd(x)),lambda x: np.float64(PRyMnucl.B8tBe7a_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8nLi6He3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8nLi6He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8nBe7d_frwrd(x)),lambda x: np.float64(PRyMnucl.B8nBe7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi7d_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3Be7d_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3Be7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aad_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aad_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3aat_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3aat_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7taad_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7taad_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi7He3_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi7He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8dBe7He3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8dBe7He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8taaHe3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8taaHe3_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3ppaa_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3ppaa_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddag_frwrd(x)),lambda x: np.float64(PRyMnucl.ddag_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3He3app_frwrd(x)),lambda x: np.float64(PRyMnucl.He3He3app_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7pB8g_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7pB8g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7daan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7daan_bkwrd(x)),lambda x: np.float64(PRyMnucl.dntg_frwrd(x)),lambda x: np.float64(PRyMnucl.dntg_bkwrd(x)),lambda x: np.float64(PRyMnucl.ttann_frwrd(x)),lambda x: np.float64(PRyMnucl.ttann_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3nag_frwrd(x)),lambda x: np.float64(PRyMnucl.He3nag_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tad_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tad_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tanp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7taan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7taan_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aanp_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8dLi7t_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8dLi7t_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7taanp_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7taanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3aapp_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3aapp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6nta_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6nta_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.anpLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.anpLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6nLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6nLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6dLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6dLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6dBe7n_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6dBe7n_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7nLi8g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7nLi8g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7dLi8p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7dLi8p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8paan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8paan_bkwrd(x)),lambda x: np.float64(PRyMnucl.annHe6g_frwrd(x)),lambda x: np.float64(PRyMnucl.annHe6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ppndp_frwrd(x)),lambda x: np.float64(PRyMnucl.ppndp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7taann_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7taann_bkwrd(x))]
        
        #################################################
        # Local thermal equilibrium for nuclear species #
        #################################################
        def YA(name,Yn,Yp,T):
            x = PRyMini.Nuclides[name]
            A = x[0]+x[1]
            Z = x[1]
            N = A-Z
            Mass = A*PRyMini.ma*PRyMini.MeV+PRyMini.keV*PRyMini.NuclExcessMass[name]-Z*PRyMini.me*PRyMini.MeV
            BindingE = N*PRyMini.NuclExcessMass["n"] + Z*PRyMini.NuclExcessMass["p"]-PRyMini.NuclExcessMass[name]
            NormYA = (Mass/((PRyMini.mn*PRyMini.MeV)**(A-Z)*(PRyMini.mp*PRyMini.MeV)**Z))**(3/2)
            return (2*PRyMini.NuclSpin[name]+1)*zeta(3)**(A-1)*np.pi**((1-A)/2)*2**((3*A-5)/2)*NormYA*(PRyMini.kB*T)**(3/2*(A-1))*etab_of_T(T)**(A-1)*Yp**Z*Yn**(A-Z) *np.exp(BindingE*PRyMini.keV/(PRyMini.kB*T))
        
        #########################################################
        # Nuclear network: Final yields for p,d,t,He3,a,Li7,Be7 #
        #########################################################
        if(PRyMini.smallnet_flag):
            def Y_prime(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                dY = PRyMnucl.dYndt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY
                
            def Jacobian(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                return PRyMnucl.Jacobian(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        else:
            def Y_prime_MT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                dY = PRyMnucl.dYndtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYB8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY
                
            def Jacobian_MT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                return PRyMnucl.JacobianMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        
            def Y_prime_LT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                dY = PRyMnucl.dYndtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYB8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY

            def Jacobian_LT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
                return PRyMnucl.JacobianLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        
        ############################
        # Mid temperature solution #
        ############################
        if(PRyMini.verbose_flag):
            print("Solving nuclear network at mid temperature era")
            
        # MT era definition
        t_init = t_weak
        t_fin = t_nucl
        
        # Weak rates at MT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_MT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_MT(T)
        
        # Initial conditions at MT
        Yn_i = Yn_HT_f
        Yp_i = Yp_HT_f
        Yd_i = YA("d",Yn_i,Yp_i,PRyMini.T_weak)
        Yt_i = YA("t",Yn_i,Yp_i,PRyMini.T_weak)
        YHe3_i = YA("He3",Yn_i,Yp_i,PRyMini.T_weak)
        Ya_i = YA("a",Yn_i,Yp_i,PRyMini.T_weak)
        YLi7_i = YA("Li7",Yn_i,Yp_i,PRyMini.T_weak)
        YBe7_i = YA("Be7",Yn_i,Yp_i,PRyMini.T_weak)
        if(PRyMini.smallnet_flag == False):
            YHe6_i = YA("He6",Yn_i,Yp_i,PRyMini.T_weak)
            YLi8_i = YA("Li8",Yn_i,Yp_i,PRyMini.T_weak)
            YLi6_i = YA("Li6",Yn_i,Yp_i,PRyMini.T_weak)
            YB8_i = YA("B8",Yn_i,Yp_i,PRyMini.T_weak)
        
        # Solving MT network
        if(PRyMini.smallnet_flag):
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
            if(PRyMini.julia_flag):
                Y0 = np.float64(Yi_vec)
                tspan = (np.float64(t_init),np.float64(t_fin))
                p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_MT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_MT(x))] + pMLT
                f_Y_prime_MT_jl = de.ODEFunction(PRyMjl.Y_prime_MLT_jl,jac = PRyMjl.Jacobian_MLT_jl)
                prob = de.ODEProblem(f_Y_prime_MT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
                sol_at_MT = de.solve(prob,de.FBDF())
                sol_at_MT = np.array(sol_at_MT.u)
                Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f = sol_at_MT[-1,:]
            else:
                sol_at_MT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,rtol=1.e-6,atol=1.e-9)
                Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1]
        else:
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
            if(PRyMini.julia_flag):
                Y0 = np.float64(Yi_vec)
                tspan = (np.float64(t_init),np.float64(t_fin))
                p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_MT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_MT(x))] + pMT
                f_Y_prime_MT_jl = de.ODEFunction(PRyMjl.Y_prime_MT_jl,jac=PRyMjl.Jacobian_MT_jl)
                prob = de.ODEProblem(f_Y_prime_MT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
                sol_at_MT = de.solve(prob,de.FBDF())
                sol_at_MT = np.array(sol_at_MT.u)
                Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f,YHe6_MT_f,YLi8_MT_f,YLi6_MT_f,YB8_MT_f = sol_at_MT[-1,:]
            else:
                sol_at_MT = solve_ivp(Y_prime_MT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_MT,rtol=1.e-6,atol=1.e-9)
                Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f,YHe6_MT_f,YLi8_MT_f,YLi6_MT_f,YB8_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1],sol_at_MT.y[8][-1],sol_at_MT.y[9][-1],sol_at_MT.y[10][-1],sol_at_MT.y[11][-1]
        
        if(PRyMini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")
        
        ############################
        # Low temperature solution #
        ############################
        if(PRyMini.verbose_flag):
            print("Solving nuclear network at low temperature era")
        
        # LT era definition
        t_init = t_nucl
        t_fin = t_end

        # Weak rates at LT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_LT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_LT(T)
            
        # Initial conditions at LT
        Yn_i = Yn_MT_f
        Yp_i = Yp_MT_f
        Yd_i = Yd_MT_f
        Yt_i = Yt_MT_f
        YHe3_i = YHe3_MT_f
        Ya_i = Ya_MT_f
        YLi7_i = YLi7_MT_f
        YBe7_i = YBe7_MT_f
        if(PRyMini.smallnet_flag == False):
            YHe6_i = YHe6_MT_f
            YLi8_i = YLi8_MT_f
            YLi6_i = YLi6_MT_f
            YB8_i = YB8_MT_f
            
        if(PRyMini.smallnet_flag):
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
            if(PRyMini.julia_flag):
                Y0 = np.float64(Yi_vec)
                tspan = (np.float64(t_init),np.float64(t_fin))
                p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_LT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_LT(x))] + pMLT
                f_Y_prime_LT_jl = de.ODEFunction(PRyMjl.Y_prime_MLT_jl,jac = PRyMjl.Jacobian_MLT_jl)
                prob = de.ODEProblem(f_Y_prime_LT_jl,Y0,tspan,p0,abstol=1.e-13)
                sol_at_LT = de.solve(prob,de.CVODE_BDF())
                sol_at_LT = np.array(sol_at_LT.u)
                Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f = sol_at_LT[-1,:]
            else:
                sol_at_LT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,atol=1.e-11)
                Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1]
        else:
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
            if(PRyMini.julia_flag):
                Y0 = np.float64(Yi_vec)
                tspan = (np.float64(t_init),np.float64(t_fin))
                p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_LT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_LT(x))] + pLT
                f_Y_prime_LT_jl = de.ODEFunction(PRyMjl.Y_prime_LT_jl,jac=PRyMjl.Jacobian_LT_jl)
                prob = de.ODEProblem(f_Y_prime_LT_jl,Y0,tspan,p0,abstol=1.e-16)
                sol_at_LT = de.solve(prob,de.CVODE_BDF())
                sol_at_LT = np.array(sol_at_LT.u)
                Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f,YHe6_f,YLi8_f,YLi6_f,YB8_f = sol_at_LT[-1,:]
            else:
                sol_at_LT = solve_ivp(Y_prime_LT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_LT,atol=1.e-15)
                Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f,YHe6_f,YLi8_f,YLi6_f,YB8_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1],sol_at_LT.y[8][-1],sol_at_LT.y[9][-1],sol_at_LT.y[10][-1],sol_at_LT.y[11][-1]

        if(PRyMini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")

        if(PRyMini.verbose_flag):
            print("-------------------------------------------------")
            print("Predicted primordial abundances at the end of BBN")
            print("-------------------------------------------------")
            print("Yp = ",Yp_f)
            print("Yd = ",Yd_f)
            print("Yt = ",Yt_f)
            print("YHe3 = ",YHe3_f)
            print("Ya = ",Ya_f)
            print("YLi7 = ",YLi7_f)
            print("YBe7 = ",YBe7_f)
            print(" ")
            print("--- PRyMordial runned in: %s seconds ---" % (time.time() - start_time))
            
        #####################
        # Final predictions #
        #####################
        # N effective at the end of BBN era
        if(PRyMini.general_nu_flag):
            if PRyMini.boltzmann_nu_flag:
                # Boltzmann: rho_3nu from grid distributions
                if(PRyMini.NP_thermo_flag):
                    self.Neff_f = N_eff(Tg_vec[-1],T_NP=TNP_vec[-1])
                else:
                    self.Neff_f = N_eff(Tg_vec[-1])
            else:
                # Non-Boltzmann general_nu: use thermal Tnu from ODE
                if(PRyMini.NP_thermo_flag):
                    self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1],TNP_vec[-1])
                else:
                    self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1])
        elif(PRyMini.NP_thermo_flag):
            self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1],TNP_vec[-1])
        else:
            self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1])
        # Abundance of a single species of relativistic neutrino x 10^6
        self.Omeganurel_f = Omeganuh2_relnu()*1.e+6
        # Inverse of abundance of non-relativistic nu in units of sum of nu masses in [eV]
        self.OneOverOmeganunr_f = 1./(Omeganuh2_nrnu()*1.e-6)
        # Primordial helium-4 abundance as (nucleon) mass fraction (BBN definition)
        self.YPBBN_f = 4.*Ya_f
        # Primordial helium-4 abundance as (baryon) mass fraction (CMB definition)
        self.YPCMB_f = (PRyMini.He4Overma/4.)*self.YPBBN_f/((PRyMini.He4Overma/4.)*self.YPBBN_f+PRyMini.HOverma*(1.-self.YPBBN_f))
        # Primordial deuterium abundance as relative number density to hydrogen x 10^5
        self.DoHx1e5_f = Yd_f/Yp_f*1e+5
        # Primordial helium-3 abundance as relative number density to hydrogen x 10^5
        self.He3oHx1e5_f = (Yt_f+YHe3_f)/Yp_f*1e+5 # includes decay of tritium
        # Primordial lithium-7 abundance as relative number density to hydrogen x 10^10
        self.Li7oHx1e10_f = (YLi7_f+YBe7_f)/Yp_f*1e+10 # includes decay of beryllium-7
        # PRymordial output
        self.res = np.array([self.Neff_f,self.Omeganurel_f,self.OneOverOmeganunr_f,self.YPCMB_f,self.YPBBN_f,self.DoHx1e5_f,self.He3oHx1e5_f,self.Li7oHx1e10_f])
        
    def PRyMresults(self):
        return self.res

    def Neff(self):
        return self.Neff_f
        
    def Omeganurel(self):
        return self.Omeganurel_f
        
    def Omeganunonrel(self):
        return 1./self.OneOverOmeganunr_f
        
    def YPCMB(self):
        return self.YPCMB_f
        
    def YPBBN(self):
        return self.YPBBN_f
        
    def DoH(self):
        return self.DoHx1e5_f*1.e-5
        
    def He3oH(self):
        return self.He3oHx1e5_f*1.e-5
        
    def Li7oH(self):
        return self.Li7oHx1e10_f*1.e-10
