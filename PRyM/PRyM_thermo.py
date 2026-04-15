# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import kv
import PRyM.PRyM_init as PRyMini
if(PRyMini.numba_flag):
    from numba import njit

my_dir = PRyMini.working_dir
if(PRyMini.verbose_flag):
    print("PRyM_thermo.py: Loading SM rates for thermal bath")
    print("Natural units adopted here. Temperatures in MeV.")
 
###########################################################
# Standard Model matrix elements & plasma QED corrections #
###########################################################
# Credit for dataset to NUDEC_BSM:
# ArXiv:1812.05605 [JCAP 1902 (2019) 007] and ArXiv:2001.04466 [JCAP 05 (2020) 048])
# Effect of finite electron mass in scattering matrix elements (standard value for me assumed)
fnu_e_scat_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"nue_scatt.txt")
fnu_e_scat = interp1d(fnu_e_scat_tab[:,0],fnu_e_scat_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
fnu_mu_scat_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"numu_scatt.txt")
fnu_mu_scat = interp1d(fnu_mu_scat_tab[:,0],fnu_mu_scat_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
# Effect of finite electron mass in annihilation matrix elements (standard value for me assumed)
fnu_e_ann_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"nue_ann.txt")
fnu_e_ann = interp1d(fnu_e_ann_tab[:,0],fnu_e_ann_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
fnu_mu_ann_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"numu_ann.txt")
fnu_mu_ann = interp1d(fnu_mu_ann_tab[:,0],fnu_mu_ann_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
# QED plasma corrections (standard value for alphaem and me assumed).
# Source: NUDEC_BSM v2 (Escudero, Jackson, Laine, Sandner 2025, arXiv:2511.04747).
# Baseline tables always loaded: O(e^2) + O(e^3).
# Optional O(e^4) two-loop correction enabled via PRyMini.two_loop_QED_flag,
# loaded as separate interp1d objects so we can clamp them to zero outside
# the tabulated range (linear extrapolation of the persistent photon-photon
# Euler-Heisenberg piece at T << table_min diverges relative to rho_gamma
# and destabilizes the ODE integration; clamping to zero is physically safe
# since the O(e^4) contribution to observables is negligible at T << 5 keV).
P_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_P_int.txt")
_PofT_base = interp1d(P_QED_tab[:,0], P_QED_tab[:,1]+P_QED_tab[:,2],
                      bounds_error=False, fill_value="extrapolate", kind='linear')
dPdT_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_dP_intdT.txt")
_dPdT_base = interp1d(dPdT_QED_tab[:,0], dPdT_QED_tab[:,1]+dPdT_QED_tab[:,2],
                      bounds_error=False, fill_value="extrapolate", kind='linear')
d2PdT2_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_d2P_intdT2.txt")
_d2PdT2_base = interp1d(d2PdT2_QED_tab[:,0], d2PdT2_QED_tab[:,1]+d2PdT2_QED_tab[:,2],
                        bounds_error=False, fill_value="extrapolate", kind='linear')

if PRyMini.two_loop_QED_flag:
    _P_e4_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_P_int_e4.txt")
    _PofT_e4 = interp1d(_P_e4_tab[:,0], _P_e4_tab[:,1],
                        bounds_error=False, fill_value=0.0, kind='linear')
    _dPdT_e4_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_dP_intdT_e4.txt")
    _dPdT_e4 = interp1d(_dPdT_e4_tab[:,0], _dPdT_e4_tab[:,1],
                        bounds_error=False, fill_value=0.0, kind='linear')
    _d2PdT2_e4_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_d2P_intdT2_e4.txt")
    _d2PdT2_e4 = interp1d(_d2PdT2_e4_tab[:,0], _d2PdT2_e4_tab[:,1],
                          bounds_error=False, fill_value=0.0, kind='linear')
    def PofT(T):   return _PofT_base(T)   + _PofT_e4(T)
    def dPdT(T):   return _dPdT_base(T)   + _dPdT_e4(T)
    def d2PdT2(T): return _d2PdT2_base(T) + _d2PdT2_e4(T)
else:
    PofT   = _PofT_base
    dPdT   = _dPdT_base
    d2PdT2 = _d2PdT2_base

##################
# Photon species #
##################
# Photon energy density
def rho_g(Tg):
    return 2.*(np.pi**2/30.)*Tg**4
# drho_g/dT
def drho_g_dT(Tg):
    return 4.*rho_g(Tg)/Tg
    
###############
# e+- species #
###############
# e+- energy density
if(PRyMini.numba_flag):
    @njit
    def rho_e_int(E,Tg):
        return E**2*(E**2-(PRyMini.me/Tg)**2)**0.5/(np.exp(E)+1.)
else:
    def rho_e_int(E,Tg):
        return E**2*(E**2-(PRyMini.me/Tg)**2)**0.5/(np.exp(E)+1.)
def rho_e(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(rho_e_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel=1e-12)[0]
        return 4./(2*np.pi**2)*Tg**4*res_int
# drho_e/dT
if(PRyMini.numba_flag):
    @njit
    def drho_e_dT_int(E,Tg):
        return E**3*(E**2-(PRyMini.me/Tg)**2)**0.5/np.cosh(E/2.0)**2
else:
    def drho_e_dT_int(E,Tg):
        return E**3*(E**2-(PRyMini.me/Tg)**2)**0.5/np.cosh(E/2.0)**2
def drho_e_dT(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(drho_e_dT_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel = 1e-12)[0]
        return 1./(2*np.pi**2)*Tg**3*res_int
# e+- pressure density
if(PRyMini.numba_flag):
    @njit
    def p_e_int(E,Tg):
        return (E**2-(PRyMini.me/Tg)**2)**1.5/(np.exp(E)+1.)
else:
    def p_e_int(E,Tg):
        return (E**2-(PRyMini.me/Tg)**2)**1.5/(np.exp(E)+1.)
def p_e(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(p_e_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel=1e-12)[0]
        return 4./(6*np.pi**2)*Tg**4*res_int

####################
# Neutrino species #
####################
# Neutrino energy density (thermal, used when general_nu_flag is False)
def rho_nu(Tnu):
    # SM contribution per flavor
    rho_nu_SM = 2.*(7./8.)*(np.pi**2)/30.*Tnu**4
    # extra contribution per flavor
    rho_nu_extra = PRyMini.DeltaNeff/3.*rho_nu_SM
    return rho_nu_SM + rho_nu_extra
# drho_nu/dT (thermal)
def drho_nu_dT(Tnu):
    return 4.*rho_nu(Tnu)/Tnu

###############################################
# General neutrino distribution functions     #
# (used when general_nu_flag is True)         #
###############################################
if(PRyMini.general_nu_flag):
    # Gauss-Legendre quadrature in dimensionless variable x = p/Tg on [0, x_max_nu]
    # At each call, physical momenta are p_i = _gl_x_nodes[i] * Tg
    _gl_nodes_raw, _gl_weights_raw = np.polynomial.legendre.leggauss(PRyMini.p_npoints_nu)
    # Transform from [-1,1] to [0, x_max_nu]
    _gl_x_nodes = 0.5*(_gl_nodes_raw + 1.)*PRyMini.x_max_nu
    _gl_x_weights = 0.5*_gl_weights_raw*PRyMini.x_max_nu

# Default distribution function placeholders (overridden by PRyMclass constructor)
# These default to thermal Fermi-Dirac so that general_nu_flag=True with no custom
# distributions recovers the standard thermal result.
def f_nue_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)
def f_nuebar_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)
def f_numu_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)
def f_numubar_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)
def f_nutau_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)
def f_nutaubar_general(p, Tg):
    return 1./(np.exp(p/Tg) + 1.)

# Energy density for massless neutrinos from a general distribution function
# rho = 1/(2*pi^2) * integral dp p^3 f(p, Tg)  [per degree of freedom]
# Uses x = p/Tg substitution: rho = Tg^4/(2*pi^2) * integral dx x^3 f(x*Tg, Tg)
def rho_nu_from_f(f_nu, Tg):
    p_nodes = _gl_x_nodes * Tg
    integrand = p_nodes**3 * f_nu(p_nodes, Tg)
    return 1./(2.*np.pi**2) * Tg * np.dot(_gl_x_weights, integrand)

# Number density from a general distribution function
# n = 1/(2*pi^2) * integral dp p^2 f(p, Tg)  [per degree of freedom]
def n_nu_from_f(f_nu, Tg):
    p_nodes = _gl_x_nodes * Tg
    integrand = p_nodes**2 * f_nu(p_nodes, Tg)
    return 1./(2.*np.pi**2) * Tg * np.dot(_gl_x_weights, integrand)

# Numerical derivative of rho_nu_from_f with respect to Tg
def drho_nu_from_f_dTg(f_nu, Tg):
    dT = 1.e-3 * Tg
    return (rho_nu_from_f(f_nu, Tg + dT) - rho_nu_from_f(f_nu, Tg - dT)) / (2.*dT)

# Total neutrino energy density (all 3 families, particles + antiparticles)
def rho_3nu(Tg, Tnue=None, Tnumu=None):
    if(PRyMini.general_nu_flag):
        return (rho_nu_from_f(f_nue_general, Tg) + rho_nu_from_f(f_nuebar_general, Tg)
              + rho_nu_from_f(f_numu_general, Tg) + rho_nu_from_f(f_numubar_general, Tg)
              + rho_nu_from_f(f_nutau_general, Tg) + rho_nu_from_f(f_nutaubar_general, Tg))
    else:
        return rho_nu(Tnue) + 2.*rho_nu(Tnumu)

# Total neutrino pressure (all 3 families) — P = rho/3 for massless
def p_3nu(Tg, Tnue=None, Tnumu=None):
    return rho_3nu(Tg, Tnue, Tnumu) / 3.

# Derivative of total neutrino energy density wrt Tg
def drho_3nu_dTg(Tg, Tnue=None, Tnumu=None):
    if(PRyMini.general_nu_flag):
        return (drho_nu_from_f_dTg(f_nue_general, Tg) + drho_nu_from_f_dTg(f_nuebar_general, Tg)
              + drho_nu_from_f_dTg(f_numu_general, Tg) + drho_nu_from_f_dTg(f_numubar_general, Tg)
              + drho_nu_from_f_dTg(f_nutau_general, Tg) + drho_nu_from_f_dTg(f_nutaubar_general, Tg))
    else:
        return drho_nu_dT(Tnue) + 2.*drho_nu_dT(Tnumu)

# Effective neutrino temperatures from general distributions
# Used for the scale factor computation (N_nu_rate) where the thermal collision
# term structure is used as an approximation. Maps general distributions to
# effective temperatures via rho = 2*(7/8)*(pi^2/30)*T_eff^4 per flavor.
def Tnu_eff_e(Tg):
    """Effective electron neutrino temperature from general distribution."""
    rho_nue = rho_nu_from_f(f_nue_general, Tg) + rho_nu_from_f(f_nuebar_general, Tg)
    return (rho_nue / (2.*(7./8.)*(np.pi**2)/30.))**0.25

def Tnu_eff_mu(Tg):
    """Effective muon neutrino temperature from general distribution."""
    rho_numu = rho_nu_from_f(f_numu_general, Tg) + rho_nu_from_f(f_numubar_general, Tg)
    return (rho_numu / (2.*(7./8.)*(np.pi**2)/30.))**0.25

def Tnu_eff_tau(Tg):
    """Effective tau neutrino temperature from general distribution."""
    rho_nutau = rho_nu_from_f(f_nutau_general, Tg) + rho_nu_from_f(f_nutaubar_general, Tg)
    return (rho_nutau / (2.*(7./8.)*(np.pi**2)/30.))**0.25

# Placeholder for user-supplied NP collision term (additional energy injection)
def delta_rho_nu_NP(Tg):
    return 0.
 
##########################
# e+- nu matrix elements #
##########################
# Pauli blocking for relativistic fermions as in [JCAP 05 (2020) 048]
fannFD, fscatFD = 0.884, 0.829
def f_nu_e(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)*fnu_e_ann(T1)+56.*fscatFD*fnu_e_scat(T1)*T1**4*T2**4*(T1-T2)
    return res
def f_nu_mu(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)*fnu_mu_ann(T1)+56.*fscatFD*fnu_mu_scat(T1)*T1**4*T2**4*(T1-T2)
    return res
def f_g(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)+56.*fscatFD*T1**4*T2**4*(T1-T2)
    return res
# Collision terms in Boltzmann equation for energy densities
def delta_rho_nue(Tg,Tnue,Tnumu):
    return PRyMini.MeV_to_secm1*PRyMini.GF**2/np.pi**5*(4.*(PRyMini.geL**2+PRyMini.geR**2)*f_nu_e(Tg,Tnue)+2.*f_g(Tnumu,Tnue))
def delta_rho_numu(Tg,Tnue,Tnumu):
    return PRyMini.MeV_to_secm1*PRyMini.GF**2/np.pi**5*(4.*(PRyMini.gmuL**2+PRyMini.gmuR**2)*f_nu_mu(Tg,Tnumu)-f_g(Tnumu,Tnue))

#######################
# Standard Model (SM) #
#######################
# Total SM energy density
def rho_SM(Tg,Tnue,Tnumu):
    rho_3nu = rho_nu(Tnue)+2.*rho_nu(Tnumu)
    rho_plasma = rho_g(Tg)+rho_e(Tg)
    delta_rho_QED = Tg*dPdT(Tg)-PofT(Tg)
    return rho_plasma+rho_3nu+delta_rho_QED
# Total SM pressure density
def p_SM(Tg,Tnue,Tnumu):
    p_3nu = (rho_nu(Tnue)+2.*rho_nu(Tnumu))/3.
    p_plasma = rho_g(Tg)/3.+p_e(Tg)
    delta_p_QED = PofT(Tg)
    return p_plasma+p_3nu+delta_p_QED

############################
# New Physics (NP) species #
############################
# NP energy density
def rho_NP(T_NP):
    return 0.
# NP pressure density
def p_NP(T_NP):
    return 0.
# drho_NP/dT
def drho_NP_dT(T_NP):
    return 0.
# Collision terms in Boltzmann equation for rho_NP
def delta_rho_NP(Tg,Tnue,Tnumu,T_NP):
    return 0.

##########################
# Plasma entropy density #
##########################
def spl(Tg):
    rho_pl = rho_g(Tg)+rho_e(Tg)
    p_pl = rho_g(Tg)/3.+p_e(Tg)
    delta_rho_QED = Tg*dPdT(Tg)-PofT(Tg)
    delta_p_QED = PofT(Tg)
    spl_T = (rho_pl+p_pl+(delta_rho_QED+delta_p_QED))/Tg
    # NP species in equilibrium with e+-, gamma (i.e. SM plasma)
    if(PRyMini.NP_e_flag):
        spl_T += (rho_NP(Tg)+p_NP(Tg))/Tg
    return spl_T
