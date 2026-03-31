# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, spence
from scipy.integrate import quad
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini
if(PRyMini.compute_nTOp_thermal_flag):
    import vegas

exp_cutoff = 3*1.e+2 # cutoff to avoid overflow warnings
epsrel_low = 1.e-1 # minimum precision sufficient to speed up some quad integrals
if(PRyMini.compute_nTOp_thermal_flag):
    # Settings for precision in vegas integration
    n_eval = 20000 # recommended max number of evaluations per iteration
    n_itn = 20 # recommended number of iterations

def FermiCoulomb(b):
    me = PRyMini.me*PRyMini.MeV # electron mass
    Gamma = np.sqrt(1.-PRyMini.alphaem**2.)-1.
    gamma1 = 1.+Gamma
    gamma2 = 3.+2.*Gamma
    Fn_Compton = PRyMini.hbar*PRyMini.clight/me
    return (1.+Gamma/2.)*4.*((2.*PRyMini.radproton*b)/Fn_Compton)**(2.*Gamma)/(gamma(gamma2)**2)* np.exp((np.pi*PRyMini.alphaem)/b)/((1.-b**2)**Gamma)*np.abs(gamma(gamma1+(PRyMini.alphaem/b)*1j))**2
    
def RadCorrResum(b,y,en):
    # Additional constants specific to radiative corrections in [Czarnecki et al. 2004]
    mA = 1.2*1.e+3 # MeV , Eq.(9) in hep-ph/0406324
    Agndecay = -0.34 # Eq.(9) in hep-ph/0406324
    Cndecay = 0.891 # Eq.(9) in hep-ph/0406324
    deltandecay = -0.00043 # Eq.(12) in hep-ph/0406324
    Lndecay = 1.02094 # Eq.(13) in hep-ph/0406324
    Sndecay = 1.02248 # Eq.(13) in hep-ph/0406324
    NLLndecay = -0.0001 # Eq.(14) in hep-ph/0406324

    # Particle masses
    me = PRyMini.me*PRyMini.MeV # electron mass
    mn = PRyMini.mn*PRyMini.MeV # neutron mass
    mp = PRyMini.mp*PRyMini.MeV # proton mass
    mA = mA*PRyMini.MeV # nucleon mass
    Q = mn - mp # Mass difference between neutrons and protons
    
    if(b == 0):
        Rd = 1.
    else:
        Rd = np.arctanh(b)/b
    # Sirlin universal function (Eq 20b of [Sirlin 1967]) + Eq. 7 of [Czarnecki et al. 2004]):
    Sirlin_fun = 3.*np.log(mp/(me))-3./4.+4.*(Rd-1.)*(y/(3.*en)-3./2.+np.log(2.*y))+Rd*(2.*(1.+b**2)+y**2/(6.*en**2)-4.*b*Rd)-(4./b)*spence(1.-(2*b)/(1.+b))
    # Eq. 15 of Czarnecki 2004 + Esposito et al. 1998
    return (1.+PRyMini.alphaem/(2.*np.pi)*(Sirlin_fun-3.*np.log(mp/(2*Q))))*(Lndecay+(PRyMini.alphaem/np.pi)*Cndecay+ PRyMini.alphaem/(2*np.pi)*deltandecay*2*np.pi/PRyMini.alphaem)*(Sndecay+1./(134.*2.*np.pi)*(np.log(mp/mA)+Agndecay)+NLLndecay)

def ComputeFn():
    # Particle masses
    me = PRyMini.me*PRyMini.MeV # electron mass
    mn = PRyMini.mn*PRyMini.MeV # neutron mass
    mp = PRyMini.mp*PRyMini.MeV # proton mass
    Q = mn - mp # Mass difference between neutrons and protons

    # Born approximation
    def Fn_Born_int(E):
        if (-1. >= E) or (E >= 1):
            return E*(E-(Q/me))**2*np.sqrt(E**2-1.)
        else:
            return 0.
    Fn_Born = quad(Fn_Born_int,1.,Q/me)[0]
    if(PRyMini.nTOpBorn_flag):
        return Fn_Born
    
    # Radiative corrections to Born approximation
    def Fn_rad_int(e):
        b = np.sqrt(e**2-1.)/e
        q = Q/me
        return e*(e-q)**2*e*b*FermiCoulomb(b)*RadCorrResum(np.sqrt(e**2-1.)/e,q-e,e)
    Fn_rad = quad(Fn_rad_int,1.,Q/me)[0]
    
    # Finite mass corrections to Born approximation
    def ChiFMnDec(en,pe):
        mnOme = mn/me
        f1n = ((1.+PRyMini.gA)**2.+2.*PRyMini.deltakappa*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
        f2n = ((1.-PRyMini.gA)**2.-2.*PRyMini.deltakappa*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
        f3n = (PRyMini.gA**2-1.)/(1.+3.*PRyMini.gA**2)
        return  f1n*(en-Q/me)**2*(pe**2/(mnOme*en))-f2n/mnOme*(en-Q/me)**3+(f1n+f2n+f3n)/(2.*mnOme)*(4.*(en-Q/me)**3+2*(en-Q/me)*pe**2)+f3n/mnOme*(en-Q/me)**2*(pe**2)/en
    def Fn_FM_int(pe):
        return pe**2*ChiFMnDec(np.sqrt(pe**2+1.),pe)*RadCorrResum(pe/np.sqrt(pe**2+1.),np.abs(np.sqrt(pe**2+1.)-Q/me),np.sqrt(pe**2+1.))*FermiCoulomb(pe/np.sqrt(pe**2+1.))
    Fn_FM = quad(Fn_FM_int,0.,np.sqrt((Q/me)**2-1.))[0]
    
    # Total correction to neutron decay constant Fn
    Fn = Fn_rad+Fn_FM
    return Fn
    
def ComputeWeakRates(Tvec):
    # Particle masses
    me = PRyMini.me*PRyMini.MeV # electron mass
    mn = PRyMini.mn*PRyMini.MeV # neutron mass
    mp = PRyMini.mp*PRyMini.MeV # proton mass
    Q = mn - mp # Mass difference between neutrons and protons

    # Input from neutrinos
    xi_nu = PRyMini.munuOverTnu # neutrino chemical potential over temperature
    my_dir = PRyMini.working_dir
    if(PRyMini.general_nu_flag):
        import PRyM.PRyM_thermo as PRyMthermo
        Tg_vec = Tvec[0]
        Tg_Kelvin = Tg_vec*PRyMini.MeV_to_Kelvin
        # Compute effective neutrino temperatures for fallback (FMCCR, thermal corrections)
        Tnu_eff = np.array([PRyMthermo.Tnu_eff_e(T) for T in Tg_vec])
        Tnu_of_Tg = Tnu_eff/Tg_vec
        T_nuOverT = interp1d(Tg_Kelvin,Tnu_of_Tg, bounds_error=False, fill_value="extrapolate", kind='linear')
    else:
        Tg_vec,Tnu_vec = Tvec # photon and neutrino temperatures
        Tg_Kelvin = Tg_vec*PRyMini.MeV_to_Kelvin
        Tnu_of_Tg = Tnu_vec/Tg_vec
        T_nuOverT = interp1d(Tg_Kelvin,Tnu_of_Tg, bounds_error=False, fill_value="extrapolate", kind='linear')

    # Auxiliary thermodynamics functions
    def FD_nu3(E,phi,x):
        if((x*E-phi)<exp_cutoff):
            return 1./(np.exp(x*E-phi)+1.)
        else:
            return 0.

    def FD2(E,x):
        if((x*E)<exp_cutoff):
            return 1./(np.exp(x*E)+1.)
        else:
            return 0.
            
    def FD_nu_e2p0(E,phi,x):
        if((x*E-phi)<exp_cutoff):
            return E**2/(np.exp(x*E-phi)+1.)
        else:
            return 0.

    def FD_nu_e3p0(E,phi,x):
        if((x*E-phi)<exp_cutoff):
            return E**3/(np.exp(x*E-phi)+1.)
        else:
            return 0.
            
    def FD_nu_e4p2(E,phi,x):
        if((2.*phi<exp_cutoff) and (E*x+phi<exp_cutoff) and (2.*E*x<exp_cutoff)):
            return (E**2*np.exp(phi)*((24.-E*x*(E*x+8.))*np.exp(E*x+phi)+np.exp(2*E*x)*(E*x-6.)*(E*x-2.)+12*np.exp(2*phi)))/(np.exp(E*x)+np.exp(phi))**3
        else:
            return 0.
            
    def FD_nu_e2p2(E,phi,x):
        if((3.*phi<exp_cutoff) and (2*E*x+phi<exp_cutoff) and (E*x<exp_cutoff)):
            return ((E*x*(E*x-4.)+2.)*np.exp(2*E*x+phi)+(4.-E*x*(E*x+4.))*np.exp(E*x+2*phi)+2*np.exp(3*phi))/(np.exp(E*x)+np.exp(phi))**3
        else:
            return 0.
            
    def FD_nu_e4p1(E,phi,x):
        if ((phi<exp_cutoff) and (E*x<exp_cutoff)):
            return (np.exp(phi)*E**3*(4*np.exp(phi)+np.exp(E*x)*(4.-E*x)))/(np.exp(E*x)+np.exp(phi))**2
        else:
            return 0.
            
    def FD_nu_e2p1(E,phi,x):
        if((phi<exp_cutoff) and (E*x<exp_cutoff)):
            return (np.exp(phi)*E*(2*np.exp(phi)+np.exp(E*x)*(2.-E*x)))/(np.exp(E*x)+np.exp(phi))**2
        else:
            return 0.
            
    def FD_nu_e3p1(E,phi,x):
        if((phi<exp_cutoff) and (E*x<exp_cutoff)):
            return (np.exp(phi)*E**2*(3*np.exp(phi)+np.exp(E*x)*(3.-E*x)))/(np.exp(E*x)+np.exp(phi))**2
        else:
            return 0.
            
    def FD_nu_e3p2(E,phi,x):
        if((2.*phi<exp_cutoff) and (E*x+phi<exp_cutoff) and (2.*E*x<exp_cutoff)):
            return (E*np.exp(phi)*((12.-E*x*(E*x+6.))*np.exp(E*x+phi)+np.exp(2.*E*x)*(E*x*(E*x-6.)+6.)+6*np.exp(2.*phi)))/(np.exp(E*x)+np.exp(phi))**3
        else:
            return 0.
            
    def D_FD2(E, x):
        if((x*E)<exp_cutoff):
            return -x*np.exp(x*E)/(1.+np.exp(x*E))**2
        else:
            return 0.
            
    # General neutrino distribution function accessors (dimensionless energy in units of me)
    if(PRyMini.general_nu_flag):
        def f_nu_general_dimless(E_nu_dimless, Tg_MeV):
            """Electron neutrino distribution as function of dimensionless energy E = p/me.
            Returns f_nue(p, Tg) where p = |E| * me [MeV]."""
            p_MeV = np.abs(E_nu_dimless) * PRyMini.me
            return PRyMthermo.f_nue_general(p_MeV, Tg_MeV)
        def f_nubar_general_dimless(E_nu_dimless, Tg_MeV):
            """Electron antineutrino distribution as function of dimensionless energy E = p/me."""
            p_MeV = np.abs(E_nu_dimless) * PRyMini.me
            return PRyMthermo.f_nuebar_general(p_MeV, Tg_MeV)

        def f_eff_nu(E_nu, Tg_MeV, sgnq):
            """Effective neutrino distribution for arbitrary neutrino energy E_nu (dimensionless).
            Handles the nu/nubar dispatch and crossing (emission = 1-f):
              E_nu > 0, sgnq=+1: f_nue (absorption)
              E_nu < 0, sgnq=+1: 1 - f_nuebar (emission)
              E_nu > 0, sgnq=-1: f_nuebar (absorption)
              E_nu < 0, sgnq=-1: 1 - f_nue (emission)
            """
            if E_nu >= 0:
                if sgnq > 0:
                    return f_nu_general_dimless(E_nu, Tg_MeV)
                else:
                    return f_nubar_general_dimless(E_nu, Tg_MeV)
            else:
                if sgnq > 0:
                    return 1. - f_nubar_general_dimless(E_nu, Tg_MeV)
                else:
                    return 1. - f_nu_general_dimless(E_nu, Tg_MeV)

        def f_eff_nu_vec(E_nu_arr, Tg_MeV, sgnq):
            """Vectorized effective neutrino distribution for numpy arrays."""
            p_MeV = np.abs(E_nu_arr) * PRyMini.me
            pos_mask = E_nu_arr >= 0
            result = np.zeros_like(E_nu_arr, dtype=float)
            if sgnq > 0:
                if np.any(pos_mask):
                    result[pos_mask] = PRyMthermo.f_nue_general(p_MeV[pos_mask], Tg_MeV)
                if np.any(~pos_mask):
                    result[~pos_mask] = 1. - PRyMthermo.f_nuebar_general(p_MeV[~pos_mask], Tg_MeV)
            else:
                if np.any(pos_mask):
                    result[pos_mask] = PRyMthermo.f_nuebar_general(p_MeV[pos_mask], Tg_MeV)
                if np.any(~pos_mask):
                    result[~pos_mask] = 1. - PRyMthermo.f_nue_general(p_MeV[~pos_mask], Tg_MeV)
            return result

        # Numerical derivatives of E^A * f_eff(E) for finite-mass corrections
        _dE_rel = 1.e-4  # relative step for finite differences
        _dE_min = 1.e-6  # minimum absolute step

        def _EA_feff(E, A, Tg_MeV, sgnq):
            """E^A * f_eff_nu(E, Tg_MeV, sgnq)"""
            return E**A * f_eff_nu(E, Tg_MeV, sgnq)

        def FD_nu_eApB_general(E, A, B, Tg_MeV, sgnq):
            """General version of FD_nu_eApB: d^B/dE^B [E^A * f_eff(E)].
            Computed via finite differences for B > 0."""
            if B == 0:
                return _EA_feff(E, A, Tg_MeV, sgnq)
            dE = max(_dE_rel * np.abs(E), _dE_min)
            if B == 1:
                return (_EA_feff(E + dE, A, Tg_MeV, sgnq)
                      - _EA_feff(E - dE, A, Tg_MeV, sgnq)) / (2.*dE)
            elif B == 2:
                return (_EA_feff(E + dE, A, Tg_MeV, sgnq)
                      - 2.*_EA_feff(E, A, Tg_MeV, sgnq)
                      + _EA_feff(E - dE, A, Tg_MeV, sgnq)) / (dE**2)
            else:
                raise ValueError("FD_nu_eApB_general: B > 2 not implemented")

    # Born rates given by Eq 2.29 in Brown & Sawyer
    if(PRyMini.general_nu_flag):
        def ChiFunc(E, p, x, znu_or_Tg, sgnq):
            """Response function for general neutrino distributions.
            znu_or_Tg is Tg in MeV when general_nu_flag is True."""
            Tg_MeV = znu_or_Tg
            E_nu = E - sgnq*(Q/me)
            if E_nu >= 0:
                # Neutrino/antineutrino absorption
                if sgnq > 0:
                    f_val = f_nu_general_dimless(E_nu, Tg_MeV)
                else:
                    f_val = f_nubar_general_dimless(E_nu, Tg_MeV)
            else:
                # Antineutrino/neutrino emission: Pauli blocking factor (1 - f)
                if sgnq > 0:
                    f_val = 1. - f_nubar_general_dimless(E_nu, Tg_MeV)
                else:
                    f_val = 1. - f_nu_general_dimless(E_nu, Tg_MeV)
            return f_val * FD2(-E, x) * E_nu**2
    else:
        def ChiFunc(E, p, x, znu, sgnq):
            return FD_nu3(E-sgnq*(Q/me),sgnq*xi_nu,znu)*FD2(-E,x)*(E-sgnq*(Q/me))**2

    # Integrands in electron momentum, w/o and w/ radiative corrections
    def IPENdpFrom_Chi_NoCCR(E, p, x, znu, sgnq):
        return p**2*(ChiFunc(E, p, x, znu, sgnq) + ChiFunc(-E, p, x, znu, sgnq))

    def FermiStat(sgnq, sgnE, b):
        if (sgnq*sgnE) > 0:
            return FermiCoulomb(b)
        else:
            return 1.

    def IPENdp(p, x, znu, sgnq):
         eOFpe = np.sqrt(p**2+1.)
         return IPENdpFrom_Chi_NoCCR(eOFpe, p, x, znu, sgnq)

    # Born rates given by Eq 2.30 in Brown & Sawyer
    if(PRyMini.general_nu_flag):
        def L_nTOpBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdp(p,x,Tg_MeV,1)
        def L_pTOnBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdp(p,x,Tg_MeV,-1)
    else:
        def L_nTOpBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdp(p,x,xnu,1)
        def L_pTOnBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdp(p,x,xnu,-1)

    def L_nTOpBORN(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_nTOpBORN_int, pemin, pemax, args=(T), epsrel = epsrel_low)[0]
    def L_pTOnBORN(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_pTOnBORN_int, pemin, pemax, args=(T), epsrel = epsrel_low)[0]

    # Finite mass effects
    def M(sgnq):
        return (mp+mn-sgnq*Q)/(2*me)

    def enu(en, sgnq):
        return en-sgnq*Q/me

    if(PRyMini.general_nu_flag):
        def ChiFunc_FM(en, pe, x, znu_or_Tg, sgnq):
            """Finite-mass correction response function for general neutrino distributions.
            znu_or_Tg is Tg in MeV when general_nu_flag is True."""
            Tg_MeV = znu_or_Tg
            Mp = mp/me
            Mn = mn/me
            M_sgnq = (mp+mn-sgnq*Q)/(2*me)
            f_1 = ((1.+sgnq*PRyMini.gA)**2.+2.*PRyMini.deltakappa*sgnq*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
            f_2 = ((1.-sgnq*PRyMini.gA)**2.-2.*PRyMini.deltakappa*sgnq*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
            f_3 = (PRyMini.gA**2-1.)/(1.+3.*PRyMini.gA**2)
            FD2_en = FD2(-en,x)
            Enu = enu(en,sgnq)
            # Shorthand for d^B/dE^B [E^A * f_eff(E)] evaluated at Enu
            def G(A,B):
                return FD_nu_eApB_general(Enu, A, B, Tg_MeV, sgnq)
            return (f_1*G(2,0)*FD2_en*(pe**2/(M_sgnq*en))
                + f_2*G(3,0)*FD2_en*(-(1./M_sgnq))
                + (f_1+f_2+f_3)/(2.*x*M_sgnq)*(G(4,2)*FD2_en + G(2,2)*FD2_en*pe**2)
                + (f_1+f_2+f_3)/(2.*M_sgnq)*(G(4,1)*FD2_en + G(2,1)*FD2_en*pe**2)
                - (f_1+f_2)/(x*M_sgnq)*(G(3,1)*FD2_en + G(2,1)*FD2_en*pe**2/(-en))
                - f_3*3./(x*M_sgnq)*G(2,0)*FD2_en
                + f_3/(3*M_sgnq)*G(3,1)*FD2_en*pe**2/en
                + f_3* 2./(2.*x*3.*M_sgnq)*G(3,2)*FD2_en*pe**2/en
                - (f_1+f_2+f_3)*3./(2.*x)*(1.-(Mn/Mp)**sgnq)*(G(2,1)*FD2_en))
    else:
        def ChiFunc_FM(en, pe, x, znu, sgnq):
            Mp = mp/me
            Mn = mn/me
            M_sgnq = (mp+mn-sgnq*Q)/(2*me)
            f_1 = ((1.+sgnq*PRyMini.gA)**2.+2.*PRyMini.deltakappa*sgnq*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
            f_2 = ((1.-sgnq*PRyMini.gA)**2.-2.*PRyMini.deltakappa*sgnq*PRyMini.gA)/(1.+3.*PRyMini.gA**2)
            f_3 = (PRyMini.gA**2-1.)/(1.+3.*PRyMini.gA**2)
            FD2_en = FD2(-en,x)
            return (f_1*FD_nu_e2p0(enu(en,sgnq),0,znu)*FD2_en*(pe**2/(M_sgnq*en))
                + f_2*FD_nu_e3p0(enu(en,sgnq),0,znu)*FD2_en*(-(1./M_sgnq))
                + (f_1+f_2+f_3)/(2.*x*M_sgnq)*(FD_nu_e4p2(enu(en,sgnq),0,znu)*FD2_en + FD_nu_e2p2(enu(en, sgnq),0,znu)*FD2_en*pe**2)
                + (f_1+f_2+f_3)/(2.*M_sgnq)*(FD_nu_e4p1(enu(en, sgnq),0,znu)*FD2_en + FD_nu_e2p1(enu(en,sgnq),0,znu)*FD2_en*pe**2)
                - (f_1+f_2)/(x*M_sgnq)*(FD_nu_e3p1(enu(en,sgnq),0,znu)*FD2_en + FD_nu_e2p1(enu(en,sgnq),0,znu)*FD2_en*pe**2/(-en))
                - f_3*3./(x*M_sgnq)*FD_nu_e2p0(enu(en, sgnq),0,znu)*FD2_en
                + f_3/(3*M_sgnq)*FD_nu_e3p1(enu(en, sgnq),0,znu)*FD2_en*pe**2/en
                + f_3* 2./(2.*x*3.*M_sgnq)*FD_nu_e3p2(enu(en, sgnq),0,znu)*FD2_en*pe**2/en
                - (f_1+f_2+f_3)*3./(2.*x)*(1.-(Mn/Mp)**sgnq)*(FD_nu_e2p1(enu(en, sgnq),0, znu)*FD2_en))

    def IPENdpFMCCR(p, x, znu, sgnq):
        eOFpe = np.sqrt(p**2+1.)
        en_ratio = p/eOFpe
        return p**2*(ChiFunc_FM(eOFpe,p,x,znu,sgnq)*RadCorrResum(en_ratio, np.abs(sgnq*Q/me-eOFpe),eOFpe)*FermiStat(sgnq,1,en_ratio) +
        ChiFunc_FM(-eOFpe,p,x,znu,sgnq)*RadCorrResum(en_ratio, np.abs(sgnq*Q/me+eOFpe),eOFpe)*FermiStat(sgnq,-1,en_ratio))

    if(PRyMini.general_nu_flag):
        def L_nTOpFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpFMCCR(p,x,Tg_MeV,1)
        def L_pTOnFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpFMCCR(p,x,Tg_MeV,-1)
    else:
        def L_nTOpFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpFMCCR(p,x,xnu,1)
        def L_pTOnFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpFMCCR(p,x,xnu,-1)

    def L_nTOpFMCCR(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_nTOpFMCCR_int,pemin,pemax, args=(T), epsrel = epsrel_low)[0]
    def L_pTOnFMCCR(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_pTOnFMCCR_int, pemin, pemax, args=(T), epsrel = epsrel_low)[0]

    # Radiative Corrections (T=0)
    def IPENdpFrom_Chi_CCR(E, p, x, znu, sgnq):
        return p**2*(ChiFunc(E, p, x, znu, sgnq)*RadCorrResum(p/E, np.abs(sgnq*Q/me - E), E)*FermiStat(sgnq, 1, p/E) + ChiFunc(-E, p, x, znu, sgnq)*RadCorrResum(p/E, np.abs(sgnq*Q/me + E), E)*FermiStat(sgnq, -1, p/E))

    def IPENdpCCR(p, x, znu, sgnq):
        eOFpe = np.sqrt(p**2+1.)
        return IPENdpFrom_Chi_CCR(eOFpe, p, x, znu, sgnq)

    if(PRyMini.general_nu_flag):
        def L_nTOpCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpCCR(p,x,Tg_MeV,1)
        def L_pTOnCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpCCR(p,x,Tg_MeV,-1)
    else:
        def L_nTOpCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpCCR(p,x,xnu,1)
        def L_pTOnCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpCCR(p,x,xnu,-1)

    def L_nTOpCCR(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_nTOpCCR_int, pemin, pemax, args=(T), epsrel = epsrel_low)[0]
    def L_pTOnCCR(T):
        pemin = 0.
        x = me/(PRyMini.kB*T)
        pemax = max(7.,30./x)
        return quad(L_pTOnCCR_int, pemin, pemax, args=(T), epsrel = epsrel_low)[0]

    # Finite-temperature Radiative Corrections
    # Brown & Sawyer for finite temperature radiative corrections + Brehmstrahlung (Eqs. 107)
    if(PRyMini.compute_nTOp_thermal_flag):
        if(PRyMini.general_nu_flag):
            def Chitilde(en, znu_or_Tg, sgnq):
                q = Q/me
                E_nu = en - sgnq*q
                return f_eff_nu(E_nu, znu_or_Tg, sgnq) * E_nu**2
        else:
            def Chitilde(en, znu, sgnq):
                q = Q/me
                return FD_nu3(en-sgnq*q,sgnq*xi_nu,znu)*(en-sgnq*q)**2

        def A(E, k):
            pE = np.sqrt(E**2 - 1.)
            return (2.*E**2 + k**2)*(np.log((E + pE)/(E - pE))) - 4.*pE*E
            
        def B(E):
            pE = np.sqrt(E**2 - 1.)
            return 2.*E*(np.log((E + pE)/(E - pE))) - 4.*pE

        def IPENCCRT(E, k, x, znu, sgnq):
            pE = np.sqrt(E**2-1.)
            def BE(EkBT):
                resvec = np.zeros(len(EkBT))
                argvec = EkBT
                my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])-1.)
                return resvec
            def FD2(en, xval):
                resvec = np.zeros(len(en))
                argvec = en*xval
                my_index = np.where(np.abs(argvec)<=exp_cutoff)[0]
                resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])+1.)
                my_index_overflow = np.where(np.abs(argvec)>exp_cutoff)[0]
                resvec[my_index_overflow[:]] = 1./(np.exp(np.sign(argvec[my_index_overflow[:]])*exp_cutoff)+1.)
                return resvec
            if(PRyMini.general_nu_flag):
                def Chitilde(en, znuval_or_Tg, sgnq):
                    q = Q/me
                    E_nu = en - sgnq*q
                    return f_eff_nu_vec(E_nu, znuval_or_Tg, sgnq) * E_nu**2
            else:
                def Chitilde(en, znuval, sgnq):
                    q = Q/me
                    resvec = np.zeros(len(en))
                    argvec = znuval*(en-sgnq*q) - (sgnq*xi_nu)
                    my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                    resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]]) + 1.)
                    return resvec*(en-sgnq*q)**2
            return  PRyMini.alphaem/(2*np.pi)*(BE(x*k)/k)*(A(E, k)*(FD2(-E,x)*FermiStat(sgnq, 1, pE/E)*(Chitilde(E - k, znu, sgnq) + Chitilde(E + k, znu, sgnq) - 2*Chitilde(E, znu, sgnq))+ FD2(E, x)*FermiStat(sgnq, -1, pE/E)*(Chitilde(-E + k, znu, sgnq) + Chitilde(-E - k, znu, sgnq) - 2*Chitilde(-E, znu, sgnq)))-k*B(E)*(FD2(-E, x)*FermiStat(sgnq, 1, pE/E)*(Chitilde(E - k, znu, sgnq) - Chitilde(E + k, znu, sgnq))+ FD2(E, x)*FermiStat(sgnq, -1, pE/E)*(Chitilde(-E + k, znu, sgnq)- Chitilde(-E - k, znu, sgnq))))

        # Bremsstrahlung corrections
        def IPENCCRDiffBremsstrahlung(E, k, x, znu, sgnq):
            q = Q/me
            pE = np.sqrt(E**2-1.)
            Fp = (2.*E**2+k**2)*(np.log((E+pE)/(E-pE)))-4.*pE*E
            Fp += k*(2.*E*(np.log((E+pE)/(E-pE)))-4.*pE)
            Fm = (2.*E**2+k**2)*(np.log((E+pE)/(E-pE)))-4.*pE*E
            Fm -= k*(2.*E*(np.log((E+pE)/(E-pE)))-4.*pE)
            def FD2(en, xval):
                resvec = np.zeros(len(en))
                argvec = en*xval
                my_index = np.where(np.abs(argvec)<=exp_cutoff)[0]
                resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])+1.)
                my_index_overflow = np.where(np.abs(argvec)>exp_cutoff)[0]
                resvec[my_index_overflow[:]] = 1./(np.exp(np.sign(argvec[my_index_overflow[:]])*exp_cutoff)+1.)
                return resvec
            if(PRyMini.general_nu_flag):
                def Chitilde(en, znuval_or_Tg, sgnq):
                    q = Q/me
                    E_nu = en - sgnq*q
                    return f_eff_nu_vec(E_nu, znuval_or_Tg, sgnq) * E_nu**2
            else:
                def Chitilde(en, znuval, sgnq):
                    q = Q/me
                    resvec = np.zeros(len(en))
                    argvec = znuval*(en-sgnq*q) - (sgnq*xi_nu)
                    my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                    resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]]) + 1.)
                    return resvec*(en-sgnq*q)**2
            if(PRyMini.general_nu_flag):
                def f_nu_dist(E_nu, znuval_or_Tg, sgnq):
                    return f_eff_nu_vec(E_nu, znuval_or_Tg, sgnq)
            else:
                def f_nu_dist(E_nu, znuval, sgnq):
                    resvec = np.zeros(len(E_nu))
                    argvec = E_nu*znuval
                    my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                    resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])+1.)
                    return resvec
            res_fac = PRyMini.alphaem/(2.*np.pi*k)
            res1_fac = FD2(-E,x)*FermiStat(sgnq,1,pE/E)
            res1vec = Fp*Chitilde(E+k,znu,sgnq)
            argvec = k
            my_index = np.where(np.abs(argvec)<np.abs(E-sgnq*q))[0]
            res1vec[my_index[:]] -= Fp[my_index[:]]*f_nu_dist(E[my_index[:]]-sgnq*q,znu,sgnq)* (np.abs(E[my_index[:]]-sgnq*q)-k[my_index[:]])**2
            res1vec[:] *= res1_fac[:]
            res2_fac = FD2(E,x)*FermiStat(sgnq,-1,pE/E)
            res2vec = Fm*Chitilde(-E+k,znu,sgnq)
            my_index = np.where(np.abs(argvec)<np.abs(E+sgnq*q))[0]
            res2vec[my_index[:]] -= Fp[my_index[:]]*f_nu_dist(-E[my_index[:]]-sgnq*q,znu,sgnq)*(np.abs(E[my_index[:]]+sgnq*q) -k[my_index[:]])**2
            res2vec[:] *= res2_fac[:]
            return res_fac*(res1vec+res2vec)

        # Mass shift and ep + ee corrections, Eq. 5.15 - 5.16 Brown & Sawyer
        def C1dE(E, x, znu, sgnq):
            pE = np.sqrt(E**2-1.)
            return -((PRyMini.alphaem*E)/(2.*np.pi*pE))*(2.*np.pi**2)/(3.*x**2)*(ChiFunc(E,pE,x,znu,sgnq)+ ChiFunc(-E,pE,x, znu,sgnq))

        def C2dE1dE2(e1v, e2v, x, znu, sgnq):
            resvec = np.zeros(len(e1v))
            e1pe2 = e1v+e2v
            e1me2 = e1v-e2v
            min_e1pe2 = 2.+np.abs(e1me2)
            max_e1pe2 = 2.+max(10.,15./x)+np.abs(e1me2)
            index_limits = np.where(((e1pe2-min_e1pe2)>0)*((max_e1pe2-e1pe2)>0))[0]
            def FD2(en, xval):
                resvec = np.zeros(len(en))
                argvec = en*xval
                my_index = np.where(np.abs(argvec)<=exp_cutoff)[0]
                resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])+1.)
                my_index_overflow = np.where(np.abs(argvec)>exp_cutoff)[0]
                resvec[my_index_overflow[:]] = 1./(np.exp(np.sign(argvec[my_index_overflow[:]])*exp_cutoff)+1.)
                return resvec
            def D_FD2(en, xval):
                resvec = np.zeros(len(en))
                argvec = en*xval
                my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                resvec[my_index[:]] = -xval*np.exp(argvec[my_index[:]])/(np.exp(argvec[my_index[:]])+1.)**2
                return resvec
            if(PRyMini.general_nu_flag):
                def ChiFunc(E, p, x, znu_or_Tg, sgnq):
                    E_nu = E - sgnq*(Q/me)
                    return f_eff_nu_vec(E_nu, znu_or_Tg, sgnq)*FD2(-E,x)*E_nu**2
            else:
                def FD_nu3(en, phi, xval):
                    resvec = np.zeros(len(en))
                    argvec = en*xval-phi
                    my_index = np.where(np.abs(argvec)<exp_cutoff)[0]
                    resvec[my_index[:]] = 1./(np.exp(argvec[my_index[:]])+1.)
                    return resvec
                def ChiFunc(E, p, x, znu, sgnq):
                    return FD_nu3(E-sgnq*(Q/me),sgnq*xi_nu,znu)*FD2(-E,x)*(E-sgnq*(Q/me))**2
            #safe_check = np.where((np.abs(p1-p2)>0)*(np.abs(p1)>0)*(np.abs(e2)>0)*(np.abs(p2)>0) *(np.abs(e1)>0))[0]
            e1 = e1v[index_limits[:]]
            e2 = e2v[index_limits[:]]
            p1 = np.sqrt(e1v[index_limits[:]]**2 - 1.)
            p2 = np.sqrt(e2v[index_limits[:]]**2 - 1.)
            L_fac = np.log((e1*e2+p1*p2+1.)/(e1*e2-p1*p2+1.))
            resvec_limits = PRyMini.alphaem/(2.*np.pi)*(ChiFunc(e1,p1,x,znu,sgnq)+ChiFunc(-e1,p1,x,znu,sgnq)) *(-(1./4.)*np.log(((p1+p2)/(p1-p2))**2)*np.log(((p1+p2)/(p1-p2))**2)*(D_FD2(e2, x)*p2/p1*e1**2/e2*(e1+e2)+FD2(e2,x)*e1**2/(p1*p2)*(e2+e1/e2**2))+np.log(((p1+ p2)/(p1-p2))**2)*(D_FD2(e2,x)*(p2**2*e1/e2*(1./p1**2+2.) -e1**2*p2/p1*L_fac)+FD2(e2,x)*(e1/(p1**2*e2**2)*(e2**2+2*p1**2+1.)-(e1**2+e2**2)/(e1+e2)-(e1**2*e2)/(p1*p2)*L_fac))-FD2(e2,x) *(4.*e1*p2/p1+2.*e2*L_fac))
            resvec[index_limits[:]] = resvec_limits[:]
            return resvec
            
        ##################################################################
        ######## TruePhoton -> real photon emission processes     ########
        ######## DiffBremsstrahlung -> bremsstrahlung corrections ########
        ######## Thermal -> mass shift and pe+ee corrections      ########
        ##################################################################
            
        if(PRyMini.general_nu_flag):
            def L_nTOpThermalTruePhoton_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return IPENCCRT(E, k, x, Tg_MeV, 1)
        else:
            def L_nTOpThermalTruePhoton_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return IPENCCRT(E, k, x, xnu, 1)
            
        def L_nTOpThermalTruePhoton(T):
            x = me/(PRyMini.kB*T)
            min_E = 1.001
            max_E = max(10.,20./x)
            min_k = 0.001
            max_k = max(10.,20./(me/(PRyMini.kB*T)))
            integ = vegas.Integrator([[min_E,max_E],[min_k,max_k]])
            @vegas.batchintegrand
            def f_batch(x):
                global store
                E_val,k_val = np.transpose(x)
                return {'myres': L_nTOpThermalTruePhoton_int(E_val,k_val,T)}
            training = integ(f_batch, nitn=n_itn, neval=n_eval)
            result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
            return result['myres'].mean
            
        if(PRyMini.general_nu_flag):
            def L_nTOpThermalDiffBremsstrahlung_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return IPENCCRDiffBremsstrahlung(E, k, x, Tg_MeV, 1)
        else:
            def L_nTOpThermalDiffBremsstrahlung_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return IPENCCRDiffBremsstrahlung(E, k, x, xnu, 1)
            
        def L_nTOpThermalDiffBremsstrahlung(T):
            min_E = 1.001
            max_E = max(10.,20./(me/(PRyMini.kB*T)))
            min_k = 0.001
            max_k = max(10.,20./(me/(PRyMini.kB*T)))
            integ = vegas.Integrator([[min_E,max_E],[min_k,max_k]])
            @vegas.batchintegrand
            def f_batch(x):
                global store
                E_val,k_val = np.transpose(x)
                return {'myres': L_nTOpThermalDiffBremsstrahlung_int(E_val,k_val,T)}
            training = integ(f_batch, nitn=n_itn, neval=n_eval)
            result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
            return result['myres'].mean

        if(PRyMini.general_nu_flag):
            def L_nTOpThermal_1_int(E, T):
                return C1dE(E, me/(PRyMini.kB*T), PRyMini.kB*T/PRyMini.MeV, 1)
        else:
            def L_nTOpThermal_1_int(E, T):
                return C1dE(E, me/(PRyMini.kB*T), me/(PRyMini.kB*T*T_nuOverT(T)), 1)

        def L_nTOpThermal_1(T):
            return quad(L_nTOpThermal_1_int, 1., max(25., 150.*(PRyMini.kB*T)/me), args=(T), epsrel = 1.e-2)[0]
            
        if(PRyMini.general_nu_flag):
            def L_nTOpThermal_2_3_int(e1pe2, e1me2, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return 0.5*C2dE1dE2((e1pe2+e1me2)/2.,(e1pe2-e1me2)/2., x, Tg_MeV, 1)
        else:
            def L_nTOpThermal_2_3_int(e1pe2, e1me2, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return 0.5*C2dE1dE2((e1pe2+e1me2)/2.,(e1pe2-e1me2)/2., x, xnu, 1)
            
        def L_nTOpThermal_2_3(T):
            x = me/(PRyMini.kB*T)
            # res_2
            min_e1me1 = -max(10.,15./x)
            max_e1me2 = -0.001
            min_e1pe2 = 2.001+min(np.abs(min_e1me1),np.abs(max_e1me2))
            max_e1pe2 = 2.+max(np.abs(min_e1me1),np.abs(max_e1me2))
            integ_2 = vegas.Integrator([[min_e1pe2,max_e1pe2],[min_e1me1,max_e1me2]])
            @vegas.batchintegrand
            def f_batch_2(x):
                global store
                e1pe2,e1me2 = np.transpose(x)
                return {'myres': L_nTOpThermal_2_3_int(e1pe2,e1me2,T)}
            training_2 = integ_2(f_batch_2, nitn=n_itn, neval=n_eval)
            result_2 = integ_2(f_batch_2, nitn=n_itn, neval=n_eval, adapt=True)
            res_2 = result_2['myres'].mean
            # res_3
            min_e1me1 = 0.001
            max_e1me2 = max(10.,15./x)
            min_e1pe2 = 2.001+min(np.abs(min_e1me1),np.abs(max_e1me2))
            max_e1pe2 = 2.+max(np.abs(min_e1me1),np.abs(max_e1me2))
            integ_3 = vegas.Integrator([[min_e1pe2,max_e1pe2],[min_e1me1,max_e1me2]])
            @vegas.batchintegrand
            def f_batch_3(x):
                global store
                e1pe2,e1me2 = np.transpose(x)
                return {'myres': L_nTOpThermal_2_3_int(e1pe2,e1me2,T)}
            training_3 = integ_3(f_batch_3, nitn=n_itn, neval=n_eval)
            result_3 = integ_3(f_batch_3, nitn=n_itn, neval=n_eval, adapt=True)
            res_3 = result_3['myres'].mean
            return res_2+res_3
            
        def L_nTOpThermal_tot(T):
            return L_nTOpThermal_1(T)+L_nTOpThermal_2_3(T)

        ####################
        # p -> n processes #
        ####################
        # p -> n real photon corrections
        if(PRyMini.general_nu_flag):
            def L_pTOnThermalTruePhoton_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return IPENCCRT(E, k, x, Tg_MeV, -1)
        else:
            def L_pTOnThermalTruePhoton_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return IPENCCRT(E, k, x, xnu, -1)
            
        def L_pTOnThermalTruePhoton(T):
            x = me/(PRyMini.kB*T)
            min_E = 1.001
            max_E = max(10.,20./x)
            min_k = 0.001
            max_k = max(10.,20./(me/(PRyMini.kB*T)))
            integ = vegas.Integrator([[min_E,max_E],[min_k,max_k]])
            @vegas.batchintegrand
            def f_batch(x):
                global store
                E_val,k_val = np.transpose(x)
                return {'myres': L_pTOnThermalTruePhoton_int(E_val,k_val,T)}
            training = integ(f_batch, nitn=n_itn, neval=n_eval)
            result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
            return result['myres'].mean
            
        # p -> n brems corrections
        if(PRyMini.general_nu_flag):
            def L_pTOnThermalDiffBremsstrahlung_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return IPENCCRDiffBremsstrahlung(E, k, x, Tg_MeV, -1)
        else:
            def L_pTOnThermalDiffBremsstrahlung_int(E, k, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return IPENCCRDiffBremsstrahlung(E, k, x, xnu, -1)
            
        def L_pTOnThermalDiffBremsstrahlung(T):
            min_E = 1.001
            max_E = max(10.,20./(me/(PRyMini.kB*T)))
            min_k = 0.001
            max_k = max(10.,20./(me/(PRyMini.kB*T)))
            integ = vegas.Integrator([[min_E,max_E],[min_k,max_k]])
            @vegas.batchintegrand
            def f_batch(x):
                global store
                E_val,k_val = np.transpose(x)
                return {'myres': L_pTOnThermalDiffBremsstrahlung_int(E_val,k_val,T)}
            training = integ(f_batch, nitn=n_itn, neval=n_eval)
            result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
            return result['myres'].mean
            
        # p -> n mass shift + pe+ee corrections
        if(PRyMini.general_nu_flag):
            def L_pTOnThermal_1_int(E, T):
                return C1dE(E, me/(PRyMini.kB*T), PRyMini.kB*T/PRyMini.MeV, -1)
        else:
            def L_pTOnThermal_1_int(E, T):
                return C1dE(E, me/(PRyMini.kB*T), me/(PRyMini.kB*T*T_nuOverT(T)), -1)

        def L_pTOnThermal_1(T):
            return quad(L_pTOnThermal_1_int, 1., max(25., 150.*(PRyMini.kB*T)/me), args=(T), epsrel = 1.e-2)[0]
            
        if(PRyMini.general_nu_flag):
            def L_pTOnThermal_2_3_int(e1pe2, e1me2, T):
                x  = me/(PRyMini.kB*T)
                Tg_MeV = PRyMini.kB*T/PRyMini.MeV
                return 0.5*C2dE1dE2((e1pe2+e1me2)/2.,(e1pe2-e1me2)/2., x, Tg_MeV, -1)
        else:
            def L_pTOnThermal_2_3_int(e1pe2, e1me2, T):
                x  = me/(PRyMini.kB*T)
                xnu = me/(PRyMini.kB*T*T_nuOverT(T))
                return 0.5*C2dE1dE2((e1pe2+e1me2)/2.,(e1pe2-e1me2)/2., x, xnu, -1)
            
        def L_pTOnThermal_2_3(T):
            x = me/(PRyMini.kB*T)
            # res_2
            min_e1me1 = -max(10.,15./x)
            max_e1me2 = -0.001
            min_e1pe2 = 2.001+min(np.abs(min_e1me1),np.abs(max_e1me2))
            max_e1pe2 = 2.+max(np.abs(min_e1me1),np.abs(max_e1me2))
            integ_2 = vegas.Integrator([[min_e1pe2,max_e1pe2],[min_e1me1,max_e1me2]])
            @vegas.batchintegrand
            def f_batch_2(x):
                global store
                e1pe2,e1me2 = np.transpose(x)
                return {'myres': L_pTOnThermal_2_3_int(e1pe2,e1me2,T)}
            training_2 = integ_2(f_batch_2, nitn=n_itn, neval=n_eval)
            result_2 = integ_2(f_batch_2, nitn=n_itn, neval=n_eval, adapt=True)
            res_2 = result_2['myres'].mean
            # res_3
            min_e1me1 = 0.001
            max_e1me2 = max(10.,15./x)
            min_e1pe2 = 2.001+min(np.abs(min_e1me1),np.abs(max_e1me2))
            max_e1pe2 = 2.+max(np.abs(min_e1me1),np.abs(max_e1me2))
            integ_3 = vegas.Integrator([[min_e1pe2,max_e1pe2],[min_e1me1,max_e1me2]])
            @vegas.batchintegrand
            def f_batch_3(x):
                global store
                e1pe2,e1me2 = np.transpose(x)
                return {'myres': L_pTOnThermal_2_3_int(e1pe2,e1me2,T)}
            training_3 = integ_3(f_batch_3, nitn=n_itn, neval=n_eval)
            result_3 = integ_3(f_batch_3, nitn=n_itn, neval=n_eval, adapt=True)
            res_3 = result_3['myres'].mean
            return res_2+res_3
        
        def L_pTOnThermal_tot(T):
            return L_pTOnThermal_1(T)+L_pTOnThermal_2_3(T)

        # Gathering all thermal corrections together
        def L_nTOpCCRTh(T):
            L_n_p_real_photon = L_nTOpThermalTruePhoton(T)
            L_n_p_thermal_brems = L_nTOpThermalDiffBremsstrahlung(T)
            L_n_p_thermal_mass = L_nTOpThermal_tot(T)
            return L_n_p_real_photon+L_n_p_thermal_brems+L_n_p_thermal_mass
        def L_pTOnCCRTh(T):
            T_threshold = 10**(8.2)
            if(T< T_threshold):
                return 0.
            else:
                L_p_n_real_photon = L_pTOnThermalTruePhoton(T)
                L_p_n_thermal_brems = L_pTOnThermalDiffBremsstrahlung(T)
                L_p_n_thermal_mass = L_pTOnThermal_tot(T)
            return L_p_n_real_photon+L_p_n_thermal_brems+L_p_n_thermal_mass
    
        if(PRyMini.verbose_flag):
            print(" ")
            print("Re-evaluating n <--> p thermal corrections")
            print("This computation may take a while ...")
        L_nTOpCCRTh_vec = np.vectorize(L_nTOpCCRTh)
        L_pTOnCCRTh_vec = np.vectorize(L_pTOnCCRTh)
        T_nTOp_thermal_interval = np.logspace(np.log10(PRyMini.T_end),np.log10(PRyMini.T_start),PRyMini.sampling_nTOp_thermal)
        L_nTOpCCRTh_res = L_nTOpCCRTh_vec(T_nTOp_thermal_interval)
        L_pTOnCCRTh_res = L_pTOnCCRTh_vec(T_nTOp_thermal_interval)
        if(PRyMini.save_nTOp_thermal_flag):
            np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_thermal_corrections.txt",np.c_[T_nTOp_thermal_interval,L_nTOpCCRTh_res])
            np.savetxt(my_dir+"/PRyMrates/nTOp/"+"pTOn_thermal_corrections.txt",np.c_[T_nTOp_thermal_interval,L_pTOnCCRTh_res])
        if(PRyMini.verbose_flag):
            print("n <--> p thermal corrections computed")
    else:
        T_nTOp_thermal_interval, L_nTOpCCRTh_res = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_thermal_corrections.txt", unpack = True)
        T_nTOp_thermal_interval, L_pTOnCCRTh_res = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"pTOn_thermal_corrections.txt", unpack = True)
    ################################
    # Splining thermal corrections #
    ################################
    L_nTOpCCRTh_interp = interp1d(T_nTOp_thermal_interval[:],L_nTOpCCRTh_res[:],bounds_error=False,fill_value="extrapolate",kind='quadratic')
    L_pTOnCCRTh_interp = interp1d(T_nTOp_thermal_interval[:],L_pTOnCCRTh_res[:],bounds_error=False,fill_value="extrapolate",kind='quadratic')

    def nTOp_frwrd_(T):
        rate_nTOp = 0.
        # pure Born approximation
        if(PRyMini.nTOpBorn_flag):
            rate_nTOp = L_nTOpBORN(T)
        else:
            # T=0 Born w/ radiative corrections
            L_nTOp_T = L_nTOpCCR(T)
            # finite nucleon mass effects
            L_nTOp_T += L_nTOpFMCCR(T)
            # interpolated thermal corrections
            L_nTOp_T += L_nTOpCCRTh_interp(T)
            # total n -> p rate
            rate_nTOp = (L_nTOp_T)
        if(PRyMini.NP_nTOp_flag):
            rate_nTOp += PRyMini.NP_delta_nTOp*L_nTOpBORN(T)
        return rate_nTOp # to be multiplied by [s-1]
       
    def nTOp_bkwrd_(T):
        rate_pTOn = 0.
                # pure Born approximation
        if(PRyMini.nTOpBorn_flag):
            rate_pTOn = L_pTOnBORN(T)
        else:
            # T=0 Born + radiative corrections
            L_pTOn_T = L_pTOnCCR(T)
            # finite nucleon mass effects
            L_pTOn_T += L_pTOnFMCCR(T)
            # interpolated thermal corrections
            L_pTOn_T += L_pTOnCCRTh_interp(T)
            # total p -> n rate
            rate_pTOn = (L_pTOn_T)
        if(PRyMini.NP_nTOp_flag):
            rate_pTOn += PRyMini.NP_delta_nTOp*L_pTOnBORN(T)
            
        return rate_pTOn # to be multiplied by [s-1]

    ##############################
    # Finalizing  n <--> p rates #
    ##############################
    # vectorization of the rates
    nTOp_frwrd_vec = np.vectorize(nTOp_frwrd_)
    nTOp_bkwrd_vec = np.vectorize(nTOp_bkwrd_)
    # saving the rates if computed from scratch
    if(PRyMini.verbose_flag):
        print(" ")
        print("Re-computing n <--> p weak rates @ high T regime")
    T_interval_HT = np.logspace(np.log10(PRyMini.T_start),np.log10(PRyMini.T_weak),PRyMini.sampling_nTOp)
    if(PRyMini.verbose_flag):
        print("Re-computing n <--> p weak rates @ mid T regime")
    T_interval_MT = np.logspace(np.log10(PRyMini.T_weak),np.log10(PRyMini.T_nucl),PRyMini.sampling_nTOp)
    if(PRyMini.verbose_flag):
        print("Re-computing n <--> p weak rates @ low T regime")
    T_interval_LT = np.logspace(np.log10(PRyMini.T_nucl),np.log10(PRyMini.T_end),PRyMini.sampling_nTOp)
    if(PRyMini.save_nTOp_flag):
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_HT.txt",np.c_[T_interval_HT,nTOp_frwrd_vec(T_interval_HT)])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_HT.txt",np.c_[T_interval_HT,nTOp_bkwrd_vec(T_interval_HT)])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_MT.txt",np.c_[T_interval_MT,nTOp_frwrd_vec(T_interval_MT)])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_MT.txt",np.c_[T_interval_MT,nTOp_bkwrd_vec(T_interval_MT)])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_LT.txt",np.c_[T_interval_LT,nTOp_frwrd_vec(T_interval_LT)])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_LT.txt",np.c_[T_interval_LT,nTOp_bkwrd_vec(T_interval_LT)])
    return [T_interval_HT,nTOp_frwrd_vec(T_interval_HT),nTOp_bkwrd_vec(T_interval_HT),T_interval_MT,nTOp_frwrd_vec(T_interval_MT),nTOp_bkwrd_vec(T_interval_MT),T_interval_LT,nTOp_frwrd_vec(T_interval_LT),nTOp_bkwrd_vec(T_interval_LT)]
