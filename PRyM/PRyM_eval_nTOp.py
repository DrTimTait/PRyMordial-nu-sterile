# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, spence
from scipy.integrate import quad
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini
if(PRyMini.compute_nTOp_thermal_flag):
    import vegas

exp_cutoff = 3*1.e+2 # cutoff to avoid overflow warnings

# Module-level state for multiprocessing (fork-inherited by child processes)
_mp_state = {}
def _mp_compute_rates(T):
    return (_mp_state['frwrd'](T), _mp_state['bkwrd'](T))
epsrel_low = 1.e-1 # minimum precision sufficient to speed up some quad integrals
if(PRyMini.compute_nTOp_thermal_flag):
    # Settings for precision in vegas integration
    n_eval = 20000 # recommended max number of evaluations per iteration
    n_itn = 20 # recommended number of iterations

###############################################################
# Numba-accelerated integrand functions for weak rate computation
###############################################################
if(PRyMini.numba_flag):
    from numba import njit

    # Pre-bake physical constants at import time
    _me = PRyMini.me*PRyMini.MeV
    _mn = PRyMini.mn*PRyMini.MeV
    _mp_ = PRyMini.mp*PRyMini.MeV
    _Q = _mn - _mp_
    _q = _Q/_me
    _gA = PRyMini.gA
    _dk = PRyMini.deltakappa
    _alpha = PRyMini.alphaem
    _Mp = _mp_/_me
    _Mn = _mn/_me

    # FermiCoulomb constants
    _FC_Gamma = np.sqrt(1.0 - _alpha**2) - 1.0
    _FC_gamma1 = 1.0 + _FC_Gamma
    _FC_prefix = (1.0 + _FC_Gamma/2.0)*4.0
    _FC_Fn_Compton = PRyMini.hbar*PRyMini.clight/_me
    _FC_radproton = PRyMini.radproton
    _FC_gamma_gamma2_sq = float(gamma(3.0 + 2.0*_FC_Gamma))**2

    @njit(cache=True)
    def _li2(x):
        """Dilogarithm Li_2(x) for real x in [0, 1]."""
        if x <= 0.0:
            return 0.0
        if x <= 0.5:
            s = 0.0
            xk = x
            for k in range(1, 50):
                s += xk/(k*k)
                xk *= x
            return s
        else:
            y = 1.0 - x
            if y <= 0.0:
                return np.pi**2/6.0
            s = 0.0
            yk = y
            for k in range(1, 50):
                s += yk/(k*k)
                yk *= y
            return np.pi**2/6.0 - np.log(x)*np.log(y) - s

    @njit(cache=True)
    def _spence_nb(z):
        """scipy.special.spence compatible: spence(z) = Li_2(1-z)."""
        return _li2(1.0 - z)

    @njit(cache=True)
    def _abs_cgamma_sq(s, t):
        """|Gamma(s+it)|^2 via Lanczos approximation (g=7, n=9)."""
        p0 = 0.99999999999980993
        p1 = 676.5203681218851
        p2 = -1259.1392167224028
        p3 = 771.32342877765313
        p4 = -176.61502916214059
        p5 = 12.507343278686905
        p6 = -0.13857109526572012
        p7 = 9.9843695780195716e-6
        p8 = 1.5056327351493116e-7
        g = 7.0
        zr = s - 1.0
        zi = t
        Ar = p0
        Ai = 0.0
        pk = (p1, p2, p3, p4, p5, p6, p7, p8)
        for k in range(8):
            dr = zr + k + 1.0
            denom = dr*dr + zi*zi
            Ar += pk[k]*dr/denom
            Ai -= pk[k]*zi/denom
        wr = zr + g + 0.5
        wi = zi
        ln_w_abs = 0.5*np.log(wr*wr + wi*wi)
        arg_w = np.arctan2(wi, wr)
        real_exp = (s - 0.5)*ln_w_abs - t*arg_w
        log_result = np.log(2.0*np.pi) + 2.0*real_exp - 2.0*wr + np.log(Ar*Ar + Ai*Ai)
        return np.exp(log_result)

    @njit(cache=True)
    def _FermiCoulomb_nb(b):
        Gamma = _FC_Gamma
        gamma1 = _FC_gamma1
        alpha = _alpha
        base = (2.0*_FC_radproton*b)/_FC_Fn_Compton
        power_part = base**(2.0*Gamma)/_FC_gamma_gamma2_sq
        exp_part = np.exp(np.pi*alpha/b)
        denom = (1.0 - b**2)**Gamma
        gamma_part = _abs_cgamma_sq(gamma1, alpha/b)
        return _FC_prefix*power_part*exp_part/denom*gamma_part

    @njit(cache=True)
    def _RadCorrResum_nb(b, y, en):
        me = _me
        mn = _mn
        mp = _mp_
        mA = 1.2e3*PRyMini.MeV
        Q = _Q
        alpha = _alpha
        Cndecay = 0.891
        deltandecay = -0.00043
        Lndecay = 1.02094
        Sndecay = 1.02248
        NLLndecay = -0.0001
        Agndecay = -0.34
        if b == 0.0:
            Rd = 1.0
        else:
            Rd = np.arctanh(b)/b
        Sirlin_fun = (3.0*np.log(mp/me) - 0.75
            + 4.0*(Rd-1.0)*(y/(3.0*en) - 1.5 + np.log(2.0*y))
            + Rd*(2.0*(1.0+b**2) + y**2/(6.0*en**2) - 4.0*b*Rd)
            - (4.0/b)*_spence_nb(1.0 - 2.0*b/(1.0+b)))
        return ((1.0 + alpha/(2.0*np.pi)*(Sirlin_fun - 3.0*np.log(mp/(2.0*Q))))
            *(Lndecay + (alpha/np.pi)*Cndecay + deltandecay)
            *(Sndecay + 1.0/(134.0*2.0*np.pi)*(np.log(mp/mA) + Agndecay) + NLLndecay))

    @njit(cache=True)
    def _FD2_nb(E, x):
        if x*E < exp_cutoff:
            return 1.0/(np.exp(x*E) + 1.0)
        return 0.0

    @njit(cache=True)
    def _FD_nu3_nb(E, phi, x):
        if x*E - phi < exp_cutoff:
            return 1.0/(np.exp(x*E - phi) + 1.0)
        return 0.0

    @njit(cache=True)
    def _FD_nu_e2p0_nb(E, phi, x):
        if x*E - phi < exp_cutoff:
            return E**2/(np.exp(x*E - phi) + 1.0)
        return 0.0

    @njit(cache=True)
    def _FD_nu_e3p0_nb(E, phi, x):
        if x*E - phi < exp_cutoff:
            return E**3/(np.exp(x*E - phi) + 1.0)
        return 0.0

    @njit(cache=True)
    def _FD_nu_e4p2_nb(E, phi, x):
        Ex = E*x
        if 2.0*phi < exp_cutoff and Ex + phi < exp_cutoff and 2.0*Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**3
            return (E**2*ep*((24.0 - Ex*(Ex + 8.0))*eEx*ep + eEx*eEx*(Ex - 6.0)*(Ex - 2.0) + 12.0*ep*ep))/d
        return 0.0

    @njit(cache=True)
    def _FD_nu_e2p2_nb(E, phi, x):
        Ex = E*x
        if 3.0*phi < exp_cutoff and 2.0*Ex + phi < exp_cutoff and Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**3
            return ((Ex*(Ex - 4.0) + 2.0)*eEx*eEx*ep + (4.0 - Ex*(Ex + 4.0))*eEx*ep*ep + 2.0*ep*ep*ep)/d
        return 0.0

    @njit(cache=True)
    def _FD_nu_e4p1_nb(E, phi, x):
        Ex = E*x
        if phi < exp_cutoff and Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**2
            return (ep*E**3*(4.0*ep + eEx*(4.0 - Ex)))/d
        return 0.0

    @njit(cache=True)
    def _FD_nu_e2p1_nb(E, phi, x):
        Ex = E*x
        if phi < exp_cutoff and Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**2
            return (ep*E*(2.0*ep + eEx*(2.0 - Ex)))/d
        return 0.0

    @njit(cache=True)
    def _FD_nu_e3p1_nb(E, phi, x):
        Ex = E*x
        if phi < exp_cutoff and Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**2
            return (ep*E**2*(3.0*ep + eEx*(3.0 - Ex)))/d
        return 0.0

    @njit(cache=True)
    def _FD_nu_e3p2_nb(E, phi, x):
        Ex = E*x
        if 2.0*phi < exp_cutoff and Ex + phi < exp_cutoff and 2.0*Ex < exp_cutoff:
            ep = np.exp(phi)
            eEx = np.exp(Ex)
            d = (eEx + ep)**3
            return (E*ep*((12.0 - Ex*(Ex + 6.0))*eEx*ep + eEx*eEx*(Ex*(Ex - 6.0) + 6.0) + 6.0*ep*ep))/d
        return 0.0

    @njit(cache=True)
    def _ChiFunc_std_nb(E, x, znu, sgnq, xi_nu):
        """Standard response function (non-general_nu)."""
        Enu = E - sgnq*_q
        return _FD_nu3_nb(Enu, sgnq*xi_nu, znu)*_FD2_nb(-E, x)*Enu**2

    @njit(cache=True)
    def _ChiFunc_FM_std_nb(en, pe, x, znu, sgnq):
        """Standard finite-mass response function (non-general_nu, phi=0)."""
        if sgnq > 0:
            f_1 = ((1.0+_gA)**2 + 2.0*_dk*_gA)/(1.0+3.0*_gA**2)
            f_2 = ((1.0-_gA)**2 - 2.0*_dk*_gA)/(1.0+3.0*_gA**2)
            M_sq = (_mp_+_mn-_Q)/(2.0*_me)
        else:
            f_1 = ((1.0-_gA)**2 - 2.0*_dk*_gA)/(1.0+3.0*_gA**2)
            f_2 = ((1.0+_gA)**2 + 2.0*_dk*_gA)/(1.0+3.0*_gA**2)
            M_sq = (_mp_+_mn+_Q)/(2.0*_me)
        f_3 = (_gA**2-1.0)/(1.0+3.0*_gA**2)
        FD2_en = _FD2_nb(-en, x)
        Enu = en - sgnq*_q
        e2p0 = _FD_nu_e2p0_nb(Enu, 0.0, znu)
        e3p0 = _FD_nu_e3p0_nb(Enu, 0.0, znu)
        e4p2 = _FD_nu_e4p2_nb(Enu, 0.0, znu)
        e2p2 = _FD_nu_e2p2_nb(Enu, 0.0, znu)
        e4p1 = _FD_nu_e4p1_nb(Enu, 0.0, znu)
        e2p1 = _FD_nu_e2p1_nb(Enu, 0.0, znu)
        e3p1 = _FD_nu_e3p1_nb(Enu, 0.0, znu)
        e3p2 = _FD_nu_e3p2_nb(Enu, 0.0, znu)
        return (f_1*e2p0*FD2_en*(pe**2/(M_sq*en))
            + f_2*e3p0*FD2_en*(-(1.0/M_sq))
            + (f_1+f_2+f_3)/(2.0*x*M_sq)*(e4p2*FD2_en + e2p2*FD2_en*pe**2)
            + (f_1+f_2+f_3)/(2.0*M_sq)*(e4p1*FD2_en + e2p1*FD2_en*pe**2)
            - (f_1+f_2)/(x*M_sq)*(e3p1*FD2_en + e2p1*FD2_en*pe**2/(-en))
            - f_3*3.0/(x*M_sq)*e2p0*FD2_en
            + f_3/(3.0*M_sq)*e3p1*FD2_en*pe**2/en
            + f_3*2.0/(2.0*x*3.0*M_sq)*e3p2*FD2_en*pe**2/en
            - (f_1+f_2+f_3)*3.0/(2.0*x)*(1.0-(_Mn/_Mp)**sgnq)*(e2p1*FD2_en))

    @njit(cache=True)
    def _FMCCR_integrand_nb(p, x, xnu, sgnq):
        """Full FMCCR integrand (standard path)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        b = p/eOFpe
        chi_p = _ChiFunc_FM_std_nb(eOFpe, p, x, xnu, sgnq)
        chi_m = _ChiFunc_FM_std_nb(-eOFpe, p, x, xnu, sgnq)
        rc_p = _RadCorrResum_nb(b, np.abs(sgnq*_q - eOFpe), eOFpe)
        rc_m = _RadCorrResum_nb(b, np.abs(sgnq*_q + eOFpe), eOFpe)
        fc = _FermiCoulomb_nb(b)
        if sgnq > 0:
            fs_p = fc
            fs_m = 1.0
        else:
            fs_p = 1.0
            fs_m = fc
        return p**2*(chi_p*rc_p*fs_p + chi_m*rc_m*fs_m)

    @njit(cache=True)
    def _CCR_integrand_nb(p, x, xnu, sgnq, xi_nu):
        """Full CCR integrand (standard path)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        b = p/eOFpe
        chi_p = _ChiFunc_std_nb(eOFpe, x, xnu, sgnq, xi_nu)
        chi_m = _ChiFunc_std_nb(-eOFpe, x, xnu, sgnq, xi_nu)
        rc_p = _RadCorrResum_nb(b, np.abs(sgnq*_q - eOFpe), eOFpe)
        rc_m = _RadCorrResum_nb(b, np.abs(sgnq*_q + eOFpe), eOFpe)
        fc = _FermiCoulomb_nb(b)
        if sgnq > 0:
            fs_p = fc
            fs_m = 1.0
        else:
            fs_p = 1.0
            fs_m = fc
        return p**2*(chi_p*rc_p*fs_p + chi_m*rc_m*fs_m)

    @njit(cache=True)
    def _Born_integrand_nb(p, x, xnu, sgnq, xi_nu):
        """Born integrand (standard path)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        chi_p = _ChiFunc_std_nb(eOFpe, x, xnu, sgnq, xi_nu)
        chi_m = _ChiFunc_std_nb(-eOFpe, x, xnu, sgnq, xi_nu)
        return p**2*(chi_p + chi_m)

    ###################################################################
    # Numba-accelerated integrands for general_nu (table-based lookup)
    ###################################################################
    # These replace the slow Python call chain through f_nue_general
    # by pre-tabulating f(E) on a uniform grid and doing fast linear
    # interpolation inside @njit compiled functions.

    _me_MeV = PRyMini.me  # electron mass in MeV (0.511)

    @njit(cache=True)
    def _interp_f_nb(E_abs, tab_E_max, tab_dE, tab_f, N):
        """Fast linear interpolation on a uniform grid.
        E_abs: absolute dimensionless neutrino energy |E_nu| (units of me).
        tab_E_max: maximum energy in table.
        tab_dE: grid spacing.
        tab_f: distribution values on uniform grid [0, tab_E_max].
        N: number of grid points."""
        if E_abs <= 0.0:
            return tab_f[0]
        if E_abs >= tab_E_max:
            return 0.0
        idx_f = E_abs / tab_dE
        i = int(idx_f)
        if i >= N - 1:
            return 0.0
        frac = idx_f - i
        return tab_f[i] * (1.0 - frac) + tab_f[i + 1] * frac

    @njit(cache=True)
    def _f_eff_nu_tab_nb(E_nu, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """Effective neutrino distribution from table with crossing symmetry.
        E_nu: signed dimensionless neutrino energy (units of me).
        Handles nu/nubar dispatch and emission (1-f) for E_nu < 0."""
        E_abs = E_nu if E_nu >= 0.0 else -E_nu
        if E_nu >= 0.0:
            if sgnq > 0:
                return _interp_f_nb(E_abs, tab_E_max, tab_dE, tab_f_nue, N)
            else:
                return _interp_f_nb(E_abs, tab_E_max, tab_dE, tab_f_nuebar, N)
        else:
            if sgnq > 0:
                return 1.0 - _interp_f_nb(E_abs, tab_E_max, tab_dE, tab_f_nuebar, N)
            else:
                return 1.0 - _interp_f_nb(E_abs, tab_E_max, tab_dE, tab_f_nue, N)

    @njit(cache=True)
    def _EA_feff_tab_nb(E_nu, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """E^A * f_eff(E) from table."""
        f_val = _f_eff_nu_tab_nb(E_nu, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        if A == 0:
            return f_val
        elif A == 1:
            return E_nu * f_val
        elif A == 2:
            return E_nu * E_nu * f_val
        elif A == 3:
            return E_nu * E_nu * E_nu * f_val
        else:
            return E_nu**A * f_val

    @njit(cache=True)
    def _FD_nu_eApB_tab_nb(E_nu, A, B, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """d^B/dE^B [E^A * f_eff(E)] via central finite differences, from table.
        Step size must span multiple table grid cells for meaningful derivatives."""
        if B == 0:
            return _EA_feff_tab_nb(E_nu, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        # Use step size of 3 table spacings to ensure derivatives sample across grid cells
        dE = 3.0 * tab_dE
        if B == 1:
            fp = _EA_feff_tab_nb(E_nu + dE, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
            fm = _EA_feff_tab_nb(E_nu - dE, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
            return (fp - fm) / (2.0 * dE)
        else:  # B == 2
            fp = _EA_feff_tab_nb(E_nu + dE, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
            f0 = _EA_feff_tab_nb(E_nu, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
            fm = _EA_feff_tab_nb(E_nu - dE, A, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
            return (fp - 2.0 * f0 + fm) / (dE * dE)

    @njit(cache=True)
    def _ChiFunc_general_tab_nb(E, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """Born response function for general distributions (table-based)."""
        E_nu = E - sgnq * _q
        f_val = _f_eff_nu_tab_nb(E_nu, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        FD2_val = _FD2_nb(-E, x)
        return f_val * FD2_val * E_nu * E_nu

    @njit(cache=True)
    def _ChiFunc_FM_general_tab_nb(en, pe, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """Finite-mass response function for general distributions (table-based)."""
        if sgnq > 0:
            f_1 = ((1.0 + _gA)**2 + 2.0 * _dk * _gA) / (1.0 + 3.0 * _gA**2)
            f_2 = ((1.0 - _gA)**2 - 2.0 * _dk * _gA) / (1.0 + 3.0 * _gA**2)
            M_sq = (_mp_ + _mn - _Q) / (2.0 * _me)
        else:
            f_1 = ((1.0 - _gA)**2 - 2.0 * _dk * _gA) / (1.0 + 3.0 * _gA**2)
            f_2 = ((1.0 + _gA)**2 + 2.0 * _dk * _gA) / (1.0 + 3.0 * _gA**2)
            M_sq = (_mp_ + _mn + _Q) / (2.0 * _me)
        f_3 = (_gA**2 - 1.0) / (1.0 + 3.0 * _gA**2)
        FD2_en = _FD2_nb(-en, x)
        Enu = en - sgnq * _q
        # Shorthand: G(A,B) = d^B/dE^B [E^A * f_eff(E)] at Enu
        e2p0 = _FD_nu_eApB_tab_nb(Enu, 2, 0, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e3p0 = _FD_nu_eApB_tab_nb(Enu, 3, 0, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e4p2 = _FD_nu_eApB_tab_nb(Enu, 4, 2, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e2p2 = _FD_nu_eApB_tab_nb(Enu, 2, 2, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e4p1 = _FD_nu_eApB_tab_nb(Enu, 4, 1, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e2p1 = _FD_nu_eApB_tab_nb(Enu, 2, 1, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e3p1 = _FD_nu_eApB_tab_nb(Enu, 3, 1, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        e3p2 = _FD_nu_eApB_tab_nb(Enu, 3, 2, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        return (f_1 * e2p0 * FD2_en * (pe**2 / (M_sq * en))
            + f_2 * e3p0 * FD2_en * (-(1.0 / M_sq))
            + (f_1 + f_2 + f_3) / (2.0 * x * M_sq) * (e4p2 * FD2_en + e2p2 * FD2_en * pe**2)
            + (f_1 + f_2 + f_3) / (2.0 * M_sq) * (e4p1 * FD2_en + e2p1 * FD2_en * pe**2)
            - (f_1 + f_2) / (x * M_sq) * (e3p1 * FD2_en + e2p1 * FD2_en * pe**2 / (-en))
            - f_3 * 3.0 / (x * M_sq) * e2p0 * FD2_en
            + f_3 / (3.0 * M_sq) * e3p1 * FD2_en * pe**2 / en
            + f_3 * 2.0 / (2.0 * x * 3.0 * M_sq) * e3p2 * FD2_en * pe**2 / en
            - (f_1 + f_2 + f_3) * 3.0 / (2.0 * x) * (1.0 - (_Mn / _Mp)**sgnq) * (e2p1 * FD2_en))

    @njit(cache=True)
    def _Born_integrand_general_tab_nb(p, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """Born integrand for general distributions (table-based)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        chi_p = _ChiFunc_general_tab_nb(eOFpe, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        chi_m = _ChiFunc_general_tab_nb(-eOFpe, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        return p**2 * (chi_p + chi_m)

    @njit(cache=True)
    def _CCR_integrand_general_tab_nb(p, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """CCR integrand for general distributions (table-based)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        b = p / eOFpe
        chi_p = _ChiFunc_general_tab_nb(eOFpe, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        chi_m = _ChiFunc_general_tab_nb(-eOFpe, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        rc_p = _RadCorrResum_nb(b, np.abs(sgnq * _q - eOFpe), eOFpe)
        rc_m = _RadCorrResum_nb(b, np.abs(sgnq * _q + eOFpe), eOFpe)
        fc = _FermiCoulomb_nb(b)
        if sgnq > 0:
            fs_p = fc
            fs_m = 1.0
        else:
            fs_p = 1.0
            fs_m = fc
        return p**2 * (chi_p * rc_p * fs_p + chi_m * rc_m * fs_m)

    @njit(cache=True)
    def _FMCCR_integrand_general_tab_nb(p, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N):
        """FMCCR integrand for general distributions (table-based)."""
        eOFpe = np.sqrt(p**2 + 1.0)
        b = p / eOFpe
        chi_p = _ChiFunc_FM_general_tab_nb(eOFpe, p, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        chi_m = _ChiFunc_FM_general_tab_nb(-eOFpe, p, x, sgnq, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, N)
        rc_p = _RadCorrResum_nb(b, np.abs(sgnq * _q - eOFpe), eOFpe)
        rc_m = _RadCorrResum_nb(b, np.abs(sgnq * _q + eOFpe), eOFpe)
        fc = _FermiCoulomb_nb(b)
        if sgnq > 0:
            fs_p = fc
            fs_m = 1.0
        else:
            fs_p = 1.0
            fs_m = fc
        return p**2 * (chi_p * rc_p * fs_p + chi_m * rc_m * fs_m)

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

    # Smart dispatch: detect thermal Fermi-Dirac distributions
    # When general_nu_flag=True but distributions are still thermal FD (e.g. SM
    # Boltzmann runs with thermal ICs), we can use the fast numba standard-path
    # integrands instead of the slow general_nu Python call chain.  This reduces
    # weak rate computation from ~12 min to ~0.06s for thermal distributions.
    # Detection: compare f(p, Tg) against FD(p/Tnu_eff) at several momenta.
    # Using Tnu_eff (not Tg) correctly handles evolved distributions where
    # neutrinos have cooled to Tnu < Tg after decoupling.
    _saved_general_nu_flag = PRyMini.general_nu_flag
    if PRyMini.general_nu_flag and PRyMini.numba_flag:
        _Tg_test = Tg_vec[len(Tg_vec)//2]  # mid-range temperature
        _Tnu_eff_test = PRyMthermo.Tnu_eff_e(_Tg_test)  # effective nu temperature
        _p_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * _Tnu_eff_test
        _fd_ref = 1./(np.exp(_p_test/_Tnu_eff_test) + 1.)
        _f_nue = PRyMthermo.f_nue_general(_p_test, _Tg_test)
        _f_nuebar = PRyMthermo.f_nuebar_general(_p_test, _Tg_test)
        _max_dev = max(np.max(np.abs(_f_nue - _fd_ref)),
                       np.max(np.abs(_f_nuebar - _fd_ref)))
        if _max_dev < 1.e-6:
            PRyMini.general_nu_flag = False
            if PRyMini.verbose_flag:
                print(" Smart dispatch: distributions are thermal FD (max dev = {:.1e})".format(_max_dev))
                print(" Using fast numba standard-path integrands for weak rates")

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

    # Table builder for numba general_nu integrands
    _N_tab = 10000  # number of grid points for distribution table
    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def _build_nu_tables(Tg_MeV, pemax):
            """Pre-tabulate f_nue and f_nuebar on a uniform dimensionless energy grid."""
            E_nu_max = np.sqrt(pemax**2 + 1.0) + Q/me + 2.0
            tab_E = np.linspace(0., E_nu_max, _N_tab)
            tab_dE = tab_E[1] - tab_E[0]
            p_MeV = tab_E * PRyMini.me  # convert dimensionless E to physical p [MeV]
            tab_f_nue = np.ascontiguousarray(PRyMthermo.f_nue_general(p_MeV, Tg_MeV))
            tab_f_nuebar = np.ascontiguousarray(PRyMthermo.f_nuebar_general(p_MeV, Tg_MeV))
            return E_nu_max, tab_dE, tab_f_nue, tab_f_nuebar

    # Born rates given by Eq 2.30 in Brown & Sawyer
    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpBORN_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _Born_integrand_general_tab_nb(p, x, 1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
        def L_pTOnBORN_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _Born_integrand_general_tab_nb(p, x, -1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
    elif(PRyMini.general_nu_flag):
        def L_nTOpBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdp(p,x,Tg_MeV,1)
        def L_pTOnBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdp(p,x,Tg_MeV,-1)
    elif(PRyMini.numba_flag):
        def L_nTOpBORN_int(p, x, xnu, xi):
            return _Born_integrand_nb(p, x, xnu, 1, xi)
        def L_pTOnBORN_int(p, x, xnu, xi):
            return _Born_integrand_nb(p, x, xnu, -1, xi)
    else:
        def L_nTOpBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdp(p,x,xnu,1)
        def L_pTOnBORN_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdp(p,x,xnu,-1)

    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpBORN(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_nTOpBORN_int, pemin, pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
        def L_pTOnBORN(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_pTOnBORN_int, pemin, pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
    elif(not PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpBORN(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_nTOpBORN_int, pemin, pemax, args=(x, xnu, xi_nu), epsrel = epsrel_low)[0]
        def L_pTOnBORN(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_pTOnBORN_int, pemin, pemax, args=(x, xnu, xi_nu), epsrel = epsrel_low)[0]
    else:
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

    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpFMCCR_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _FMCCR_integrand_general_tab_nb(p, x, 1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
        def L_pTOnFMCCR_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _FMCCR_integrand_general_tab_nb(p, x, -1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
    elif(PRyMini.general_nu_flag):
        def L_nTOpFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpFMCCR(p,x,Tg_MeV,1)
        def L_pTOnFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpFMCCR(p,x,Tg_MeV,-1)
    elif(PRyMini.numba_flag):
        def L_nTOpFMCCR_int(p, x, xnu):
            return _FMCCR_integrand_nb(p, x, xnu, 1)
        def L_pTOnFMCCR_int(p, x, xnu):
            return _FMCCR_integrand_nb(p, x, xnu, -1)
    else:
        def L_nTOpFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpFMCCR(p,x,xnu,1)
        def L_pTOnFMCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpFMCCR(p,x,xnu,-1)

    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpFMCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_nTOpFMCCR_int,pemin,pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
        def L_pTOnFMCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_pTOnFMCCR_int, pemin, pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
    elif(not PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpFMCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_nTOpFMCCR_int,pemin,pemax, args=(x, xnu), epsrel = epsrel_low)[0]
        def L_pTOnFMCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_pTOnFMCCR_int, pemin, pemax, args=(x, xnu), epsrel = epsrel_low)[0]
    else:
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

    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpCCR_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _CCR_integrand_general_tab_nb(p, x, 1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
        def L_pTOnCCR_int(p, x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar):
            return _CCR_integrand_general_tab_nb(p, x, -1, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar, _N_tab)
    elif(PRyMini.general_nu_flag):
        def L_nTOpCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpCCR(p,x,Tg_MeV,1)
        def L_pTOnCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            return IPENdpCCR(p,x,Tg_MeV,-1)
    elif(PRyMini.numba_flag):
        def L_nTOpCCR_int(p, x, xnu, xi):
            return _CCR_integrand_nb(p, x, xnu, 1, xi)
        def L_pTOnCCR_int(p, x, xnu, xi):
            return _CCR_integrand_nb(p, x, xnu, -1, xi)
    else:
        def L_nTOpCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpCCR(p,x,xnu,1)
        def L_pTOnCCR_int(p,T):
            x = me/(PRyMini.kB*T)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return IPENdpCCR(p,x,xnu,-1)

    if(PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_nTOpCCR_int, pemin, pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
        def L_pTOnCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            Tg_MeV = PRyMini.kB*T/PRyMini.MeV
            tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar = _build_nu_tables(Tg_MeV, pemax)
            return quad(L_pTOnCCR_int, pemin, pemax, args=(x, tab_E_max, tab_dE, tab_f_nue, tab_f_nuebar), epsrel = epsrel_low)[0]
    elif(not PRyMini.general_nu_flag and PRyMini.numba_flag):
        def L_nTOpCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_nTOpCCR_int, pemin, pemax, args=(x, xnu, xi_nu), epsrel = epsrel_low)[0]
        def L_pTOnCCR(T):
            pemin = 0.
            x = me/(PRyMini.kB*T)
            pemax = max(7.,30./x)
            xnu = me/(PRyMini.kB*T*T_nuOverT(T))
            return quad(L_pTOnCCR_int, pemin, pemax, args=(x, xnu, xi_nu), epsrel = epsrel_low)[0]
    else:
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
    T_interval_HT = np.logspace(np.log10(PRyMini.T_start),np.log10(PRyMini.T_weak),PRyMini.sampling_nTOp)
    T_interval_MT = np.logspace(np.log10(PRyMini.T_weak),np.log10(PRyMini.T_nucl),PRyMini.sampling_nTOp)
    T_interval_LT = np.logspace(np.log10(PRyMini.T_nucl),np.log10(PRyMini.T_end),PRyMini.sampling_nTOp)
    T_all = np.concatenate([T_interval_HT, T_interval_MT, T_interval_LT])

    # Determine parallelism
    import os
    n_cores = PRyMini.n_cores_nTOp
    if n_cores == 0:
        n_cores = os.cpu_count() or 1
    n_cores = min(n_cores, len(T_all))

    if(PRyMini.verbose_flag):
        print(" ")
        print("Re-computing n <--> p weak rates ({0} temperature points, {1} core{2})".format(
            len(T_all), n_cores, "s" if n_cores > 1 else ""))

    if n_cores > 1:
        import multiprocessing as _multiproc
        _mp_state['frwrd'] = nTOp_frwrd_
        _mp_state['bkwrd'] = nTOp_bkwrd_
        ctx = _multiproc.get_context('fork')
        with ctx.Pool(n_cores) as pool:
            results = pool.map(_mp_compute_rates, T_all)
        _mp_state.clear()
        frwrd_all = np.array([r[0] for r in results])
        bkwrd_all = np.array([r[1] for r in results])
    else:
        results = [(nTOp_frwrd_(T), nTOp_bkwrd_(T)) for T in T_all]
        frwrd_all = np.array([r[0] for r in results])
        bkwrd_all = np.array([r[1] for r in results])

    # Split into HT/MT/LT segments
    n = PRyMini.sampling_nTOp
    nTOp_frwrdvec_HT = frwrd_all[:n]
    nTOp_bkwrdvec_HT = bkwrd_all[:n]
    nTOp_frwrdvec_MT = frwrd_all[n:2*n]
    nTOp_bkwrdvec_MT = bkwrd_all[n:2*n]
    nTOp_frwrdvec_LT = frwrd_all[2*n:]
    nTOp_bkwrdvec_LT = bkwrd_all[2*n:]

    if(PRyMini.save_nTOp_flag):
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_HT.txt",np.c_[T_interval_HT,nTOp_frwrdvec_HT])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_HT.txt",np.c_[T_interval_HT,nTOp_bkwrdvec_HT])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_MT.txt",np.c_[T_interval_MT,nTOp_frwrdvec_MT])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_MT.txt",np.c_[T_interval_MT,nTOp_bkwrdvec_MT])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_LT.txt",np.c_[T_interval_LT,nTOp_frwrdvec_LT])
        np.savetxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_LT.txt",np.c_[T_interval_LT,nTOp_bkwrdvec_LT])
    # Restore general_nu_flag (may have been temporarily cleared by smart dispatch)
    PRyMini.general_nu_flag = _saved_general_nu_flag
    return [T_interval_HT,nTOp_frwrdvec_HT,nTOp_bkwrdvec_HT,T_interval_MT,nTOp_frwrdvec_MT,nTOp_bkwrdvec_MT,T_interval_LT,nTOp_frwrdvec_LT,nTOp_bkwrdvec_LT]
