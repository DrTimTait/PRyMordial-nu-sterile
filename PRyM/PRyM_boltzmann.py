# -*- coding: utf-8 -*-
"""
PRyM_boltzmann.py — Internal Boltzmann solver for neutrino distribution functions.

Evolves f_nu(y, t) on a comoving momentum grid (y = p*a), computing SM 2-to-2
collision integrals from first principles following Sabti et al. (Appendix E).

The collision integrals are reduced from 9D to 2D using symmetry and analytic
integration of angular variables, yielding piecewise polynomial D-functions.
"""
import numpy as np
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini

# Try to import numba for JIT compilation of inner loops
try:
    if PRyMini.numba_flag:
        from numba import njit, prange
        _has_numba = True
    else:
        _has_numba = False
except ImportError:
    _has_numba = False

if not _has_numba:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


###############################################################################
# D-functions: Angular integration kernels from Sabti-BBN Appendix E.2       #
# (Eqs. E.22-E.29)                                                           #
#                                                                             #
# These are piecewise polynomial functions of (y1,y2,y3,y4) that encode the  #
# result of analytically integrating out the angular variables and the        #
# auxiliary lambda integration in the 4-particle collision integral.          #
#                                                                             #
# Convention: yi >= yj and yk >= yl (enforced by taking abs and sorting).     #
# Five momentum-ordering cases determine the polynomial form.                 #
###############################################################################

@njit
def _D1_raw(yi, yj, yk, yl):
    """
    D1(yi,yj,yk,yl) from Eq. E.22, with yi >= yj and yk >= yl assumed.
    D1 = (4/pi) * integral dλ/λ² sin(yi*λ) sin(yj*λ) sin(yk*λ) sin(yl*λ)
    Piecewise polynomial with 5 cases.
    """
    # Case 1: yi > yj + yk + yl or yk > yi + yj + yl
    if yi > yj + yk + yl or yk > yi + yj + yl:
        return 0.0

    # Case 2: yi + yj > yk + yl and yi + yl < yj + yk
    if yi + yj > yk + yl and yi + yl < yj + yk:
        return yl

    # Case 3: yi + yj > yk + yl and yi + yl > yj + yk
    if yi + yj > yk + yl and yi + yl >= yj + yk:
        return 0.5 * (yj + yk + yl - yi)

    # Case 4: yi + yj < yk + yl and yi + yl > yj + yk
    if yi + yj <= yk + yl and yi + yl > yj + yk:
        return yj

    # Case 5: yi + yj < yk + yl and yi + yl < yj + yk
    if yi + yj <= yk + yl and yi + yl <= yj + yk:
        return 0.5 * (yi + yj + yl - yk)

    return 0.0


@njit
def _D2_raw(yi, yj, yk, yl):
    """
    D2(yi,yj,yk,yl) from Eq. E.23, with yi >= yj and yk >= yl assumed.
    D2 = (4*yk*yl/pi) * integral dλ/λ² sin(yi*λ) sin(yj*λ)
         × [cos(yk*λ) - sin(yk*λ)/(yk*λ)] × [cos(yl*λ) - sin(yl*λ)/(yl*λ)]
    """
    if yi > yj + yk + yl or yk > yi + yj + yl:
        return 0.0

    if yi + yj > yk + yl and yi + yl < yj + yk:
        return yl**3 / 3.0

    if yi + yj > yk + yl and yi + yl >= yj + yk:
        return ((yi - yj) * ((yi - yj)**2 - 3.0 * (yk**2 + yl**2))
                + 2.0 * (yk**3 + yl**3)) / 12.0

    if yi + yj <= yk + yl and yi + yl > yj + yk:
        return yj * (3.0 * (yk**2 + yl**2) - yi**2 - yj**2) / 6.0

    if yi + yj <= yk + yl and yi + yl <= yj + yk:
        return -((yi + yj) * ((yi + yj)**2 - 3.0 * (yk**2 + yl**2))
                 + 2.0 * (yk**3 - yl**3)) / 12.0

    return 0.0


@njit
def _D3_raw(yi, yj, yk, yl):
    """
    D3(yi,yj,yk,yl) from Eq. E.24, with yi >= yj and yk >= yl assumed.
    D3 = (4*yi*yj*yk*yl/pi) * integral dλ/λ²
         × [cos(yi*λ) - sin(yi*λ)/(yi*λ)] × [cos(yj*λ) - sin(yj*λ)/(yj*λ)]
         × [cos(yk*λ) - sin(yk*λ)/(yk*λ)] × [cos(yl*λ) - sin(yl*λ)/(yl*λ)]
    """
    if yi > yj + yk + yl or yk > yi + yj + yl:
        return 0.0

    if yi + yj > yk + yl and yi + yl < yj + yk:
        return yl**3 * (5.0 * (yi**2 + yj**2 + yk**2) - yl**2) / 30.0

    if yi + yj > yk + yl and yi + yl >= yj + yk:
        a, b, c, d = yi, yj, yk, yl
        val = (a**5 - b**5 - c**5 - d**5
               + 5.0 * (-a**3 * b**2 + a**2 * b**3
                        - a**3 * c**2 + a**2 * c**3
                        - a**3 * d**2 + a**2 * d**3
                        + b**3 * c**2 + b**2 * c**3
                        + b**3 * d**2 + b**2 * d**3
                        + c**3 * d**2 + c**2 * d**3))
        return val / 60.0

    if yi + yj <= yk + yl and yi + yl > yj + yk:
        return yj**3 * (5.0 * (yi**2 + yk**2 + yl**2) - yj**2) / 30.0

    if yi + yj <= yk + yl and yi + yl <= yj + yk:
        a, b, c, d = yk, yi, yj, yl
        val = (a**5 - b**5 - c**5 - d**5
               + 5.0 * (-a**3 * b**2 + a**2 * b**3
                        - a**3 * c**2 + a**2 * c**3
                        - a**3 * d**2 + a**2 * d**3
                        + b**3 * c**2 + b**2 * c**3
                        + b**3 * d**2 + b**2 * d**3
                        + c**3 * d**2 + c**2 * d**3))
        return val / 60.0

    return 0.0


@njit
def D1(y1, y2, y3, y4):
    """D1 with proper ordering enforced: yi >= yj and yk >= yl."""
    yi = max(y1, y2)
    yj = min(y1, y2)
    yk = max(y3, y4)
    yl = min(y3, y4)
    return _D1_raw(yi, yj, yk, yl)


@njit
def D2(y1, y2, y3, y4, s3, s4):
    """
    D2 from Eq. E.23. Includes sign factors sk*sl from the angular integrals.
    Arguments s3, s4 are the reaction-side signs (+1 for final, -1 for initial).
    """
    yi = max(y1, y2)
    yj = min(y1, y2)
    yk = max(y3, y4)
    yl = min(y3, y4)
    return s3 * s4 * _D2_raw(yi, yj, yk, yl)


@njit
def D3(y1, y2, y3, y4, s1, s2, s3, s4):
    """
    D3 from Eq. E.24. Includes sign factors si*sj*sk*sl.
    """
    yi = max(y1, y2)
    yj = min(y1, y2)
    yk = max(y3, y4)
    yl = min(y3, y4)
    return s1 * s2 * s3 * s4 * _D3_raw(yi, yj, yk, yl)


###############################################################################
# Full D(Y1,Y2,Y3,Y4) kernel from Eq. E.21                                   #
# For massless neutrinos: E_tilde = y, mi=mj=0 so K2 terms vanish.           #
###############################################################################

@njit
def D_kernel_massless(y1, y2, y3, y4, c_D1, c_D2, c_D3):
    """
    Compute the full angular kernel for massless 4-particle processes.

    The squared matrix element has the form:
      |M|^2 / (32 G_F^2 a^{-4}) = c_12_34 * (Y1.Y2)(Y3.Y4)
                                  + c_13_24 * (Y1.Y3)(Y2.Y4)
                                  + c_14_23 * (Y1.Y4)(Y2.Y3)

    Arguments c_D1, c_D2, c_D3 are the coefficients for the three structures
    (Y1.Y2)(Y3.Y4), (Y1.Y3)(Y2.Y4), (Y1.Y4)(Y2.Y3) respectively.

    Sign convention: 1+2 -> 3+4, so s1=-1, s2=-1, s3=+1, s4=+1.
    """
    s1, s2, s3, s4 = -1.0, -1.0, 1.0, 1.0
    result = 0.0

    if c_D1 != 0.0:
        # (Y1.Y2)(Y3.Y4) structure
        d1_val = D1(y1, y2, y3, y4)
        d2_12 = D2(y1, y2, y3, y4, s3, s4)
        d2_34 = D2(y3, y4, y1, y2, s1, s2)
        d3_val = D3(y1, y2, y3, y4, s1, s2, s3, s4)
        result += c_D1 * (y1*y2*y3*y4 * d1_val + y1*y2 * d2_12
                          + y3*y4 * d2_34 + d3_val)

    if c_D2 != 0.0:
        # (Y1.Y3)(Y2.Y4) structure: swap 2<->3
        d1_val = D1(y1, y3, y2, y4)
        d2_13 = D2(y1, y3, y2, y4, s2, s4)
        d2_24 = D2(y2, y4, y1, y3, s1, s3)
        d3_val = D3(y1, y3, y2, y4, s1, s3, s2, s4)
        result += c_D2 * (y1*y3*y2*y4 * d1_val + y1*y3 * d2_13
                          + y2*y4 * d2_24 + d3_val)

    if c_D3 != 0.0:
        # (Y1.Y4)(Y2.Y3) structure: swap 2<->4
        d1_val = D1(y1, y4, y2, y3)
        d2_14 = D2(y1, y4, y2, y3, s2, s3)
        d2_23 = D2(y2, y3, y1, y4, s1, s4)
        d3_val = D3(y1, y4, y2, y3, s1, s4, s2, s3)
        result += c_D3 * (y1*y4*y2*y3 * d1_val + y1*y4 * d2_14
                          + y2*y3 * d2_23 + d3_val)

    return result


###############################################################################
# Pre-computed D-kernel tables and collision integral computation              #
###############################################################################

@njit
def _fit_tail_params(y_grid, f_grid):
    """
    Fit log(1/f - 1) = a + b*y to the high-momentum tail of the distribution.
    Uses the last n_fit points where f > f_min. Returns (a, b).
    This is exact for Fermi-Dirac distributions: log(1/f - 1) = y/(T_com) gives
    a=0, b=1/T_com.
    """
    Ny = len(y_grid)
    n_fit = min(10, Ny)
    f_min = 1.0e-12
    # Collect valid points from the tail
    yy = np.empty(n_fit)
    ll = np.empty(n_fit)
    count = 0
    for i in range(Ny - 1, -1, -1):
        fi = f_grid[i]
        if fi > f_min and fi < 1.0 - f_min:
            yy[count] = y_grid[i]
            ll[count] = np.log(1.0 / fi - 1.0)
            count += 1
            if count >= n_fit:
                break
    if count < 2:
        # Can't fit, return parameters that give f -> 0 for large y
        return 0.0, 1.0
    # Linear regression: ll = a + b*y
    sum_y = 0.0
    sum_l = 0.0
    sum_yy = 0.0
    sum_yl = 0.0
    n = float(count)
    for k in range(count):
        sum_y += yy[k]
        sum_l += ll[k]
        sum_yy += yy[k] * yy[k]
        sum_yl += yy[k] * ll[k]
    det = n * sum_yy - sum_y * sum_y
    if abs(det) < 1.0e-30:
        return 0.0, 1.0
    b = (n * sum_yl - sum_y * sum_l) / det
    a = (sum_l - b * sum_y) / n
    # Ensure b > 0 (distribution must decay at large y)
    if b <= 0.0:
        b = 1.0 / y_grid[-1]
        a = np.log(1.0 / max(f_grid[-1], f_min) - 1.0) - b * y_grid[-1]
    return a, b


@njit
def _compute_all_tail_params(y_grid, f_all):
    """Compute tail fit parameters for all species. Returns shape (n_species, 2)."""
    n_species = f_all.shape[0]
    params = np.empty((n_species, 2))
    for s in range(n_species):
        a, b = _fit_tail_params(y_grid, f_all[s])
        params[s, 0] = a
        params[s, 1] = b
    return params


@njit
def _interp_grid(y, y_grid, f_grid, tail_a, tail_b):
    """
    Linear interpolation of f on the grid with FD-tail extrapolation.
    For y > y_grid[-1], uses f = 1/(exp(a + b*y) + 1) fitted from the tail.
    """
    Ny = len(y_grid)
    if y <= 0.0:
        return 0.0
    if y > y_grid[-1]:
        # FD-tail extrapolation
        arg = tail_a + tail_b * y
        if arg > 500.0:
            return 0.0
        return 1.0 / (np.exp(arg) + 1.0)
    if y <= y_grid[0]:
        return f_grid[0] * y / y_grid[0]
    # Find index
    dy = y_grid[1] - y_grid[0]
    idx = int((y - y_grid[0]) / dy)
    if idx >= Ny - 1:
        # FD-tail extrapolation
        arg = tail_a + tail_b * y
        if arg > 500.0:
            return 0.0
        return 1.0 / (np.exp(arg) + 1.0)
    frac = (y - y_grid[idx]) / dy
    return f_grid[idx] * (1.0 - frac) + f_grid[idx + 1] * frac


@njit
def _precompute_D_tables(y_grid):
    """
    Pre-compute D-kernel basis values for all (i1,i2,i3) grid triples.

    Returns D_k0, D_k1, D_k2: three (Ny,Ny,Ny) arrays corresponding to
    D_kernel_massless(y1,y2,y3,y4, 1,0,0), (0,1,0), (0,0,1) respectively,
    where y4 = y1+y2-y3 (energy conservation).

    This eliminates all D-function branching from the collision integral
    inner loops, providing a major speedup since the grid is fixed.
    """
    Ny = len(y_grid)
    D_k0 = np.zeros((Ny, Ny, Ny))
    D_k1 = np.zeros((Ny, Ny, Ny))
    D_k2 = np.zeros((Ny, Ny, Ny))
    for i1 in range(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10:
            continue
        for i2 in range(Ny):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            for i3 in range(Ny):
                y3 = y_grid[i3]
                y4 = y1 + y2 - y3
                if y4 <= 0.0 or y3 < 1.0e-10:
                    continue
                D_k0[i1, i2, i3] = D_kernel_massless(
                    y1, y2, y3, y4, 1.0, 0.0, 0.0)
                D_k1[i1, i2, i3] = D_kernel_massless(
                    y1, y2, y3, y4, 0.0, 1.0, 0.0)
                D_k2[i1, i2, i3] = D_kernel_massless(
                    y1, y2, y3, y4, 0.0, 0.0, 1.0)
    return D_k0, D_k1, D_k2


# Species indices: 0=nue, 1=nuebar, 2=numu (=nutau), 3=numubar (=nutaubar)
# With mu-tau symmetry: only 3 independent species (0,1,2)
# numu represents numu+nutau, numubar represents numubar+nutaubar

@njit
def _collision_integral_nu_nu(f_all, y_grid, dy, a, GF2_prefactor, tail_params,
                               D_k0, D_k2, Ny_coll):
    """
    Compute collision integrals for all neutrino species from nu-nu processes.
    Uses pre-computed D-kernel tables (D_k0 and D_k2) for speed.

    Ny_coll: number of grid points to include in the i2/i3 summation loops.
    This can be smaller than Ny to exclude high-y contributions where the
    D-kernel polynomial growth overwhelms the exponentially small f values,
    producing spurious Riemann sum artifacts.  The outer i1 loop still runs
    over the full grid.

    Processes:
    A) nu_a + nu_b -> nu_a + nu_b: D_A = D_k0 (c_D1=1)
    B) nu_a + nubar_a -> nu_a + nubar_a: D_B = 4*D_k2 (c_D3=4)
    C) nu_a + nubar_a -> nu_b + nubar_b: D_C = D_k2 (c_D3=1)
    """
    Ny = len(y_grid)
    I_coll = np.zeros((3, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1_nue = f_all[0, i1]
        f1_nuebar = f_all[1, i1]
        f1_numu = f_all[2, i1]

        I_nue = 0.0
        I_nuebar = 0.0
        I_numu = 0.0

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            f2_nue = f_all[0, i2]
            f2_nuebar = f_all[1, i2]
            f2_numu = f_all[2, i2]

            for i3 in range(Ny_coll):
                y4 = y1 + y2 - y_grid[i3]
                if y4 <= 0.0 or y_grid[i3] < 1.0e-10:
                    continue

                f3_nue = f_all[0, i3]
                f3_nuebar = f_all[1, i3]
                f3_numu = f_all[2, i3]

                f4_nue = _interp_grid(y4, y_grid, f_all[0],
                                      tail_params[0, 0], tail_params[0, 1])
                f4_nuebar = _interp_grid(y4, y_grid, f_all[1],
                                         tail_params[1, 0], tail_params[1, 1])
                f4_numu = _interp_grid(y4, y_grid, f_all[2],
                                       tail_params[2, 0], tail_params[2, 1])

                wt = dy * dy

                # Pre-computed D-kernels (no branching!)
                D_A = D_k0[i1, i2, i3]        # Process A: c_D1=1
                D_B = 4.0 * D_k2[i1, i2, i3]  # Process B: c_D3=4
                D_C = D_k2[i1, i2, i3]        # Process C: c_D3=1

                # Process A: different-flavor scattering
                # nue(1) + numu(2): 4x (numu, numubar, nutau, nutaubar)
                F_stat = (f3_nue * f4_numu * (1.0 - f1_nue) * (1.0 - f2_numu)
                          - f1_nue * f2_numu * (1.0 - f3_nue) * (1.0 - f4_numu))
                I_nue += 4.0 * wt * D_A * F_stat

                # nuebar(1) + numu(2): 4x
                F_stat = (f3_nuebar * f4_numu * (1.0 - f1_nuebar) * (1.0 - f2_numu)
                          - f1_nuebar * f2_numu * (1.0 - f3_nuebar) * (1.0 - f4_numu))
                I_nuebar += 4.0 * wt * D_A * F_stat

                # numu(1) + nue(2)
                F_stat = (f3_numu * f4_nue * (1.0 - f1_numu) * (1.0 - f2_nue)
                          - f1_numu * f2_nue * (1.0 - f3_numu) * (1.0 - f4_nue))
                I_numu += wt * D_A * F_stat

                # numu(1) + nuebar(2)
                F_stat = (f3_numu * f4_nuebar * (1.0 - f1_numu) * (1.0 - f2_nuebar)
                          - f1_numu * f2_nuebar * (1.0 - f3_numu) * (1.0 - f4_nuebar))
                I_numu += wt * D_A * F_stat

                # numu(1) + nutau(2) + nutaubar(2): 2x, nutau=numu
                F_stat = (f3_numu * f4_numu * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_numu += 2.0 * wt * D_A * F_stat

                # Process B: same-flavor forward scattering
                # nue(1) + nuebar(2)
                F_stat = (f3_nue * f4_nuebar * (1.0 - f1_nue) * (1.0 - f2_nuebar)
                          - f1_nue * f2_nuebar * (1.0 - f3_nue) * (1.0 - f4_nuebar))
                I_nue += wt * D_B * F_stat

                # nuebar(1) + nue(2)
                F_stat = (f3_nuebar * f4_nue * (1.0 - f1_nuebar) * (1.0 - f2_nue)
                          - f1_nuebar * f2_nue * (1.0 - f3_nuebar) * (1.0 - f4_nue))
                I_nuebar += wt * D_B * F_stat

                # numu(1) + numubar(2), numubar=numu
                F_stat = (f3_numu * f4_numu * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_numu += wt * D_B * F_stat

                # Process C: pair annihilation to different flavor
                # nue(1) + nuebar(2) -> numu(3) + numubar(4): 2x
                F_stat = (f3_numu * f4_numu * (1.0 - f1_nue) * (1.0 - f2_nuebar)
                          - f1_nue * f2_nuebar * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_nue += 2.0 * wt * D_C * F_stat

                # nuebar(1) + nue(2) -> numubar(3) + numu(4): 2x
                F_stat = (f3_numu * f4_numu * (1.0 - f1_nuebar) * (1.0 - f2_nue)
                          - f1_nuebar * f2_nue * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_nuebar += 2.0 * wt * D_C * F_stat

                # numu(1) + numubar(2) -> nue(3) + nuebar(4)
                F_stat = (f3_nue * f4_nuebar * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_nue) * (1.0 - f4_nuebar))
                I_numu += wt * D_C * F_stat

                # numu(1) + numubar(2) -> nutau(3) + nutaubar(4)
                F_stat = (f3_numu * f4_numu * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_numu += wt * D_C * F_stat

        I_coll[0, i1] = prefactor / (y1 * y1) * I_nue
        I_coll[1, i1] = prefactor / (y1 * y1) * I_nuebar
        I_coll[2, i1] = prefactor / (y1 * y1) * I_numu

    return I_coll


@njit
def _collision_integral_nu_e(f_all, y_grid, dy, a, Tg, GF2_prefactor,
                              geL2, geR2, geLgeR, gmuL2, gmuR2, gmuLgmuR,
                              me, fnu_e_scat_val, fnu_e_ann_val,
                              fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                              D_k0, D_k1, D_k2, Ny_coll):
    """
    Compute collision integrals from neutrino-electron processes.
    Uses pre-computed D-kernel tables D_k0, D_k1, and D_k2.

    Ny_coll: number of grid points to include in the i2/i3 summation loops.

    Scattering nu(1)+e(2)->nu(3)+e(4): |M|^2 ~ gL^2*s^2 + gR^2*u^2
      -> D_scat = gL^2 * D_k0 + gR^2 * D_k2  (s-channel and u-channel)
    Annihilation nu(1)+nubar(2)->e+(3)+e-(4): |M|^2 ~ gL^2*t^2 + gR^2*u^2
      -> D_ann = gL^2 * D_k1 + gR^2 * D_k2  (t-channel and u-channel)
    """
    Ny = len(y_grid)
    I_coll = np.zeros((3, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_comoving = Tg * a  # comoving temperature for electron distribution

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1_nue = f_all[0, i1]
        f1_nuebar = f_all[1, i1]
        f1_numu = f_all[2, i1]

        I_nue = 0.0
        I_nuebar = 0.0
        I_numu = 0.0

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            # Electron distribution (thermal, massless approx in comoving coords)
            x_e2 = y2 / Te_comoving
            if x_e2 > 500.0:
                f2_e = 0.0
            else:
                f2_e = 1.0 / (np.exp(x_e2) + 1.0)

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                y4 = y1 + y2 - y3
                if y4 <= 0.0 or y3 < 1.0e-10:
                    continue

                f3_nue = f_all[0, i3]
                f3_nuebar = f_all[1, i3]
                f3_numu = f_all[2, i3]
                x_e3 = y3 / Te_comoving
                if x_e3 > 500.0:
                    f3_e = 0.0
                else:
                    f3_e = 1.0 / (np.exp(x_e3) + 1.0)

                f4_nue = _interp_grid(y4, y_grid, f_all[0],
                                      tail_params[0, 0], tail_params[0, 1])
                f4_nuebar = _interp_grid(y4, y_grid, f_all[1],
                                         tail_params[1, 0], tail_params[1, 1])
                f4_numu = _interp_grid(y4, y_grid, f_all[2],
                                       tail_params[2, 0], tail_params[2, 1])
                x_e4 = y4 / Te_comoving
                if x_e4 > 500.0:
                    f4_e = 0.0
                else:
                    f4_e = 1.0 / (np.exp(x_e4) + 1.0)

                wt = dy * dy

                # Pre-computed D-kernel basis values
                dk0 = D_k0[i1, i2, i3]
                dk1 = D_k1[i1, i2, i3]
                dk2 = D_k2[i1, i2, i3]

                # --- Scattering: nu(1) + e(2) -> nu(3) + e(4) ---
                # From Sabti Table 3: S*GF^{-2}*a^{-4}*|M|^2 = 128[...]
                # In our 32*GF^2 convention, coupling coeffs are 128/32 = 4x
                # e-: 4*gL^2*D_k0 + 4*gR^2*D_k2
                # e+: 4*gR^2*D_k0 + 4*gL^2*D_k2 (gL<->gR swapped)
                # Combined e- + e+: 4*(gL^2+gR^2)*(D_k0+D_k2)
                D_scat_nue = 4.0 * (geL2 + geR2) * (dk0 + dk2) * fnu_e_scat_val
                D_scat_nuebar = D_scat_nue  # same after e-/e+ combination
                D_scat_numu = 4.0 * (gmuL2 + gmuR2) * (dk0 + dk2) * fnu_mu_scat_val

                # nu_e(1) + e(2) -> nu_e(3) + e(4): e- and e+ combined
                F_stat = (f3_nue * f4_e * (1.0 - f1_nue) * (1.0 - f2_e)
                          - f1_nue * f2_e * (1.0 - f3_nue) * (1.0 - f4_e))
                I_nue += wt * D_scat_nue * F_stat

                # nuebar(1) + e(2): e- and e+ combined
                F_stat = (f3_nuebar * f4_e * (1.0 - f1_nuebar) * (1.0 - f2_e)
                          - f1_nuebar * f2_e * (1.0 - f3_nuebar) * (1.0 - f4_e))
                I_nuebar += wt * D_scat_nuebar * F_stat

                # numu(1) + e(2): e- and e+ combined
                F_stat = (f3_numu * f4_e * (1.0 - f1_numu) * (1.0 - f2_e)
                          - f1_numu * f2_e * (1.0 - f3_numu) * (1.0 - f4_e))
                I_numu += wt * D_scat_numu * F_stat

                # --- Annihilation: nu(1) + nubar(2) -> e+(3) + e-(4) ---
                # From Sabti Table 3: 128[gL^2*(Y1.Y3)(Y2.Y4) + gR^2*(Y1.Y4)(Y2.Y3)]
                # Coupling coeffs: 4*gL^2 for D_k1, 4*gR^2 for D_k2
                f2_nuebar_ann = f_all[1, i2]
                f2_nue_ann = f_all[0, i2]
                f2_numu_ann = f_all[2, i2]

                D_ann_nue = 4.0 * (geL2 * dk1 + geR2 * dk2) * fnu_e_ann_val
                D_ann_numu = 4.0 * (gmuL2 * dk1 + gmuR2 * dk2) * fnu_mu_ann_val

                # nue(1) + nuebar(2) -> e+ + e-
                F_stat = (f3_e * f4_e * (1.0 - f1_nue) * (1.0 - f2_nuebar_ann)
                          - f1_nue * f2_nuebar_ann * (1.0 - f3_e) * (1.0 - f4_e))
                I_nue += wt * D_ann_nue * F_stat

                # nuebar(1) + nue(2) -> e+ + e-
                F_stat = (f3_e * f4_e * (1.0 - f1_nuebar) * (1.0 - f2_nue_ann)
                          - f1_nuebar * f2_nue_ann * (1.0 - f3_e) * (1.0 - f4_e))
                I_nuebar += wt * D_ann_nue * F_stat

                # numu(1) + numubar(2) -> e+ + e-
                F_stat = (f3_e * f4_e * (1.0 - f1_numu) * (1.0 - f2_numu_ann)
                          - f1_numu * f2_numu_ann * (1.0 - f3_e) * (1.0 - f4_e))
                I_numu += wt * D_ann_numu * F_stat

        I_coll[0, i1] += prefactor / (y1 * y1) * I_nue
        I_coll[1, i1] += prefactor / (y1 * y1) * I_nuebar
        I_coll[2, i1] += prefactor / (y1 * y1) * I_numu

    return I_coll


###############################################################################
# BoltzmannSolver class                                                        #
###############################################################################

class BoltzmannSolver(object):
    """
    Boltzmann solver for neutrino distribution functions on a comoving
    momentum grid. Computes SM 2-to-2 collision integrals and evolves
    f_nu(y, t) coupled with the photon temperature Tg(t).
    """

    def __init__(self, Ny=None, y_max=None, y_coll_max=None, C_NP_funcs=None):
        """
        Initialize the solver.

        Parameters
        ----------
        Ny : int, optional
            Number of grid points (default: PRyMini.Ny_boltz)
        y_max : float, optional
            Maximum comoving momentum in MeV (default: PRyMini.y_max_boltz)
        y_coll_max : float, optional
            Maximum comoving momentum for collision integral summation.
            Beyond this, D-kernel polynomial growth overwhelms exponentially
            small f, producing spurious Riemann sum artifacts. The grid still
            extends to y_max for energy density representation. If None,
            defaults to y_max (no cutoff).
        C_NP_funcs : dict, optional
            NP collision term callbacks: {'nue': func, 'nuebar': func, 'numu': func}
            Each func(y_grid, a, Tg, f_all) -> array(Ny)
        """
        self.Ny = Ny if Ny is not None else PRyMini.Ny_boltz
        self.y_max = y_max if y_max is not None else PRyMini.y_max_boltz
        self.dy = self.y_max / self.Ny
        self.y_grid = np.linspace(self.dy / 2.0, self.y_max - self.dy / 2.0, self.Ny)
        self.n_species = 3  # nue, nuebar, numu (mu-tau symmetric)

        # Collision integral summation cutoff
        if y_coll_max is not None and y_coll_max < self.y_max:
            self.Ny_coll = int(np.searchsorted(self.y_grid, y_coll_max)) + 1
            self.Ny_coll = min(self.Ny_coll, self.Ny)
        else:
            self.Ny_coll = self.Ny

        # Coupling constants
        self.GF2_prefactor = 32.0 * PRyMini.GF**2
        self.geL2 = PRyMini.geL**2
        self.geR2 = PRyMini.geR**2
        self.geLgeR = PRyMini.geL * PRyMini.geR
        self.gmuL2 = PRyMini.gmuL**2
        self.gmuR2 = PRyMini.gmuR**2
        self.gmuLgmuR = PRyMini.gmuL * PRyMini.gmuR

        # NP collision terms
        self.C_NP_funcs = C_NP_funcs if C_NP_funcs is not None else {}

        # Load tabulated finite-mass correction factors
        import PRyM.PRyM_thermo as PRyMthermo
        self._fnu_e_scat = PRyMthermo.fnu_e_scat
        self._fnu_e_ann = PRyMthermo.fnu_e_ann
        self._fnu_mu_scat = PRyMthermo.fnu_mu_scat
        self._fnu_mu_ann = PRyMthermo.fnu_mu_ann

        # Pre-compute D-kernel lookup tables (one-time cost, grid-dependent only)
        if PRyMini.verbose_flag:
            print(f"BoltzmannSolver: Ny={self.Ny}, y_max={self.y_max:.1f} MeV, "
                  f"dy={self.dy:.2f} MeV")
            print(f"  Pre-computing D-kernel tables ({self.Ny}^3 = "
                  f"{self.Ny**3} entries)...")
        self.D_k0, self.D_k1, self.D_k2 = _precompute_D_tables(self.y_grid)
        if PRyMini.verbose_flag:
            print(f"  D-kernel tables ready.")

        # Oscillation mixing matrix (identity if disabled)
        if PRyMini.nu_oscillation_flag:
            self._setup_oscillation_matrix()
        else:
            self.P_osc = np.eye(self.n_species)

    def _setup_oscillation_matrix(self):
        """Set up time-averaged PMNS oscillation probability matrix."""
        # PMNS mixing angles (PDG 2023)
        s12_2 = 0.307  # sin^2(theta_12)
        s23_2 = 0.546  # sin^2(theta_23)
        s13_2 = 0.0220 # sin^2(theta_13)
        c12_2 = 1.0 - s12_2
        c13_2 = 1.0 - s13_2
        c13_4 = c13_2**2
        # Time-averaged probabilities (vacuum, 3-flavor)
        P_ee = 1.0 - 0.5 * (4.0 * s12_2 * c12_2 * c13_4 + 4.0 * s13_2 * c13_2)
        P_emu = 0.5 * (4.0 * s12_2 * c12_2 * c13_4 * s23_2
                       + 4.0 * s13_2 * c13_2 * s23_2)  # simplified
        P_etau = 1.0 - P_ee - P_emu
        # With mu-tau symmetry: P_mumu ~ P_tautau, P_mutau ~ etc.
        # Simplified 3x3 for (nue, nuebar, numu_equiv)
        # nuebar has same mixing as nue (CPT), numu_equiv averages mu+tau
        self.P_osc = np.array([
            [P_ee, P_ee, P_emu],       # nue mixes with itself, nuebar, numu
            [P_ee, P_ee, P_emu],       # nuebar ~ nue
            [P_emu, P_emu, 1.0 - P_emu], # numu
        ])

    def initial_conditions(self, Tg, a):
        """
        Return thermal Fermi-Dirac distributions on the comoving grid.

        f_alpha(y_i) = 1 / (exp(y_i / (Tnu * a)) + 1)

        where Tnu is the neutrino temperature (= Tg at early times before
        significant e+e- annihilation heating).
        """
        f_all = np.zeros((self.n_species, self.Ny))
        # At T_boltz_start ~ 5 MeV, neutrinos are still nearly thermal at Tg
        Tnu_com = Tg * a  # comoving neutrino temperature
        for i in range(self.Ny):
            x = self.y_grid[i] / Tnu_com
            if x < 500.0:
                f_all[:, i] = 1.0 / (np.exp(x) + 1.0)
        return f_all

    def collision_integrals(self, f_all, a, Tg):
        """
        Compute total collision integrals for all species.

        Returns array of shape (n_species, Ny).
        """
        # Convert units: GF in MeV^{-2}, collision integral in MeV * s^{-1}
        GF2_pref = self.GF2_prefactor * PRyMini.MeV_to_secm1

        # Compute FD-tail extrapolation parameters for off-grid interpolation
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        # Nu-nu processes (uses D_k0, D_k2)
        I_nu_nu = _collision_integral_nu_nu(
            f_all, self.y_grid, self.dy, a, GF2_pref, tail_params,
            self.D_k0, self.D_k2, self.Ny_coll)

        # Nu-e processes (uses D_k0 for scattering, D_k1 for annihilation, D_k2 for both)
        fnu_e_scat_val = float(self._fnu_e_scat(Tg))
        fnu_e_ann_val = float(self._fnu_e_ann(Tg))
        fnu_mu_scat_val = float(self._fnu_mu_scat(Tg))
        fnu_mu_ann_val = float(self._fnu_mu_ann(Tg))

        I_nu_e = _collision_integral_nu_e(
            f_all, self.y_grid, self.dy, a, Tg, GF2_pref,
            self.geL2, self.geR2, self.geLgeR,
            self.gmuL2, self.gmuR2, self.gmuLgmuR,
            PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
            fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
            self.D_k0, self.D_k1, self.D_k2, self.Ny_coll)

        I_total = I_nu_nu + I_nu_e

        # Apply oscillation mixing if enabled
        if PRyMini.nu_oscillation_flag:
            I_mixed = np.zeros_like(I_total)
            for alpha in range(self.n_species):
                for beta in range(self.n_species):
                    I_mixed[alpha] += self.P_osc[alpha, beta] * I_total[beta]
            I_total = I_mixed

        # Add NP collision terms
        if 'nue' in self.C_NP_funcs:
            I_total[0] += self.C_NP_funcs['nue'](self.y_grid, a, Tg, f_all)
        if 'nuebar' in self.C_NP_funcs:
            I_total[1] += self.C_NP_funcs['nuebar'](self.y_grid, a, Tg, f_all)
        if 'numu' in self.C_NP_funcs:
            I_total[2] += self.C_NP_funcs['numu'](self.y_grid, a, Tg, f_all)

        return I_total

    def energy_transfer_rate(self, I_coll, a):
        """
        Compute the net energy transfer rate from neutrinos to plasma.

        delta_rho = sum_alpha g_alpha/(2*pi^2*a^4) * integral dy y^2 E_tilde * I_alpha

        For massless neutrinos E_tilde = y, so:
        delta_rho = sum_alpha 1/(2*pi^2*a^4) * integral dy y^3 * I_alpha

        Returns delta_rho in MeV^4 / s (after MeV_to_secm1 conversion in I_coll).
        """
        delta_rho = 0.0
        for alpha in range(self.n_species):
            # g_alpha = 1 per species (particle+antiparticle counted separately)
            # Multiply by 2 for numu to account for nutau
            mult = 2.0 if alpha == 2 else 1.0
            integrand = self.y_grid**3 * I_coll[alpha]
            delta_rho += mult * self.dy / (2.0 * np.pi**2 * a**4) * np.sum(integrand)
        return delta_rho

    def update_thermo_distributions(self, f_all, a, a_of_T_func=None):
        """
        Convert comoving grid values to f_nu(p, Tg) callables and update
        PRyMthermo module-level globals.

        The mapping is: physical momentum p = y / a, so for given (p, Tg),
        y = p * a(Tg). We interpolate f_all on the y_grid.

        If a_of_T_func is provided, the callables use a(Tg) dynamically
        (for frozen distributions where a changes but f(y) doesn't).
        Otherwise, a is used as a fixed constant.
        """
        import PRyM.PRyM_thermo as PRyMthermo

        # Create interpolators for each species on the comoving grid
        y_grid = self.y_grid
        f_nue_grid = f_all[0].copy()
        f_nuebar_grid = f_all[1].copy()
        f_numu_grid = f_all[2].copy()
        current_a = a
        a_func = a_of_T_func

        def _make_f_callable(f_grid, a_val, a_of_T=None):
            # Fit FD-tail extrapolation parameters: log(1/f - 1) = tail_a + tail_b * y
            f_min = 1.0e-12
            tail_y = []
            tail_l = []
            for i in range(len(y_grid) - 1, -1, -1):
                fi = f_grid[i]
                if fi > f_min and fi < 1.0 - f_min:
                    tail_y.append(y_grid[i])
                    tail_l.append(np.log(1.0/fi - 1.0))
                    if len(tail_y) >= 10:
                        break
            if len(tail_y) >= 2:
                tail_y = np.array(tail_y)
                tail_l = np.array(tail_l)
                coeffs = np.polyfit(tail_y, tail_l, 1)
                _tail_b, _tail_a = coeffs[0], coeffs[1]
                if _tail_b <= 0:
                    _tail_b = 1.0 / y_grid[-1]
                    _tail_a = np.log(1.0/max(f_grid[-1], f_min) - 1.0) - _tail_b * y_grid[-1]
            else:
                _tail_a, _tail_b = 0.0, 1.0
            y_max_grid = y_grid[-1]

            f_interp = interp1d(y_grid, f_grid, bounds_error=False,
                                fill_value=(f_grid[0], 0.0), kind='linear')

            def _eval_f(y_arr):
                """Evaluate f with FD-tail extrapolation beyond grid."""
                y_arr = np.asarray(y_arr)
                result = f_interp(y_arr)
                # Apply FD-tail extrapolation where y > grid max
                mask = y_arr > y_max_grid
                if np.any(mask):
                    arg = _tail_a + _tail_b * y_arr[mask]
                    arg = np.clip(arg, -500.0, 500.0)
                    result[mask] = 1.0 / (np.exp(arg) + 1.0)
                return result

            if a_of_T is not None:
                def f_nu(p, Tg):
                    y = np.asarray(p, dtype=float) * a_of_T(Tg)
                    return _eval_f(y)
            else:
                def f_nu(p, Tg):
                    y = np.asarray(p, dtype=float) * a_val
                    return _eval_f(y)
            return f_nu

        PRyMthermo.f_nue_general = _make_f_callable(f_nue_grid, current_a, a_func)
        PRyMthermo.f_nuebar_general = _make_f_callable(f_nuebar_grid, current_a, a_func)
        PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
        PRyMthermo.f_numubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
        PRyMthermo.f_nutau_general = _make_f_callable(f_numu_grid, current_a, a_func)
        PRyMthermo.f_nutaubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
