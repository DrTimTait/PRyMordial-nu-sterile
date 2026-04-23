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
from scipy.linalg import expm
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

@njit(cache=True)
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


@njit(cache=True)
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
        return yj * (3.0 * (yk**2 + yl**2 - yi**2) - yj**2) / 6.0

    if yi + yj <= yk + yl and yi + yl <= yj + yk:
        return -((yi + yj) * ((yi + yj)**2 - 3.0 * (yk**2 + yl**2))
                 + 2.0 * (yk**3 - yl**3)) / 12.0

    return 0.0


@njit(cache=True)
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


@njit(cache=True)
def D1(y1, y2, y3, y4):
    """D1 with proper ordering enforced: yi >= yj and yk >= yl."""
    yi = max(y1, y2)
    yj = min(y1, y2)
    yk = max(y3, y4)
    yl = min(y3, y4)
    return _D1_raw(yi, yj, yk, yl)


@njit(cache=True)
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


@njit(cache=True)
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

@njit(cache=True)
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


@njit(cache=True)
def D_kernel_massive(y1, y2, y3, y4, E1, E2, E3, E4, c_D1, c_D2, c_D3):
    """
    Compute the full angular kernel with massive particles (Sabti E.21).

    Same structure as D_kernel_massless but uses separate energy prefactors
    Ei instead of assuming Ei = yi (massless). The D1, D2, D3 piecewise
    polynomials still operate on 3-momenta yi only.

    For massive particle i: Ei = sqrt(yi^2 + mi^2*a^2)
    For massless particle i: Ei = yi
    Setting all Ei = yi recovers D_kernel_massless exactly.
    """
    s1, s2, s3, s4 = -1.0, -1.0, 1.0, 1.0
    result = 0.0

    if c_D1 != 0.0:
        # (Y1.Y2)(Y3.Y4) structure
        d1_val = D1(y1, y2, y3, y4)
        d2_12 = D2(y1, y2, y3, y4, s3, s4)
        d2_34 = D2(y3, y4, y1, y2, s1, s2)
        d3_val = D3(y1, y2, y3, y4, s1, s2, s3, s4)
        result += c_D1 * (E1*E2*E3*E4 * d1_val + E1*E2 * d2_12
                          + E3*E4 * d2_34 + d3_val)

    if c_D2 != 0.0:
        # (Y1.Y3)(Y2.Y4) structure: swap 2<->3
        d1_val = D1(y1, y3, y2, y4)
        d2_13 = D2(y1, y3, y2, y4, s2, s4)
        d2_24 = D2(y2, y4, y1, y3, s1, s3)
        d3_val = D3(y1, y3, y2, y4, s1, s3, s2, s4)
        result += c_D2 * (E1*E3*E2*E4 * d1_val + E1*E3 * d2_13
                          + E2*E4 * d2_24 + d3_val)

    if c_D3 != 0.0:
        # (Y1.Y4)(Y2.Y3) structure: swap 2<->4
        d1_val = D1(y1, y4, y2, y3)
        d2_14 = D2(y1, y4, y2, y3, s2, s3)
        d2_23 = D2(y2, y3, y1, y4, s1, s4)
        d3_val = D3(y1, y4, y2, y3, s1, s4, s2, s3)
        result += c_D3 * (E1*E4*E2*E3 * d1_val + E1*E4 * d2_14
                          + E2*E3 * d2_23 + d3_val)

    return result


###############################################################################
# Pre-computed D-kernel tables and collision integral computation              #
###############################################################################

@njit(cache=True)
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


@njit(cache=True)
def _compute_all_tail_params(y_grid, f_all):
    """Compute tail fit parameters for all species. Returns shape (n_species, 2)."""
    n_species = f_all.shape[0]
    params = np.empty((n_species, 2))
    for s in range(n_species):
        a, b = _fit_tail_params(y_grid, f_all[s])
        params[s, 0] = a
        params[s, 1] = b
    return params


@njit(cache=True)
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


@njit(cache=True)
def _quad_weights(N, dy):
    """
    Composite Simpson's quadrature weights for N equally-spaced points.

    For even N-1 intervals: pure Simpson's 1/3 (O(dy^4) error).
    For odd N-1 intervals: Simpson's 1/3 for first N-4 intervals,
    Simpson's 3/8 for last 3 intervals.
    Falls back to midpoint rule (O(dy^2)) for N < 3.
    """
    w = np.empty(N)
    if N < 3:
        w[:] = dy
        return w
    n_int = N - 1  # number of intervals
    if n_int % 2 == 0:
        # Pure Simpson's 1/3
        w[0] = dy / 3.0
        for i in range(1, N - 1):
            if i % 2 == 1:
                w[i] = 4.0 * dy / 3.0
            else:
                w[i] = 2.0 * dy / 3.0
        w[N - 1] = dy / 3.0
    else:
        # Mixed: Simpson 1/3 for first N-4 intervals, 3/8 for last 3
        if N >= 5:
            m = N - 3  # points in 1/3 section (indices 0..m-1)
            w[0] = dy / 3.0
            for i in range(1, m - 1):
                if i % 2 == 1:
                    w[i] = 4.0 * dy / 3.0
                else:
                    w[i] = 2.0 * dy / 3.0
            # Shared point: end of 1/3 + start of 3/8
            w[m - 1] = dy / 3.0 + 3.0 * dy / 8.0
            # 3/8 rule for last 4 points
            w[m] = 9.0 * dy / 8.0
            w[m + 1] = 9.0 * dy / 8.0
            w[N - 1] = 3.0 * dy / 8.0
        elif N == 4:
            w[0] = 3.0 * dy / 8.0
            w[1] = 9.0 * dy / 8.0
            w[2] = 9.0 * dy / 8.0
            w[3] = 3.0 * dy / 8.0
        else:
            # N == 3
            w[0] = dy / 3.0
            w[1] = 4.0 * dy / 3.0
            w[2] = dy / 3.0
    return w


@njit(cache=True)
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

@njit(cache=True)
def _collision_integral_nu_nu(f_all, y_grid, quad_w, a, GF2_prefactor, tail_params,
                               D_k0, D_k2, Ny_coll,
                               scale_nunu_A=1.0, scale_nunu_B=1.0, scale_nunu_C=1.0):
    """
    Compute collision integrals for all neutrino species from nu-nu processes.
    Uses pre-computed D-kernel tables (D_k0 and D_k2) for speed.

    Ny_coll: number of grid points to include in the i2/i3 summation loops.
    This can be smaller than Ny to exclude high-y contributions where the
    D-kernel polynomial growth overwhelms the exponentially small f values,
    producing spurious Riemann sum artifacts.  The outer i1 loop still runs
    over the full grid.

    Processes (Sabti et al. Table 3):
    A) Different-flavor scattering: nu+nubar uses D_k0, nu+nu uses D_k2
    B) Same-flavor nu+nubar forward: 2*D_k0
    C) Pair annihilation nu+nubar -> nu'+nubar': D_k2
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

                wt = quad_w[i2] * quad_w[i3]

                # Pre-computed D-kernels
                # Process A has two sub-processes (Sabti Table 3):
                #   nu+nubar (diff-flavor): |M|^2 ~ (Y1.Y2)(Y3.Y4) -> D_k0
                #   nu+nu    (diff-flavor): |M|^2 ~ (Y1.Y4)(Y2.Y3) -> D_k2
                # Each species has equal numbers of nu and nubar partners,
                # so the average D-kernel per partner is (D_k0 + D_k2)/2.
                D_k0_val = D_k0[i1, i2, i3]
                D_k2_val = D_k2[i1, i2, i3]
                D_A_sum = scale_nunu_A * (D_k0_val + D_k2_val)
                D_B = scale_nunu_B * 2.0 * D_k0_val              # Process B: nu+nubar same-flavor (Table 3 row 3)
                D_C = scale_nunu_C * D_k2_val                     # Process C: pair annihilation

                # Process A: different-flavor scattering
                # nue(1) + numu(2): 4 partners (numu, numubar, nutau, nutaubar)
                # 2 are nu+nubar (D_k0) + 2 are nu+nu (D_k2) = 2*(D_k0+D_k2)
                F_stat = (f3_nue * f4_numu * (1.0 - f1_nue) * (1.0 - f2_numu)
                          - f1_nue * f2_numu * (1.0 - f3_nue) * (1.0 - f4_numu))
                I_nue += 2.0 * wt * D_A_sum * F_stat

                # nuebar(1) + numu(2): 4 partners, same split
                F_stat = (f3_nuebar * f4_numu * (1.0 - f1_nuebar) * (1.0 - f2_numu)
                          - f1_nuebar * f2_numu * (1.0 - f3_nuebar) * (1.0 - f4_numu))
                I_nuebar += 2.0 * wt * D_A_sum * F_stat

                # numu(1) + nue(2): averaged over nu_mu + nubar_mu
                F_stat = (f3_numu * f4_nue * (1.0 - f1_numu) * (1.0 - f2_nue)
                          - f1_numu * f2_nue * (1.0 - f3_numu) * (1.0 - f4_nue))
                I_numu += 0.5 * wt * D_A_sum * F_stat

                # numu(1) + nuebar(2): averaged over nu_mu + nubar_mu
                F_stat = (f3_numu * f4_nuebar * (1.0 - f1_numu) * (1.0 - f2_nuebar)
                          - f1_numu * f2_nuebar * (1.0 - f3_numu) * (1.0 - f4_nuebar))
                I_numu += 0.5 * wt * D_A_sum * F_stat

                # numu(1) + nutau(2) + nutaubar(2): 2 partners, each avg'd
                F_stat = (f3_numu * f4_numu * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_numu += wt * D_A_sum * F_stat

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


@njit(parallel=True, cache=True)
def _collision_integral_nu_nu_asym4(f_all, y_grid, quad_w, a, GF2_prefactor,
                                     tail_params, D_k0, D_k2, Ny_coll,
                                     scale_nunu_A=1.0, scale_nunu_B=1.0, scale_nunu_C=1.0):
    """n=4 asymmetric nu-nu collision integral (mu-tau symmetry broken).

    Species layout: f_all shape (4, Ny) = [nue, nuebar, numu_eff, nutau_eff].
    Each of numu_eff, nutau_eff still aggregates particle+antiparticle for that
    flavor (shared distribution under CPT in SM BBN). The function is derived
    from _collision_integral_nu_nu by splitting every mu-tau-combined factor
    into explicit mu and tau pieces, so that when f_all[2] == f_all[3] the
    result reproduces the n=3 case exactly (I_coll[2] == I_coll[3] == n=3 I_numu).

    See _collision_integral_nu_nu for the Sabti Process A/B/C conventions.
    """
    Ny = len(y_grid)
    I_coll = np.zeros((4, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1_nue = f_all[0, i1]
        f1_nuebar = f_all[1, i1]
        f1_numu = f_all[2, i1]
        f1_nutau = f_all[3, i1]

        I_nue = 0.0
        I_nuebar = 0.0
        I_numu = 0.0
        I_nutau = 0.0

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            f2_nue = f_all[0, i2]
            f2_nuebar = f_all[1, i2]
            f2_numu = f_all[2, i2]
            f2_nutau = f_all[3, i2]

            for i3 in range(Ny_coll):
                y4 = y1 + y2 - y_grid[i3]
                if y4 <= 0.0 or y_grid[i3] < 1.0e-10:
                    continue

                f3_nue = f_all[0, i3]
                f3_nuebar = f_all[1, i3]
                f3_numu = f_all[2, i3]
                f3_nutau = f_all[3, i3]

                f4_nue = _interp_grid(y4, y_grid, f_all[0],
                                      tail_params[0, 0], tail_params[0, 1])
                f4_nuebar = _interp_grid(y4, y_grid, f_all[1],
                                         tail_params[1, 0], tail_params[1, 1])
                f4_numu = _interp_grid(y4, y_grid, f_all[2],
                                       tail_params[2, 0], tail_params[2, 1])
                f4_nutau = _interp_grid(y4, y_grid, f_all[3],
                                        tail_params[3, 0], tail_params[3, 1])

                wt = quad_w[i2] * quad_w[i3]
                D_k0_val = D_k0[i1, i2, i3]
                D_k2_val = D_k2[i1, i2, i3]
                D_A_sum = scale_nunu_A * (D_k0_val + D_k2_val)
                D_B = scale_nunu_B * 2.0 * D_k0_val
                D_C = scale_nunu_C * D_k2_val

                # --- Process A (different-flavor scattering) ---
                # nue(1) with mu-sector partner: 2 partners (numu + numubar) = D_A_sum
                F_stat = (f3_nue * f4_numu * (1.0 - f1_nue) * (1.0 - f2_numu)
                          - f1_nue * f2_numu * (1.0 - f3_nue) * (1.0 - f4_numu))
                I_nue += wt * D_A_sum * F_stat
                # nue(1) with tau-sector partner: 2 partners (nutau + nutaubar) = D_A_sum
                F_stat = (f3_nue * f4_nutau * (1.0 - f1_nue) * (1.0 - f2_nutau)
                          - f1_nue * f2_nutau * (1.0 - f3_nue) * (1.0 - f4_nutau))
                I_nue += wt * D_A_sum * F_stat

                # nuebar(1) with mu-sector partner
                F_stat = (f3_nuebar * f4_numu * (1.0 - f1_nuebar) * (1.0 - f2_numu)
                          - f1_nuebar * f2_numu * (1.0 - f3_nuebar) * (1.0 - f4_numu))
                I_nuebar += wt * D_A_sum * F_stat
                # nuebar(1) with tau-sector partner
                F_stat = (f3_nuebar * f4_nutau * (1.0 - f1_nuebar) * (1.0 - f2_nutau)
                          - f1_nuebar * f2_nutau * (1.0 - f3_nuebar) * (1.0 - f4_nutau))
                I_nuebar += wt * D_A_sum * F_stat

                # numu(1) with nue(2): 0.5*D_A_sum (averaging over mu pcle/antipcle)
                F_stat = (f3_numu * f4_nue * (1.0 - f1_numu) * (1.0 - f2_nue)
                          - f1_numu * f2_nue * (1.0 - f3_numu) * (1.0 - f4_nue))
                I_numu += 0.5 * wt * D_A_sum * F_stat
                # numu(1) with nuebar(2)
                F_stat = (f3_numu * f4_nuebar * (1.0 - f1_numu) * (1.0 - f2_nuebar)
                          - f1_numu * f2_nuebar * (1.0 - f3_numu) * (1.0 - f4_nuebar))
                I_numu += 0.5 * wt * D_A_sum * F_stat
                # numu(1) with tau-sector partner (nutau + nutaubar = 2 partners): D_A_sum
                F_stat = (f3_numu * f4_nutau * (1.0 - f1_numu) * (1.0 - f2_nutau)
                          - f1_numu * f2_nutau * (1.0 - f3_numu) * (1.0 - f4_nutau))
                I_numu += wt * D_A_sum * F_stat

                # nutau(1) with nue(2)
                F_stat = (f3_nutau * f4_nue * (1.0 - f1_nutau) * (1.0 - f2_nue)
                          - f1_nutau * f2_nue * (1.0 - f3_nutau) * (1.0 - f4_nue))
                I_nutau += 0.5 * wt * D_A_sum * F_stat
                # nutau(1) with nuebar(2)
                F_stat = (f3_nutau * f4_nuebar * (1.0 - f1_nutau) * (1.0 - f2_nuebar)
                          - f1_nutau * f2_nuebar * (1.0 - f3_nutau) * (1.0 - f4_nuebar))
                I_nutau += 0.5 * wt * D_A_sum * F_stat
                # nutau(1) with mu-sector partner: D_A_sum
                F_stat = (f3_nutau * f4_numu * (1.0 - f1_nutau) * (1.0 - f2_numu)
                          - f1_nutau * f2_numu * (1.0 - f3_nutau) * (1.0 - f4_numu))
                I_nutau += wt * D_A_sum * F_stat

                # --- Process B (same-flavor forward scattering, nu+nubar) ---
                # nue(1) + nuebar(2)
                F_stat = (f3_nue * f4_nuebar * (1.0 - f1_nue) * (1.0 - f2_nuebar)
                          - f1_nue * f2_nuebar * (1.0 - f3_nue) * (1.0 - f4_nuebar))
                I_nue += wt * D_B * F_stat
                # nuebar(1) + nue(2)
                F_stat = (f3_nuebar * f4_nue * (1.0 - f1_nuebar) * (1.0 - f2_nue)
                          - f1_nuebar * f2_nue * (1.0 - f3_nuebar) * (1.0 - f4_nue))
                I_nuebar += wt * D_B * F_stat
                # numu(1) + numubar(2): same flavor = f1_numu, f2_numu
                F_stat = (f3_numu * f4_numu * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_numu += wt * D_B * F_stat
                # nutau(1) + nutaubar(2)
                F_stat = (f3_nutau * f4_nutau * (1.0 - f1_nutau) * (1.0 - f2_nutau)
                          - f1_nutau * f2_nutau * (1.0 - f3_nutau) * (1.0 - f4_nutau))
                I_nutau += wt * D_B * F_stat

                # --- Process C (pair annihilation to different flavor) ---
                # nue + nuebar -> numu + numubar:  2x (counts both final-state particle types)
                F_stat = (f3_numu * f4_numu * (1.0 - f1_nue) * (1.0 - f2_nuebar)
                          - f1_nue * f2_nuebar * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_nue += wt * D_C * F_stat
                # nue + nuebar -> nutau + nutaubar
                F_stat = (f3_nutau * f4_nutau * (1.0 - f1_nue) * (1.0 - f2_nuebar)
                          - f1_nue * f2_nuebar * (1.0 - f3_nutau) * (1.0 - f4_nutau))
                I_nue += wt * D_C * F_stat

                # nuebar + nue -> numubar + numu
                F_stat = (f3_numu * f4_numu * (1.0 - f1_nuebar) * (1.0 - f2_nue)
                          - f1_nuebar * f2_nue * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_nuebar += wt * D_C * F_stat
                # nuebar + nue -> nutaubar + nutau
                F_stat = (f3_nutau * f4_nutau * (1.0 - f1_nuebar) * (1.0 - f2_nue)
                          - f1_nuebar * f2_nue * (1.0 - f3_nutau) * (1.0 - f4_nutau))
                I_nuebar += wt * D_C * F_stat

                # numu + numubar -> nue + nuebar (and inverse contributes to numu loss)
                F_stat = (f3_nue * f4_nuebar * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_nue) * (1.0 - f4_nuebar))
                I_numu += wt * D_C * F_stat
                # numu + numubar -> nutau + nutaubar
                F_stat = (f3_nutau * f4_nutau * (1.0 - f1_numu) * (1.0 - f2_numu)
                          - f1_numu * f2_numu * (1.0 - f3_nutau) * (1.0 - f4_nutau))
                I_numu += wt * D_C * F_stat

                # nutau + nutaubar -> nue + nuebar
                F_stat = (f3_nue * f4_nuebar * (1.0 - f1_nutau) * (1.0 - f2_nutau)
                          - f1_nutau * f2_nutau * (1.0 - f3_nue) * (1.0 - f4_nuebar))
                I_nutau += wt * D_C * F_stat
                # nutau + nutaubar -> numu + numubar
                F_stat = (f3_numu * f4_numu * (1.0 - f1_nutau) * (1.0 - f2_nutau)
                          - f1_nutau * f2_nutau * (1.0 - f3_numu) * (1.0 - f4_numu))
                I_nutau += wt * D_C * F_stat

        I_coll[0, i1] = prefactor / (y1 * y1) * I_nue
        I_coll[1, i1] = prefactor / (y1 * y1) * I_nuebar
        I_coll[2, i1] = prefactor / (y1 * y1) * I_numu
        I_coll[3, i1] = prefactor / (y1 * y1) * I_nutau

    return I_coll


@njit(parallel=True, cache=True)
def _collision_integral_nu_nu_asym6(f_all, y_grid, quad_w, a, GF2_prefactor,
                                     tail_params, D_k0, D_k2, Ny_coll,
                                     scale_nunu_A=1.0, scale_nunu_B=1.0, scale_nunu_C=1.0):
    """n=6 full nu/nu-bar-per-flavor collision integral (Stage 3).

    Species layout: f_all shape (6, Ny) = [nue, nuebar, numu, numubar,
    nutau, nutaubar]. Each I[alpha] is the per-species collision rate
    (not aggregated, no D-kernel averaging approximation).

    Kinematic D-kernels:
      - same-sign (nu+nu or nubar+nubar), different flavor: D_k2
      - opposite-sign (nu+nubar), different flavor: D_k0
      - same-flavor forward (nu+nubar): D_B = 2*D_k0
      - pair annihilation (nu+nubar -> nu'+nubar'), different flavor: D_C = D_k2

    In the symmetric limit (f[0]==f[1], f[2]==f[3]==f[4]==f[5]) the
    per-species rates here are NOT identical to n=3 I[2]/2 because n=3
    averages D_k0 and D_k2 per partner. n=6 preserves the true kinematics,
    so there is a small (~few%) integrated shift in the per-species rate
    compared with the n=3 average — but the total energy-transfer rate
    (sum over species with weight 1 each) agrees to within that same
    margin because averaged and un-averaged integrals match at leading
    order for thermal distributions.
    """
    Ny = len(y_grid)
    I_coll = np.zeros((6, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)

    # Species index map (documentation)
    # 0: nue,      particle  (+)
    # 1: nuebar,   antiparticle (-)
    # 2: numu,     particle  (+)
    # 3: numubar,  antiparticle (-)
    # 4: nutau,    particle  (+)
    # 5: nutaubar, antiparticle (-)

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1 = np.empty(6)
        for s in range(6):
            f1[s] = f_all[s, i1]

        I = np.zeros(6)

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            f2 = np.empty(6)
            for s in range(6):
                f2[s] = f_all[s, i2]

            for i3 in range(Ny_coll):
                y4 = y1 + y2 - y_grid[i3]
                if y4 <= 0.0 or y_grid[i3] < 1.0e-10:
                    continue

                f3 = np.empty(6)
                f4 = np.empty(6)
                for s in range(6):
                    f3[s] = f_all[s, i3]
                    f4[s] = _interp_grid(y4, y_grid, f_all[s],
                                          tail_params[s, 0], tail_params[s, 1])

                wt = quad_w[i2] * quad_w[i3]
                dk0 = D_k0[i1, i2, i3]
                dk2 = D_k2[i1, i2, i3]
                D_A_2 = scale_nunu_A * dk2   # same-sign diff-flavor
                D_A_0 = scale_nunu_A * dk0   # opp-sign diff-flavor
                D_B_val = scale_nunu_B * 2.0 * dk0   # same-flavor forward
                D_C_val = scale_nunu_C * dk2         # pair annihilation

                # --- Process A: different-flavor scattering ---
                # For each species alpha, sum over partner species beta (different flavor).
                # Kernel: D_k2 if sign(alpha)==sign(beta), else D_k0.
                #
                # We enumerate by pair (alpha, beta) where beta has flavor != alpha.
                # Partner sets:
                #   e-flavor (alpha in {0,1}): beta in {2,3,4,5}
                #   mu-flavor (alpha in {2,3}): beta in {0,1,4,5}
                #   tau-flavor (alpha in {4,5}): beta in {0,1,2,3}

                # Partner lists (indices and sign)
                # Using signs: +1 for even alpha, -1 for odd alpha
                # Precompute: loop over (alpha, beta_list)
                # For brevity, unroll the 4x6=24 scattering terms:

                # nue (alpha=0, sign=+) with partners 2,3,4,5
                Fs = f3[0]*f4[2]*(1.0-f1[0])*(1.0-f2[2]) - f1[0]*f2[2]*(1.0-f3[0])*(1.0-f4[2])
                I[0] += wt * D_A_2 * Fs
                Fs = f3[0]*f4[3]*(1.0-f1[0])*(1.0-f2[3]) - f1[0]*f2[3]*(1.0-f3[0])*(1.0-f4[3])
                I[0] += wt * D_A_0 * Fs
                Fs = f3[0]*f4[4]*(1.0-f1[0])*(1.0-f2[4]) - f1[0]*f2[4]*(1.0-f3[0])*(1.0-f4[4])
                I[0] += wt * D_A_2 * Fs
                Fs = f3[0]*f4[5]*(1.0-f1[0])*(1.0-f2[5]) - f1[0]*f2[5]*(1.0-f3[0])*(1.0-f4[5])
                I[0] += wt * D_A_0 * Fs

                # nuebar (alpha=1, sign=-)
                Fs = f3[1]*f4[2]*(1.0-f1[1])*(1.0-f2[2]) - f1[1]*f2[2]*(1.0-f3[1])*(1.0-f4[2])
                I[1] += wt * D_A_0 * Fs
                Fs = f3[1]*f4[3]*(1.0-f1[1])*(1.0-f2[3]) - f1[1]*f2[3]*(1.0-f3[1])*(1.0-f4[3])
                I[1] += wt * D_A_2 * Fs
                Fs = f3[1]*f4[4]*(1.0-f1[1])*(1.0-f2[4]) - f1[1]*f2[4]*(1.0-f3[1])*(1.0-f4[4])
                I[1] += wt * D_A_0 * Fs
                Fs = f3[1]*f4[5]*(1.0-f1[1])*(1.0-f2[5]) - f1[1]*f2[5]*(1.0-f3[1])*(1.0-f4[5])
                I[1] += wt * D_A_2 * Fs

                # numu (alpha=2, sign=+)
                Fs = f3[2]*f4[0]*(1.0-f1[2])*(1.0-f2[0]) - f1[2]*f2[0]*(1.0-f3[2])*(1.0-f4[0])
                I[2] += wt * D_A_2 * Fs
                Fs = f3[2]*f4[1]*(1.0-f1[2])*(1.0-f2[1]) - f1[2]*f2[1]*(1.0-f3[2])*(1.0-f4[1])
                I[2] += wt * D_A_0 * Fs
                Fs = f3[2]*f4[4]*(1.0-f1[2])*(1.0-f2[4]) - f1[2]*f2[4]*(1.0-f3[2])*(1.0-f4[4])
                I[2] += wt * D_A_2 * Fs
                Fs = f3[2]*f4[5]*(1.0-f1[2])*(1.0-f2[5]) - f1[2]*f2[5]*(1.0-f3[2])*(1.0-f4[5])
                I[2] += wt * D_A_0 * Fs

                # numubar (alpha=3, sign=-)
                Fs = f3[3]*f4[0]*(1.0-f1[3])*(1.0-f2[0]) - f1[3]*f2[0]*(1.0-f3[3])*(1.0-f4[0])
                I[3] += wt * D_A_0 * Fs
                Fs = f3[3]*f4[1]*(1.0-f1[3])*(1.0-f2[1]) - f1[3]*f2[1]*(1.0-f3[3])*(1.0-f4[1])
                I[3] += wt * D_A_2 * Fs
                Fs = f3[3]*f4[4]*(1.0-f1[3])*(1.0-f2[4]) - f1[3]*f2[4]*(1.0-f3[3])*(1.0-f4[4])
                I[3] += wt * D_A_0 * Fs
                Fs = f3[3]*f4[5]*(1.0-f1[3])*(1.0-f2[5]) - f1[3]*f2[5]*(1.0-f3[3])*(1.0-f4[5])
                I[3] += wt * D_A_2 * Fs

                # nutau (alpha=4, sign=+)
                Fs = f3[4]*f4[0]*(1.0-f1[4])*(1.0-f2[0]) - f1[4]*f2[0]*(1.0-f3[4])*(1.0-f4[0])
                I[4] += wt * D_A_2 * Fs
                Fs = f3[4]*f4[1]*(1.0-f1[4])*(1.0-f2[1]) - f1[4]*f2[1]*(1.0-f3[4])*(1.0-f4[1])
                I[4] += wt * D_A_0 * Fs
                Fs = f3[4]*f4[2]*(1.0-f1[4])*(1.0-f2[2]) - f1[4]*f2[2]*(1.0-f3[4])*(1.0-f4[2])
                I[4] += wt * D_A_2 * Fs
                Fs = f3[4]*f4[3]*(1.0-f1[4])*(1.0-f2[3]) - f1[4]*f2[3]*(1.0-f3[4])*(1.0-f4[3])
                I[4] += wt * D_A_0 * Fs

                # nutaubar (alpha=5, sign=-)
                Fs = f3[5]*f4[0]*(1.0-f1[5])*(1.0-f2[0]) - f1[5]*f2[0]*(1.0-f3[5])*(1.0-f4[0])
                I[5] += wt * D_A_0 * Fs
                Fs = f3[5]*f4[1]*(1.0-f1[5])*(1.0-f2[1]) - f1[5]*f2[1]*(1.0-f3[5])*(1.0-f4[1])
                I[5] += wt * D_A_2 * Fs
                Fs = f3[5]*f4[2]*(1.0-f1[5])*(1.0-f2[2]) - f1[5]*f2[2]*(1.0-f3[5])*(1.0-f4[2])
                I[5] += wt * D_A_0 * Fs
                Fs = f3[5]*f4[3]*(1.0-f1[5])*(1.0-f2[3]) - f1[5]*f2[3]*(1.0-f3[5])*(1.0-f4[3])
                I[5] += wt * D_A_2 * Fs

                # --- Process B: same-flavor nu + nubar forward scattering ---
                # nue(1)+nuebar(2) -> nue(3)+nuebar(4); contributes to I_nue when alpha=1 is slot 1
                Fs = f3[0]*f4[1]*(1.0-f1[0])*(1.0-f2[1]) - f1[0]*f2[1]*(1.0-f3[0])*(1.0-f4[1])
                I[0] += wt * D_B_val * Fs
                Fs = f3[1]*f4[0]*(1.0-f1[1])*(1.0-f2[0]) - f1[1]*f2[0]*(1.0-f3[1])*(1.0-f4[0])
                I[1] += wt * D_B_val * Fs
                Fs = f3[2]*f4[3]*(1.0-f1[2])*(1.0-f2[3]) - f1[2]*f2[3]*(1.0-f3[2])*(1.0-f4[3])
                I[2] += wt * D_B_val * Fs
                Fs = f3[3]*f4[2]*(1.0-f1[3])*(1.0-f2[2]) - f1[3]*f2[2]*(1.0-f3[3])*(1.0-f4[2])
                I[3] += wt * D_B_val * Fs
                Fs = f3[4]*f4[5]*(1.0-f1[4])*(1.0-f2[5]) - f1[4]*f2[5]*(1.0-f3[4])*(1.0-f4[5])
                I[4] += wt * D_B_val * Fs
                Fs = f3[5]*f4[4]*(1.0-f1[5])*(1.0-f2[4]) - f1[5]*f2[4]*(1.0-f3[5])*(1.0-f4[4])
                I[5] += wt * D_B_val * Fs

                # --- Process C: pair annihilation alpha + alpha-bar -> beta + beta-bar ---
                # For each flavor alpha and each other flavor beta, the process
                # alpha+alphabar(slots 1,2) -> beta+betabar(slots 3,4) contributes
                # loss to I[alpha] and I[alphabar] at their respective slot-1
                # integrals, and gain to I[beta] and I[betabar] in reverse
                # process integrals. The F_stat already includes both forward
                # (loss) and inverse (gain) via its detailed-balance structure.
                # When alpha's slot 1 is the subject of the current integral,
                # the slot-2 is its antiparticle-partner at y2, and the final
                # state betas are at y3, y4.
                #
                # Forward: loss term for I[alpha] when f1[alpha] large
                # F_C_stat_loss(alpha, beta) = f3[beta]*f4[beta-bar]*(1-f1[a])*(1-f2[a-bar]) - f1[a]*f2[a-bar]*(1-f3[b])*(1-f4[b-bar])

                # nue(1) + nuebar(2) -> numu + numubar  (loss for nue)
                Fs = f3[2]*f4[3]*(1.0-f1[0])*(1.0-f2[1]) - f1[0]*f2[1]*(1.0-f3[2])*(1.0-f4[3])
                I[0] += wt * D_C_val * Fs
                # nue(1) + nuebar(2) -> nutau + nutaubar
                Fs = f3[4]*f4[5]*(1.0-f1[0])*(1.0-f2[1]) - f1[0]*f2[1]*(1.0-f3[4])*(1.0-f4[5])
                I[0] += wt * D_C_val * Fs

                # nuebar(1) + nue(2) -> numubar + numu
                Fs = f3[3]*f4[2]*(1.0-f1[1])*(1.0-f2[0]) - f1[1]*f2[0]*(1.0-f3[3])*(1.0-f4[2])
                I[1] += wt * D_C_val * Fs
                # nuebar(1) + nue(2) -> nutaubar + nutau
                Fs = f3[5]*f4[4]*(1.0-f1[1])*(1.0-f2[0]) - f1[1]*f2[0]*(1.0-f3[5])*(1.0-f4[4])
                I[1] += wt * D_C_val * Fs

                # numu(1) + numubar(2) -> nue + nuebar
                Fs = f3[0]*f4[1]*(1.0-f1[2])*(1.0-f2[3]) - f1[2]*f2[3]*(1.0-f3[0])*(1.0-f4[1])
                I[2] += wt * D_C_val * Fs
                # numu(1) + numubar(2) -> nutau + nutaubar
                Fs = f3[4]*f4[5]*(1.0-f1[2])*(1.0-f2[3]) - f1[2]*f2[3]*(1.0-f3[4])*(1.0-f4[5])
                I[2] += wt * D_C_val * Fs

                # numubar(1) + numu(2) -> nuebar + nue
                Fs = f3[1]*f4[0]*(1.0-f1[3])*(1.0-f2[2]) - f1[3]*f2[2]*(1.0-f3[1])*(1.0-f4[0])
                I[3] += wt * D_C_val * Fs
                # numubar(1) + numu(2) -> nutaubar + nutau
                Fs = f3[5]*f4[4]*(1.0-f1[3])*(1.0-f2[2]) - f1[3]*f2[2]*(1.0-f3[5])*(1.0-f4[4])
                I[3] += wt * D_C_val * Fs

                # nutau(1) + nutaubar(2) -> nue + nuebar
                Fs = f3[0]*f4[1]*(1.0-f1[4])*(1.0-f2[5]) - f1[4]*f2[5]*(1.0-f3[0])*(1.0-f4[1])
                I[4] += wt * D_C_val * Fs
                # nutau(1) + nutaubar(2) -> numu + numubar
                Fs = f3[2]*f4[3]*(1.0-f1[4])*(1.0-f2[5]) - f1[4]*f2[5]*(1.0-f3[2])*(1.0-f4[3])
                I[4] += wt * D_C_val * Fs

                # nutaubar(1) + nutau(2) -> nuebar + nue
                Fs = f3[1]*f4[0]*(1.0-f1[5])*(1.0-f2[4]) - f1[5]*f2[4]*(1.0-f3[1])*(1.0-f4[0])
                I[5] += wt * D_C_val * Fs
                # nutaubar(1) + nutau(2) -> numubar + numu
                Fs = f3[3]*f4[2]*(1.0-f1[5])*(1.0-f2[4]) - f1[5]*f2[4]*(1.0-f3[3])*(1.0-f4[2])
                I[5] += wt * D_C_val * Fs

        for s in range(6):
            I_coll[s, i1] = prefactor / (y1 * y1) * I[s]

    return I_coll


@njit(cache=True)
def _F_stat_stable(f1, f2, f3, f4):
    """Numerically stable statistical factor for collision integral.

    Computes f3*f4*(1-f1)*(1-f2) - f1*f2*(1-f3)*(1-f4) using the
    reformulation: f1*f2*(1-f3)*(1-f4) * expm1(mu1+mu2-mu3-mu4)
    where mu_i = log((1-fi)/fi).

    This avoids catastrophic cancellation when distributions are near
    equilibrium (all fi close to FD at the same temperature).
    """
    # Clamp to avoid log(0) at the f = 0 and f = 1 singularities. Prior to
    # Stage E.2 sprint 7, _hi was 1.0 - 1.0e-20, which rounds to exactly 1.0
    # in float64 (1e-20 is far below machine eps(1) ≈ 2.22e-16), so the
    # upper clamp was a no-op. When a diagonal briefly overshoots f > 1
    # during the D.7.1 half-diag Strang split (seen at extended windows
    # with the V_nunu active-only projection on, before the end-of-step
    # diagonal clip fires), c = 1.0 gives log(0/1) = -inf in mu, so
    # d_mu = +inf + (-inf) = NaN. At cold T the _fnu_*_scat/ann scalars
    # are exactly 0, and the product 0 × NaN = NaN poisoned
    # I_nu_e[species, y_idx=0] at T ~ a few keV. Fix: pick _hi large
    # enough that 1 - _hi ≠ 1 in float64 (1.0 - 1e-15 ≈ 0.999999999999999).
    # _lo is unchanged, so the low-edge clamp behaviour for f ∈ [1e-30,
    # 1e-20] is identical; the only numerical change is for f ≥ 1-1e-15,
    # which never occurs in the physical [0, 1] range enforced by the
    # diagonal clip at end-of-step.
    _lo = 1.0e-20
    _hi = 1.0 - 1.0e-15
    c1 = min(max(f1, _lo), _hi)
    c2 = min(max(f2, _lo), _hi)
    c3 = min(max(f3, _lo), _hi)
    c4 = min(max(f4, _lo), _hi)

    mu1 = np.log((1.0 - c1) / c1)
    mu2 = np.log((1.0 - c2) / c2)
    mu3 = np.log((1.0 - c3) / c3)
    mu4 = np.log((1.0 - c4) / c4)
    d_mu = mu1 + mu2 - mu3 - mu4

    # For large |d_mu|, expm1 overflows; use the dominant term directly
    if d_mu > 500.0:
        return f3 * f4 * (1.0 - f1) * (1.0 - f2)
    elif d_mu < -500.0:
        return -f1 * f2 * (1.0 - f3) * (1.0 - f4)
    else:
        return f1 * f2 * (1.0 - f3) * (1.0 - f4) * np.expm1(d_mu)


@njit(cache=True)
def _collision_integral_nu_e(f_all, y_grid, quad_w, a, Tg, GF2_prefactor,
                              geL2, geR2, geLgeR, gmuL2, gmuR2, gmuLgmuR,
                              me, fnu_e_scat_val, fnu_e_ann_val,
                              fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                              D_k0, D_k1, D_k2, Ny_coll,
                              scale_scat=1.0, scale_ann=1.0,
                              scale_nue=1.0, scale_numu=1.0):
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

                wt = quad_w[i2] * quad_w[i3]

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
                D_scat_nue = scale_scat * scale_nue * 4.0 * (geL2 + geR2) * (dk0 + dk2) * fnu_e_scat_val
                D_scat_nuebar = D_scat_nue  # same after e-/e+ combination
                D_scat_numu = scale_scat * scale_numu * 4.0 * (gmuL2 + gmuR2) * (dk0 + dk2) * fnu_mu_scat_val

                # nu_e(1) + e(2) -> nu_e(3) + e(4): e- and e+ combined
                I_nue += wt * D_scat_nue * _F_stat_stable(f1_nue, f2_e, f3_nue, f4_e)

                # nuebar(1) + e(2): e- and e+ combined
                I_nuebar += wt * D_scat_nuebar * _F_stat_stable(f1_nuebar, f2_e, f3_nuebar, f4_e)

                # numu(1) + e(2): e- and e+ combined
                I_numu += wt * D_scat_numu * _F_stat_stable(f1_numu, f2_e, f3_numu, f4_e)

                # --- Annihilation: nu(1) + nubar(2) -> e+(3) + e-(4) ---
                # From Sabti Table 3: 128[gL^2*(Y1.Y3)(Y2.Y4) + gR^2*(Y1.Y4)(Y2.Y3)]
                # Coupling coeffs: 4*gL^2 for D_k1, 4*gR^2 for D_k2
                f2_nuebar_ann = f_all[1, i2]
                f2_nue_ann = f_all[0, i2]
                f2_numu_ann = f_all[2, i2]

                D_ann_nue = scale_ann * scale_nue * 4.0 * (geL2 * dk1 + geR2 * dk2) * fnu_e_ann_val
                D_ann_numu = scale_ann * scale_numu * 4.0 * (gmuL2 * dk1 + gmuR2 * dk2) * fnu_mu_ann_val

                # nue(1) + nuebar(2) -> e+ + e-
                I_nue += wt * D_ann_nue * _F_stat_stable(f1_nue, f2_nuebar_ann, f3_e, f4_e)

                # nuebar(1) + nue(2) -> e+ + e-
                I_nuebar += wt * D_ann_nue * _F_stat_stable(f1_nuebar, f2_nue_ann, f3_e, f4_e)

                # numu(1) + numubar(2) -> e+ + e-
                I_numu += wt * D_ann_numu * _F_stat_stable(f1_numu, f2_numu_ann, f3_e, f4_e)

        I_coll[0, i1] += prefactor / (y1 * y1) * I_nue
        I_coll[1, i1] += prefactor / (y1 * y1) * I_nuebar
        I_coll[2, i1] += prefactor / (y1 * y1) * I_numu

    return I_coll


@njit(parallel=True, cache=True)
def _collision_integral_nu_e_asym6(f_all, y_grid, quad_w, a, Tg, GF2_prefactor,
                                    geL2, geR2, geLgeR, gmuL2, gmuR2, gmuLgmuR,
                                    me, fnu_e_scat_val, fnu_e_ann_val,
                                    fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                                    D_k0, D_k1, D_k2, Ny_coll):
    """n=6 nu-e collision integral with explicit nu/nubar per flavor.

    Species layout: [nue, nuebar, numu, numubar, nutau, nutaubar].

    Scattering nu(1)+e(2)->nu(3)+e(4) is diagonal in species (slot 1 =
    slot 3). Each species feels the same rate (massless-electron variant).

    Annihilation nu_alpha(1)+nu_alpha_bar(2)->e+(3)+e-(4) pairs a species
    with its CPT partner. With f_nu != f_nubar the annihilation uses
    DIFFERENT distributions on slots 1 and 2 — this is the key change
    vs. n=3 (where f_all[2] stands in for both particle and antiparticle).
    """
    Ny = len(y_grid)
    I_coll = np.zeros((6, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_comoving = Tg * a

    D_scat_e_coeff = 4.0 * (geL2 + geR2) * fnu_e_scat_val
    D_scat_mu_coeff = 4.0 * (gmuL2 + gmuR2) * fnu_mu_scat_val
    D_ann_e_L = 4.0 * geL2 * fnu_e_ann_val
    D_ann_e_R = 4.0 * geR2 * fnu_e_ann_val
    D_ann_mu_L = 4.0 * gmuL2 * fnu_mu_ann_val
    D_ann_mu_R = 4.0 * gmuR2 * fnu_mu_ann_val

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1 = np.empty(6)
        for s in range(6):
            f1[s] = f_all[s, i1]

        I_local = np.zeros(6)

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue
            x_e2 = y2 / Te_comoving
            f2_e = 0.0 if x_e2 > 500.0 else 1.0 / (np.exp(x_e2) + 1.0)
            f2 = np.empty(6)
            for s in range(6):
                f2[s] = f_all[s, i2]

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                y4 = y1 + y2 - y3
                if y4 <= 0.0 or y3 < 1.0e-10:
                    continue
                x_e3 = y3 / Te_comoving
                f3_e = 0.0 if x_e3 > 500.0 else 1.0 / (np.exp(x_e3) + 1.0)
                x_e4 = y4 / Te_comoving
                f4_e = 0.0 if x_e4 > 500.0 else 1.0 / (np.exp(x_e4) + 1.0)

                f3 = np.empty(6)
                f4 = np.empty(6)
                for s in range(6):
                    f3[s] = f_all[s, i3]
                    f4[s] = _interp_grid(y4, y_grid, f_all[s],
                                          tail_params[s, 0], tail_params[s, 1])

                wt = quad_w[i2] * quad_w[i3]
                dk0 = D_k0[i1, i2, i3]
                dk1 = D_k1[i1, i2, i3]
                dk2 = D_k2[i1, i2, i3]

                D_scat_e = D_scat_e_coeff * (dk0 + dk2)
                D_scat_mu = D_scat_mu_coeff * (dk0 + dk2)
                D_ann_e = D_ann_e_L * dk1 + D_ann_e_R * dk2
                D_ann_mu = D_ann_mu_L * dk1 + D_ann_mu_R * dk2

                # Scattering — diagonal in species (slot 1 == slot 3)
                for alpha in range(6):
                    D_scat = D_scat_e if alpha < 2 else D_scat_mu
                    I_local[alpha] += wt * D_scat * _F_stat_stable(
                        f1[alpha], f2_e, f3[alpha], f4_e)

                # Annihilation nu_alpha + nubar_alpha -> e+ e-
                # Pair (alpha, alpha_bar) with alpha_bar = alpha^1.
                I_local[0] += wt * D_ann_e * _F_stat_stable(f1[0], f2[1], f3_e, f4_e)
                I_local[1] += wt * D_ann_e * _F_stat_stable(f1[1], f2[0], f3_e, f4_e)
                I_local[2] += wt * D_ann_mu * _F_stat_stable(f1[2], f2[3], f3_e, f4_e)
                I_local[3] += wt * D_ann_mu * _F_stat_stable(f1[3], f2[2], f3_e, f4_e)
                I_local[4] += wt * D_ann_mu * _F_stat_stable(f1[4], f2[5], f3_e, f4_e)
                I_local[5] += wt * D_ann_mu * _F_stat_stable(f1[5], f2[4], f3_e, f4_e)

        for alpha in range(6):
            I_coll[alpha, i1] = prefactor / (y1 * y1) * I_local[alpha]

    return I_coll


@njit(cache=True)
def _collision_integral_nu_e_massive(f_all, y_grid, quad_w, a, Tg, GF2_prefactor,
                                      geL2, geR2, gmuL2, gmuR2,
                                      me, Ny_coll):
    """
    Collision integrals from nu-e processes with massive electron kinematics.

    Unlike _collision_integral_nu_e, this computes D-kernels on-the-fly
    (no pre-computed tables) because y4 depends on the electron mass through
    energy conservation. The electron Fermi-Dirac distributions use the
    relativistic energy E = sqrt(y^2 + me^2*a^2) instead of y.

    This eliminates the need for:
    - Pre-computed D-kernel tables (D_k0, D_k1, D_k2) for nu-e processes
    - fnu_e_scat/ann finite-mass correction factors
    - The coll_scale fudge factor

    Processes:
    Scattering: nu(1) + e(2) -> nu(3) + e(4)  [positions 2,4 massive]
    Annihilation: nu(1) + nubar(2) -> e+(3) + e-(4)  [positions 3,4 massive]
    """
    Ny = len(y_grid)
    I_coll = np.zeros((3, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_comoving = Tg * a
    me_a = me * a
    me_a2 = me_a * me_a

    # Combined coupling coefficients (e- + e+ summed)
    # Scattering: 4*(gL^2+gR^2) for both s-channel (D1) and u-channel (D3)
    c_scat_nue_D13 = 4.0 * (geL2 + geR2)
    c_scat_numu_D13 = 4.0 * (gmuL2 + gmuR2)
    # Annihilation: 4*gL^2 for t-channel (D2), 4*gR^2 for u-channel (D3)
    c_ann_nue_D2 = 4.0 * geL2
    c_ann_nue_D3 = 4.0 * geR2
    c_ann_numu_D2 = 4.0 * gmuL2
    c_ann_numu_D3 = 4.0 * gmuR2

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

            # --- Scattering: particle 2 is electron (massive) ---
            E2_e = np.sqrt(y2 * y2 + me_a2)
            x_e2 = E2_e / Te_comoving
            if x_e2 > 500.0:
                f2_e = 0.0
            else:
                f2_e = 1.0 / (np.exp(x_e2) + 1.0)

            # --- Annihilation: particle 2 is antineutrino (massless) ---
            f2_nuebar_ann = f_all[1, i2]
            f2_nue_ann = f_all[0, i2]
            f2_numu_ann = f_all[2, i2]

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                if y3 < 1.0e-10:
                    continue

                wt = quad_w[i2] * quad_w[i3]

                # ========== SCATTERING: nu(1)+e(2)->nu(3)+e(4) ==========
                # Positions 1,3: neutrino (massless, E=y)
                # Positions 2,4: electron (massive, E=sqrt(y^2+me_a^2))
                E4_scat = y1 + E2_e - y3
                if E4_scat > me_a:
                    y4s_sq = E4_scat * E4_scat - me_a2
                    if y4s_sq > 0.0:
                        y4_scat = np.sqrt(y4s_sq)

                        # Electron FD at position 4
                        x_e4s = E4_scat / Te_comoving
                        if x_e4s > 500.0:
                            f4_e_s = 0.0
                        else:
                            f4_e_s = 1.0 / (np.exp(x_e4s) + 1.0)

                        # Neutrino distributions at position 3 (on grid)
                        f3_nue = f_all[0, i3]
                        f3_nuebar = f_all[1, i3]
                        f3_numu = f_all[2, i3]

                        # D-kernel with massive electron energies.
                        # Phase space factor (y2*y3)/(E2*E3) from Sabti E.14:
                        # particle 2 massive (y2/E2), particle 3 massless (1)
                        ps_scat = y2 / E2_e

                        D_scat_nue = ps_scat * D_kernel_massive(
                            y1, y2, y3, y4_scat,
                            y1, E2_e, y3, E4_scat,
                            c_scat_nue_D13, 0.0, c_scat_nue_D13)

                        D_scat_numu = ps_scat * D_kernel_massive(
                            y1, y2, y3, y4_scat,
                            y1, E2_e, y3, E4_scat,
                            c_scat_numu_D13, 0.0, c_scat_numu_D13)

                        # nue(1) + e(2) -> nue(3) + e(4)
                        I_nue += wt * D_scat_nue * _F_stat_stable(f1_nue, f2_e, f3_nue, f4_e_s)

                        # nuebar(1) + e(2) -> nuebar(3) + e(4)
                        I_nuebar += wt * D_scat_nue * _F_stat_stable(f1_nuebar, f2_e, f3_nuebar, f4_e_s)

                        # numu(1) + e(2) -> numu(3) + e(4)
                        I_numu += wt * D_scat_numu * _F_stat_stable(f1_numu, f2_e, f3_numu, f4_e_s)

                # ========== ANNIHILATION: nu(1)+nubar(2)->e+(3)+e-(4) ==========
                # Positions 1,2: neutrino (massless, E=y)
                # Positions 3,4: electron (massive, E=sqrt(y^2+me_a^2))
                E3_e = np.sqrt(y3 * y3 + me_a2)
                E4_ann = y1 + y2 - E3_e
                if E4_ann > me_a:
                    y4a_sq = E4_ann * E4_ann - me_a2
                    if y4a_sq > 0.0:
                        y4_ann = np.sqrt(y4a_sq)

                        # Electron FD at positions 3 and 4
                        x_e3a = E3_e / Te_comoving
                        if x_e3a > 500.0:
                            f3_e_a = 0.0
                        else:
                            f3_e_a = 1.0 / (np.exp(x_e3a) + 1.0)

                        x_e4a = E4_ann / Te_comoving
                        if x_e4a > 500.0:
                            f4_e_a = 0.0
                        else:
                            f4_e_a = 1.0 / (np.exp(x_e4a) + 1.0)

                        # D-kernel with massive electron energies.
                        # Phase space factor (y2*y3)/(E2*E3) from Sabti E.14:
                        # particle 2 massless (1), particle 3 massive (y3/E3)
                        ps_ann = y3 / E3_e

                        D_ann_nue = ps_ann * D_kernel_massive(
                            y1, y2, y3, y4_ann,
                            y1, y2, E3_e, E4_ann,
                            0.0, c_ann_nue_D2, c_ann_nue_D3)

                        D_ann_numu = ps_ann * D_kernel_massive(
                            y1, y2, y3, y4_ann,
                            y1, y2, E3_e, E4_ann,
                            0.0, c_ann_numu_D2, c_ann_numu_D3)

                        # nue(1) + nuebar(2) -> e+ + e-
                        I_nue += wt * D_ann_nue * _F_stat_stable(f1_nue, f2_nuebar_ann, f3_e_a, f4_e_a)

                        # nuebar(1) + nue(2) -> e+ + e-
                        I_nuebar += wt * D_ann_nue * _F_stat_stable(f1_nuebar, f2_nue_ann, f3_e_a, f4_e_a)

                        # numu(1) + numubar(2) -> e+ + e-
                        I_numu += wt * D_ann_numu * _F_stat_stable(f1_numu, f2_numu_ann, f3_e_a, f4_e_a)

        I_coll[0, i1] += prefactor / (y1 * y1) * I_nue
        I_coll[1, i1] += prefactor / (y1 * y1) * I_nuebar
        I_coll[2, i1] += prefactor / (y1 * y1) * I_numu

    return I_coll


@njit(parallel=True, cache=True)
def _collision_integral_nu_e_massive_asym6(f_all, y_grid, quad_w, a, Tg,
                                            GF2_prefactor, geL2, geR2,
                                            gmuL2, gmuR2, me, Ny_coll):
    """n=6 nu-e collision integral with MASSIVE electron kinematics.

    Port of `_collision_integral_nu_e_massive` (n=3) to the
    [nue, nuebar, numu, numubar, nutau, nutaubar] species layout, with
    the same ν/ν̄-per-flavor-distinct annihilation logic as the massless
    `_collision_integral_nu_e_asym6`. Scattering is diagonal in species;
    annihilation pairs slot α with slot α^1 (its CPT partner) so
    f_ν(y1) != f_ν̄(y2) is handled correctly.

    Species 0,1 use e-flavor couplings (geL², geR²); species 2..5 use
    mu-flavor (gmuL², gmuR²). μ and τ carry identical couplings in the SM.

    D-kernels are computed on-the-fly via `D_kernel_massive` since
    energy conservation with E = sqrt(y² + me²a²) makes y4 table-lookup
    impractical. See the n=3 function for the phase-space factor
    derivation (Sabti Eq. E.14).
    """
    Ny = len(y_grid)
    I_coll = np.zeros((6, Ny))
    prefactor = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_comoving = Tg * a
    me_a = me * a
    me_a2 = me_a * me_a

    c_scat_e_D13 = 4.0 * (geL2 + geR2)
    c_scat_mu_D13 = 4.0 * (gmuL2 + gmuR2)
    c_ann_e_D2 = 4.0 * geL2
    c_ann_e_D3 = 4.0 * geR2
    c_ann_mu_D2 = 4.0 * gmuL2
    c_ann_mu_D3 = 4.0 * gmuR2

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        f1 = np.empty(6)
        for s in range(6):
            f1[s] = f_all[s, i1]

        I_local = np.zeros(6)

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue

            # Slot 2 is MASSIVE electron for scattering (y2 -> E2_e)
            E2_e = np.sqrt(y2 * y2 + me_a2)
            x_e2 = E2_e / Te_comoving
            f2_e = 0.0 if x_e2 > 500.0 else 1.0 / (np.exp(x_e2) + 1.0)

            # Slot 2 is neutrino for annihilation (massless)
            f2 = np.empty(6)
            for s in range(6):
                f2[s] = f_all[s, i2]

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                if y3 < 1.0e-10:
                    continue
                wt = quad_w[i2] * quad_w[i3]

                # --- Scattering: nu(1) + e(2) -> nu(3) + e(4) ---
                E4_scat = y1 + E2_e - y3
                if E4_scat > me_a:
                    y4s_sq = E4_scat * E4_scat - me_a2
                    if y4s_sq > 0.0:
                        y4_scat = np.sqrt(y4s_sq)
                        x_e4s = E4_scat / Te_comoving
                        f4_e_s = 0.0 if x_e4s > 500.0 else 1.0 / (np.exp(x_e4s) + 1.0)

                        # Neutrino distribution at slot 3 (y3, on grid)
                        f3 = np.empty(6)
                        for s in range(6):
                            f3[s] = f_all[s, i3]

                        ps_scat = y2 / E2_e

                        D_scat_e = ps_scat * D_kernel_massive(
                            y1, y2, y3, y4_scat,
                            y1, E2_e, y3, E4_scat,
                            c_scat_e_D13, 0.0, c_scat_e_D13)
                        D_scat_mu = ps_scat * D_kernel_massive(
                            y1, y2, y3, y4_scat,
                            y1, E2_e, y3, E4_scat,
                            c_scat_mu_D13, 0.0, c_scat_mu_D13)

                        # Scattering is diagonal in species
                        for alpha in range(6):
                            D_scat = D_scat_e if alpha < 2 else D_scat_mu
                            I_local[alpha] += wt * D_scat * _F_stat_stable(
                                f1[alpha], f2_e, f3[alpha], f4_e_s)

                # --- Annihilation: nu(1) + nubar(2) -> e+(3) + e-(4) ---
                # Positions 3,4 are MASSIVE electrons (E = sqrt(y^2 + me_a^2))
                E3_e = np.sqrt(y3 * y3 + me_a2)
                E4_ann = y1 + y2 - E3_e
                if E4_ann > me_a:
                    y4a_sq = E4_ann * E4_ann - me_a2
                    if y4a_sq > 0.0:
                        y4_ann = np.sqrt(y4a_sq)
                        x_e3a = E3_e / Te_comoving
                        f3_e_a = 0.0 if x_e3a > 500.0 else 1.0 / (np.exp(x_e3a) + 1.0)
                        x_e4a = E4_ann / Te_comoving
                        f4_e_a = 0.0 if x_e4a > 500.0 else 1.0 / (np.exp(x_e4a) + 1.0)

                        ps_ann = y3 / E3_e

                        D_ann_e = ps_ann * D_kernel_massive(
                            y1, y2, y3, y4_ann,
                            y1, y2, E3_e, E4_ann,
                            0.0, c_ann_e_D2, c_ann_e_D3)
                        D_ann_mu = ps_ann * D_kernel_massive(
                            y1, y2, y3, y4_ann,
                            y1, y2, E3_e, E4_ann,
                            0.0, c_ann_mu_D2, c_ann_mu_D3)

                        # Annihilation nu_α + nubar_α -> e+ e-: slot 1 = α,
                        # slot 2 = α_bar (= α XOR 1 in our layout)
                        I_local[0] += wt * D_ann_e * _F_stat_stable(
                            f1[0], f2[1], f3_e_a, f4_e_a)
                        I_local[1] += wt * D_ann_e * _F_stat_stable(
                            f1[1], f2[0], f3_e_a, f4_e_a)
                        I_local[2] += wt * D_ann_mu * _F_stat_stable(
                            f1[2], f2[3], f3_e_a, f4_e_a)
                        I_local[3] += wt * D_ann_mu * _F_stat_stable(
                            f1[3], f2[2], f3_e_a, f4_e_a)
                        I_local[4] += wt * D_ann_mu * _F_stat_stable(
                            f1[4], f2[5], f3_e_a, f4_e_a)
                        I_local[5] += wt * D_ann_mu * _F_stat_stable(
                            f1[5], f2[4], f3_e_a, f4_e_a)

        for alpha in range(6):
            I_coll[alpha, i1] = prefactor / (y1 * y1) * I_local[alpha]

    return I_coll


###############################################################################
# Off-diagonal collision gain (transport) terms                                #
#                                                                              #
# Computes the gain part of the off-diagonal QKE collision integral:           #
#   C_αβ(p₁) = gain_αβ(p₁)  −  ½(Γ_α + Γ_β) ρ_αβ(p₁)                       #
# The gain scatters coherences from p₃ → p₁ via the same D-kernel formalism   #
# as the diagonal collision integrals, with modified coupling constants.       #
#                                                                              #
# From the anticommutator form (Sigl & Raffelt 1993):                          #
#   gain_αβ = ½[(1-f₁_α)+(1-f₁_β)] × Σ g_α g_β D × ρ₃_αβ × stat_factors    #
# For ν-ν: g_α g_β = ¼ (universal Z-exchange)                                 #
# For ν-e: g_α g_β = g_{L,α} g_{L,β} + g_R² (flavor-asymmetric)              #
###############################################################################

@njit(cache=True)
def _offdiag_collision_gain(rho_offdiag, f_all, y_grid, quad_w, a, Tg,
                             GF2_prefactor,
                             c_emu_scat, c_mutau_scat,
                             fnu_emu_scat_val, fnu_mutau_scat_val,
                             B_spectator_e_idx,
                             tail_params, D_k0, D_k2, Ny_coll):
    """
    Off-diagonal collision gain (transport) for one sector (ν or ν̄).

    Computes the gain rate for each off-diagonal component, arising from
    coherences at momentum p₃ scattered to p₁ by ν-ν and ν-e processes.

    Parameters
    ----------
    rho_offdiag : ndarray, shape (6, Ny)
        Off-diagonal components for this sector:
        [Re(ρ_eμ), Im(ρ_eμ), Re(ρ_eτ), Im(ρ_eτ), Re(ρ_μτ), Im(ρ_μτ)]
    f_all : ndarray, shape (3, Ny)
        Diagonal distributions [f_νe, f_ν̄e, f_νμ_eff].
    B_spectator_e_idx : int
        Index into f_all for the Process B spectator of the "e" flavour.
        For ν sector: 1 (ν̄_e). For ν̄ sector: 0 (ν_e).
    c_emu_scat : float
        Off-diagonal ν-e scattering coupling for e-μ pair:
        4 × (g_{L,e} g_{L,μ} + g_R²).
    c_mutau_scat : float
        Off-diagonal ν-e scattering coupling for μ-τ pair:
        4 × (g_{L,μ}² + g_R²) = diagonal μ coupling.
    fnu_emu_scat_val, fnu_mutau_scat_val : float
        Finite-mass correction factors for the off-diagonal couplings.

    Returns
    -------
    gain : ndarray, shape (6, Ny)
        Gain rates in 1/s for each off-diagonal component.
    """
    Ny = len(y_grid)
    gain = np.zeros((6, Ny))
    pref_base = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_com = Tg * a

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        pref = pref_base / (y1 * y1)

        f1_e = f_all[0, i1]
        f1_mu = f_all[2, i1]

        # Pauli blocking at p₁: ½[(1-f₁_α) + (1-f₁_β)]
        pauli_emu = 0.5 * ((1.0 - f1_e) + (1.0 - f1_mu))   # e-μ and e-τ
        pauli_mutau = 1.0 - f1_mu                            # μ-τ (f_τ ≈ f_μ)

        # Accumulators for 6 off-diagonal components
        G0 = 0.0; G1 = 0.0; G2 = 0.0; G3 = 0.0; G4 = 0.0; G5 = 0.0

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue

            # Electron at p₂ (for ν-e scattering)
            x_e2 = y2 / Te_com
            f2_e = 0.0
            if x_e2 < 500.0:
                f2_e = 1.0 / (np.exp(x_e2) + 1.0)

            # Neutrino distributions at p₂ (for ν-ν scattering)
            f2_nue = f_all[0, i2]
            f2_nuebar = f_all[1, i2]
            f2_numu = f_all[2, i2]
            f2_B_e = f_all[B_spectator_e_idx, i2]

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                y4 = y1 + y2 - y3
                if y4 <= 0.0 or y3 < 1.0e-10:
                    continue

                wt = quad_w[i2] * quad_w[i3]
                dk0 = D_k0[i1, i2, i3]
                dk2 = D_k2[i1, i2, i3]

                # Off-diagonal at p₃
                r0 = rho_offdiag[0, i3]  # Re(ρ_eμ)
                r1 = rho_offdiag[1, i3]  # Im(ρ_eμ)
                r2 = rho_offdiag[2, i3]  # Re(ρ_eτ)
                r3 = rho_offdiag[3, i3]  # Im(ρ_eτ)
                r4 = rho_offdiag[4, i3]  # Re(ρ_μτ)
                r5 = rho_offdiag[5, i3]  # Im(ρ_μτ)

                # ============ ν-e scattering gain ============
                # ν(1)+e(2)→ν(3)+e(4)
                x_e4 = y4 / Te_com
                f4_e = 0.0
                if x_e4 < 500.0:
                    f4_e = 1.0 / (np.exp(x_e4) + 1.0)

                K_scat = wt * (dk0 + dk2) * f2_e * (1.0 - f4_e)

                # e-μ and e-τ pairs (same coupling)
                K_emu = K_scat * c_emu_scat * fnu_emu_scat_val
                G0 += K_emu * r0
                G1 += K_emu * r1
                G2 += K_emu * r2
                G3 += K_emu * r3

                # μ-τ pair (diagonal-μ coupling)
                K_mt = K_scat * c_mutau_scat * fnu_mutau_scat_val
                G4 += K_mt * r4
                G5 += K_mt * r5

                # ============ ν-ν scattering gain ============
                # Spectator distributions at p₄ (off-grid)
                f4_numu = _interp_grid(y4, y_grid, f_all[2],
                                       tail_params[2, 0], tail_params[2, 1])
                f4_nue = _interp_grid(y4, y_grid, f_all[0],
                                      tail_params[0, 0], tail_params[0, 1])
                f4_nuebar = _interp_grid(y4, y_grid, f_all[1],
                                         tail_params[1, 0], tail_params[1, 1])
                f4_B_e = _interp_grid(y4, y_grid, f_all[B_spectator_e_idx],
                                      tail_params[B_spectator_e_idx, 0],
                                      tail_params[B_spectator_e_idx, 1])

                # Process A: different-flavour scattering, D_k0.
                # For ρ_eμ / ρ_eτ: 4 μ/τ-type spectators (same as diag ν_e).
                K_A_mu = wt * dk0 * f2_numu * (1.0 - f4_numu)
                G0 += 4.0 * K_A_mu * r0
                G1 += 4.0 * K_A_mu * r1
                G2 += 4.0 * K_A_mu * r2
                G3 += 4.0 * K_A_mu * r3

                # For ρ_μτ: ν_e + ν̄_e + 2×ν_μ spectators (same as diag ν_μ).
                K_A_mt = wt * dk0 * (f2_nue * (1.0 - f4_nue)
                                     + f2_nuebar * (1.0 - f4_nuebar)
                                     + 2.0 * f2_numu * (1.0 - f4_numu))
                G4 += K_A_mt * r4
                G5 += K_A_mt * r5

                # Process B: same-flavour ν+ν̄ forward, 4×D_k2.
                # For ρ_eμ / ρ_eτ: ½(ν̄_e + ν̄_μ) spectators (averaging e,μ perspectives)
                K_B_em = wt * 4.0 * dk2 * 0.5 * (
                    f2_B_e * (1.0 - f4_B_e)
                    + f2_numu * (1.0 - f4_numu))
                G0 += K_B_em * r0
                G1 += K_B_em * r1
                G2 += K_B_em * r2
                G3 += K_B_em * r3

                # For ρ_μτ: ν̄_μ spectator (≈ f_numu)
                K_B_mt = wt * 4.0 * dk2 * f2_numu * (1.0 - f4_numu)
                G4 += K_B_mt * r4
                G5 += K_B_mt * r5

                # Process C (pair annihilation to different flavour):
                # No off-diagonal gain — inverse produces definite flavour.

        gain[0, i1] = pref * pauli_emu * G0
        gain[1, i1] = pref * pauli_emu * G1
        gain[2, i1] = pref * pauli_emu * G2
        gain[3, i1] = pref * pauli_emu * G3
        gain[4, i1] = pref * pauli_mutau * G4
        gain[5, i1] = pref * pauli_mutau * G5

    return gain


@njit(cache=True)
def _offdiag_collision_gain_massive(rho_offdiag, f_all, y_grid, quad_w, a, Tg,
                                     GF2_prefactor,
                                     c_emu_scat, c_mutau_scat,
                                     B_spectator_e_idx,
                                     tail_params, D_k0, D_k2, Ny_coll, me):
    """
    Off-diagonal collision gain with massive electron kinematics for nu-e.

    Same as _offdiag_collision_gain but computes nu-e scattering D-kernels
    on-the-fly using D_kernel_massive with E_e = sqrt(y^2 + me^2*a^2) for
    electron legs. The nu-nu part is unchanged (all particles massless).
    """
    Ny = len(y_grid)
    gain = np.zeros((6, Ny))
    pref_base = GF2_prefactor / (64.0 * np.pi**3 * a**5)
    Te_com = Tg * a
    me_a = me * a
    me_a2 = me_a * me_a

    for i1 in prange(Ny):
        y1 = y_grid[i1]
        if y1 < 1.0e-10 or i1 >= Ny_coll:
            continue

        pref = pref_base / (y1 * y1)

        f1_e = f_all[0, i1]
        f1_mu = f_all[2, i1]

        pauli_emu = 0.5 * ((1.0 - f1_e) + (1.0 - f1_mu))
        pauli_mutau = 1.0 - f1_mu

        G0 = 0.0; G1 = 0.0; G2 = 0.0; G3 = 0.0; G4 = 0.0; G5 = 0.0

        for i2 in range(Ny_coll):
            y2 = y_grid[i2]
            if y2 < 1.0e-10:
                continue

            # Massive electron at p2: E2 = sqrt(y2^2 + me_a^2)
            E2_e = np.sqrt(y2 * y2 + me_a2)
            x_e2 = E2_e / Te_com
            f2_e = 0.0
            if x_e2 < 500.0:
                f2_e = 1.0 / (np.exp(x_e2) + 1.0)

            # Neutrino distributions at p2 (for nu-nu scattering, unchanged)
            f2_nue = f_all[0, i2]
            f2_nuebar = f_all[1, i2]
            f2_numu = f_all[2, i2]
            f2_B_e = f_all[B_spectator_e_idx, i2]

            for i3 in range(Ny_coll):
                y3 = y_grid[i3]
                if y3 < 1.0e-10:
                    continue

                wt = quad_w[i2] * quad_w[i3]

                # Off-diagonal at p3
                r0 = rho_offdiag[0, i3]
                r1 = rho_offdiag[1, i3]
                r2 = rho_offdiag[2, i3]
                r3 = rho_offdiag[3, i3]
                r4 = rho_offdiag[4, i3]
                r5 = rho_offdiag[5, i3]

                # ============ nu-e scattering gain (massive electron) ============
                # nu(1) + e(2) -> nu(3) + e(4)
                # Particles 1,3: massless neutrino; 2,4: massive electron
                E4_scat = y1 + E2_e - y3
                if E4_scat > me_a:
                    y4s_sq = E4_scat * E4_scat - me_a2
                    if y4s_sq > 0.0:
                        y4_scat = np.sqrt(y4s_sq)

                        x_e4s = E4_scat / Te_com
                        f4_e_s = 0.0
                        if x_e4s < 500.0:
                            f4_e_s = 1.0 / (np.exp(x_e4s) + 1.0)

                        # Phase space factor y2/E2 for massive particle 2
                        ps_scat = y2 / E2_e

                        # D-kernel with massive energies (D1+D3 channels for scattering)
                        dk_scat = ps_scat * D_kernel_massive(
                            y1, y2, y3, y4_scat,
                            y1, E2_e, y3, E4_scat,
                            1.0, 0.0, 1.0)

                        K_scat = wt * dk_scat * f2_e * (1.0 - f4_e_s)

                        K_emu = K_scat * c_emu_scat
                        G0 += K_emu * r0
                        G1 += K_emu * r1
                        G2 += K_emu * r2
                        G3 += K_emu * r3

                        K_mt = K_scat * c_mutau_scat
                        G4 += K_mt * r4
                        G5 += K_mt * r5

                # ============ nu-nu scattering gain (unchanged) ============
                # Uses massless kinematics and pre-computed tables
                y4 = y1 + y2 - y3
                if y4 <= 0.0:
                    continue

                dk0 = D_k0[i1, i2, i3]
                dk2 = D_k2[i1, i2, i3]

                # Process A: different-flavour scattering
                f4_numu = _interp_grid(y4, y_grid, f_all[2],
                                       tail_params[2, 0], tail_params[2, 1])
                f4_nue = _interp_grid(y4, y_grid, f_all[0],
                                      tail_params[0, 0], tail_params[0, 1])
                f4_nuebar = _interp_grid(y4, y_grid, f_all[1],
                                         tail_params[1, 0], tail_params[1, 1])
                f4_B_e = _interp_grid(y4, y_grid, f_all[B_spectator_e_idx],
                                      tail_params[B_spectator_e_idx, 0],
                                      tail_params[B_spectator_e_idx, 1])

                K_A_mu = wt * dk0 * f2_numu * (1.0 - f4_numu)
                G0 += 4.0 * K_A_mu * r0
                G1 += 4.0 * K_A_mu * r1
                G2 += 4.0 * K_A_mu * r2
                G3 += 4.0 * K_A_mu * r3

                K_A_mt = wt * dk0 * (f2_nue * (1.0 - f4_nue)
                                     + f2_nuebar * (1.0 - f4_nuebar)
                                     + 2.0 * f2_numu * (1.0 - f4_numu))
                G4 += K_A_mt * r4
                G5 += K_A_mt * r5

                # Process B: same-flavour forward
                K_B_em = wt * 4.0 * dk2 * 0.5 * (
                    f2_B_e * (1.0 - f4_B_e)
                    + f2_numu * (1.0 - f4_numu))
                G0 += K_B_em * r0
                G1 += K_B_em * r1
                G2 += K_B_em * r2
                G3 += K_B_em * r3

                K_B_mt = wt * 4.0 * dk2 * f2_numu * (1.0 - f4_numu)
                G4 += K_B_mt * r4
                G5 += K_B_mt * r5

        gain[0, i1] = pref * pauli_emu * G0
        gain[1, i1] = pref * pauli_emu * G1
        gain[2, i1] = pref * pauli_emu * G2
        gain[3, i1] = pref * pauli_emu * G3
        gain[4, i1] = pref * pauli_mutau * G4
        gain[5, i1] = pref * pauli_mutau * G5

    return gain


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
        # Species layout selected via two flags:
        #   mu_tau_symmetric=True                        -> n=3 [nue, nuebar, numu_eff]
        #   mu_tau_symmetric=False, nu_nubar_symmetric=True  -> n=4 [nue, nuebar, numu_eff, nutau_eff]
        #   mu_tau_symmetric=False, nu_nubar_symmetric=False -> n=6 [nue, nuebar, numu, numubar, nutau, nutaubar]
        # (nu_nubar_symmetric=False with mu_tau=True is forced to mu_tau=False
        #  since lepton asymmetry implies full species resolution.)
        self.mu_tau_symmetric = PRyMini.mu_tau_symmetric_flag
        self.nu_nubar_symmetric = PRyMini.nu_nubar_symmetric_flag
        if not self.nu_nubar_symmetric:
            self.mu_tau_symmetric = False
        if self.mu_tau_symmetric:
            self.n_species = 3
        elif self.nu_nubar_symmetric:
            self.n_species = 4
        else:
            self.n_species = 6

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

        # Quadrature weights for collision integral inner sums (Simpson's rule)
        self.quad_w = _quad_weights(self.Ny_coll, self.dy)

        # Pre-compute D-kernel lookup tables (one-time cost, grid-dependent only)
        if PRyMini.verbose_flag:
            print(f"BoltzmannSolver: Ny={self.Ny}, y_max={self.y_max:.1f} MeV, "
                  f"dy={self.dy:.2f} MeV")
            print(f"  Pre-computing D-kernel tables ({self.Ny}^3 = "
                  f"{self.Ny**3} entries)...")
        self.D_k0, self.D_k1, self.D_k2 = _precompute_D_tables(self.y_grid)
        if PRyMini.verbose_flag:
            print(f"  D-kernel tables ready.")

        # Oscillation-induced flavor mixing
        if PRyMini.nu_oscillation_flag:
            method = getattr(PRyMini, 'nu_oscillation_method', 'relaxation')
            if method == 'collision_mixing':
                self._setup_collision_mixing()
            else:
                self._setup_oscillation_relaxation()

    def _setup_collision_mixing(self):
        """
        Compute time-averaged PMNS transition probabilities for the
        rapid-oscillation limit (Sabti Eq. 3.18):

            df_a/dt = sum_b P_ab * C_b[f],   P_ab = sum_i |V_ai|^2 |V_bi|^2.

        For n=3 [nue, nuebar, numu_eff] only P_ee appears (the column sum
        P_μe + P_τe = 1 - P_ee already collapses the μ-τ average).

        For n=4 [nue, nuebar, numu_eff, nutau_eff] and n=6 [nue, nuebar,
        numu, numubar, nutau, nutaubar] the full 3×3 P_αβ matrix is used.
        (The earlier "maximal-θ₂₃, no-CP" approximation P_μe = P_τe =
        (1-P_ee)/2 and P_μμ = P_ττ = P_μτ = (1+P_ee)/4 is only exact at
        θ₂₃ = π/4 and δ_CP = 0; PDG 2024 values deviate by ~14%.)

        Stores self.P_ee (n=3 fast path) and self.P_PMNS (full 3x3, used
        by n=4 and n=6 paths).
        """
        s12 = np.sin(PRyMini.theta_12); c12 = np.cos(PRyMini.theta_12)
        s13 = np.sin(PRyMini.theta_13); c13 = np.cos(PRyMini.theta_13)
        s23 = np.sin(PRyMini.theta_23); c23 = np.cos(PRyMini.theta_23)
        cd = np.cos(PRyMini.delta_CP)

        # PDG PMNS parameterization: |V_ei|^2 is real, |V_μi|^2 and |V_τi|^2
        # pick up interference terms proportional to
        #   K = s12 c12 s23 c23 s13 cos(delta_CP).
        Ve1 = c12**2 * c13**2
        Ve2 = s12**2 * c13**2
        Ve3 = s13**2

        K = s12 * c12 * s23 * c23 * s13 * cd
        Vm1 = s12**2 * c23**2 + c12**2 * s23**2 * s13**2 + 2.0 * K
        Vm2 = c12**2 * c23**2 + s12**2 * s23**2 * s13**2 - 2.0 * K
        Vm3 = s23**2 * c13**2
        Vt1 = s12**2 * s23**2 + c12**2 * c23**2 * s13**2 - 2.0 * K
        Vt2 = c12**2 * s23**2 + s12**2 * c23**2 * s13**2 + 2.0 * K
        Vt3 = c23**2 * c13**2

        # Assemble the 3x3 P_αβ matrix (rows/cols in order e, μ, τ).
        P = np.empty((3, 3))
        rows = ((Ve1, Ve2, Ve3), (Vm1, Vm2, Vm3), (Vt1, Vt2, Vt3))
        for i, ai in enumerate(rows):
            for j, bj in enumerate(rows):
                P[i, j] = ai[0] * bj[0] + ai[1] * bj[1] + ai[2] * bj[2]
        self.P_PMNS = P
        self.P_ee = P[0, 0]  # backwards-compat for the n=3 path

        if PRyMini.verbose_flag:
            print(f"  Collision mixing (Sabti + full 3x3 PMNS):")
            print(f"    P_ee = {P[0,0]:.4f}  1-P_ee = {1-P[0,0]:.4f}")
            print(f"    P_eμ = {P[0,1]:.4f}  P_eτ = {P[0,2]:.4f}")
            print(f"    P_μμ = {P[1,1]:.4f}  P_ττ = {P[2,2]:.4f}  "
                  f"P_μτ = {P[1,2]:.4f}")

    def _apply_collision_mixing(self, I_total):
        """
        Apply PMNS time-averaged oscillation mixing to collision integrals.

        n=3 [nue, nuebar, numu_eff] — exact, only P_ee appears because the
        mu-tau aggregation in I_total[2] collapses the column sum:
          I_mixed[0] = P_ee * I[0] + (1-P_ee) * I[2]
          I_mixed[1] = P_ee * I[1] + (1-P_ee) * I[2]
          I_mixed[2] = (1-P_ee)/2 * (I[0]+I[1])/2 + (1+P_ee)/2 * I[2]

        n=4 [nue, nuebar, numu_eff, nutau_eff] — uses the full 3x3 P_αβ:
          I_mixed[0]  = P_ee I[0] + P_eμ I[2] + P_eτ I[3]
          I_mixed[1]  = P_ee I[1] + P_eμ I[2] + P_eτ I[3]
          I_mixed[2]  = P_μe avg(I[0],I[1]) + P_μμ I[2] + P_μτ I[3]
          I_mixed[3]  = P_τe avg(I[0],I[1]) + P_τμ I[2] + P_ττ I[3]

        n=6 [nue, nuebar, numu, numubar, nutau, nutaubar] — apply 3x3 PMNS
        separately to the nu sector (slots 0,2,4) and nubar sector (1,3,5).
        |V_αi|^2 is identical for nu and nubar under CPT so the P_αβ matrix
        is shared.

        With PDG 2024 (sin²θ₂₃ = 0.546, δ_CP = 1.36π) the exact mu/tau
        probabilities deviate by ~14% from the maximal-θ₂₃, no-CP
        approximation, so this upgrade matters for BSM scenarios with
        I[2] != I[3] or asymmetric nu/nubar distributions.

        Returns a new array (same shape as I_total).
        """
        P = self.P_PMNS  # 3x3 matrix, rows/cols (e, μ, τ)
        P_ee = P[0, 0]
        P_eμ = P[0, 1]
        P_eτ = P[0, 2]
        P_μe = P[1, 0]
        P_μμ = P[1, 1]
        P_μτ = P[1, 2]
        P_τe = P[2, 0]
        P_τμ = P[2, 1]
        P_ττ = P[2, 2]

        if self.n_species == 3:
            # numu_eff = <I_μ, I_τ> symmetric average; reduces to the same
            # P_ee-only formula as before (independent of θ₂₃ or δ_CP).
            P_off = 1.0 - P_ee
            I_mixed = np.empty_like(I_total)
            I_mixed[0] = P_ee * I_total[0] + P_off * I_total[2]
            I_mixed[1] = P_ee * I_total[1] + P_off * I_total[2]
            I_mixed[2] = (P_off / 2.0) * (I_total[0] + I_total[1]) / 2.0 \
                        + (1.0 + P_ee) / 2.0 * I_total[2]
            return I_mixed

        if self.n_species == 4:
            # Average of pcle/antipcle e-flavor collision rates, used to mix
            # into the (pcle+antipcle)-aggregated mu/tau slots.
            avg_e = 0.5 * (I_total[0] + I_total[1])
            I_mixed = np.empty_like(I_total)
            I_mixed[0] = P_ee * I_total[0] + P_eμ * I_total[2] + P_eτ * I_total[3]
            I_mixed[1] = P_ee * I_total[1] + P_eμ * I_total[2] + P_eτ * I_total[3]
            I_mixed[2] = P_μe * avg_e + P_μμ * I_total[2] + P_μτ * I_total[3]
            I_mixed[3] = P_τe * avg_e + P_τμ * I_total[2] + P_ττ * I_total[3]
            return I_mixed

        # n=6 path: full per-species mixing, applied independently to
        # particle (slots 0,2,4) and antiparticle (slots 1,3,5) sectors.
        I_mixed = np.empty_like(I_total)
        # Particle sector
        I_mixed[0] = P_ee * I_total[0] + P_eμ * I_total[2] + P_eτ * I_total[4]
        I_mixed[2] = P_μe * I_total[0] + P_μμ * I_total[2] + P_μτ * I_total[4]
        I_mixed[4] = P_τe * I_total[0] + P_τμ * I_total[2] + P_ττ * I_total[4]
        # Antiparticle sector (CPT-identical P_αβ)
        I_mixed[1] = P_ee * I_total[1] + P_eμ * I_total[3] + P_eτ * I_total[5]
        I_mixed[3] = P_μe * I_total[1] + P_μμ * I_total[3] + P_μτ * I_total[5]
        I_mixed[5] = P_τe * I_total[1] + P_τμ * I_total[3] + P_ττ * I_total[5]
        return I_mixed

    def _setup_oscillation_relaxation(self):
        """
        Set up oscillation-induced flavor relaxation parameters.

        In the early universe, neutrino flavor oscillations are damped by
        collisions. The interplay produces an effective flavor relaxation
        (Sigl & Raffelt 1993, Dolgov 2002):

            (df_alpha/dt)_osc = Gamma_flavor(y, Tg) * (f_target - f_alpha)

        The effective rate per momentum mode (quasi-static density matrix):

            Gamma_flavor = sin^2(2theta) * omega^2 * D / (2*(omega_eff^2 + D^2))

        where:
            omega = Dm2 / (2E)               vacuum oscillation frequency
            omega_eff = omega*cos(2theta) - V  in-matter frequency
            D = C_D * GF^2 * T^4 * E          collision damping rate
            V = 8*sqrt(2)*GF*E*rho_e/(3*mW^2)  CC thermal potential (Notzold-Raffelt)

        Three regimes:
            - T > 3 MeV: V >> omega, MSW-suppressed
            - T ~ 1-3 MeV: V ~ omega, fast equilibration (Gamma/H ~ 1-4)
            - T < 0.5 MeV: D << omega, collisions freeze out

        The dominant channel is nue <-> numu/nutau via Dm2_21 and theta_12.
        numu <-> nutau is enforced by mu-tau symmetry in the collision integrals.
        """
        # Mixing parameters
        self.sin2_2theta12 = np.sin(2.0 * PRyMini.theta_12)**2
        self.cos_2theta12 = np.cos(2.0 * PRyMini.theta_12)
        self.sin2_2theta13 = np.sin(2.0 * PRyMini.theta_13)**2
        self.cos_2theta13 = np.cos(2.0 * PRyMini.theta_13)

        # Mass splittings [eV^2]
        self.Dm2_21 = PRyMini.Dm2_21
        self.Dm2_31 = PRyMini.Dm2_31

        # Collision damping coefficients (de Salas & Pastor 2016)
        # D_alpha = C_D_alpha * GF^2 * T^4 * E  (all in natural units)
        self.C_D_nue = 3.06   # nue: CC + NC scattering
        self.C_D_numu = 2.22  # numu: NC only

        # W boson mass for matter potential
        self.mW2 = (PRyMini.mZ * np.sqrt(1.0 - PRyMini.sW2))**2  # MeV^2

    def _oscillation_relaxation(self, f_all, a, Tg):
        """
        Compute the oscillation-induced flavor relaxation term.

        Returns array of shape (n_species, Ny) to be added to I_total.

        Uses the quasi-static density matrix result (Sigl & Raffelt 1993):

            Gamma_flavor = sin^2(2theta) * omega^2 * D / (2*(omega_eff^2 + D^2))

        Drives nue toward numu (and vice versa), conserving total number.
        """
        if not self.mu_tau_symmetric:
            raise NotImplementedError(
                "Sigl-Raffelt relaxation not yet generalized to n=4 asymmetric "
                "diagonal mode. Use nu_oscillation_method='collision_mixing' "
                "(default) or enable qke_density_matrix_flag for full 3x3 QKE.")
        I_osc = np.zeros_like(f_all)

        # Physical momentum: p = y / a  [MeV]
        p_MeV = self.y_grid / a
        # Avoid division by zero for very small momenta
        p_MeV = np.maximum(p_MeV, 1.0e-10)
        E_eV = p_MeV * 1.0e6  # eV (massless neutrinos: E = p)

        # --- Vacuum oscillation frequency ---
        # omega = Dm2 / (2E)  [eV]
        omega_21 = self.Dm2_21 / (2.0 * E_eV)
        omega_31 = self.Dm2_31 / (2.0 * E_eV)

        # --- CC matter potential (Notzold-Raffelt 1988) ---
        # V = 8*sqrt(2) * GF * E * rho_e / (3 * m_W^2)
        # rho_e = 7*pi^2/60 * T^4 for relativistic e+e-  [MeV^4]
        rho_e = 7.0 * np.pi**2 / 60.0 * Tg**4  # MeV^4
        # V in MeV, then convert to eV
        V_MeV = 8.0 * np.sqrt(2.0) * PRyMini.GF * p_MeV * rho_e / (3.0 * self.mW2)
        V_eV = V_MeV * 1.0e6  # eV

        # --- Collision damping rate ---
        # D = C_D * GF^2 * T^4 * E  [eV, all in eV natural units]
        GF_eV = PRyMini.GF * 1.0e-12  # MeV^-2 -> eV^-2
        T_eV = Tg * 1.0e6
        D_eV = self.C_D_nue * GF_eV**2 * T_eV**4 * E_eV  # eV

        # --- Solar channel: Dm2_21, theta_12 ---
        omega_eff_21 = omega_21 * self.cos_2theta12 - V_eV
        Gamma_21 = (self.sin2_2theta12 * omega_21**2 * D_eV
                    / (2.0 * (omega_eff_21**2 + D_eV**2)))  # eV

        # --- Atmospheric channel (subdominant via theta_13): Dm2_31 ---
        omega_eff_31 = omega_31 * self.cos_2theta13 - V_eV
        Gamma_31 = (self.sin2_2theta13 * omega_31**2 * D_eV
                    / (2.0 * (omega_eff_31**2 + D_eV**2)))  # eV

        # Total relaxation rate [1/s]
        eV_to_sec = PRyMini.MeV_to_secm1 * 1.0e-6
        Gamma_tot = (Gamma_21 + Gamma_31) * eV_to_sec  # 1/s

        # --- Flavor relaxation toward weighted average ---
        # f_all[0]=nue, f_all[1]=nuebar, f_all[2]=numu_equiv (2 DOFs: mu+tau)
        # Equilibrium target: f_eq = (f_e + 2*f_mu) / 3
        f_eq_nu = (f_all[0] + 2.0 * f_all[2]) / 3.0
        f_eq_nubar = (f_all[1] + 2.0 * f_all[2]) / 3.0

        # Relaxation: df_alpha/dt = Gamma * (f_eq - f_alpha)
        I_osc[0] = Gamma_tot * (f_eq_nu - f_all[0])
        I_osc[1] = Gamma_tot * (f_eq_nubar - f_all[1])
        # Conservation: 1*I_nue + 1*I_nuebar + 2*I_numu = 0
        I_osc[2] = -(I_osc[0] + I_osc[1]) / 2.0

        return I_osc

    def initial_conditions(self, Tnu, a, f_initial=None):
        """
        Return initial distributions on the comoving grid.

        Default (f_initial is None or symmetric mode): thermal Fermi-Dirac at
        Tnu for all species, f_alpha(y_i) = 1 / (exp(y_i / (Tnu * a)) + 1).

        Tnu is the (physical) neutrino temperature at the handoff moment.
        Callers must pass Tnu_A[-1], NOT Tg, or the initial rho_nu will be
        biased.

        Asymmetric mode (mu_tau_symmetric_flag=False, n_species=4): if
        f_initial is provided, it may include 'nue', 'nuebar', 'numu',
        'nutau' callables f(p_MeV, Tnu_MeV) used to initialize each slot.
        Any missing key falls back to thermal FD. 'numubar' and 'nutaubar'
        callables are NOT used in the diagonal n=4 path because particle
        and antiparticle are still aggregated per flavor — for full ν/ν̄
        asymmetry use the QKE density-matrix path (Stage 3 will extend
        the diagonal solver to n=6).
        """
        f_all = np.zeros((self.n_species, self.Ny))
        Tnu_com = Tnu * a
        p_grid = self.y_grid / a

        # Thermal FD default
        fd_default = np.zeros(self.Ny)
        for i in range(self.Ny):
            x = self.y_grid[i] / Tnu_com
            if x < 500.0:
                fd_default[i] = 1.0 / (np.exp(x) + 1.0)

        if self.n_species == 3:
            f_all[:, :] = fd_default
            if f_initial is not None:
                _slot_map = {'nue': 0, 'nuebar': 1, 'numu': 2}
                for key, idx in _slot_map.items():
                    if f_initial.get(key) is not None:
                        vals = np.asarray(f_initial[key](p_grid, Tnu), dtype=float)
                        f_all[idx] = vals
        elif self.n_species == 4:
            _slot_map = {'nue': 0, 'nuebar': 1, 'numu': 2, 'nutau': 3}
            for key, idx in _slot_map.items():
                cb = f_initial.get(key) if f_initial else None
                if cb is not None:
                    vals = np.asarray(cb(p_grid, Tnu), dtype=float)
                    if vals.shape != (self.Ny,):
                        raise ValueError(
                            f"initial_conditions: callable for {key} returned "
                            f"shape {vals.shape}, expected {(self.Ny,)}")
                    f_all[idx] = vals
                else:
                    f_all[idx] = fd_default
        else:  # n_species == 6
            _slot_map = {
                'nue': 0, 'nuebar': 1,
                'numu': 2, 'numubar': 3,
                'nutau': 4, 'nutaubar': 5,
            }
            for key, idx in _slot_map.items():
                cb = f_initial.get(key) if f_initial else None
                if cb is not None:
                    vals = np.asarray(cb(p_grid, Tnu), dtype=float)
                    if vals.shape != (self.Ny,):
                        raise ValueError(
                            f"initial_conditions: callable for {key} returned "
                            f"shape {vals.shape}, expected {(self.Ny,)}")
                    f_all[idx] = vals
                else:
                    f_all[idx] = fd_default
        return f_all

    def apply_oscillation_mixing(self, f_all, a, Tg, dt):
        """
        Apply oscillation-induced flavor relaxation using exact exponential decay.

        This is operator-split from the collision step for stability: the
        oscillation relaxation rate can spike at MSW near-resonance momenta
        (Gamma_osc >> Gamma_coll), which would destabilize the explicit
        exponential-Euler stepper if included in C_f.

        The exact solution of df/dt = Gamma*(f_eq - f) over interval dt is:
            f_new = f_eq + (f_old - f_eq) * exp(-Gamma*dt)

        This is unconditionally stable for any Gamma*dt.

        Parameters
        ----------
        f_all : ndarray, shape (n_species, Ny)
            Current distributions (modified in-place).
        a : float
            Scale factor at current time.
        Tg : float
            Photon temperature in MeV.
        dt : float
            Time step in seconds.

        Returns
        -------
        f_all : ndarray
            Modified distributions (same array, modified in-place).
        """
        if not PRyMini.nu_oscillation_flag:
            return f_all
        if not self.mu_tau_symmetric:
            raise NotImplementedError(
                "apply_oscillation_mixing (Sigl-Raffelt relaxation) not yet "
                "generalized to n=4 asymmetric mode. Use "
                "nu_oscillation_method='collision_mixing' or enable "
                "qke_density_matrix_flag for full 3x3 QKE.")

        # Compute per-momentum relaxation rate Gamma_tot [1/s]
        p_MeV = np.maximum(self.y_grid / a, 1.0e-10)
        E_eV = p_MeV * 1.0e6

        # Vacuum oscillation frequency
        omega_21 = self.Dm2_21 / (2.0 * E_eV)
        omega_31 = self.Dm2_31 / (2.0 * E_eV)

        # CC matter potential (Notzold-Raffelt)
        rho_e = 7.0 * np.pi**2 / 60.0 * Tg**4
        V_MeV = 8.0 * np.sqrt(2.0) * PRyMini.GF * p_MeV * rho_e / (3.0 * self.mW2)
        V_eV = V_MeV * 1.0e6

        # Collision damping rate
        GF_eV = PRyMini.GF * 1.0e-12
        T_eV = Tg * 1.0e6
        D_eV = self.C_D_nue * GF_eV**2 * T_eV**4 * E_eV

        # Solar channel
        omega_eff_21 = omega_21 * self.cos_2theta12 - V_eV
        Gamma_21 = (self.sin2_2theta12 * omega_21**2 * D_eV
                    / (2.0 * (omega_eff_21**2 + D_eV**2)))

        # Atmospheric channel (subdominant)
        omega_eff_31 = omega_31 * self.cos_2theta13 - V_eV
        Gamma_31 = (self.sin2_2theta13 * omega_31**2 * D_eV
                    / (2.0 * (omega_eff_31**2 + D_eV**2)))

        # Total rate in 1/s
        eV_to_sec = PRyMini.MeV_to_secm1 * 1.0e-6
        Gamma_tot = (Gamma_21 + Gamma_31) * eV_to_sec

        # Exact exponential decay: f_new = f_eq + (f_old - f_eq)*exp(-Gamma*dt)
        decay = np.exp(-Gamma_tot * dt)  # shape (Ny,)

        # Equilibrium targets for the particle and antiparticle sectors:
        #   particle: nue ↔ numu+nutau, target = (f_nue + 2*f_numu)/3
        #   antiparticle: nuebar ↔ numubar+nutaubar, target = (f_nuebar + 2*f_numu)/3
        # numu_equiv carries both sectors; its update is fixed by conservation.
        f_eq_p = (f_all[0] + 2.0 * f_all[2]) / 3.0
        f_eq_a = (f_all[1] + 2.0 * f_all[2]) / 3.0

        # Compute changes (keeping originals for numu conservation)
        df_nue = (f_eq_p - f_all[0]) * (1.0 - decay)
        df_nuebar = (f_eq_a - f_all[1]) * (1.0 - decay)

        # Apply to e-sector
        f_all[0] += df_nue
        f_all[1] += df_nuebar
        # numu: conservation at each y requires Δf_nue + Δf_nuebar + 2*Δf_numu = 0
        f_all[2] -= (df_nue + df_nuebar) / 2.0

        return f_all

    # -------- Helpers for collision_integrals() dispatch --------

    def _fnu_corrections(self, Tg):
        """Return (fnu_e_scat, fnu_e_ann, fnu_mu_scat, fnu_mu_ann) floats.

        These are the finite-m_e correction factors read from the NUDEC_BSM
        thermal tables; needed only by the massless ν-e integrals (the
        massive variants compute D-kernels on-the-fly with E = sqrt(y²+m²)).
        """
        return (float(self._fnu_e_scat(Tg)),  float(self._fnu_e_ann(Tg)),
                float(self._fnu_mu_scat(Tg)), float(self._fnu_mu_ann(Tg)))

    def _nu_e_n3(self, f_all, a, Tg, GF2_pref, tail_params):
        """Nu-e collision integrals for the n=3 path (mu-tau symmetric)."""
        if PRyMini.massive_electron_flag:
            return _collision_integral_nu_e_massive(
                f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.gmuL2, self.gmuR2,
                PRyMini.me, self.Ny_coll)
        f1, f2, f3, f4 = self._fnu_corrections(Tg)
        return _collision_integral_nu_e(
            f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
            self.geL2, self.geR2, self.geLgeR,
            self.gmuL2, self.gmuR2, self.gmuLgmuR,
            PRyMini.me, f1, f2, f3, f4, tail_params,
            self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
            1.0, 1.0, 1.0, 1.0)

    def _nu_e_n4(self, f_all, a, Tg, GF2_pref, tail_params):
        """Nu-e collision integrals for the n=4 path (μ-τ split, ν/ν̄ merged
        per flavor).

        Assembled from two calls into the n=3 ν-e function — one with
        slot-2 = f_numu_eff, one with slot-2 = f_nutau_eff — since I_nue
        and I_nuebar don't depend on slot 2 and the muon-sector couplings
        are identical for μ and τ.
        """
        f3_mu = np.stack([f_all[0], f_all[1], f_all[2]], axis=0)
        f3_tau = np.stack([f_all[0], f_all[1], f_all[3]], axis=0)
        tp_mu = tail_params[[0, 1, 2]]
        tp_tau = tail_params[[0, 1, 3]]

        if PRyMini.massive_electron_flag:
            I_mu = _collision_integral_nu_e_massive(
                f3_mu, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.gmuL2, self.gmuR2,
                PRyMini.me, self.Ny_coll)
            I_tau = _collision_integral_nu_e_massive(
                f3_tau, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.gmuL2, self.gmuR2,
                PRyMini.me, self.Ny_coll)
        else:
            f1, f2, f3, f4 = self._fnu_corrections(Tg)
            I_mu = _collision_integral_nu_e(
                f3_mu, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.geLgeR,
                self.gmuL2, self.gmuR2, self.gmuLgmuR,
                PRyMini.me, f1, f2, f3, f4, tp_mu,
                self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
                1.0, 1.0, 1.0, 1.0)
            I_tau = _collision_integral_nu_e(
                f3_tau, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.geLgeR,
                self.gmuL2, self.gmuR2, self.gmuLgmuR,
                PRyMini.me, f1, f2, f3, f4, tp_tau,
                self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
                1.0, 1.0, 1.0, 1.0)

        # Assemble n=4 output. I_nue, I_nuebar are identical across both
        # calls (they don't depend on slot 2); we average for symmetry/safety.
        I_nu_e = np.zeros((4, self.Ny))
        I_nu_e[0] = 0.5 * (I_mu[0] + I_tau[0])
        I_nu_e[1] = 0.5 * (I_mu[1] + I_tau[1])
        I_nu_e[2] = I_mu[2]   # numu_eff
        I_nu_e[3] = I_tau[2]  # nutau_eff
        return I_nu_e

    def _nu_e_n6(self, f_all, a, Tg, GF2_pref, tail_params):
        """Nu-e collision integrals for the n=6 path (full ν/ν̄ per flavor)."""
        if PRyMini.massive_electron_flag:
            return _collision_integral_nu_e_massive_asym6(
                f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                self.geL2, self.geR2, self.gmuL2, self.gmuR2,
                PRyMini.me, self.Ny_coll)
        f1, f2, f3, f4 = self._fnu_corrections(Tg)
        return _collision_integral_nu_e_asym6(
            f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
            self.geL2, self.geR2, self.geLgeR,
            self.gmuL2, self.gmuR2, self.gmuLgmuR,
            PRyMini.me, f1, f2, f3, f4, tail_params,
            self.D_k0, self.D_k1, self.D_k2, self.Ny_coll)

    def _nu_nu_dispatch(self, f_all, a, GF2_pref, tail_params):
        """Call the nu-nu collision integral for the current n_species."""
        if self.n_species == 3:
            return _collision_integral_nu_nu(
                f_all, self.y_grid, self.quad_w, a, GF2_pref, tail_params,
                self.D_k0, self.D_k2, self.Ny_coll)
        elif self.n_species == 4:
            return _collision_integral_nu_nu_asym4(
                f_all, self.y_grid, self.quad_w, a, GF2_pref, tail_params,
                self.D_k0, self.D_k2, self.Ny_coll)
        else:  # n_species == 6
            return _collision_integral_nu_nu_asym6(
                f_all, self.y_grid, self.quad_w, a, GF2_pref, tail_params,
                self.D_k0, self.D_k2, self.Ny_coll)

    def _nu_e_dispatch(self, f_all, a, Tg, GF2_pref, tail_params):
        """Call the nu-e collision integral for the current n_species.

        When PRyMini.nlo_weak_flag is True, multiplies the nu-e prefactor
        by PRyMini.nlo_weak_rate_scale — a single-number placeholder for
        the one-loop electroweak corrections to nu-e scattering and
        annihilation. This scale only affects the nu-e channel, NOT nu-nu,
        mirroring the structure of the NLO corrections in Akita &
        Yamaguchi 2020.
        """
        if PRyMini.nlo_weak_flag:
            GF2_pref = GF2_pref * PRyMini.nlo_weak_rate_scale
        if self.n_species == 3:
            return self._nu_e_n3(f_all, a, Tg, GF2_pref, tail_params)
        elif self.n_species == 4:
            return self._nu_e_n4(f_all, a, Tg, GF2_pref, tail_params)
        else:  # n_species == 6
            return self._nu_e_n6(f_all, a, Tg, GF2_pref, tail_params)

    # Species-label arrays: which NP callback key maps to which f_all row.
    # Used by collision_integrals to add user-supplied NP collision terms.
    _NP_SLOT_MAP_N3 = (('nue', 0), ('nuebar', 1), ('numu', 2))
    _NP_SLOT_MAP_N4 = (('nue', 0), ('nuebar', 1), ('numu', 2), ('nutau', 3))
    _NP_SLOT_MAP_N6 = (('nue', 0), ('nuebar', 1),
                       ('numu', 2), ('numubar', 3),
                       ('nutau', 4), ('nutaubar', 5))

    def _apply_NP_collisions(self, I_total, f_all, a, Tg):
        """Add user-supplied NP collision terms to each relevant species."""
        if self.n_species == 3:
            mapping = self._NP_SLOT_MAP_N3
        elif self.n_species == 4:
            mapping = self._NP_SLOT_MAP_N4
        else:
            mapping = self._NP_SLOT_MAP_N6
        for name, idx in mapping:
            fn = self.C_NP_funcs.get(name)
            if fn is not None:
                I_total[idx] += fn(self.y_grid, a, Tg, f_all)
        return I_total

    def collision_integrals(self, f_all, a, Tg):
        """
        Compute total collision integrals for all species.

        Returns an array of shape (n_species, Ny). Row layout:
          n=3: [I_nue, I_nuebar, I_numu_eff]
          n=4: [I_nue, I_nuebar, I_numu_eff, I_nutau_eff]
          n=6: [I_nue, I_nuebar, I_numu, I_numubar, I_nutau, I_nutaubar]

        Dispatches the nu-nu and nu-e integrals to the per-n_species
        helpers (_nu_nu_dispatch, _nu_e_dispatch), then applies oscillation
        mixing (Sabti Eq. 3.18, when nu_oscillation_flag is set and the
        method is 'collision_mixing') and adds user-supplied NP terms from
        self.C_NP_funcs.

        `coll_scale` in PRyMini scales the overall GF^2 prefactor so both
        nu-nu and nu-e contributions are rescaled uniformly.
        """
        GF2_pref = self.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        I_nu_nu = self._nu_nu_dispatch(f_all, a, GF2_pref, tail_params)
        I_nu_e = self._nu_e_dispatch(f_all, a, Tg, GF2_pref, tail_params)
        I_total = I_nu_nu + I_nu_e

        if PRyMini.nu_oscillation_flag and \
                getattr(PRyMini, 'nu_oscillation_method', 'relaxation') == 'collision_mixing':
            I_total = self._apply_collision_mixing(I_total)

        if self.C_NP_funcs:
            I_total = self._apply_NP_collisions(I_total, f_all, a, Tg)

        return I_total

    def energy_transfer_rate(self, I_coll, a):
        """
        Compute the net energy transfer rate from neutrinos to plasma.

        delta_rho = sum_alpha g_alpha/(2*pi^2*a^4) * integral dy y^2 E_tilde * I_alpha

        For massless neutrinos E_tilde = y, so:
        delta_rho = sum_alpha 1/(2*pi^2*a^4) * integral dy y^3 * I_alpha

        Returns delta_rho in MeV^4 / s (after MeV_to_secm1 conversion in I_coll).
        """
        # Species weights:
        #   n=3 {1, 1, 2} — weight 2 sums over 4 DOF (mu,mubar,tau,taubar)
        #   n=4 {1, 1, 1, 1} — each mu_eff/tau_eff slot aggregates pcle+antipcle (2 DOF)
        #   n=6 {1, 1, 1, 1, 1, 1} — one species per slot (1 DOF each)
        if self.n_species == 3:
            _weights = (1.0, 1.0, 2.0)
        elif self.n_species == 4:
            _weights = (1.0, 1.0, 1.0, 1.0)
        else:
            _weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        delta_rho = 0.0
        for alpha in range(self.n_species):
            integrand = self.y_grid**3 * I_coll[alpha]
            delta_rho += _weights[alpha] * self.dy / (2.0 * np.pi**2 * a**4) * np.sum(integrand)
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

            # Log-FD interpolation: interpolate L(y) = log(1/f - 1) linearly in y.
            # This is EXACT for any Fermi-Dirac distribution (thermal or chemical-
            # potential-shifted) since L(y) = (y - mu)/T is linear. For mildly
            # perturbed distributions, it is near-exact and dramatically reduces
            # the O(dy^2) bias of direct linear interpolation of f on a concave
            # curve. Kills a +0.26% rho_nu readout bias at Ny=100 (see
            # diagnose_readout.py).
            f_safe = np.clip(f_grid, f_min, 1.0 - f_min)
            L_grid = np.log(1.0/f_safe - 1.0)
            L_interp = interp1d(y_grid, L_grid, bounds_error=False,
                                fill_value=(L_grid[0], L_grid[-1]), kind='linear')

            def _eval_f(y_arr):
                """Evaluate f via log-FD interpolation with FD-tail extrapolation."""
                y_arr = np.asarray(y_arr, dtype=float)
                result = np.empty_like(y_arr)
                # In-grid: use log-FD interpolation
                in_grid = y_arr <= y_max_grid
                if np.any(in_grid):
                    L = L_interp(y_arr[in_grid])
                    L = np.clip(L, -500.0, 500.0)
                    result[in_grid] = 1.0 / (np.exp(L) + 1.0)
                # Out-of-grid: use tail fit (also a log-FD extrapolation)
                mask = ~in_grid
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
        if self.n_species == 3:
            PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_numubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutau_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutaubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
        elif self.n_species == 4:
            f_nutau_grid = f_all[3].copy()
            PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_numubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutau_general = _make_f_callable(f_nutau_grid, current_a, a_func)
            PRyMthermo.f_nutaubar_general = _make_f_callable(f_nutau_grid, current_a, a_func)
        else:  # n_species == 6
            f_numubar_grid = f_all[3].copy()
            f_nutau_grid = f_all[4].copy()
            f_nutaubar_grid = f_all[5].copy()
            PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_numubar_general = _make_f_callable(f_numubar_grid, current_a, a_func)
            PRyMthermo.f_nutau_general = _make_f_callable(f_nutau_grid, current_a, a_func)
            PRyMthermo.f_nutaubar_general = _make_f_callable(f_nutaubar_grid, current_a, a_func)

    def make_f_callable(self, f_grid, a, a_of_T_func=None):
        """Public interface to create a f(p, Tg) callable from a grid array.

        Used by DensityMatrixSolver to create callables for each of the 6
        neutrino flavors independently.
        """
        import PRyM.PRyM_thermo as PRyMthermo  # noqa: F811
        y_grid = self.y_grid
        f_min = 1.0e-12

        # Fit FD-tail extrapolation
        tail_y, tail_l = [], []
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

        f_safe = np.clip(f_grid, f_min, 1.0 - f_min)
        L_grid = np.log(1.0/f_safe - 1.0)
        L_interp = interp1d(y_grid, L_grid, bounds_error=False,
                            fill_value=(L_grid[0], L_grid[-1]), kind='linear')

        def _eval_f(y_arr):
            y_arr = np.asarray(y_arr, dtype=float)
            result = np.empty_like(y_arr)
            in_grid = y_arr <= y_max_grid
            if np.any(in_grid):
                L = L_interp(y_arr[in_grid])
                L = np.clip(L, -500.0, 500.0)
                result[in_grid] = 1.0 / (np.exp(L) + 1.0)
            mask = ~in_grid
            if np.any(mask):
                arg = _tail_a + _tail_b * y_arr[mask]
                arg = np.clip(arg, -500.0, 500.0)
                result[mask] = 1.0 / (np.exp(arg) + 1.0)
            return result

        if a_of_T_func is not None:
            a_of_T = a_of_T_func
            def f_nu(p, Tg):
                y = np.asarray(p, dtype=float) * a_of_T(Tg)
                return _eval_f(y)
        else:
            a_val = a
            def f_nu(p, Tg):
                y = np.asarray(p, dtype=float) * a_val
                return _eval_f(y)
        return f_nu


###############################################################################
# Density matrix representation helpers                                        #
###############################################################################

def _rho_vec_batch_to_matrix(rho_vec):
    """Convert (9, Ny) real array to (Ny, 3, 3) complex Hermitian matrices.

    Layout: [rho_ee, rho_mumu, rho_tautau,
             Re(rho_emu), Im(rho_emu),
             Re(rho_etau), Im(rho_etau),
             Re(rho_mutau), Im(rho_mutau)]
    """
    Ny = rho_vec.shape[1]
    rho = np.zeros((Ny, 3, 3), dtype=complex)
    rho[:, 0, 0] = rho_vec[0]
    rho[:, 1, 1] = rho_vec[1]
    rho[:, 2, 2] = rho_vec[2]
    rho[:, 0, 1] = rho_vec[3] + 1j * rho_vec[4]
    rho[:, 1, 0] = rho_vec[3] - 1j * rho_vec[4]
    rho[:, 0, 2] = rho_vec[5] + 1j * rho_vec[6]
    rho[:, 2, 0] = rho_vec[5] - 1j * rho_vec[6]
    rho[:, 1, 2] = rho_vec[7] + 1j * rho_vec[8]
    rho[:, 2, 1] = rho_vec[7] - 1j * rho_vec[8]
    return rho


def _matrix_batch_to_rho_vec(rho):
    """Convert (Ny, 3, 3) complex Hermitian matrices to (9, Ny) real array."""
    Ny = rho.shape[0]
    vec = np.zeros((9, Ny))
    vec[0] = rho[:, 0, 0].real
    vec[1] = rho[:, 1, 1].real
    vec[2] = rho[:, 2, 2].real
    vec[3] = rho[:, 0, 1].real
    vec[4] = rho[:, 0, 1].imag
    vec[5] = rho[:, 0, 2].real
    vec[6] = rho[:, 0, 2].imag
    vec[7] = rho[:, 1, 2].real
    vec[8] = rho[:, 1, 2].imag
    return vec


def _rho_vec_batch_to_matrix_4(rho_vec):
    """Convert (16, Ny) real array to (Ny, 4, 4) complex Hermitian matrices.

    Layout for 3+1 sterile extension:
      [rho_ee, rho_mumu, rho_tautau, rho_ss,           # diag: 0-3
       Re(rho_emu), Im(rho_emu),                        # 4-5
       Re(rho_etau), Im(rho_etau),                      # 6-7
       Re(rho_es), Im(rho_es),                          # 8-9
       Re(rho_mutau), Im(rho_mutau),                    # 10-11
       Re(rho_mus), Im(rho_mus),                        # 12-13
       Re(rho_taus), Im(rho_taus)]                      # 14-15
    """
    Ny = rho_vec.shape[1]
    rho = np.zeros((Ny, 4, 4), dtype=complex)
    # Diagonals
    for d in range(4):
        rho[:, d, d] = rho_vec[d]
    # Off-diagonals: pairs (i,j) with i<j, stored as (Re, Im) at indices
    # 4+2*k, 5+2*k where k enumerates the 6 pairs in order:
    # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    _pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for k, (i, j) in enumerate(_pairs):
        re_idx = 4 + 2 * k
        im_idx = 5 + 2 * k
        rho[:, i, j] = rho_vec[re_idx] + 1j * rho_vec[im_idx]
        rho[:, j, i] = rho_vec[re_idx] - 1j * rho_vec[im_idx]
    return rho


def _matrix_batch_to_rho_vec_4(rho):
    """Convert (Ny, 4, 4) complex Hermitian matrices to (16, Ny) real array."""
    Ny = rho.shape[0]
    vec = np.zeros((16, Ny))
    for d in range(4):
        vec[d] = rho[:, d, d].real
    _pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for k, (i, j) in enumerate(_pairs):
        vec[4 + 2 * k] = rho[:, i, j].real
        vec[5 + 2 * k] = rho[:, i, j].imag
    return vec


###############################################################################
# DensityMatrixSolver class                                                    #
###############################################################################

class DensityMatrixSolver(object):
    """
    Full 3x3 density matrix QKE solver for neutrino flavor evolution.

    Tracks the complete 3x3 Hermitian density matrix rho(y) for neutrinos
    and rho_bar(y) for antineutrinos at each comoving momentum mode y.
    Off-diagonal elements encode flavor coherences from neutrino oscillations.

    The Quantum Kinetic Equations (QKE) are solved via Strang operator splitting:
      1. Half oscillation step: exact unitary rotation exp(-iHdt/2) rho exp(iHdt/2)
      2. Full collision step: diagonal rates from D-kernel integrals,
         off-diagonal damping C_ij = -1/2 (Gamma_i + Gamma_j) rho_ij
      3. Half oscillation step

    State representation: rho_all shape (2, 9, Ny)
      sector 0 = neutrinos, sector 1 = antineutrinos
      9 components per mode: [rho_ee, rho_mumu, rho_tautau,
        Re(rho_emu), Im(rho_emu), Re(rho_etau), Im(rho_etau),
        Re(rho_mutau), Im(rho_mutau)]
    """

    def __init__(self, Ny=None, y_max=None, y_coll_max=None, C_NP_funcs=None):
        # Create internal BoltzmannSolver for collision integrals and grid
        self._boltz = BoltzmannSolver(Ny=Ny, y_max=y_max, y_coll_max=y_coll_max,
                                       C_NP_funcs=C_NP_funcs)
        self.Ny = self._boltz.Ny
        self.y_grid = self._boltz.y_grid
        self.dy = self._boltz.dy
        self.y_max = self._boltz.y_max
        self.n_species = self._boltz.n_species  # 3 for collision integrals

        # Sterile extension: 3×3 or 4×4 density matrix
        self.n_flavor = 4 if PRyMini.sterile_flag else 3
        self.n_components = self.n_flavor**2  # 9 or 16 real components

        # Build PMNS mixing matrix (3×3 or 4×4)
        self._build_PMNS()

        # Precompute vacuum Hamiltonian base matrices (eV^2)
        # H_vac_nu(y) = Omega_nu / E(y),  H_vac_nubar(y) = Omega_nubar / E(y)
        if self.n_flavor == 4:
            Dm2 = np.array([0.0, PRyMini.Dm2_21, PRyMini.Dm2_31,
                            PRyMini.Dm2_41])
        else:
            Dm2 = np.array([0.0, PRyMini.Dm2_21, PRyMini.Dm2_31])
        Dm2_half = np.diag(Dm2 / 2.0)
        self._Omega_nu = self.U_PMNS @ Dm2_half @ self.U_PMNS.conj().T
        self._Omega_nubar = self.U_PMNS.conj() @ Dm2_half @ self.U_PMNS.T

        # Electron-flavor projector for matter potential
        N = self.n_flavor
        self._diag_e = np.zeros((N, N), dtype=complex)
        self._diag_e[0, 0] = 1.0

        # W boson mass squared for thermal matter potential
        self.mW2 = (PRyMini.mZ * np.sqrt(1.0 - PRyMini.sW2))**2  # MeV^2

        # Collision damping coefficients (de Salas & Pastor 2016)
        # D_alpha = C_D_alpha * GF^2 * T^4 * E  [natural units]
        # Sterile: Gamma_s = 0 (no SM gauge coupling)
        if self.n_flavor == 4:
            self.C_D = np.array([3.06, 2.22, 2.22, 0.0])
        else:
            self.C_D = np.array([3.06, 2.22, 2.22])

        # Annihilation coefficients (g_alpha^a)^2 from Mirizzi+2012 Eq. 29
        # (Hannestad ref. [33]). Used with C_D = (g_alpha^s)^2 by the Mirizzi
        # pair-specific damping formula, PRyMini.qke_damping_formula = "mirizzi".
        if self.n_flavor == 4:
            self.C_A = np.array([0.50, 0.28, 0.28, 0.0])
        else:
            self.C_A = np.array([0.50, 0.28, 0.28])

        # Off-diagonal ν-e scattering couplings (for collision gain terms).
        geL = PRyMini.geL
        gmuL = PRyMini.gmuL
        geR2 = PRyMini.geR**2
        self.c_emu_scat = 4.0 * (geL * gmuL + geR2)
        self.c_mutau_scat = 4.0 * (gmuL**2 + geR2)

        # eV <-> seconds conversion
        self._eV_to_secm1 = PRyMini.MeV_to_secm1 * 1.0e-6

        # Helper dispatchers for vec <-> matrix conversion
        if self.n_flavor == 4:
            self._to_mat = _rho_vec_batch_to_matrix_4
            self._to_vec = _matrix_batch_to_rho_vec_4
        else:
            self._to_mat = _rho_vec_batch_to_matrix
            self._to_vec = _matrix_batch_to_rho_vec

        # Off-diagonal component indices for each flavor pair.
        #
        # Layouts:
        #   3-flavor: diag 0,1,2; pairs (0,1)@3-4, (0,2)@5-6, (1,2)@7-8
        #   4-flavor: diag 0,1,2,3; pairs (0,1)@4-5, (0,2)@6-7, (0,3)@8-9,
        #              (1,2)@10-11, (1,3)@12-13, (2,3)@14-15
        #
        # _active_offdiag_idx: indices of the 3 active-active pairs (e-μ,
        #   e-τ, μ-τ), used when passing to _offdiag_collision_gain which
        #   expects exactly 3 pairs.
        # _active_pair_write: [(Re, Im)] write targets for the 3 active pairs.
        # _all_pair_flavors:  list of (α, β) for ALL pairs (3 or 6).
        # _all_pair_write:    [(Re, Im)] write targets for ALL pairs.
        if self.n_flavor == 4:
            self._active_offdiag_idx = [4, 5, 6, 7, 10, 11]
            self._active_pair_write = [(4, 5), (6, 7), (10, 11)]
            # All 6 pairs: active-active first, then active-sterile
            self._all_pair_flavors = [(0, 1), (0, 2), (1, 2),
                                      (0, 3), (1, 3), (2, 3)]
            self._all_pair_write = [(4, 5), (6, 7), (10, 11),
                                    (8, 9), (12, 13), (14, 15)]
        else:
            self._active_offdiag_idx = [3, 4, 5, 6, 7, 8]
            self._active_pair_write = [(3, 4), (5, 6), (7, 8)]
            self._all_pair_flavors = [(0, 1), (0, 2), (1, 2)]
            self._all_pair_write = [(3, 4), (5, 6), (7, 8)]

        if PRyMini.verbose_flag:
            tag = "4x4 (3+1 sterile)" if self.n_flavor == 4 else "3x3"
            print(f"  DensityMatrixSolver: {tag} QKE, "
                  f"{2*self.n_components*self.Ny} real DOFs")

        # Stage E.2 sprint 8: per-step energy accounting for the active-
        # sterile 2x2 blocks. Populated only when PRyMini.qke_energy_diag_flag
        # is True (guarded in evolve_step_ode_etdrk2 snapshots). Empty
        # default costs ~nothing and keeps solver state deterministic.
        self._energy_hist = []
        self._energy_step_idx = 0
        # Stage E.2 sprint 9: per-step per-y-mode MSW-passage snapshots.
        # Populated only when PRyMini.qke_msw_diag_flag is True (guarded in
        # evolve_step_ode_etdrk2 snapshots). Empty default is bit-identical.
        self._msw_hist = []
        self._msw_step_idx = 0

    def _build_PMNS(self):
        """Construct the PMNS mixing matrix from oscillation parameters.

        When sterile_flag=False: standard 3×3 PDG parameterization.
        When sterile_flag=True: 4×4 using the 3+1 convention
          U₄ₓ₄ = R₃₄(θ₃₄) × R₂₄(θ₂₄, δ₁₄) × R₁₄(θ₁₄) × [U₃ₓ₃ ⊕ 1]
        which recovers the 3-flavor matrix when θ₁₄=θ₂₄=θ₃₄=0.
        """
        s12 = np.sin(PRyMini.theta_12)
        c12 = np.cos(PRyMini.theta_12)
        s13 = np.sin(PRyMini.theta_13)
        c13 = np.cos(PRyMini.theta_13)
        s23 = np.sin(PRyMini.theta_23)
        c23 = np.cos(PRyMini.theta_23)
        eidCP = np.exp(1j * PRyMini.delta_CP)
        emidCP = np.exp(-1j * PRyMini.delta_CP)

        U3 = np.array([
            [c12*c13,                        s12*c13,                        s13*emidCP],
            [-s12*c23 - c12*s23*s13*eidCP,   c12*c23 - s12*s23*s13*eidCP,   s23*c13],
            [s12*s23 - c12*c23*s13*eidCP,   -c12*s23 - s12*c23*s13*eidCP,   c23*c13]
        ], dtype=complex)

        if self.n_flavor == 3:
            self.U_PMNS = U3
            return

        # 4×4 extension: embed U3 into the upper-left block, then apply
        # the three active-sterile rotation matrices.
        U4 = np.eye(4, dtype=complex)
        U4[:3, :3] = U3

        # R_14(theta_14): rotation in the 1-4 plane
        s14 = np.sin(PRyMini.theta_14)
        c14 = np.cos(PRyMini.theta_14)
        R14 = np.eye(4, dtype=complex)
        R14[0, 0] = c14;  R14[0, 3] = s14
        R14[3, 0] = -s14; R14[3, 3] = c14

        # R_24(theta_24, delta_14): rotation in 2-4 plane with CP phase
        s24 = np.sin(PRyMini.theta_24)
        c24 = np.cos(PRyMini.theta_24)
        eid14 = np.exp(1j * PRyMini.delta_14)
        emid14 = np.exp(-1j * PRyMini.delta_14)
        R24 = np.eye(4, dtype=complex)
        R24[1, 1] = c24;          R24[1, 3] = s24 * emid14
        R24[3, 1] = -s24 * eid14; R24[3, 3] = c24

        # R_34(theta_34): rotation in 3-4 plane
        s34 = np.sin(PRyMini.theta_34)
        c34 = np.cos(PRyMini.theta_34)
        R34 = np.eye(4, dtype=complex)
        R34[2, 2] = c34;  R34[2, 3] = s34
        R34[3, 2] = -s34; R34[3, 3] = c34

        # U₄ₓ₄ = R₃₄ × R₂₄ × R₁₄ × [U₃ₓ₃ ⊕ 1]
        self.U_PMNS = R34 @ R24 @ R14 @ U4

    def initial_conditions(self, Tnu, a, f_initial=None):
        """Return initial density matrices for the QKE solver.

        Parameters
        ----------
        Tnu : float
            Neutrino temperature at the start of the Boltzmann phase (MeV).
        a : float
            Scale factor at start.
        f_initial : dict, optional
            User-supplied initial distribution callables, keyed by species
            name. Missing keys fall back to thermal FD at Tnu. The 'nus'
            and 'nusbar' keys (sterile) default to ZERO (empty sterile
            sector) rather than thermal FD.

        Returns
        -------
        rho_all : ndarray, shape (2, n_components, Ny)
            Density matrices. Off-diagonals initialized to zero.
        """
        rho_all = np.zeros((2, self.n_components, self.Ny))
        Tnu_com = Tnu * a

        # Active species map (sector, diagonal index)
        _species_to_slot = {
            'nue':      (0, 0), 'numu':      (0, 1), 'nutau':    (0, 2),
            'nuebar':   (1, 0), 'numubar':   (1, 1), 'nutaubar': (1, 2),
        }
        if self.n_flavor == 4:
            _species_to_slot['nus'] = (0, 3)
            _species_to_slot['nusbar'] = (1, 3)

        # Default thermal FD for active flavors
        p_grid = self.y_grid / a
        fd_default = np.zeros(self.Ny)
        for i in range(self.Ny):
            x = self.y_grid[i] / Tnu_com
            if x < 500.0:
                fd_default[i] = 1.0 / (np.exp(x) + 1.0)

        # Stage C (Shi-Fuller): per-flavor asymmetric FD from ξ_α.
        # f_ν(y)  = 1 / (exp(y/T − ξ) + 1)   (neutrino:       positive ξ ⇒ excess ν)
        # f_ν̄(y) = 1 / (exp(y/T + ξ) + 1)   (antineutrino)
        # Only active when sterile_flag=True. Zero ξ everywhere ⇒ fd_default.
        def fd_xi(xi):
            if xi == 0.0:
                return fd_default, fd_default
            f_nu = np.zeros(self.Ny)
            f_nubar = np.zeros(self.Ny)
            for i in range(self.Ny):
                x = self.y_grid[i] / Tnu_com
                if x - xi < 500.0:
                    f_nu[i] = 1.0 / (np.exp(x - xi) + 1.0)
                if x + xi < 500.0:
                    f_nubar[i] = 1.0 / (np.exp(x + xi) + 1.0)
            return f_nu, f_nubar

        if getattr(PRyMini, 'sterile_flag', False):
            fd_e_nu,  fd_e_nubar  = fd_xi(getattr(PRyMini, 'xi_nue_init',   0.0))
            fd_mu_nu, fd_mu_nubar = fd_xi(getattr(PRyMini, 'xi_numu_init',  0.0))
            fd_ta_nu, fd_ta_nubar = fd_xi(getattr(PRyMini, 'xi_nutau_init', 0.0))
        else:
            fd_e_nu  = fd_e_nubar  = fd_default
            fd_mu_nu = fd_mu_nubar = fd_default
            fd_ta_nu = fd_ta_nubar = fd_default

        _fd_by_species = {
            'nue':      fd_e_nu,  'nuebar':   fd_e_nubar,
            'numu':     fd_mu_nu, 'numubar':  fd_mu_nubar,
            'nutau':    fd_ta_nu, 'nutaubar': fd_ta_nubar,
        }

        for species, (sector, flavor) in _species_to_slot.items():
            if f_initial is not None and species in f_initial and f_initial[species] is not None:
                f_vals = np.asarray(f_initial[species](p_grid, Tnu), dtype=float)
                if f_vals.shape != (self.Ny,):
                    raise ValueError(
                        f"initial_conditions: callable for {species} returned "
                        f"shape {f_vals.shape}, expected {(self.Ny,)}")
                rho_all[sector, flavor] = f_vals
            elif species in ('nus', 'nusbar'):
                # Sterile starts EMPTY by default (not thermal)
                rho_all[sector, flavor] = 0.0
            else:
                rho_all[sector, flavor] = _fd_by_species[species]

        return rho_all

    def oscillation_step(self, rho_all, dt, a, Tg):
        """Apply exact unitary oscillation evolution for time interval dt.

        For each momentum mode and each sector (nu, nubar), diagonalizes
        the effective Hamiltonian H = H_vac + H_matter, computes the exact
        phase rotation, and transforms back to the flavor basis.

        Vectorized over all Ny momentum modes using batch eigendecomposition.

        Parameters
        ----------
        rho_all : ndarray, shape (2, 9, Ny)
            Modified in-place.
        dt : float
            Time interval in seconds.
        a : float
            Scale factor.
        Tg : float
            Photon temperature in MeV.
        """
        Ny = self.Ny
        # Convert dt to natural units: phase = H[eV] * dt_nat is dimensionless
        dt_nat = dt * self._eV_to_secm1

        # Physical energies in eV: E = y/a (massless neutrinos)
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)  # (Ny,) eV
        inv_E = 1.0 / E_eV  # (Ny,) 1/eV

        # Thermal matter potential (Notzold-Raffelt 1988):
        #   V_th(E) = -8*sqrt(2)/3 * GF * E * rho_e / mW^2
        # where rho_e = 7*pi^2/60 * T^4 (relativistic e+e- energy density).
        # V_th suppresses oscillations at high T via matter-induced mass.
        rho_e = 7.0 * np.pi**2 / 60.0 * Tg**4  # MeV^4
        V0 = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e / (3.0 * self.mW2)  # dimless
        V_eV = V0 * E_eV  # (Ny,) eV

        # Neutrino self-interaction potential (Sigl & Raffelt 1993):
        #   V_nunu = sqrt(2) * GF / (2*pi^2 * a^3) * int (rho_y - rhobar_y) y^2 dy
        # This 3x3 matrix arises from forward nu-nu scattering and creates
        # synchronized oscillation effects that enhance flavor conversion.
        rho_nu_mat = self._to_mat(rho_all[0])   # (Ny, N, N)
        rho_nubar_mat = self._to_mat(rho_all[1])  # (Ny, N, N)
        diff_mat = rho_nu_mat - rho_nubar_mat  # (Ny, 3, 3)
        # Quadrature: sum w_i * y_i^2 * diff_mat[i] over collision grid
        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2  # (Ny_coll,) MeV^3
        V_nunu_mat = np.einsum('i,ijk->jk', y2w, diff_mat[:Ny_coll])  # (N,N) MeV^3
        V_nunu_prefactor = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3)
        V_nunu_eV = V_nunu_prefactor * V_nunu_mat * 1.0e6  # (N,N) eV

        # Build Hamiltonians: shape (Ny, N, N) where N = self.n_flavor.
        # H = Omega/E + V_thermal*diag(1,0,...,0) + V_nunu
        # In our convention (phase_signs = [-1,+1]):
        #   nu:  exp(-iH_nu dt) rho exp(+iH_nu dt) -> i*drho/dt = [H_nu, rho]
        #   nubar: exp(+iH_nubar dt) rhobar exp(-iH_nubar dt)
        N = self.n_flavor
        H_list = [None, None]
        for s in range(2):
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, N, N), dtype=complex)
            for k in range(N):
                for l in range(N):
                    H[:, k, l] = Omega[k, l] * inv_E + V_nunu_eV[k, l]
            H[:, 0, 0] += V_eV
            H_list[s] = H

        # Phase sign convention:
        # Neutrinos:     i*drho/dt = [H, rho]   -> rho(t+dt) = exp(-iHdt) rho exp(+iHdt)
        # Antineutrinos: i*drhobar/dt = -[Hbar, rhobar]
        #                             -> rhobar(t+dt) = exp(+iHbar*dt) rhobar exp(-iHbar*dt)
        phase_signs = [-1.0, +1.0]

        for sector in range(2):
            H_eff = H_list[sector]
            sign = phase_signs[sector]

            # Batch eigendecompose: H = P diag(lambda) P^dagger
            eigenvalues, P = np.linalg.eigh(H_eff)  # (Ny,3), (Ny,3,3)

            # Phase factors: exp(i * sign * lambda * dt_nat)
            phases = np.exp(1j * sign * eigenvalues * dt_nat)  # (Ny, 3)

            # Reconstruct density matrices from vec9
            rho_mat = self._to_mat(rho_all[sector])  # (Ny, N, N)

            # Transform to H eigenbasis: rho_H = P^dag @ rho @ P
            Pdag = P.conj().transpose(0, 2, 1)  # (Ny, 3, 3)
            rho_H = np.einsum('nij,njk,nkl->nil', Pdag, rho_mat, P)

            # Apply phase rotations: rho'_H[n,i,j] *= phase[n,i] * conj(phase[n,j])
            phase_ij = phases[:, :, None] * phases[:, None, :].conj()  # (Ny, 3, 3)
            rho_H *= phase_ij

            # Transform back: rho' = P @ rho'_H @ P^dag
            rho_new = np.einsum('nij,njk,nkl->nil', P, rho_H, Pdag)

            # Store back as vec9
            rho_all[sector] = self._to_vec(rho_new)

    def _compute_D_pair_matrix(self, T_eV, E_eV, units="eV"):
        """Off-diagonal pair damping D[alpha, beta, :] for alpha != beta.

        Selected by PRyMini.qke_damping_formula:
          - "symmetric" (legacy): D_ab = 0.5 * (Gamma_a + Gamma_b)
                where Gamma_a = C_D[a] * GF^2 * T^4 * E.
          - "mirizzi" (Mirizzi+2012 Eq. 28, Hannestad-style coefs):
                D_ab = 0.5 * GF^2 * T^4 * E
                       * [(g_a^s - g_b^s)^2 + (g_a^a + g_b^a)^2]
                with g^s = sqrt(C_D), g^a = sqrt(C_A). Gives active-sterile
                damping ~2.3x active-active damping; used to fix the
                small-mixing DW overproduction relative to Hannestad+2012.
          - "gariazzo" (Gariazzo+2019 App. A.17-A.20): NotImplementedError
                until Step 8 of Stage E.1 (dimensional calibration).

        Parameters
        ----------
        T_eV : float
            Photon temperature in eV.
        E_eV : ndarray, shape (Ny,)
            Per-mode physical energy in eV.
        units : {"eV", "si"}
            Output units: "eV" (natural, for L assembly) or "si" (1/s,
            for collision-RHS assembly).

        Returns
        -------
        D_off : ndarray, shape (N, N, Ny)
            Diagonal entries D[a, a] are zero; off-diagonals hold the
            pair damping rate in the requested units.
        """
        N = self.n_flavor
        Ny = E_eV.shape[0]
        prefac = PRyMini.GF * 1.0e-12  # GF in eV^-2
        base = (prefac**2) * (T_eV**4) * E_eV  # (Ny,) eV, common factor

        formula = getattr(PRyMini, "qke_damping_formula", "symmetric")
        D_off = np.zeros((N, N, Ny))

        if formula == "symmetric":
            Gamma = np.zeros((N, Ny))
            for alpha in range(N):
                Gamma[alpha] = self.C_D[alpha] * base
            for alpha in range(N):
                for beta in range(N):
                    if alpha != beta:
                        D_off[alpha, beta] = 0.5 * (Gamma[alpha] + Gamma[beta])
        elif formula == "mirizzi":
            g_s = np.sqrt(self.C_D)  # (g_alpha^s), shape (N,)
            g_a = np.sqrt(self.C_A)  # (g_alpha^a), shape (N,)
            for alpha in range(N):
                for beta in range(N):
                    if alpha == beta:
                        continue
                    scat = (g_s[alpha] - g_s[beta])**2
                    anni = (g_a[alpha] + g_a[beta])**2
                    D_off[alpha, beta] = 0.5 * (scat + anni) * base
        elif formula == "gariazzo":
            raise NotImplementedError(
                "Gariazzo damping form requires dimensional calibration "
                "(Stage E.1 Step 8). Use 'mirizzi' or 'symmetric'."
            )
        else:
            raise ValueError(
                f"Unknown qke_damping_formula={formula!r}; "
                "expected one of {'symmetric', 'mirizzi', 'gariazzo'}."
            )

        if units == "si":
            D_off *= self._eV_to_secm1
        elif units != "eV":
            raise ValueError(f"units must be 'eV' or 'si', got {units!r}")

        # Stage E.2 sprint 4 diagnostic knob: global D scaling.
        _damp_scale = getattr(PRyMini, "qke_damping_scale", 1.0)
        if _damp_scale != 1.0:
            D_off = D_off * _damp_scale

        return D_off

    def collision_step(self, rho_all, phi1_dt, dt, a, Tg):
        """Apply collision integrals to the density matrix.

        Diagonal elements: updated using existing D-kernel collision integrals
        from the wrapped BoltzmannSolver (no collision_mixing, since the QKE
        oscillation step handles flavor mixing directly).

        Off-diagonal elements: gain (transport) + damping via exponential Euler:
            rho_ij(t+dt) = exp(-D*dt) * rho_ij(t) + phi1(D*dt)*dt * gain_ij
        where D = 1/2(Gamma_i + Gamma_j) and gain_ij scatters coherences
        from other momenta to p_i via the D-kernel collision formalism.

        Parameters
        ----------
        rho_all : ndarray, shape (2, 9, Ny)
            Modified in-place.
        phi1_dt : float
            phi_1(z) * dt for exponential Euler regularization of diagonals.
        dt : float
            Actual time step in seconds (for off-diagonal damping).
        a : float
            Scale factor at midpoint.
        Tg : float
            Photon temperature in MeV at midpoint.
        """
        # --- Extract diagonal distributions for 3-species collision integrals ---
        f_all = np.zeros((3, self.Ny))
        f_all[0] = rho_all[0, 0]  # f_nue = rho_ee (neutrino sector)
        f_all[1] = rho_all[1, 0]  # f_nuebar = rho_bar_ee (antineutrino sector)
        # f_numu_eff: average over all 4 mu/tau-type species
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]    # rho_mumu + rho_tautau (nu)
                          + rho_all[1, 1] + rho_all[1, 2])    # rho_bar_mumu + rho_bar_tautau (nubar)

        # Compute collision integrals via raw functions (bypassing collision_mixing)
        GF2_pref = self._boltz.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        I_nu_nu = _collision_integral_nu_nu(
            f_all, self.y_grid, self._boltz.quad_w, a, GF2_pref, tail_params,
            self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

        if PRyMini.massive_electron_flag:
            I_nu_e = _collision_integral_nu_e_massive(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2,
                self._boltz.gmuL2, self._boltz.gmuR2,
                PRyMini.me, self._boltz.Ny_coll)
        else:
            fnu_e_scat_val = float(self._boltz._fnu_e_scat(Tg))
            fnu_e_ann_val = float(self._boltz._fnu_e_ann(Tg))
            fnu_mu_scat_val = float(self._boltz._fnu_mu_scat(Tg))
            fnu_mu_ann_val = float(self._boltz._fnu_mu_ann(Tg))
            I_nu_e = _collision_integral_nu_e(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2, self._boltz.geLgeR,
                self._boltz.gmuL2, self._boltz.gmuR2, self._boltz.gmuLgmuR,
                PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                self._boltz.D_k0, self._boltz.D_k1, self._boltz.D_k2,
                self._boltz.Ny_coll, 1.0, 1.0, 1.0, 1.0)

        I_total = I_nu_nu + I_nu_e  # shape (3, Ny)

        # Add NP collision terms
        C_NP = self._boltz.C_NP_funcs
        if 'nue' in C_NP:
            I_total[0] += C_NP['nue'](self.y_grid, a, Tg, f_all)
        if 'nuebar' in C_NP:
            I_total[1] += C_NP['nuebar'](self.y_grid, a, Tg, f_all)
        if 'numu' in C_NP:
            I_total[2] += C_NP['numu'](self.y_grid, a, Tg, f_all)

        # --- Update diagonal elements with exponential Euler ---
        # I_total[0] = nue, I_total[1] = nuebar, I_total[2] = numu_eff
        # phi1_dt can be a scalar or array of shape (3, Ny).
        if np.ndim(phi1_dt) == 0:
            _p0 = phi1_dt
            _p1 = phi1_dt
            _p2 = phi1_dt
        else:
            _p0 = phi1_dt[0]
            _p1 = phi1_dt[1]
            _p2 = phi1_dt[2]
        rho_all[0, 0] += _p0 * I_total[0]   # rho_ee neutrino
        rho_all[1, 0] += _p1 * I_total[1]   # rho_bar_ee antineutrino
        rho_all[0, 1] += _p2 * I_total[2]   # rho_mumu neutrino
        rho_all[0, 2] += _p2 * I_total[2]   # rho_tautau neutrino
        rho_all[1, 1] += _p2 * I_total[2]   # rho_bar_mumu antineutrino
        rho_all[1, 2] += _p2 * I_total[2]   # rho_bar_tautau antineutrino

        # --- Off-diagonal: damping + gain (transport) ---
        # Gamma_alpha = C_D_alpha * GF^2 * T^4 * E  [eV], converted to 1/s
        GF_eV = PRyMini.GF * 1.0e-12  # MeV^{-2} -> eV^{-2}
        T_eV = Tg * 1.0e6
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)

        Gamma = np.zeros((3, self.Ny))  # [nue, numu, nutau] collision rates in 1/s
        for alpha in range(3):
            Gamma[alpha] = self.C_D[alpha] * GF_eV**2 * T_eV**4 * E_eV * self._eV_to_secm1

        # Damping rates D_αβ = ½(Γ_α + Γ_β) for each off-diagonal pair
        D_emu = 0.5 * (Gamma[0] + Gamma[1])    # e-μ (components 3,4)
        D_etau = 0.5 * (Gamma[0] + Gamma[2])   # e-τ (components 5,6)
        D_mutau = 0.5 * (Gamma[1] + Gamma[2])  # μ-τ (components 7,8)

        # Damping factors: exp(-D * dt)
        damp_emu = np.exp(-D_emu * dt)
        damp_etau = np.exp(-D_etau * dt)
        damp_mutau = np.exp(-D_mutau * dt)

        # Exponential Euler gain factor: phi1(D*dt)*dt = (1-exp(-D*dt))/D
        # For D*dt→0: phi1*dt → dt; for D*dt→∞: phi1*dt → 1/D
        z_emu = D_emu * dt
        z_etau = D_etau * dt
        z_mutau = D_mutau * dt
        _eps = 1.0e-8
        phi1dt_emu = np.where(z_emu > 1.0e-4,
                              (1.0 - damp_emu) / np.maximum(D_emu, _eps),
                              dt * (1.0 - 0.5 * z_emu))
        phi1dt_etau = np.where(z_etau > 1.0e-4,
                               (1.0 - damp_etau) / np.maximum(D_etau, _eps),
                               dt * (1.0 - 0.5 * z_etau))
        phi1dt_mutau = np.where(z_mutau > 1.0e-4,
                                (1.0 - damp_mutau) / np.maximum(D_mutau, _eps),
                                dt * (1.0 - 0.5 * z_mutau))

        # Finite-mass correction for off-diagonal ν-e scattering
        if PRyMini.massive_electron_flag:
            # Massive electron kinematics already exact; no correction needed
            fnu_emu_scat_val = 1.0
            fnu_mutau_scat_val = 1.0
        else:
            fnu_emu_scat_val = np.sqrt(max(fnu_e_scat_val * fnu_mu_scat_val, 0.0))
            fnu_mutau_scat_val = fnu_mu_scat_val  # μ-τ = diagonal μ

        # Compute off-diagonal gain (transport) for each sector.
        # _active_offdiag_idx maps the 6 active-active off-diagonal
        # components to the correct rho_all indices (contiguous for 3-flavor,
        # non-contiguous for 4-flavor where sterile pairs interleave).
        _oidx = self._active_offdiag_idx
        _pw = self._active_pair_write
        for sector in range(2):
            rho_offdiag = rho_all[sector, _oidx, :]  # shape (6, Ny)
            B_idx = 1 if sector == 0 else 0

            gain = _offdiag_collision_gain(
                rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                GF2_pref,
                self.c_emu_scat, self.c_mutau_scat,
                fnu_emu_scat_val, fnu_mutau_scat_val,
                B_idx, tail_params,
                self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            # Update off-diagonals: ρ_new = damp × ρ_old + phi1dt × gain
            # e-μ
            rho_all[sector, _pw[0][0]] = damp_emu * rho_all[sector, _pw[0][0]] + phi1dt_emu * gain[0]
            rho_all[sector, _pw[0][1]] = damp_emu * rho_all[sector, _pw[0][1]] + phi1dt_emu * gain[1]
            # e-τ
            rho_all[sector, _pw[1][0]] = damp_etau * rho_all[sector, _pw[1][0]] + phi1dt_etau * gain[2]
            rho_all[sector, _pw[1][1]] = damp_etau * rho_all[sector, _pw[1][1]] + phi1dt_etau * gain[3]
            # μ-τ
            rho_all[sector, _pw[2][0]] = damp_mutau * rho_all[sector, _pw[2][0]] + phi1dt_mutau * gain[4]
            rho_all[sector, _pw[2][1]] = damp_mutau * rho_all[sector, _pw[2][1]] + phi1dt_mutau * gain[5]

        # Clip diagonal elements to valid range [0, 1]
        f_min = 1.0e-30
        f_max = 1.0 - f_min
        for sector in range(2):
            for d in range(3):
                rho_all[sector, d] = np.clip(rho_all[sector, d], f_min, f_max)

    def evolve_step(self, rho_all, dt, phi1_dt, a, Tg):
        """Combined oscillation + collision step (implicit off-diagonal).

        Instead of Strang splitting (half-osc, collision, half-osc), this
        solves the combined oscillation + damping ODE for off-diagonal
        elements analytically using a complex exponential Euler scheme:

            dρ_αβ/dt = -(D_αβ + i*s*ω_αβ)*ρ_αβ + S_αβ

        where s = +1 for ν, -1 for ν̄, ω_αβ = (H_αα - H_ββ) is the
        effective oscillation frequency, and S_αβ includes both the
        oscillation source (-i*s*H_αβ*(ρ_ββ-ρ_αα)) and collision gain.

        This correctly captures the steady-state ρ_αβ = S/(D + iω), which
        produces the Sigl-Raffelt relaxation rate ω²D/(ω²+D²) for the
        diagonal elements, without the splitting error that occurs when
        oscillation and collision are applied sequentially.

        Diagonal elements are updated with exponential Euler from collision
        integrals plus the oscillation-induced relaxation feedback.

        Parameters
        ----------
        rho_all : ndarray, shape (2, 9, Ny)
            Modified in-place.
        dt : float
            Physical time step in seconds.
        phi1_dt : float
            phi_1(z) * dt for exponential Euler regularization of diagonals.
        a : float
            Scale factor at midpoint.
        Tg : float
            Photon temperature in MeV at midpoint.
        """
        Ny = self.Ny

        # ================================================================
        # 1. Diagonal collision integrals (same as collision_step)
        # ================================================================
        f_all = np.zeros((3, Ny))
        f_all[0] = rho_all[0, 0]
        f_all[1] = rho_all[1, 0]
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                          + rho_all[1, 1] + rho_all[1, 2])

        GF2_pref = self._boltz.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        I_nu_nu = _collision_integral_nu_nu(
            f_all, self.y_grid, self._boltz.quad_w, a, GF2_pref, tail_params,
            self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

        if PRyMini.massive_electron_flag:
            I_nu_e = _collision_integral_nu_e_massive(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2,
                self._boltz.gmuL2, self._boltz.gmuR2,
                PRyMini.me, self._boltz.Ny_coll)
        else:
            fnu_e_scat_val = float(self._boltz._fnu_e_scat(Tg))
            fnu_e_ann_val = float(self._boltz._fnu_e_ann(Tg))
            fnu_mu_scat_val = float(self._boltz._fnu_mu_scat(Tg))
            fnu_mu_ann_val = float(self._boltz._fnu_mu_ann(Tg))
            I_nu_e = _collision_integral_nu_e(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2, self._boltz.geLgeR,
                self._boltz.gmuL2, self._boltz.gmuR2, self._boltz.gmuLgmuR,
                PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                self._boltz.D_k0, self._boltz.D_k1, self._boltz.D_k2,
                self._boltz.Ny_coll, 1.0, 1.0, 1.0, 1.0)

        I_total = I_nu_nu + I_nu_e
        C_NP = self._boltz.C_NP_funcs
        if 'nue' in C_NP:
            I_total[0] += C_NP['nue'](self.y_grid, a, Tg, f_all)
        if 'nuebar' in C_NP:
            I_total[1] += C_NP['nuebar'](self.y_grid, a, Tg, f_all)
        if 'numu' in C_NP:
            I_total[2] += C_NP['numu'](self.y_grid, a, Tg, f_all)

        # Update diagonals with collision integrals (exponential Euler).
        # phi1_dt can be a scalar or array of shape (3, Ny) for mode-dependent
        # regularization (species order: nue, nuebar, numu_equiv).
        if np.ndim(phi1_dt) == 0:
            # Scalar phi1_dt: apply uniformly
            _p0 = phi1_dt
            _p1 = phi1_dt
            _p2 = phi1_dt
        else:
            # Array phi1_dt[species, y]: mode-dependent
            _p0 = phi1_dt[0]
            _p1 = phi1_dt[1]
            _p2 = phi1_dt[2]
        rho_all[0, 0] += _p0 * I_total[0]
        rho_all[1, 0] += _p1 * I_total[1]
        rho_all[0, 1] += _p2 * I_total[2]
        rho_all[0, 2] += _p2 * I_total[2]
        rho_all[1, 1] += _p2 * I_total[2]
        rho_all[1, 2] += _p2 * I_total[2]

        # ================================================================
        # 2. Build Hamiltonians (flavor basis, per momentum mode)
        # ================================================================
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)
        inv_E = 1.0 / E_eV

        # Thermal matter potential (Notzold-Raffelt 1988). Two pieces from forward
        # scattering on the CP-symmetric e+/e- plasma:
        #   * V_thermal_eV  (CC, via W exchange; only ν_e)                1/m_W^2
        #   * V_NC_eV       (NC, via Z exchange; flavor-universal active) 1/m_Z^2
        # Same sign for ν and ν̄ (CP-even). Adding V_NC·I_3 to the active block is
        # a global phase for 3-flavor SM (observable-neutral), but in 3+1 sterile
        # it creates the active-sterile gap that suppresses in-medium mixing at
        # small θ_{14,24,34} -- the canonical V_z that enters the DW formula.
        rho_e_th = 7.0 * np.pi**2 / 60.0 * Tg**4
        V_pref = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e_th / 3.0
        V_thermal_eV = V_pref / self.mW2 * E_eV
        V_NC_eV = V_pref / (PRyMini.mZ**2) * E_eV
        # Stage E.2 sprint 5 diagnostic scale knobs.
        V_thermal_eV = V_thermal_eV * getattr(PRyMini, "qke_v_thermal_scale", 1.0)
        V_NC_eV = V_NC_eV * getattr(PRyMini, "qke_v_nc_scale", 1.0)

        # CC matter potential: V_CC = sqrt(2) GF (n_e- - n_e+)
        # Charge neutrality: n_e- - n_e+ = n_p ~ eta_b * n_gamma
        # n_gamma = 2 zeta(3)/pi^2 * T^3, in comoving: T -> Tg
        from scipy.special import zeta as _zeta
        n_gamma = 2.0 * _zeta(3) / np.pi**2 * Tg**3  # MeV^3
        n_e_asym = PRyMini.eta0b * n_gamma  # MeV^3
        V_CC_MeV = np.sqrt(2.0) * PRyMini.GF * n_e_asym  # MeV
        V_CC_eV = V_CC_MeV * 1.0e6  # eV (scalar, same for all modes)

        # V_nunu self-interaction potential (full N×N matrix, N = n_flavor).
        N = self.n_flavor
        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2
        V_nunu_pref = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3) * 1.0e6  # eV/MeV³
        rho_nu_mat = self._to_mat(rho_all[0])     # (Ny, N, N)
        rho_nubar_mat = self._to_mat(rho_all[1])  # (Ny, N, N)
        diff_mat = rho_nu_mat[:Ny_coll] - rho_nubar_mat[:Ny_coll]
        V_nunu_eV = V_nunu_pref * np.einsum('i,ijk->jk', y2w, diff_mat)  # (N, N) eV
        V_nunu_eV = V_nunu_eV * getattr(PRyMini, "qke_v_nunu_scale", 1.0)
        if getattr(PRyMini, "qke_v_nunu_active_only", False) and self.n_flavor == 4:
            # Project onto active 3x3 block: sterile has no NC charge.
            V_nunu_eV[3, :] = 0.0
            V_nunu_eV[:, 3] = 0.0

        # Active-flavor trace of n_ξ (zero identically in 3-flavor because
        # ρ = ρ̄ by symmetry there, nonzero under Stage C asymmetry). The
        # trace shifts active H_αα uniformly, which is invisible to
        # active↔active differences but changes H_αα − H_ss and drives
        # the Shi-Fuller MSW resonance.
        if N >= 3:
            trace_nxi_eV = V_nunu_eV[0, 0] + V_nunu_eV[1, 1] + V_nunu_eV[2, 2]
        else:
            trace_nxi_eV = 0.0

        # Build H for each sector. SF-complete matter potential:
        #   * V_CC (baryon charged-current) flips sign for ν̄.
        #   * V_nunu (matrix from ∫y²(ρ−ρ̄)dy) flips sign for ν̄ because
        #     ν̄ sees (ρ̄−ρ) = −(ρ−ρ̄).
        #   * V_thermal (symmetric Notzold-Raffelt thermal term) is NOT
        #     sign-flipped: it comes from the charge-symmetric electron
        #     plasma, not from the asymmetry.
        #   * For n_flavor=4, the active-trace term is added on active
        #     diagonals only, with the same ν/ν̄ sign flip.
        H_list = [None, None]
        for s in range(2):
            V_sign = 1.0 if s == 0 else -1.0
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, N, N), dtype=complex)
            for k in range(N):
                for l in range(N):
                    H[:, k, l] = Omega[k, l] * inv_E + V_sign * V_nunu_eV[k, l]
            H[:, 0, 0] += V_thermal_eV + V_sign * V_CC_eV
            # NC thermal piece on all 3 active diagonals. In 3-flavor mode this
            # is V_NC · I_3 on the active block (global phase, no observable).
            # In 4-flavor mode it creates the active-sterile gap.
            if self.n_flavor >= 3:
                H[:, 0, 0] += V_NC_eV
                H[:, 1, 1] += V_NC_eV
                H[:, 2, 2] += V_NC_eV
            if self.n_flavor == 4:
                for alpha in range(3):
                    H[:, alpha, alpha] += V_sign * trace_nxi_eV
            H_list[s] = H

        # ================================================================
        # 3. Damping rates (C_D parameterization, de Salas & Pastor 2016)
        # ================================================================
        # Total interaction rate Γ_α = C_D_α × GF² × T⁴ × E.
        # The C_D coefficients encode the total scattering rate (not net),
        # which doesn't vanish at equilibrium. Using the net collision
        # integral would underestimate damping near equilibrium.
        GF_eV = PRyMini.GF * 1.0e-12
        T_eV = Tg * 1.0e6
        # Off-diagonal pair damping D_αβ in 1/s via shared helper. Formula
        # selected by PRyMini.qke_damping_formula (legacy symmetric default).
        # NOTE: do NOT name the loop variable `a` here — that would shadow
        # the scale factor `a` used by the oscillation-relaxation block below.
        D_off_si = self._compute_D_pair_matrix(T_eV, E_eV, units="si")
        n_pairs = len(self._all_pair_flavors)
        D_pairs = np.zeros((n_pairs, Ny))
        for p_idx, (_fa, _fb) in enumerate(self._all_pair_flavors):
            D_pairs[p_idx] = D_off_si[_fa, _fb]

        # Finite-mass correction for off-diagonal ν-e scattering
        if PRyMini.massive_electron_flag:
            fnu_emu_scat_val = 1.0
            fnu_mutau_scat_val = 1.0
        else:
            fnu_emu_scat_val = np.sqrt(max(fnu_e_scat_val * fnu_mu_scat_val, 0.0))
            fnu_mutau_scat_val = fnu_mu_scat_val

        # ================================================================
        # 4. Oscillation relaxation (channel-separated Sigl-Raffelt)
        # ================================================================
        # The 3-flavor Hamiltonian H_ee - H_μμ is dominated by the atmospheric
        # mass splitting Δm²₃₁, which suppresses the solar-channel relaxation
        # if used naively. Instead, we separate the solar (Δm²₂₁, θ₁₂) and
        # atmospheric (Δm²₃₁, θ₁₃) channels, each with its own ω_eff:
        #   Γ = sin²(2θ) × ω² × D / (2(ω_eff² + D²))  per channel
        # This is the proven Sigl-Raffelt quasi-static approximation.

        p_MeV = np.maximum(self.y_grid / a, 1.0e-10)
        E_eV_osc = p_MeV * 1.0e6

        # Vacuum oscillation frequencies (eV)
        omega_21 = PRyMini.Dm2_21 / (2.0 * E_eV_osc)
        omega_31 = PRyMini.Dm2_31 / (2.0 * E_eV_osc)

        # Matter potential in eV (already computed as V_thermal_eV)
        V_eV_osc = V_thermal_eV + V_CC_eV

        # Damping rate in eV (nue coefficient for oscillation relaxation)
        D_eV_osc = self.C_D[0] * GF_eV**2 * T_eV**4 * E_eV_osc

        # Solar channel: Δm²₂₁, θ₁₂
        sin2_2theta12 = np.sin(2.0 * PRyMini.theta_12)**2
        cos_2theta12 = np.cos(2.0 * PRyMini.theta_12)
        omega_eff_21 = omega_21 * cos_2theta12 - V_eV_osc
        Gamma_21_eV = sin2_2theta12 * omega_21**2 * D_eV_osc / (
            2.0 * (omega_eff_21**2 + D_eV_osc**2))

        # Atmospheric channel: Δm²₃₁, θ₁₃
        sin2_2theta13 = np.sin(2.0 * PRyMini.theta_13)**2
        cos_2theta13 = np.cos(2.0 * PRyMini.theta_13)
        omega_eff_31 = omega_31 * cos_2theta13 - V_eV_osc
        Gamma_31_eV = sin2_2theta13 * omega_31**2 * D_eV_osc / (
            2.0 * (omega_eff_31**2 + D_eV_osc**2))

        # Total relaxation rate (1/s)
        Gamma_osc = (Gamma_21_eV + Gamma_31_eV) * self._eV_to_secm1

        # Apply relaxation using 3-species pooling (matching diagonal solver).
        # Oscillation connects ν_e↔ν_μ within each CP sector, but the collision
        # integral treats all mu/tau as a single effective species (numu_equiv).
        # Using combined conservation ensures the effective numu seen by the next
        # collision step is correctly updated.
        f_nue = rho_all[0, 0]       # nue (neutrino)
        f_nuebar = rho_all[1, 0]    # nue (antineutrino)
        f_mu = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                       + rho_all[1, 1] + rho_all[1, 2])

        # Equilibrium targets for particle and antiparticle sectors
        f_eq_p = (f_nue + 2.0 * f_mu) / 3.0
        f_eq_a = (f_nuebar + 2.0 * f_mu) / 3.0

        # Exact exponential decay (unconditionally stable)
        decay = np.exp(-Gamma_osc * dt)

        # Compute changes
        df_nue = (f_eq_p - f_nue) * (1.0 - decay)
        df_nuebar = (f_eq_a - f_nuebar) * (1.0 - decay)
        df_numu = -(df_nue + df_nuebar) / 2.0  # combined conservation

        # Apply to density matrix
        rho_all[0, 0] += df_nue       # nue (nu)
        rho_all[1, 0] += df_nuebar    # nue (nubar)
        # Spread numu change equally across all 4 mu/tau sectors
        rho_all[0, 1] += df_numu
        rho_all[0, 2] += df_numu
        rho_all[1, 1] += df_numu
        rho_all[1, 2] += df_numu

        # ================================================================
        # 5. Off-diagonal evolution (combined oscillation + damping)
        # ================================================================
        # Off-diagonal density matrix elements track flavor coherences.
        # Combined osc+damping: dρ_αβ/dt = -(D + isω)ρ_αβ + source
        # In 3-flavor mode we track 3 pairs (e-μ, e-τ, μ-τ). In 4-flavor
        # mode we additionally track 3 active-sterile pairs (e-s, μ-s, τ-s)
        # which have zero collision gain (no SM vertex connects active and
        # sterile neutrinos) but non-zero damping D_αs = ½Γ_α.
        pair_flavors = self._all_pair_flavors
        pair_write = self._all_pair_write
        n_pairs = len(pair_flavors)
        n_active_pairs = 3  # e-μ, e-τ, μ-τ handled by _offdiag_collision_gain
        osc_signs = [+1.0, -1.0]
        dt_nat = dt * self._eV_to_secm1

        for sector in range(2):
            s = osc_signs[sector]
            H = H_list[sector]

            # Off-diagonal collision gain. _offdiag_collision_gain returns
            # shape (6, Ny) covering the 3 ACTIVE pairs only. For 4-flavor
            # we pad with zeros for the 3 active-sterile pairs.
            rho_offdiag = rho_all[sector, self._active_offdiag_idx, :]  # (6, Ny)
            B_idx = 1 if sector == 0 else 0
            if PRyMini.massive_electron_flag:
                gain_active = _offdiag_collision_gain_massive(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll,
                    PRyMini.me)
            else:
                gain_active = _offdiag_collision_gain(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    fnu_emu_scat_val, fnu_mutau_scat_val,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            # Pad with zeros for active-sterile pairs in 4-flavor mode.
            if n_pairs > n_active_pairs:
                gain = np.zeros((2 * n_pairs, Ny))
                gain[:2 * n_active_pairs] = gain_active
                # Remaining (active-sterile) entries stay at zero.
            else:
                gain = gain_active

            for p_idx, (alpha, beta) in enumerate(pair_flavors):
                re_idx = pair_write[p_idx][0]
                im_idx = pair_write[p_idx][1]
                rho_ab = rho_all[sector, re_idx] + 1j * rho_all[sector, im_idx]

                # Hamiltonian elements (eV)
                H_ab_eV = H[:, alpha, beta]
                omega_eV = (H[:, alpha, alpha] - H[:, beta, beta]).real
                D_nat = D_pairs[p_idx] / self._eV_to_secm1

                # Complex stiffness
                z_c = (D_nat + 1j * s * omega_eV) * dt_nat

                # Source: oscillation drive + collision gain
                rho_aa = rho_all[sector, alpha]
                rho_bb = rho_all[sector, beta]
                S_osc_eV = -1j * s * H_ab_eV * (rho_bb - rho_aa)
                S_gain_eV = (gain[2*p_idx] + 1j*gain[2*p_idx+1]) / self._eV_to_secm1
                S_total_eV = S_osc_eV + S_gain_eV

                # Complex exponential Euler
                exp_neg_z = np.exp(-z_c)
                _small = np.abs(z_c) < 1.0e-4
                phi1_c = np.where(_small,
                                  1.0 - 0.5*z_c + z_c**2/6.0,
                                  (1.0 - exp_neg_z) / np.where(_small, 1.0, z_c))

                rho_ab_new = exp_neg_z * rho_ab + phi1_c * dt_nat * S_total_eV

                # Clamp off-diagonal magnitude
                ab_mag = np.abs(rho_ab_new)
                max_mag = np.minimum(0.5, np.sqrt(np.maximum(rho_aa*rho_bb, 0.0)) + 1e-10)
                scale = np.where(ab_mag > max_mag,
                                 max_mag / np.maximum(ab_mag, 1e-30), 1.0)
                rho_ab_new *= scale

                rho_all[sector, re_idx] = rho_ab_new.real
                rho_all[sector, im_idx] = rho_ab_new.imag

        # ================================================================
        # Dodelson-Widrow: active ↔ sterile diagonal transfer, quasi-static
        # Sigl-Raffelt limit.
        #
        # The off-diagonal ODE dρ_αs/dt = -(D + isω)ρ_αs - is H_αs (ρ_ss − ρ_αα)
        # reaches its steady state on timescale 1/D, which is typically much
        # shorter than the integration timestep dt. In that regime
        #     ρ_αs^{eq} = is H_αs (ρ_αα − ρ_ss) / (D + isω)
        # and the diagonal rate that comes from 2 Im(H_αβ ρ_βα) reduces to
        #     dρ_αα/dt = -dρ_ss/dt = -Γ_DW (ρ_αα − ρ_ss)
        # with the Dodelson-Widrow rate
        #     Γ_DW = 2 |H_αs|² D / (D² + ω²)                     [eV units]
        # applied independently per momentum mode. This gives smooth
        # exponential-relaxation dynamics that are stable even when the
        # mass-splitting frequency ω_es exceeds 1/dt (as for eV-scale m_4).
        #
        # Active-active pairs are already handled by the Sigl-Raffelt
        # block above, so we restrict this update to active-sterile.
        # ================================================================
        if self.n_flavor == 4:
            for sector in range(2):
                H = H_list[sector]
                for alpha in range(3):
                    pair_idx = 3 + alpha  # active-sterile pair
                    H_as = H[:, alpha, 3]                                  # complex (Ny,), eV
                    abs_H_as_sq = H_as.real**2 + H_as.imag**2              # eV²
                    omega = (H[:, alpha, alpha] - H[:, 3, 3]).real         # eV
                    D_nat = D_pairs[pair_idx] / self._eV_to_secm1          # eV
                    denom = D_nat**2 + omega**2
                    denom_safe = np.where(denom > 0.0, denom, 1.0)
                    gamma_DW = 2.0 * abs_H_as_sq * D_nat / denom_safe      # eV
                    # Exponential update: Δρ = (ρ_αα − ρ_ss)·(1 − exp(−Γ·dt))
                    # keeps the transfer bounded when Γ·dt > 1.
                    x = gamma_DW * dt_nat
                    frac = np.where(x < 1.0e-4, x - 0.5 * x * x, 1.0 - np.exp(-x))
                    delta = frac * (rho_all[sector, alpha] - rho_all[sector, 3])
                    rho_all[sector, alpha] -= delta
                    rho_all[sector, 3]     += delta

        # Clip diagonal elements to valid range [0, 1]
        f_min = 1.0e-30
        f_max = 1.0 - f_min
        for sector in range(2):
            for d in range(self.n_flavor):
                rho_all[sector, d] = np.clip(rho_all[sector, d], f_min, f_max)

    def _apply_unitary(self, rho_all, H_list, dt_nat):
        """In-place unitary evolution of rho_all per sector per mode.

        PRyM's convention (baked into evolve_step's osc_signs = [+1, -1])
        is that rho_all[0] stores ρ and rho_all[1] stores ρ̄* (complex
        conjugate of the antineutrino density matrix, which equals ρ̄ᵀ
        for Hermitian ρ̄). The physical QKE is

            dρ /dt = -i [H_ν , ρ ]    ⇒  ρ  → U_ν  ρ  U_ν†
            dρ̄/dt = -i [H_ν̄, ρ̄]    ⇒  ρ̄ → U_ν̄ ρ̄ U_ν̄†

        and taking * of the antineutrino update gives the stored form:

            (ρ̄)*_new = (U_ν̄ ρ̄ U_ν̄†)* = U_ν̄* · (ρ̄)* · U_ν̄ᵀ

        so the unitary conjugation for sector 1 uses U* on the left and
        Uᵀ on the right, NOT U on the left and U† on the right.

        Preserves Hermiticity of the stored matrix via symmetrization.
        """
        for sector in range(2):
            H = H_list[sector]                               # (Ny, N, N)
            lam, V = np.linalg.eigh(H)                       # (Ny, N), (Ny, N, N)
            phase = np.exp(-1j * lam * dt_nat)               # (Ny, N)
            # Physical U_ν (or U_ν̄) = V · diag(exp(-iλ dt)) · V†
            U = np.einsum('iak,ik,ibk->iab', V, phase, V.conj())

            rho_mat = self._to_mat(rho_all[sector])          # (Ny, N, N)
            if sector == 0:
                # ρ_new = U · ρ · U†
                rho_new = U @ rho_mat @ U.conj().swapaxes(-1, -2)
            else:
                # Stored is ρ̄*, so (ρ̄)*_new = U* · (ρ̄)* · Uᵀ
                rho_new = U.conj() @ rho_mat @ U.swapaxes(-1, -2)
            # Numerical cleanup: force exact Hermiticity
            rho_new = 0.5 * (rho_new + rho_new.conj().swapaxes(-1, -2))
            rho_all[sector] = self._to_vec(rho_new)

    def evolve_step_ode(self, rho_all, dt, phi1_dt, a, Tg):
        """Strang-split QKE evolution with exact unitary conjugation.

        Stage D variant of evolve_step, gated by PRyMini.qke_full_ode_flag.
        Replaces the two quasi-static diagonal-transfer approximations used
        by evolve_step (Sigl-Raffelt active-active relaxation and the
        Stage B Dodelson-Widrow active-sterile transfer) with a single
        exact unitary conjugation ρ → U ρ U†,  U = exp(-iH dt),  applied
        Strang-symmetrically around a collision step:

            ρ ← U^(½) ρ (U^(½))†                          (first half-step)
            ρ ← collision step (diag + off-diag damping + gain, NO H)
            ρ ← U^(½) ρ (U^(½))†                          (second half-step)

        Because the half-step unitaries handle the full H commutator
        exactly (mixing diagonals and off-diagonals naturally per mode),
        the collision step must DROP the H-oscillation piece to avoid
        double counting:
        * Diagonals: Euler with phi_1 damping (identical to evolve_step).
        * Off-diagonals: exp-Euler with pure damping D_αβ and the
          collision gain source; the ω_αβ oscillation term is removed.

        This is physically cleaner than evolve_step's two quasi-static
        approximations: no need to split H into solar/atmospheric channels
        (the Sigl-Raffelt hack), and no need for a separate DW transfer in
        4-flavor mode. The unitary handles all H-driven mixing, including
        active-sterile, automatically.

        Expected parity with evolve_step: SM observables (mode 5) within
        the 10⁻³ tolerance, because the current quasi-static blocks are
        already accurate in their respective fast-oscillation limits.

        Parameters
        ----------
        rho_all : ndarray, shape (2, n_components, Ny)
            Modified in-place.
        dt : float
            Physical time step in seconds.
        phi1_dt : float or (3, Ny) array
            phi_1(z) * dt for exponential-Euler regularization of diagonals.
        a : float
            Scale factor at midpoint.
        Tg : float
            Photon temperature in MeV at midpoint.
        """
        Ny = self.Ny
        N = self.n_flavor

        # ================================================================
        # 1. Diagonal collision integrals (identical to evolve_step)
        # ================================================================
        f_all = np.zeros((3, Ny))
        f_all[0] = rho_all[0, 0]
        f_all[1] = rho_all[1, 0]
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                           + rho_all[1, 1] + rho_all[1, 2])

        GF2_pref = self._boltz.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        I_nu_nu = _collision_integral_nu_nu(
            f_all, self.y_grid, self._boltz.quad_w, a, GF2_pref, tail_params,
            self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

        if PRyMini.massive_electron_flag:
            I_nu_e = _collision_integral_nu_e_massive(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2,
                self._boltz.gmuL2, self._boltz.gmuR2,
                PRyMini.me, self._boltz.Ny_coll)
        else:
            fnu_e_scat_val = float(self._boltz._fnu_e_scat(Tg))
            fnu_e_ann_val = float(self._boltz._fnu_e_ann(Tg))
            fnu_mu_scat_val = float(self._boltz._fnu_mu_scat(Tg))
            fnu_mu_ann_val = float(self._boltz._fnu_mu_ann(Tg))
            I_nu_e = _collision_integral_nu_e(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2, self._boltz.geLgeR,
                self._boltz.gmuL2, self._boltz.gmuR2, self._boltz.gmuLgmuR,
                PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                self._boltz.D_k0, self._boltz.D_k1, self._boltz.D_k2,
                self._boltz.Ny_coll, 1.0, 1.0, 1.0, 1.0)

        I_total = I_nu_nu + I_nu_e
        C_NP = self._boltz.C_NP_funcs
        if 'nue' in C_NP:
            I_total[0] += C_NP['nue'](self.y_grid, a, Tg, f_all)
        if 'nuebar' in C_NP:
            I_total[1] += C_NP['nuebar'](self.y_grid, a, Tg, f_all)
        if 'numu' in C_NP:
            I_total[2] += C_NP['numu'](self.y_grid, a, Tg, f_all)

        # ================================================================
        # 2. Build Hamiltonians (identical to evolve_step)
        # ================================================================
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)

        # Notzold-Raffelt thermal self-energy: CC (1/m_W^2, ν_e only) and NC
        # (1/m_Z^2, all active). See evolve_step for extended rationale.
        rho_e_th = 7.0 * np.pi**2 / 60.0 * Tg**4
        V_pref = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e_th / 3.0
        V_thermal_eV = V_pref / self.mW2 * E_eV
        V_NC_eV = V_pref / (PRyMini.mZ**2) * E_eV
        # Stage E.2 sprint 5 diagnostic scale knobs.
        V_thermal_eV = V_thermal_eV * getattr(PRyMini, "qke_v_thermal_scale", 1.0)
        V_NC_eV = V_NC_eV * getattr(PRyMini, "qke_v_nc_scale", 1.0)

        from scipy.special import zeta as _zeta
        n_gamma = 2.0 * _zeta(3) / np.pi**2 * Tg**3
        n_e_asym = PRyMini.eta0b * n_gamma
        V_CC_MeV = np.sqrt(2.0) * PRyMini.GF * n_e_asym
        V_CC_eV = V_CC_MeV * 1.0e6

        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2
        V_nunu_pref = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3) * 1.0e6
        rho_nu_mat = self._to_mat(rho_all[0])
        rho_nubar_mat = self._to_mat(rho_all[1])
        diff_mat = rho_nu_mat[:Ny_coll] - rho_nubar_mat[:Ny_coll]
        V_nunu_eV = V_nunu_pref * np.einsum('i,ijk->jk', y2w, diff_mat)
        V_nunu_eV = V_nunu_eV * getattr(PRyMini, "qke_v_nunu_scale", 1.0)
        if getattr(PRyMini, "qke_v_nunu_active_only", False) and self.n_flavor == 4:
            # Project onto active 3x3 block: sterile has no NC charge.
            V_nunu_eV[3, :] = 0.0
            V_nunu_eV[:, 3] = 0.0

        if N >= 3:
            trace_nxi_eV = V_nunu_eV[0, 0] + V_nunu_eV[1, 1] + V_nunu_eV[2, 2]
        else:
            trace_nxi_eV = 0.0

        inv_E = 1.0 / E_eV
        H_list = [None, None]
        for s in range(2):
            V_sign = 1.0 if s == 0 else -1.0
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, N, N), dtype=complex)
            for k in range(N):
                for l in range(N):
                    H[:, k, l] = Omega[k, l] * inv_E + V_sign * V_nunu_eV[k, l]
            H[:, 0, 0] += V_thermal_eV + V_sign * V_CC_eV
            # NC thermal on all active diagonals (observable-neutral in 3-flavor,
            # lifts active-sterile gap in 4-flavor). See evolve_step for detail.
            if self.n_flavor >= 3:
                H[:, 0, 0] += V_NC_eV
                H[:, 1, 1] += V_NC_eV
                H[:, 2, 2] += V_NC_eV
            if self.n_flavor == 4:
                for alpha in range(3):
                    H[:, alpha, alpha] += V_sign * trace_nxi_eV
            H_list[s] = H

        # ================================================================
        # 3. Damping rates (identical to evolve_step, minus the D_eV_osc
        #    Sigl-Raffelt prep which is no longer needed). Off-diagonal
        #    pair damping via shared helper; PRyMini.qke_damping_formula
        #    selects symmetric (legacy) / mirizzi / gariazzo.
        # ================================================================
        GF_eV = PRyMini.GF * 1.0e-12
        T_eV = Tg * 1.0e6
        D_off_si = self._compute_D_pair_matrix(T_eV, E_eV, units="si")
        n_pairs = len(self._all_pair_flavors)
        D_pairs = np.zeros((n_pairs, Ny))
        for p_idx, (_fa, _fb) in enumerate(self._all_pair_flavors):
            D_pairs[p_idx] = D_off_si[_fa, _fb]

        if PRyMini.massive_electron_flag:
            fnu_emu_scat_val = 1.0
            fnu_mutau_scat_val = 1.0
        else:
            fnu_emu_scat_val = np.sqrt(max(fnu_e_scat_val * fnu_mu_scat_val, 0.0))
            fnu_mutau_scat_val = fnu_mu_scat_val

        # ================================================================
        # 4. First half-step unitary: ρ → U^(½) ρ (U^(½))†
        # ================================================================
        half_dt_nat = 0.5 * dt * self._eV_to_secm1
        self._apply_unitary(rho_all, H_list, half_dt_nat)

        # ================================================================
        # 5. Collision step (no H)
        # ================================================================
        # 5a. Diagonals — exponential Euler identical to evolve_step.
        if np.ndim(phi1_dt) == 0:
            _p0 = phi1_dt
            _p1 = phi1_dt
            _p2 = phi1_dt
        else:
            _p0 = phi1_dt[0]
            _p1 = phi1_dt[1]
            _p2 = phi1_dt[2]
        rho_all[0, 0] += _p0 * I_total[0]
        rho_all[1, 0] += _p1 * I_total[1]
        rho_all[0, 1] += _p2 * I_total[2]
        rho_all[0, 2] += _p2 * I_total[2]
        rho_all[1, 1] += _p2 * I_total[2]
        rho_all[1, 2] += _p2 * I_total[2]

        # 5b. Off-diagonals — exp-Euler with DAMPING ONLY (no H oscillation).
        # Full ODE:  dρ_αβ/dt = -D_αβ ρ_αβ + S_gain
        # Closed-form exp-Euler:
        #   ρ_αβ(t+dt) = exp(-D dt) ρ_αβ(t) + phi_1(-D dt) dt S_gain
        # with S_gain from _offdiag_collision_gain for the 3 active-active
        # pairs; active-sterile pairs see zero collision gain (no SM vertex).
        dt_nat = dt * self._eV_to_secm1

        # Refresh f_all after unitary + diagonal Euler so the collision-gain
        # integrals see the up-to-date occupations.
        f_all[0] = rho_all[0, 0]
        f_all[1] = rho_all[1, 0]
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                           + rho_all[1, 1] + rho_all[1, 2])

        for sector in range(2):
            rho_offdiag = rho_all[sector, self._active_offdiag_idx, :]  # (6, Ny)
            B_idx = 1 if sector == 0 else 0
            if PRyMini.massive_electron_flag:
                gain_active = _offdiag_collision_gain_massive(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll,
                    PRyMini.me)
            else:
                gain_active = _offdiag_collision_gain(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    fnu_emu_scat_val, fnu_mutau_scat_val,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            if n_pairs > 3:
                gain = np.zeros((2 * n_pairs, Ny))
                gain[:6] = gain_active
            else:
                gain = gain_active

            for p_idx, (alpha, beta) in enumerate(self._all_pair_flavors):
                re_idx, im_idx = self._all_pair_write[p_idx]
                rho_ab = rho_all[sector, re_idx] + 1j * rho_all[sector, im_idx]

                D_nat = D_pairs[p_idx] / self._eV_to_secm1   # eV
                z = D_nat * dt_nat                           # real, ≥ 0
                exp_mz = np.exp(-z)
                phi1 = np.where(z < 1.0e-4,
                                1.0 - 0.5 * z + z * z / 6.0,
                                (1.0 - exp_mz) / np.where(z > 0.0, z, 1.0))
                S_gain_eV = (gain[2 * p_idx] + 1j * gain[2 * p_idx + 1]) / self._eV_to_secm1
                rho_ab_new = exp_mz * rho_ab + phi1 * dt_nat * S_gain_eV

                # Clamp off-diagonal magnitude (same as evolve_step)
                rho_aa = rho_all[sector, alpha]
                rho_bb = rho_all[sector, beta]
                ab_mag = np.abs(rho_ab_new)
                max_mag = np.minimum(0.5, np.sqrt(np.maximum(rho_aa * rho_bb, 0.0)) + 1e-10)
                scale = np.where(ab_mag > max_mag,
                                 max_mag / np.maximum(ab_mag, 1e-30), 1.0)
                rho_ab_new *= scale

                rho_all[sector, re_idx] = rho_ab_new.real
                rho_all[sector, im_idx] = rho_ab_new.imag

        # ================================================================
        # 6. Second half-step unitary
        # ================================================================
        self._apply_unitary(rho_all, H_list, half_dt_nat)

        # ================================================================
        # 7. Clip diagonals
        # ================================================================
        f_min = 1.0e-30
        f_max = 1.0 - f_min
        for sector in range(2):
            for d in range(self.n_flavor):
                rho_all[sector, d] = np.clip(rho_all[sector, d], f_min, f_max)

    # --------------------------------------------------------------------
    # Stage D.7: full ETDRK2 with off-diagonal damping in L (qke_ode_etdrk2_flag)
    # --------------------------------------------------------------------
    # Restores O(dt^2) convergence at the Shi-Fuller MSW resonance by absorbing
    # the real flavor-basis off-diagonal pair damping D_ab = 1/2(Gamma_a + Gamma_b)
    # into the linear part L = -i[H, .] - D_off o . , then computing e^{L dt},
    # dt*phi_1(L dt), dt*phi_2(L dt) via a per-mode matrix exponential on the
    # 9x9 (or 16x16) vectorised Liouvillian superoperator (Al-Mohy & Higham
    # 2011 augmented-matrix trick). This bypasses the H-eigenbasis rotation that
    # broke nu-nubar symmetry in the D.6 Cox-Matthews corrector.
    #
    # Hybrid scope: damping-in-L is applied ONLY to off-diagonal entries. Diagonal
    # occupations continue on the existing flavor-basis exp-Euler step (phi1_dt
    # times I_total) identical to evolve_step_ode. Rationale: off-diagonal
    # coherence is what drives O(dt^2) at MSW, and PRyMordial's I_total is the
    # full collision kernel, not a schematic -Gamma_alpha relaxation, so a
    # I_total_alpha + Gamma_alpha*rho_alpha_alpha subtraction would be a large
    # cancellation during the SF sweep when rho_ee departs far from FD.
    #
    # These helpers are used ONLY by evolve_step_ode_etdrk2 below. The existing
    # evolve_step and evolve_step_ode are untouched; default runs (qke_ode_etdrk2_flag=False)
    # remain bit-identical.

    def _build_H_list(self, rho_all, a, Tg):
        """Assemble the per-sector Hamiltonian H in eV, shape (Ny, N, N) each.

        Block is duplicated from evolve_step_ode lines 3984-4024 so that the
        ODE driver can remain byte-for-byte unchanged. If you modify one,
        mirror the change in the other (and ideally fold them back together
        once Stage D.6 is validated).
        """
        from scipy.special import zeta as _zeta

        Ny = self.Ny
        N = self.n_flavor
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)

        # Notzold-Raffelt thermal self-energy: CC (1/m_W^2, ν_e only) and NC
        # (1/m_Z^2, all active). See evolve_step for extended rationale.
        rho_e_th = 7.0 * np.pi**2 / 60.0 * Tg**4
        V_pref = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e_th / 3.0
        V_thermal_eV = V_pref / self.mW2 * E_eV
        V_NC_eV = V_pref / (PRyMini.mZ**2) * E_eV
        # Stage E.2 sprint 5 diagnostic scale knobs.
        V_thermal_eV = V_thermal_eV * getattr(PRyMini, "qke_v_thermal_scale", 1.0)
        V_NC_eV = V_NC_eV * getattr(PRyMini, "qke_v_nc_scale", 1.0)

        n_gamma = 2.0 * _zeta(3) / np.pi**2 * Tg**3
        n_e_asym = PRyMini.eta0b * n_gamma
        V_CC_MeV = np.sqrt(2.0) * PRyMini.GF * n_e_asym
        V_CC_eV = V_CC_MeV * 1.0e6

        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2
        V_nunu_pref = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3) * 1.0e6
        rho_nu_mat = self._to_mat(rho_all[0])
        rho_nubar_mat = self._to_mat(rho_all[1])
        diff_mat = rho_nu_mat[:Ny_coll] - rho_nubar_mat[:Ny_coll]
        V_nunu_eV = V_nunu_pref * np.einsum('i,ijk->jk', y2w, diff_mat)
        V_nunu_eV = V_nunu_eV * getattr(PRyMini, "qke_v_nunu_scale", 1.0)
        if getattr(PRyMini, "qke_v_nunu_active_only", False) and self.n_flavor == 4:
            # Project onto active 3x3 block: sterile has no NC charge.
            V_nunu_eV[3, :] = 0.0
            V_nunu_eV[:, 3] = 0.0

        if N >= 3:
            trace_nxi_eV = V_nunu_eV[0, 0] + V_nunu_eV[1, 1] + V_nunu_eV[2, 2]
        else:
            trace_nxi_eV = 0.0

        inv_E = 1.0 / E_eV
        H_list = [None, None]
        for s in range(2):
            V_sign = 1.0 if s == 0 else -1.0
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, N, N), dtype=complex)
            for k in range(N):
                for l in range(N):
                    H[:, k, l] = Omega[k, l] * inv_E + V_sign * V_nunu_eV[k, l]
            H[:, 0, 0] += V_thermal_eV + V_sign * V_CC_eV
            # NC thermal on all active diagonals. Observable-neutral in 3-flavor
            # (V_NC · I_3 is a global phase on the active block); in 4-flavor
            # this lifts the active-sterile energy gap -- the canonical V_z that
            # controls in-medium mixing suppression at small θ_{14,24,34}.
            if self.n_flavor >= 3:
                H[:, 0, 0] += V_NC_eV
                H[:, 1, 1] += V_NC_eV
                H[:, 2, 2] += V_NC_eV
            if self.n_flavor == 4:
                for alpha in range(3):
                    H[:, alpha, alpha] += V_sign * trace_nxi_eV
            H_list[s] = H
        return H_list

    def _assemble_collision_N(self, rho_all, a, Tg):
        """Return (dro/dt)_collision as per-sector Hermitian (Ny, N, N) matrices.

        Output convention matches the natural-units choice _apply_unitary uses
        for H: matrix entries are in eV = (SI rate in s^-1) / _eV_to_secm1.
        Then dt_nat [1/eV] * N [eV] is dimensionless, ready to add to rho.

        Components
        ----------
        * Diagonals from I_total[0..2] = (I_nu_e + I_nu_nu)_[nue, nuebar, numu_eff]
          plus any C_NP new-physics additions. Sector 0 row 0 gets I_total[0];
          sector 1 row 0 gets I_total[1]; all four mu/tau diagonals get
          I_total[2] (the mu-tau symmetric channel).
          Sterile (row 3, 4-flavor) is zero: no SM vertex.
        * Off-diagonals for the 3 active-active pairs: gain from
          _offdiag_collision_gain[_massive] minus damping D_ab * rho_ab.
          Sector 0 uses rho_nubar as the "B" occupation (B_idx=1); sector 1
          uses rho_nu (B_idx=0), matching evolve_step_ode.
        * Off-diagonals for the 3 active-sterile pairs (4-flavor only):
          damping only, zero collision gain. (Sterile has no SM vertex but
          active decoherence still suppresses coherence.)

        Notes
        -----
        - Does not mutate rho_all.
        - Duplicates a fair amount of kernel plumbing from evolve_step_ode
          (lines 3941-4046 and 4088-4106). Factoring both call sites through
          this helper is a follow-up janitorial task.
        """
        Ny = self.Ny
        N = self.n_flavor

        # 1. Diagonal collision integrals
        f_all = np.zeros((3, Ny))
        f_all[0] = rho_all[0, 0]
        f_all[1] = rho_all[1, 0]
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                           + rho_all[1, 1] + rho_all[1, 2])

        GF2_pref = self._boltz.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        I_nu_nu = _collision_integral_nu_nu(
            f_all, self.y_grid, self._boltz.quad_w, a, GF2_pref, tail_params,
            self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

        if PRyMini.massive_electron_flag:
            I_nu_e = _collision_integral_nu_e_massive(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2,
                self._boltz.gmuL2, self._boltz.gmuR2,
                PRyMini.me, self._boltz.Ny_coll)
            fnu_emu_scat_val = 1.0
            fnu_mutau_scat_val = 1.0
        else:
            fnu_e_scat_val = float(self._boltz._fnu_e_scat(Tg))
            fnu_e_ann_val = float(self._boltz._fnu_e_ann(Tg))
            fnu_mu_scat_val = float(self._boltz._fnu_mu_scat(Tg))
            fnu_mu_ann_val = float(self._boltz._fnu_mu_ann(Tg))
            I_nu_e = _collision_integral_nu_e(
                f_all, self.y_grid, self._boltz.quad_w, a, Tg, GF2_pref,
                self._boltz.geL2, self._boltz.geR2, self._boltz.geLgeR,
                self._boltz.gmuL2, self._boltz.gmuR2, self._boltz.gmuLgmuR,
                PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                self._boltz.D_k0, self._boltz.D_k1, self._boltz.D_k2,
                self._boltz.Ny_coll, 1.0, 1.0, 1.0, 1.0)
            fnu_emu_scat_val = np.sqrt(max(fnu_e_scat_val * fnu_mu_scat_val, 0.0))
            fnu_mutau_scat_val = fnu_mu_scat_val

        I_total = I_nu_nu + I_nu_e
        C_NP = self._boltz.C_NP_funcs
        if 'nue' in C_NP:
            I_total[0] += C_NP['nue'](self.y_grid, a, Tg, f_all)
        if 'nuebar' in C_NP:
            I_total[1] += C_NP['nuebar'](self.y_grid, a, Tg, f_all)
        if 'numu' in C_NP:
            I_total[2] += C_NP['numu'](self.y_grid, a, Tg, f_all)

        # 2. Damping rates (1/s) via shared helper; PRyMini.qke_damping_formula
        #    selects symmetric (legacy) / mirizzi / gariazzo.
        GF_eV = PRyMini.GF * 1.0e-12
        T_eV = Tg * 1.0e6
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)
        D_off_si = self._compute_D_pair_matrix(T_eV, E_eV, units="si")
        n_pairs = len(self._all_pair_flavors)
        D_pairs = np.zeros((n_pairs, Ny))
        for p_idx, (_fa, _fb) in enumerate(self._all_pair_flavors):
            D_pairs[p_idx] = D_off_si[_fa, _fb]

        # 3. Per-sector assembly
        N_list = [np.zeros((Ny, N, N), dtype=complex) for _ in range(2)]
        inv_rate = 1.0 / self._eV_to_secm1

        for sector in range(2):
            # Diagonals (1/s -> eV via inv_rate)
            diag_idx = 0 if sector == 0 else 1
            N_list[sector][:, 0, 0] = I_total[diag_idx] * inv_rate
            N_list[sector][:, 1, 1] = I_total[2] * inv_rate
            N_list[sector][:, 2, 2] = I_total[2] * inv_rate
            # Sterile (index 3) stays zero.

            # Active-active off-diagonal gain
            rho_offdiag = rho_all[sector, self._active_offdiag_idx, :]  # (6, Ny)
            B_idx = 1 if sector == 0 else 0
            if PRyMini.massive_electron_flag:
                gain_active = _offdiag_collision_gain_massive(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll,
                    PRyMini.me)
            else:
                gain_active = _offdiag_collision_gain(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    fnu_emu_scat_val, fnu_mutau_scat_val,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            # Pack gain + damping into the Hermitian matrix
            for p_idx, (alpha, beta) in enumerate(self._all_pair_flavors):
                re_idx, im_idx = self._all_pair_write[p_idx]
                rho_ab_stored = rho_all[sector, re_idx] + 1j * rho_all[sector, im_idx]

                if p_idx < 3:
                    S_gain_si = (gain_active[2 * p_idx]
                                 + 1j * gain_active[2 * p_idx + 1])
                else:
                    S_gain_si = 0.0  # active-sterile pairs: no SM gain

                rhs_si = -D_pairs[p_idx] * rho_ab_stored + S_gain_si   # 1/s
                rhs_eV = rhs_si * inv_rate                              # eV

                N_list[sector][:, alpha, beta] = rhs_eV
                N_list[sector][:, beta, alpha] = np.conj(rhs_eV)

        return N_list, I_total

    def _build_L_list(self, rho_all, a, Tg):
        """Assemble vectorised superoperator L and gain-only N per sector.

        Returns
        -------
        L_list : list of length 2, each shape (Ny, N*N, N*N), complex, in eV
            Per-sector Liouvillian L = -i[H, .] - D_off o . on the row-major
            vec(rho) space (vec(A)_{alpha*N+beta} = A_{alpha,beta}).
            Sector 0: L = -i (H kron I - I kron H^T) - diag(vec(D_off))
            Sector 1: L = +i (H^T kron I - I kron H) - diag(vec(D_off))
              (stored-rho-bar* convention; the sign flip comes from taking the
               complex conjugate of the physical nu-bar equation.)
            D_off is the off-diagonal pair damping D_{alpha,beta} = (Gamma_a+Gamma_b)/2
            for alpha != beta, zero on the diagonal (hybrid D.7 scope:
            diagonal damping stays in the flavor-basis exp-Euler step).
        N_gain : list of length 2, each shape (Ny, N, N), complex, in eV
            Per-sector collision RHS with the -D_off o rho damping term added
            BACK on off-diagonals (since damping now lives in L). Diagonals are
            left as I_total * inv_rate, consumed by the diagonal exp-Euler step.
        I_total : shape (3, Ny), 1/s
            Raw collision integrals passed through from _assemble_collision_N
            for use by the diagonal exp-Euler step.
        """
        Ny = self.Ny
        N = self.n_flavor

        H_list = self._build_H_list(rho_all, a, Tg)

        # Off-diagonal pair damping in eV via shared helper; mirrors the
        # _assemble_collision_N formula but drops the _eV_to_secm1 factor
        # that converts 1/eV to 1/s. PRyMini.qke_damping_formula selects
        # symmetric (legacy) / mirizzi / gariazzo. Both halves of the D.7.1
        # cancellation (-D·ρ in _assemble_collision_N, +D·ρ add-back here)
        # must use the same formula — the shared helper guarantees this.
        T_eV = Tg * 1.0e6
        E_eV = np.maximum(self.y_grid / a * 1.0e6, 1.0e-4)
        D_off_eV = self._compute_D_pair_matrix(T_eV, E_eV, units="eV")

        # Build L[s, i] in the row-major vec(rho) space.
        I_N = np.eye(N, dtype=complex)
        L_list = [np.zeros((Ny, N*N, N*N), dtype=complex) for _ in range(2)]
        for s in range(2):
            H_s = H_list[s]
            for i in range(Ny):
                H_i = H_s[i]
                D_diag_vec = D_off_eV[:, :, i].reshape(N*N)  # row-major
                if s == 0:
                    comm = np.kron(H_i, I_N) - np.kron(I_N, H_i.T)
                    L_list[s][i] = -1j * comm - np.diag(D_diag_vec)
                else:
                    comm = np.kron(H_i.T, I_N) - np.kron(I_N, H_i)
                    L_list[s][i] = +1j * comm - np.diag(D_diag_vec)

        # Gain-only off-diagonal N: add back D_off * rho_ab_stored to cancel the
        # -D*rho damping piece inside _assemble_collision_N (see line 4367 of the
        # D.6 landing commit). Diagonals untouched.
        N_full, I_total = self._assemble_collision_N(rho_all, a, Tg)
        N_gain = [arr.copy() for arr in N_full]
        for s in range(2):
            rho_mat = self._to_mat(rho_all[s])
            for (alpha, beta) in self._all_pair_flavors:
                add_back = D_off_eV[alpha, beta] * rho_mat[:, alpha, beta]
                N_gain[s][:, alpha, beta] += add_back
                N_gain[s][:, beta, alpha] = np.conj(N_gain[s][:, alpha, beta])

        return L_list, N_gain, I_total

    def _etdrk2_expm_phi(self, L, dt_nat):
        """Compute Phi0 = e^{L dt}, Phi1 = dt*phi_1(L dt), Phi2 = dt*phi_2(L dt)
        per mode via the Al-Mohy & Higham (2011) augmented-matrix exponential.

        The 3N^2-sized augmented block has the property that a single expm call
        yields all three phi functions simultaneously (Thm 2.1):
            M = [[L*dt, I, 0], [0, 0, I], [0, 0, 0]]
            exp(M)[0:N^2, 0:N^2]       = e^{L dt}
            exp(M)[0:N^2, N^2:2N^2]    = dt * phi_1(L dt)
            exp(M)[0:N^2, 2N^2:3N^2]   = dt^2 * phi_2(L dt)
        The ETDRK2 corrector uses dt*phi_2, so divide the third block by dt_nat.

        Parameters
        ----------
        L : (Ny, N^2, N^2) complex, in eV
            Vectorised Liouvillian for one sector.
        dt_nat : float
            Time step in inverse-eV units (dt_seconds * _eV_to_secm1).

        Returns
        -------
        Phi0, Phi1, Phi2 : (Ny, N^2, N^2) complex
            Phi0 dimensionless, Phi1 and Phi2 in [1/eV].
        """
        Ny = L.shape[0]
        Nsq = L.shape[1]
        I_Nsq = np.eye(Nsq, dtype=complex)

        Phi0 = np.zeros_like(L)
        Phi1 = np.zeros_like(L)
        Phi2 = np.zeros_like(L)

        M = np.zeros((3 * Nsq, 3 * Nsq), dtype=complex)
        M[:Nsq, Nsq:2 * Nsq] = I_Nsq
        M[Nsq:2 * Nsq, 2 * Nsq:3 * Nsq] = I_Nsq

        for i in range(Ny):
            M[:Nsq, :Nsq] = L[i] * dt_nat
            # the other blocks of M are constant across modes (see pre-loop init).
            E = expm(M)
            Phi0[i] = E[:Nsq, :Nsq]
            Phi1[i] = E[:Nsq, Nsq:2 * Nsq]
            Phi2[i] = E[:Nsq, 2 * Nsq:3 * Nsq] / dt_nat

        return Phi0, Phi1, Phi2

    def _energy_snapshot(self, rho_all, a, Tg, label):
        """Stage E.2 sprint 8: append per-(sector, active-sterile pair) N/E/C
        integrals for the current rho_all to self._energy_hist.

        Gated by PRyMini.qke_energy_diag_flag. Caller increments
        self._energy_step_idx once per evolve_step_ode_etdrk2 call.

        Metrics per (sector s, pair (alpha, sterile_idx=3)) under the
        uniform midpoint rule (dy constant across y_grid):
          N_as = dy * sum_y y^2 * (rho_aa(y) + rho_ss(y))
          E_as = dy * sum_y y^3 * (rho_aa(y) + rho_ss(y))
          C_as = dy * sum_y y^3 * |rho_as(y)|

        N/E conservation is the diagnostic for Suspect 1 (active-sterile
        off-diagonal damping energy-balance). C is a coherence sidebar.
        Active-sterile pairs only: 4-flavor runs append data; 3-flavor
        runs skip silently.
        """
        if self.n_flavor != 4:
            return
        # Uniform midpoint rule over the full Ny grid (dy constant by
        # construction at PRyM_boltzmann.py:1988-1989). The collision-integral
        # Gauss-Legendre weights self._boltz.quad_w are only Ny_coll long;
        # we want the full comoving distribution here, so use dy directly.
        dy = self.dy
        y = self.y_grid
        y2 = y * y
        y3 = y2 * y
        row = {
            "step": int(self._energy_step_idx),
            "label": str(label),
            "a": float(a),
            "Tg": float(Tg),
        }
        rho_mat0 = self._to_mat(rho_all[0])
        rho_mat1 = self._to_mat(rho_all[1])
        rho_mats = (rho_mat0, rho_mat1)
        for s in (0, 1):
            rho_mat = rho_mats[s]
            for alpha in (0, 1, 2):
                rho_aa = rho_mat[:, alpha, alpha].real
                rho_ss = rho_mat[:, 3, 3].real
                rho_as = rho_mat[:, alpha, 3]
                N_as = float(dy * np.sum(y2 * (rho_aa + rho_ss)))
                E_as = float(dy * np.sum(y3 * (rho_aa + rho_ss)))
                C_as = float(dy * np.sum(y3 * np.abs(rho_as)))
                row[f"N_{s}_{alpha}s"] = N_as
                row[f"E_{s}_{alpha}s"] = E_as
                row[f"C_{s}_{alpha}s"] = C_as
        self._energy_hist.append(row)

    def _msw_snapshot(self, rho_all, a, Tg, label):
        """Stage E.2 sprint 9: append per-y-mode per-sector Hamiltonian and
        diagonal-population snapshot for the active-sterile pair selected by
        PRyMini.qke_msw_diag_pair_idx to self._msw_hist.

        Gated by PRyMini.qke_msw_diag_flag. Caller increments
        self._msw_step_idx once per evolve_step_ode_etdrk2 call.

        Per (sector s, y-mode), records for the pair (alpha, sterile=3):
          H_aa(y), H_ss(y)      diagonals in eV
          Re_H_as(y), Im_H_as(y) off-diagonal in eV
          rho_aa(y), rho_ss(y)  diagonal populations (dimensionless)

        Purpose: localise which y-modes receive anomalous sterile deposition
        around the MSW resonance crossing (|H_aa - H_ss| minimum). Combined
        with the post-processor in validation/diagnostics/diag_msw_passage.py,
        this extracts per-y resonance temperature, Landau-Zener adiabaticity,
        and end-state ρ_ss(y) residual vs. a thermal target.

        Active-sterile pairs only: 4-flavor runs append; 3-flavor runs skip
        silently (no sterile index 3 to reference).
        """
        if self.n_flavor != 4:
            return
        pair_idx = int(getattr(PRyMini, "qke_msw_diag_pair_idx", 4))
        if pair_idx < 0 or pair_idx >= len(self._all_pair_flavors):
            return
        alpha, beta = self._all_pair_flavors[pair_idx]
        if beta != 3:
            # Only active-sterile pairs have a meaningful sterile diagonal.
            return

        # Build the per-sector Hamiltonian at the current rho_all, a, Tg.
        # Non-mutating: _build_H_list only reads rho_all.
        H_list = self._build_H_list(rho_all, a, Tg)
        rho_mat0 = self._to_mat(rho_all[0])
        rho_mat1 = self._to_mat(rho_all[1])
        rho_mats = (rho_mat0, rho_mat1)

        row = {
            "step": int(self._msw_step_idx),
            "label": str(label),
            "a": float(a),
            "Tg": float(Tg),
            "pair_idx": int(pair_idx),
            "alpha": int(alpha),
            "sterile": int(beta),
        }
        for s in (0, 1):
            H = H_list[s]
            rho_mat = rho_mats[s]
            H_aa = H[:, alpha, alpha].real.astype(np.float64)
            H_ss = H[:, beta, beta].real.astype(np.float64)
            H_as = H[:, alpha, beta].astype(np.complex128)
            rho_aa = rho_mat[:, alpha, alpha].real.astype(np.float64)
            rho_ss = rho_mat[:, beta, beta].real.astype(np.float64)
            row[f"H_aa_{s}"] = H_aa
            row[f"H_ss_{s}"] = H_ss
            row[f"Re_H_as_{s}"] = H_as.real.astype(np.float64)
            row[f"Im_H_as_{s}"] = H_as.imag.astype(np.float64)
            row[f"rho_aa_{s}"] = rho_aa
            row[f"rho_ss_{s}"] = rho_ss
        self._msw_hist.append(row)

    def evolve_step_ode_etdrk2(self, rho_all, dt, phi1_dt, a, Tg):
        """Stage D.7.1: Strang-symmetric diagonal split around ETDRK2 off-diag.

        Gated by PRyMini.qke_ode_etdrk2_flag (inside qke_full_ode_flag). Restores
        O(dt^2) convergence at the Shi-Fuller MSW resonance.

        D.7 (predictor -> diag -> corrector) closed the D.6 ν-ν̄-symmetry
        instability and gave a ~4e-3 Richardson-limit match to Strang, but its
        Lie-Trotter-style diagonal/off-diagonal split was formally O(dt) and
        the SF drift ratio landed at 0.36 (vs 0.25 target for pure O(dt^2)).
        D.7.1 symmetrises the split: ½-diag -> predictor+corrector -> ½-diag,
        with I_total re-evaluated at the state going into the second half so
        the composition is genuinely second-order in the frozen-coefficient
        sense.

        Structure (per time step, given rho_n)
        --------------------------------------
        1. Build L(rho_n), gain-only N(rho_n), and I_total(rho_n) via
           _build_L_list. Compute a local phi_1(Γ·dt/2)·(dt/2) regulariser
           for the half-step diagonal exp-Euler (NOT the full-dt phi1_dt that
           PRyM_main supplies -- factor-2 wrong in the stiff limit if halved).
        2. Cache (Phi0, Phi1, Phi2) per sector at L(rho_n).
        3. Half-diag #1: rho_αα += phi_half · I_total(rho_n).
        4. Off-diag ETDRK2 predictor using L(rho_n) and N_off(rho_n).
        5. Re-evaluate N at rho_star via _build_L_list; off-diag corrector
           uses Phi2 at L(rho_n) (cache) and (N_off_star - N_off_n).
        6. Recompute I_total at rho_after_corrector via _assemble_collision_N
           (just the collision integrals; do not rebuild L).
        7. Half-diag #2: rho_αα += phi_half · I_total(rho_after_corrector).
        8. Off-diagonal magnitude clamp + diagonal clip.

        Parameters
        ----------
        phi1_dt : scalar or (3, Ny)
            Consumed by non-D.7 paths (evolve_step_ode). D.7.1 computes its
            own half-step regulariser locally; this parameter is accepted for
            dispatcher signature stability and otherwise ignored.
        """
        N = self.n_flavor
        Ny = self.Ny
        _ = phi1_dt  # see docstring: ignored in D.7.1 (half-step phi computed locally)

        # Stage E.2 sprint 8: per-step energy-accounting diagnostic.
        _diag_on = getattr(PRyMini, "qke_energy_diag_flag", False)
        if _diag_on:
            self._energy_step_idx += 1
            self._energy_snapshot(rho_all, a, Tg, label="pre_step")

        # Stage E.2 sprint 9: per-step per-y-mode MSW-passage diagnostic.
        _msw_on = getattr(PRyMini, "qke_msw_diag_flag", False)
        if _msw_on:
            self._msw_step_idx += 1
            self._msw_snapshot(rho_all, a, Tg, label="pre_step")

        # 1. L, gain-only N, and I_total at rho_n.
        L_list, N_gain_n, I_total_n = self._build_L_list(rho_all, a, Tg)
        dt_nat = dt * self._eV_to_secm1

        # Half-step diagonal regulariser phi_1(Γ_α · dt/2) · (dt/2).
        # Three channels per PRyM_main convention: [nue, nuebar, numu_eff].
        # nue/nuebar both use the electron C_D (self.C_D[0] = 3.06); numu_eff
        # is the mu-tau-symmetric average so uses the mu C_D (self.C_D[1]).
        half_dt = 0.5 * dt
        GF2_secm1 = PRyMini.GF**2 * PRyMini.MeV_to_secm1
        rate_base_h = GF2_secm1 * Tg**4 / a * half_dt
        C_D_chan = np.array([self.C_D[0], self.C_D[0], self.C_D[1]])
        z_h = np.outer(C_D_chan, self.y_grid) * rate_base_h
        z_h = np.maximum(z_h, 1.0e-15)
        phi1_h = np.where(z_h < 1.0e-4,
                          1.0 - 0.5 * z_h + z_h * z_h / 6.0,
                          (1.0 - np.exp(-z_h)) / z_h)
        phi_half = phi1_h * half_dt  # shape (3, Ny)

        # 2. Cache (Phi0, Phi1, Phi2) per sector at L(rho_n).
        Phi_cache = [self._etdrk2_expm_phi(L_list[s], dt_nat) for s in (0, 1)]

        # 3. Half-diag #1 using I_total(rho_n).
        rho_all[0, 0] += phi_half[0] * I_total_n[0]
        rho_all[1, 0] += phi_half[1] * I_total_n[1]
        rho_all[0, 1] += phi_half[2] * I_total_n[2]
        rho_all[0, 2] += phi_half[2] * I_total_n[2]
        rho_all[1, 1] += phi_half[2] * I_total_n[2]
        rho_all[1, 2] += phi_half[2] * I_total_n[2]

        # 4. ETDRK2 predictor -- off-diagonal coherence only. Applied to the
        #    state AFTER the first half-diag; Phi0/Phi1 use L(rho_n) (frozen).
        for s in range(2):
            Phi0, Phi1, _p2 = Phi_cache[s]
            rho_mat = self._to_mat(rho_all[s])
            rho_vec = rho_mat.reshape(Ny, N * N)
            N_off_vec = N_gain_n[s].reshape(Ny, N * N).copy()
            for alpha in range(N):
                N_off_vec[:, alpha * N + alpha] = 0.0
            rho_star_vec = (np.einsum('ijk,ik->ij', Phi0, rho_vec)
                            + np.einsum('ijk,ik->ij', Phi1, N_off_vec))
            rho_star_mat = rho_star_vec.reshape(Ny, N, N)
            rho_star_mat = 0.5 * (rho_star_mat + rho_star_mat.conj().swapaxes(-1, -2))
            rho_all[s] = self._to_vec(rho_star_mat)

        # 5. Corrector -- evaluate N at rho_star; Phi2 reuses cache at L(rho_n).
        _, N_gain_star, _ = self._build_L_list(rho_all, a, Tg)
        for s in range(2):
            _p0, _p1, Phi2 = Phi_cache[s]
            dN_off = (N_gain_star[s] - N_gain_n[s]).reshape(Ny, N * N).copy()
            for alpha in range(N):
                dN_off[:, alpha * N + alpha] = 0.0
            rho_vec = self._to_mat(rho_all[s]).reshape(Ny, N * N)
            rho_new_vec = rho_vec + np.einsum('ijk,ik->ij', Phi2, dN_off)
            rho_new_mat = rho_new_vec.reshape(Ny, N, N)
            rho_new_mat = 0.5 * (rho_new_mat + rho_new_mat.conj().swapaxes(-1, -2))
            rho_all[s] = self._to_vec(rho_new_mat)

        # Stage E.2 sprint 8: post-corrector energy snapshot (pre half-diag #2).
        if _diag_on:
            self._energy_snapshot(rho_all, a, Tg, label="post_corrector")

        # 6. Re-evaluate I_total at rho_after_corrector for the second half-diag.
        #    We only need I_total, not the full L; skip _build_L_list and call
        #    _assemble_collision_N directly to keep cost down.
        _, I_total_post = self._assemble_collision_N(rho_all, a, Tg)

        # 7. Half-diag #2 using I_total(rho_after_corrector).
        rho_all[0, 0] += phi_half[0] * I_total_post[0]
        rho_all[1, 0] += phi_half[1] * I_total_post[1]
        rho_all[0, 1] += phi_half[2] * I_total_post[2]
        rho_all[0, 2] += phi_half[2] * I_total_post[2]
        rho_all[1, 1] += phi_half[2] * I_total_post[2]
        rho_all[1, 2] += phi_half[2] * I_total_post[2]

        # 6. Off-diagonal magnitude clamp (same as evolve_step_ode).
        for sector in range(2):
            for p_idx, (alpha, beta) in enumerate(self._all_pair_flavors):
                re_idx, im_idx = self._all_pair_write[p_idx]
                rho_ab = rho_all[sector, re_idx] + 1j * rho_all[sector, im_idx]
                rho_aa = rho_all[sector, alpha]
                rho_bb = rho_all[sector, beta]
                ab_mag = np.abs(rho_ab)
                # Stage E.2 sprint 6: zero non-finite off-diagonals so the
                # clamp below doesn't silently no-op (np.where treats NaN
                # as False, letting NaN propagate into _etdrk2_expm_phi).
                rho_ab = np.where(np.isfinite(ab_mag), rho_ab, 0.0 + 0.0j)
                ab_mag = np.abs(rho_ab)
                max_mag = np.minimum(
                    0.5,
                    np.sqrt(np.maximum(rho_aa * rho_bb, 0.0)) + 1e-10)
                scale = np.where(
                    ab_mag > max_mag,
                    max_mag / np.maximum(ab_mag, 1e-30),
                    1.0)
                rho_ab = rho_ab * scale
                rho_all[sector, re_idx] = rho_ab.real
                rho_all[sector, im_idx] = rho_ab.imag

        # 7. Clip diagonals to [f_min, f_max].
        # Stage E.2 sprint 6: nan_to_num before clip. At very cold T (~5 keV)
        # in extended-window runs the collision integral I_total_post
        # occasionally produces a single-mode non-finite entry that
        # half-diag-2 writes into a diagonal; np.clip passes NaN through,
        # so without the sanitisation the next step's _build_L_list
        # inherits NaN and scipy.linalg.expm crashes inside its
        # norm-estimate.
        f_min = 1.0e-30
        f_max = 1.0 - f_min
        for sector in range(2):
            for d in range(self.n_flavor):
                rho_all[sector, d] = np.clip(
                    np.nan_to_num(rho_all[sector, d],
                                  nan=f_min, posinf=f_max, neginf=f_min),
                    f_min, f_max)

        # Stage E.2 sprint 8: post-clip energy snapshot (end-of-step state).
        if _diag_on:
            self._energy_snapshot(rho_all, a, Tg, label="post_clip")

        # Stage E.2 sprint 9: post-clip MSW-passage snapshot.
        if _msw_on:
            self._msw_snapshot(rho_all, a, Tg, label="post_clip")

    def extract_f_all_3species(self, rho_all):
        """Extract 3-species f_all array compatible with BoltzmannSolver.

        Returns shape (3, Ny): [nue, nuebar, numu_eff].
        """
        f_all = np.zeros((3, self.Ny))
        f_all[0] = rho_all[0, 0]
        f_all[1] = rho_all[1, 0]
        f_all[2] = 0.25 * (rho_all[0, 1] + rho_all[0, 2]
                          + rho_all[1, 1] + rho_all[1, 2])
        return f_all

    def update_thermo_distributions(self, rho_all, a, a_of_T_func=None):
        """Patch PRyMthermo with all 6 flavor distributions from density matrix.

        Unlike the diagonal BoltzmannSolver which uses mu-tau symmetry,
        this extracts all 6 independent distributions: nue, nuebar,
        numu, numubar, nutau, nutaubar.
        """
        import PRyM.PRyM_thermo as PRyMthermo

        f_nue = rho_all[0, 0].copy()
        f_nuebar = rho_all[1, 0].copy()
        f_numu = rho_all[0, 1].copy()
        f_numubar = rho_all[1, 1].copy()
        f_nutau = rho_all[0, 2].copy()
        f_nutaubar = rho_all[1, 2].copy()

        PRyMthermo.f_nue_general = self._boltz.make_f_callable(f_nue, a, a_of_T_func)
        PRyMthermo.f_nuebar_general = self._boltz.make_f_callable(f_nuebar, a, a_of_T_func)
        PRyMthermo.f_numu_general = self._boltz.make_f_callable(f_numu, a, a_of_T_func)
        PRyMthermo.f_numubar_general = self._boltz.make_f_callable(f_numubar, a, a_of_T_func)
        PRyMthermo.f_nutau_general = self._boltz.make_f_callable(f_nutau, a, a_of_T_func)
        PRyMthermo.f_nutaubar_general = self._boltz.make_f_callable(f_nutaubar, a, a_of_T_func)

        if self.n_flavor == 4:
            f_nus = rho_all[0, 3].copy()
            f_nusbar = rho_all[1, 3].copy()
            PRyMthermo.f_nus_general = self._boltz.make_f_callable(f_nus, a, a_of_T_func)
            PRyMthermo.f_nusbar_general = self._boltz.make_f_callable(f_nusbar, a, a_of_T_func)
