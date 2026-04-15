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
        return yj * (3.0 * (yk**2 + yl**2 - yi**2) - yj**2) / 6.0

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


@njit
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


@njit(parallel=True)
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


@njit
def _F_stat_stable(f1, f2, f3, f4):
    """Numerically stable statistical factor for collision integral.

    Computes f3*f4*(1-f1)*(1-f2) - f1*f2*(1-f3)*(1-f4) using the
    reformulation: f1*f2*(1-f3)*(1-f4) * expm1(mu1+mu2-mu3-mu4)
    where mu_i = log((1-fi)/fi).

    This avoids catastrophic cancellation when distributions are near
    equilibrium (all fi close to FD at the same temperature).
    """
    # Clamp to avoid log(0)
    _lo = 1.0e-20
    _hi = 1.0 - 1.0e-20
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


@njit
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


@njit
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

@njit
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


@njit
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
        # Species layout:
        #   mu_tau_symmetric_flag=True  -> n_species=3 [nue, nuebar, numu_eff]
        #       (numu_eff aggregates {numu, numubar, nutau, nutaubar})
        #   mu_tau_symmetric_flag=False -> n_species=4 [nue, nuebar, numu_eff, nutau_eff]
        #       (numu_eff aggregates pcle+antipcle of mu; nutau_eff same for tau)
        self.mu_tau_symmetric = PRyMini.mu_tau_symmetric_flag
        self.n_species = 3 if self.mu_tau_symmetric else 4

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
        effective 2-flavor system (Sabti Eq. 3.18).

        In the rapid-oscillation limit, the collision integral for each
        flavor is mixed with the PMNS transition matrix:

            df_a/dt = sum_b P_ab * C_b[f]

        where P_ab = sum_i |V_ai|^2 |V_bi|^2 and C_b is the collision
        integral for flavor b evaluated with the current distributions.

        For our [nue, nuebar, numu_eff] system with mu-tau symmetry:

            I_mixed[0] = P_ee * I[0] + (1-P_ee) * I[2]
            I_mixed[1] = P_ee * I[1] + (1-P_ee) * I[2]
            I_mixed[2] = (1-P_ee)/2 * (I[0]+I[1])/2 + (1+P_ee)/2 * I[2]

        The numu_eff row automatically averages over mu and tau contributions.
        """
        # Build PMNS |V_ai|^2 from mixing angles
        s12 = np.sin(PRyMini.theta_12)
        c12 = np.cos(PRyMini.theta_12)
        s13 = np.sin(PRyMini.theta_13)
        c13 = np.cos(PRyMini.theta_13)

        # Electron row: |V_ei|^2 (independent of theta_23 and delta_CP)
        Ve1_sq = c12**2 * c13**2
        Ve2_sq = s12**2 * c13**2
        Ve3_sq = s13**2

        # P_ee = sum_i |V_ei|^4 (electron survival probability)
        self.P_ee = Ve1_sq**2 + Ve2_sq**2 + Ve3_sq**2

        if PRyMini.verbose_flag:
            print(f"  Collision mixing (Sabti): P_ee = {self.P_ee:.4f}, "
                  f"1-P_ee = {1-self.P_ee:.4f}")

    def _apply_collision_mixing(self, I_total):
        """
        Apply PMNS time-averaged oscillation mixing to collision integrals.

        n=3 (mu_tau_symmetric=True):
          I_mixed[0] = P_ee * I[0] + (1-P_ee) * I[2]          # nue
          I_mixed[1] = P_ee * I[1] + (1-P_ee) * I[2]          # nuebar
          I_mixed[2] = (1-P_ee)/2 * (I[0]+I[1])/2
                       + (1+P_ee)/2 * I[2]                    # numu_eff

        n=4 (mu_tau_symmetric=False): split the mu-tau sector. We use the
        maximal-theta_23, no-CP-phase PMNS approximation for the mu/tau
        mixing (P_mu_tau = P_mu_mu = P_tau_tau = (1+P_ee)/4). This keeps
        the vacuum oscillation physics mu-tau-symmetric even when the
        distributions are not — consistent with the user intent that the
        flag breaks COLLISION dynamics, not the SM PMNS. Exact PMNS could
        be reintroduced in a future pass by computing the full 3x3 |V|^4
        matrix from the current sin^2(theta_ij) values.

        Returns the mixed collision integral array (same shape as I_total).
        """
        P_ee = self.P_ee
        P_off = 1.0 - P_ee

        if self.mu_tau_symmetric:
            I_mixed = np.empty_like(I_total)
            I_mixed[0] = P_ee * I_total[0] + P_off * I_total[2]
            I_mixed[1] = P_ee * I_total[1] + P_off * I_total[2]
            I_mixed[2] = (P_off / 2.0) * (I_total[0] + I_total[1]) / 2.0 \
                        + (1.0 + P_ee) / 2.0 * I_total[2]
            return I_mixed

        # n=4 path
        I_mixed = np.empty_like(I_total)
        half_off = 0.5 * P_off
        I_mu_plus_tau = I_total[2] + I_total[3]
        avg_e = 0.5 * (I_total[0] + I_total[1])
        I_mixed[0] = P_ee * I_total[0] + half_off * I_mu_plus_tau
        I_mixed[1] = P_ee * I_total[1] + half_off * I_mu_plus_tau
        # P_mu_e = P_tau_e = (1-P_ee)/2; P_mu_mu = P_tau_tau = P_mu_tau = (1+P_ee)/4
        I_mixed[2] = half_off * avg_e + 0.25 * (1.0 + P_ee) * I_mu_plus_tau
        I_mixed[3] = half_off * avg_e + 0.25 * (1.0 + P_ee) * I_mu_plus_tau
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

        if self.mu_tau_symmetric:
            # n=3: single thermal FD for all 3 species
            f_all[:, :] = fd_default
            # Optional override: only 'nue', 'nuebar', 'numu' are honored in n=3
            if f_initial is not None:
                _slot_map = {'nue': 0, 'nuebar': 1, 'numu': 2}
                for key, idx in _slot_map.items():
                    if f_initial.get(key) is not None:
                        vals = np.asarray(f_initial[key](p_grid, Tnu), dtype=float)
                        f_all[idx] = vals
        else:
            # n=4: separate mu and tau slots
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

    def collision_integrals(self, f_all, a, Tg):
        """
        Compute total collision integrals for all species.

        Returns array of shape (n_species, Ny).
        When mu_tau_symmetric_flag=False, n_species=4 and the returned array
        has rows [I_nue, I_nuebar, I_numu, I_nutau].
        """
        # Convert units: GF in MeV^{-2}, collision integral in MeV * s^{-1}
        # coll_scale applied to the overall prefactor, scaling both nu-nu
        # and nu-e collision integrals uniformly.
        GF2_pref = self.GF2_prefactor * PRyMini.MeV_to_secm1 * PRyMini.coll_scale

        # Compute FD-tail extrapolation parameters for off-grid interpolation
        tail_params = _compute_all_tail_params(self.y_grid, f_all)

        if self.mu_tau_symmetric:
            # Nu-nu processes (uses D_k0, D_k2)
            I_nu_nu = _collision_integral_nu_nu(
                f_all, self.y_grid, self.quad_w, a, GF2_pref, tail_params,
                self.D_k0, self.D_k2, self.Ny_coll)

            # Nu-e processes: massive or massless electron kinematics
            if PRyMini.massive_electron_flag:
                I_nu_e = _collision_integral_nu_e_massive(
                    f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                    self.geL2, self.geR2, self.gmuL2, self.gmuR2,
                    PRyMini.me, self.Ny_coll)
            else:
                fnu_e_scat_val = float(self._fnu_e_scat(Tg))
                fnu_e_ann_val = float(self._fnu_e_ann(Tg))
                fnu_mu_scat_val = float(self._fnu_mu_scat(Tg))
                fnu_mu_ann_val = float(self._fnu_mu_ann(Tg))
                I_nu_e = _collision_integral_nu_e(
                    f_all, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                    self.geL2, self.geR2, self.geLgeR,
                    self.gmuL2, self.gmuR2, self.gmuLgmuR,
                    PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                    fnu_mu_scat_val, fnu_mu_ann_val, tail_params,
                    self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
                    1.0, 1.0, 1.0, 1.0)
        else:
            # n=4 asymmetric path. nu-nu uses the dedicated asym4 function;
            # nu-e is assembled from two n=3 calls (one for mu slot, one for
            # tau slot) since I_nue, I_nuebar don't depend on the mu/tau
            # distribution and I_numu/I_nutau use identical couplings.
            I_nu_nu = _collision_integral_nu_nu_asym4(
                f_all, self.y_grid, self.quad_w, a, GF2_pref, tail_params,
                self.D_k0, self.D_k2, self.Ny_coll)

            # Build 3-slot arrays for mu and tau calls into the n=3 nu-e
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
                fnu_e_scat_val = float(self._fnu_e_scat(Tg))
                fnu_e_ann_val = float(self._fnu_e_ann(Tg))
                fnu_mu_scat_val = float(self._fnu_mu_scat(Tg))
                fnu_mu_ann_val = float(self._fnu_mu_ann(Tg))
                I_mu = _collision_integral_nu_e(
                    f3_mu, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                    self.geL2, self.geR2, self.geLgeR,
                    self.gmuL2, self.gmuR2, self.gmuLgmuR,
                    PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                    fnu_mu_scat_val, fnu_mu_ann_val, tp_mu,
                    self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
                    1.0, 1.0, 1.0, 1.0)
                I_tau = _collision_integral_nu_e(
                    f3_tau, self.y_grid, self.quad_w, a, Tg, GF2_pref,
                    self.geL2, self.geR2, self.geLgeR,
                    self.gmuL2, self.gmuR2, self.gmuLgmuR,
                    PRyMini.me, fnu_e_scat_val, fnu_e_ann_val,
                    fnu_mu_scat_val, fnu_mu_ann_val, tp_tau,
                    self.D_k0, self.D_k1, self.D_k2, self.Ny_coll,
                    1.0, 1.0, 1.0, 1.0)

            I_nu_e = np.zeros((4, self.Ny))
            # I_nue, I_nuebar come from either call; average for symmetry/safety
            # (they should be bit-identical as I[0,1] don't depend on slot 2).
            I_nu_e[0] = 0.5 * (I_mu[0] + I_tau[0])
            I_nu_e[1] = 0.5 * (I_mu[1] + I_tau[1])
            I_nu_e[2] = I_mu[2]
            I_nu_e[3] = I_tau[2]

        I_total = I_nu_nu + I_nu_e

        # Apply oscillation mixing to SM collision integrals (Sabti Eq. 3.18).
        # For 'collision_mixing' method, this replaces the operator-split
        # relaxation step. For 'relaxation' method, mixing is applied
        # separately via apply_oscillation_mixing() after the collision step.
        if PRyMini.nu_oscillation_flag and \
                getattr(PRyMini, 'nu_oscillation_method', 'relaxation') == 'collision_mixing':
            I_total = self._apply_collision_mixing(I_total)

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
        # Species weights convention (see __init__):
        #   n=3 [nue, nuebar, numu_eff]            → {1, 1, 2}
        #     numu_eff encodes I[2] = 2*per-species rate AND represents the
        #     shared value of all 4 mu-sector DOF (mu,mubar,tau,taubar).
        #     Weight 2 gives sum over 4 DOF: 2*(2*per) = 4*per.
        #   n=4 [nue, nuebar, numu_eff, nutau_eff] → {1, 1, 1, 1}
        #     I[2], I[3] each encode 2*per-species rate for one flavor
        #     (pcle+antipcle aggregated). Weight 1 each gives sum over 2 DOF.
        # In the symmetric limit f_numu==f_nutau: I[2]_n4 == I[2]_n3, so the
        # n=4 sum (I[0]+I[1]+I[2]+I[3]) equals the n=3 sum (I[0]+I[1]+2*I[2]).
        if self.mu_tau_symmetric:
            _weights = (1.0, 1.0, 2.0)
        else:
            _weights = (1.0, 1.0, 1.0, 1.0)

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
        if self.mu_tau_symmetric:
            PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_numubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutau_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutaubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
        else:
            # n=4: mu_eff and tau_eff aggregate pcle+antipcle per flavor, so
            # f_numu_general == f_numubar_general and f_nutau_general == f_nutaubar_general
            # Full nu/nu-bar asymmetry is Stage 3 (n=6 diagonal) or use QKE now.
            f_nutau_grid = f_all[3].copy()
            PRyMthermo.f_numu_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_numubar_general = _make_f_callable(f_numu_grid, current_a, a_func)
            PRyMthermo.f_nutau_general = _make_f_callable(f_nutau_grid, current_a, a_func)
            PRyMthermo.f_nutaubar_general = _make_f_callable(f_nutau_grid, current_a, a_func)

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

        # Build 3x3 PMNS mixing matrix from PDG parameters
        self._build_PMNS()

        # Precompute vacuum Hamiltonian base matrices (eV^2)
        # H_vac_nu(y) = Omega_nu / E(y),  H_vac_nubar(y) = Omega_nubar / E(y)
        Dm2 = np.array([0.0, PRyMini.Dm2_21, PRyMini.Dm2_31])  # eV^2
        Dm2_half = np.diag(Dm2 / 2.0)
        self._Omega_nu = self.U_PMNS @ Dm2_half @ self.U_PMNS.conj().T      # eV^2
        self._Omega_nubar = self.U_PMNS.conj() @ Dm2_half @ self.U_PMNS.T   # eV^2

        # Electron-flavor projector for matter potential
        self._diag_e = np.zeros((3, 3), dtype=complex)
        self._diag_e[0, 0] = 1.0

        # W boson mass squared for thermal matter potential
        self.mW2 = (PRyMini.mZ * np.sqrt(1.0 - PRyMini.sW2))**2  # MeV^2

        # Collision damping coefficients (de Salas & Pastor 2016)
        # D_alpha = C_D_alpha * GF^2 * T^4 * E  [natural units]
        self.C_D = np.array([3.06, 2.22, 2.22])  # [nue, numu, nutau]

        # Off-diagonal ν-e scattering couplings (for collision gain terms).
        # Diagonal: 4(g_{L,α}² + g_R²); off-diagonal: 4(g_{L,α} g_{L,β} + g_R²).
        geL = PRyMini.geL    # ½ + sW²
        gmuL = PRyMini.gmuL  # -½ + sW²
        geR2 = PRyMini.geR**2  # sW⁴
        self.c_emu_scat = 4.0 * (geL * gmuL + geR2)   # e-μ and e-τ (g_{L,τ} = g_{L,μ})
        self.c_mutau_scat = 4.0 * (gmuL**2 + geR2)    # μ-τ = diagonal μ coupling

        # eV <-> seconds conversion: 1 eV = eV_to_secm1 s^{-1}
        self._eV_to_secm1 = PRyMini.MeV_to_secm1 * 1.0e-6

        if PRyMini.verbose_flag:
            print(f"  DensityMatrixSolver: 3x3 QKE, {2*9*self.Ny} real DOFs")

    def _build_PMNS(self):
        """Construct the 3x3 PMNS mixing matrix from oscillation parameters."""
        s12 = np.sin(PRyMini.theta_12)
        c12 = np.cos(PRyMini.theta_12)
        s13 = np.sin(PRyMini.theta_13)
        c13 = np.cos(PRyMini.theta_13)
        s23 = np.sin(PRyMini.theta_23)
        c23 = np.cos(PRyMini.theta_23)
        eidCP = np.exp(1j * PRyMini.delta_CP)
        emidCP = np.exp(-1j * PRyMini.delta_CP)

        self.U_PMNS = np.array([
            [c12*c13,                        s12*c13,                        s13*emidCP],
            [-s12*c23 - c12*s23*s13*eidCP,   c12*c23 - s12*s23*s13*eidCP,   s23*c13],
            [s12*s23 - c12*c23*s13*eidCP,   -c12*s23 - s12*c23*s13*eidCP,   c23*c13]
        ], dtype=complex)

    def initial_conditions(self, Tnu, a, f_initial=None):
        """Return initial density matrices for the QKE solver.

        Parameters
        ----------
        Tnu : float
            Neutrino temperature at the start of the Boltzmann phase (MeV).
            Used to construct the default thermal FD distribution.
        a : float
            Scale factor at start.
        f_initial : dict, optional
            User-supplied initial distribution callables, keyed by species
            name: 'nue', 'nuebar', 'numu', 'numubar', 'nutau', 'nutaubar'.
            Each callable has signature f(p_MeV, Tnu_MeV) -> array_like. Any
            missing key falls back to thermal FD at Tnu. Used when
            mu_tau_symmetric_flag=False to launch asymmetric initial
            conditions.

        Returns
        -------
        rho_all : ndarray, shape (2, 9, Ny)
            Density matrices in flavor basis. Off-diagonals initialized to
            zero (no initial flavor coherences).
        """
        rho_all = np.zeros((2, 9, self.Ny))
        Tnu_com = Tnu * a

        # Sector 0 = neutrinos (rho_ee, rho_mumu, rho_tautau at indices 0,1,2)
        # Sector 1 = antineutrinos (same layout)
        # f_initial maps species -> (sector, flavor_idx)
        _species_to_slot = {
            'nue':      (0, 0), 'numu':      (0, 1), 'nutau':    (0, 2),
            'nuebar':   (1, 0), 'numubar':   (1, 1), 'nutaubar': (1, 2),
        }

        # Default thermal-FD diagonal
        p_grid = self.y_grid / a  # physical momenta at scale factor a
        fd_default = np.zeros(self.Ny)
        for i in range(self.Ny):
            x = self.y_grid[i] / Tnu_com
            if x < 500.0:
                fd_default[i] = 1.0 / (np.exp(x) + 1.0)

        for species, (sector, flavor) in _species_to_slot.items():
            if f_initial is not None and species in f_initial and f_initial[species] is not None:
                # User callable: evaluate at physical momenta + Tnu
                f_vals = np.asarray(f_initial[species](p_grid, Tnu), dtype=float)
                if f_vals.shape != (self.Ny,):
                    raise ValueError(
                        f"initial_conditions: callable for {species} returned "
                        f"shape {f_vals.shape}, expected {(self.Ny,)}")
                rho_all[sector, flavor] = f_vals
            else:
                rho_all[sector, flavor] = fd_default

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
        rho_nu_mat = _rho_vec_batch_to_matrix(rho_all[0])   # (Ny, 3, 3)
        rho_nubar_mat = _rho_vec_batch_to_matrix(rho_all[1])  # (Ny, 3, 3)
        diff_mat = rho_nu_mat - rho_nubar_mat  # (Ny, 3, 3)
        # Quadrature: sum w_i * y_i^2 * diff_mat[i] over collision grid
        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2  # (Ny_coll,) MeV^3
        V_nunu_mat = np.einsum('i,ijk->jk', y2w, diff_mat[:Ny_coll])  # (3,3) MeV^3
        V_nunu_prefactor = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3)  # MeV^{-2} / MeV^0 = MeV^{-2}... no
        # GF [MeV^{-2}] * MeV^3 / a^3 = MeV / a^3 -> multiply by 1e6 for eV
        V_nunu_eV = V_nunu_prefactor * V_nunu_mat * 1.0e6  # (3,3) eV

        # Build Hamiltonians: shape (Ny, 3, 3)
        # H = Omega/E + V_thermal*diag(1,0,0) + V_nunu
        # In our convention (phase_signs = [-1,+1]):
        #   nu:  exp(-iH_nu dt) rho exp(+iH_nu dt) -> i*drho/dt = [H_nu, rho]
        #   nubar: exp(+iH_nubar dt) rhobar exp(-iH_nubar dt) -> i*drhobar/dt = -[H_nubar, rhobar]
        # This matches the standard QKE when H_nubar uses Omega_nubar and same-sign potentials.
        H_list = [None, None]
        for s in range(2):
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, 3, 3), dtype=complex)
            for k in range(3):
                for l in range(3):
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
            rho_mat = _rho_vec_batch_to_matrix(rho_all[sector])  # (Ny, 3, 3)

            # Transform to H eigenbasis: rho_H = P^dag @ rho @ P
            Pdag = P.conj().transpose(0, 2, 1)  # (Ny, 3, 3)
            rho_H = np.einsum('nij,njk,nkl->nil', Pdag, rho_mat, P)

            # Apply phase rotations: rho'_H[n,i,j] *= phase[n,i] * conj(phase[n,j])
            phase_ij = phases[:, :, None] * phases[:, None, :].conj()  # (Ny, 3, 3)
            rho_H *= phase_ij

            # Transform back: rho' = P @ rho'_H @ P^dag
            rho_new = np.einsum('nij,njk,nkl->nil', P, rho_H, Pdag)

            # Store back as vec9
            rho_all[sector] = _matrix_batch_to_rho_vec(rho_new)

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

        # Compute off-diagonal gain (transport) for each sector
        for sector in range(2):
            # Extract off-diagonal: shape (6, Ny) = components 3..8
            rho_offdiag = rho_all[sector, 3:, :]
            B_idx = 1 if sector == 0 else 0  # ν̄_e for ν sector, ν_e for ν̄

            gain = _offdiag_collision_gain(
                rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                GF2_pref,
                self.c_emu_scat, self.c_mutau_scat,
                fnu_emu_scat_val, fnu_mutau_scat_val,
                B_idx, tail_params,
                self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            # Update off-diagonals: ρ_new = damp × ρ_old + phi1dt × gain
            # e-μ (components 3,4 → gain indices 0,1)
            rho_all[sector, 3] = damp_emu * rho_all[sector, 3] + phi1dt_emu * gain[0]
            rho_all[sector, 4] = damp_emu * rho_all[sector, 4] + phi1dt_emu * gain[1]
            # e-τ (components 5,6 → gain indices 2,3)
            rho_all[sector, 5] = damp_etau * rho_all[sector, 5] + phi1dt_etau * gain[2]
            rho_all[sector, 6] = damp_etau * rho_all[sector, 6] + phi1dt_etau * gain[3]
            # μ-τ (components 7,8 → gain indices 4,5)
            rho_all[sector, 7] = damp_mutau * rho_all[sector, 7] + phi1dt_mutau * gain[4]
            rho_all[sector, 8] = damp_mutau * rho_all[sector, 8] + phi1dt_mutau * gain[5]

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

        # Thermal matter potential (Notzold-Raffelt)
        rho_e_th = 7.0 * np.pi**2 / 60.0 * Tg**4
        V0 = 8.0 * np.sqrt(2.0) * PRyMini.GF * rho_e_th / (3.0 * self.mW2)
        V_thermal_eV = V0 * E_eV

        # CC matter potential: V_CC = sqrt(2) GF (n_e- - n_e+)
        # Charge neutrality: n_e- - n_e+ = n_p ~ eta_b * n_gamma
        # n_gamma = 2 zeta(3)/pi^2 * T^3, in comoving: T -> Tg
        from scipy.special import zeta as _zeta
        n_gamma = 2.0 * _zeta(3) / np.pi**2 * Tg**3  # MeV^3
        n_e_asym = PRyMini.eta0b * n_gamma  # MeV^3
        V_CC_MeV = np.sqrt(2.0) * PRyMini.GF * n_e_asym  # MeV
        V_CC_eV = V_CC_MeV * 1.0e6  # eV (scalar, same for all modes)

        # V_nunu self-interaction potential (full 3x3 matrix).
        # Includes off-diagonal elements from flavor coherences in ρ - ρ̄.
        Ny_coll = min(self._boltz.Ny_coll, Ny)
        y2w = self._boltz.quad_w[:Ny_coll] * self.y_grid[:Ny_coll]**2
        V_nunu_pref = np.sqrt(2.0) * PRyMini.GF / (2.0 * np.pi**2 * a**3) * 1.0e6  # eV/MeV³
        rho_nu_mat = _rho_vec_batch_to_matrix(rho_all[0])    # (Ny, 3, 3)
        rho_nubar_mat = _rho_vec_batch_to_matrix(rho_all[1]) # (Ny, 3, 3)
        diff_mat = rho_nu_mat[:Ny_coll] - rho_nubar_mat[:Ny_coll]  # (Ny_coll, 3, 3)
        V_nunu_eV = V_nunu_pref * np.einsum('i,ijk->jk', y2w, diff_mat)  # (3, 3) eV

        # Build H for each sector: H = Omega/E + V_thermal*diag_e + V_CC*diag_e + V_nunu
        H_list = [None, None]
        for s in range(2):
            Omega = self._Omega_nu if s == 0 else self._Omega_nubar
            H = np.zeros((Ny, 3, 3), dtype=complex)
            for k in range(3):
                for l in range(3):
                    H[:, k, l] = Omega[k, l] * inv_E + V_nunu_eV[k, l]
            H[:, 0, 0] += V_thermal_eV + V_CC_eV
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
        Gamma = np.zeros((3, Ny))
        for alpha in range(3):
            Gamma[alpha] = self.C_D[alpha] * GF_eV**2 * T_eV**4 * E_eV * self._eV_to_secm1

        # D_αβ = ½(Γ_α + Γ_β) in 1/s, for each off-diagonal pair
        D_pairs = np.zeros((3, Ny))
        D_pairs[0] = 0.5 * (Gamma[0] + Gamma[1])   # e-μ
        D_pairs[1] = 0.5 * (Gamma[0] + Gamma[2])   # e-τ
        D_pairs[2] = 0.5 * (Gamma[1] + Gamma[2])   # μ-τ

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
        pair_flavors = [(0, 1), (0, 2), (1, 2)]
        osc_signs = [+1.0, -1.0]
        dt_nat = dt * self._eV_to_secm1

        for sector in range(2):
            s = osc_signs[sector]
            H = H_list[sector]

            # Off-diagonal collision gain
            rho_offdiag = rho_all[sector, 3:, :]
            B_idx = 1 if sector == 0 else 0
            if PRyMini.massive_electron_flag:
                gain = _offdiag_collision_gain_massive(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll,
                    PRyMini.me)
            else:
                gain = _offdiag_collision_gain(
                    rho_offdiag, f_all, self.y_grid, self._boltz.quad_w, a, Tg,
                    GF2_pref,
                    self.c_emu_scat, self.c_mutau_scat,
                    fnu_emu_scat_val, fnu_mutau_scat_val,
                    B_idx, tail_params,
                    self._boltz.D_k0, self._boltz.D_k2, self._boltz.Ny_coll)

            for p_idx, (alpha, beta) in enumerate(pair_flavors):
                re_idx = 2 * p_idx + 3
                im_idx = 2 * p_idx + 4
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

        # Clip diagonal elements to valid range [0, 1]
        f_min = 1.0e-30
        f_max = 1.0 - f_min
        for sector in range(2):
            for d in range(3):
                rho_all[sector, d] = np.clip(rho_all[sector, d], f_min, f_max)

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
