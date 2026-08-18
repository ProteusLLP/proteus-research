"""Float64 evaluation of the Dirichlet log-product density.

For m<=51, coefficients are formed lazily with NumPy and SciPy. For m=52..201,
the central Glaser series is combined with a packaged tilted log-density grid
generated offline by pointwise inversion. Runtime evaluation performs no
Fourier inversion and uses no extended arithmetic.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import interpolate, special


MAX_M = 201
GLASER_TERMS = 180
HIGH_M_DENSITY_PATH = (
    Path(__file__).resolve().parent / "data" / "high_m_log_density.npz"
)


@lru_cache(maxsize=1)
def _high_m_density_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(HIGH_M_DENSITY_PATH, allow_pickle=False) as values:
        return (
            values["m_grid"].copy(),
            values["t_grid"].copy(),
            values["log_ell_tilde"].copy(),
        )


@lru_cache(maxsize=None)
def _high_m_log_density_interpolator(m: int) -> interpolate.PchipInterpolator:
    m_grid, t_grid, log_values = _high_m_density_data()
    index = int(np.searchsorted(m_grid, int(m)))
    if index >= len(m_grid) or m_grid[index] != int(m):
        raise ValueError(f"high-order density resource does not contain m={m}")
    return interpolate.PchipInterpolator(np.log(t_grid), log_values[index])


def _high_m_log_ell_tilde(m: int, t: float) -> float:
    _, t_grid, _ = _high_m_density_data()
    if t < t_grid[0]:
        if t <= 0.0:
            return -math.inf
        nu, leading, coefficients = glaser_coefficients(m, GLASER_TERMS)
        polynomial = float(np.polynomial.polynomial.polyval(t, coefficients))
        if polynomial <= 0.0:
            raise ArithmeticError(f"invalid Glaser density for m={m}, t={t}")
        return math.log(leading) + (nu - 1.0) * math.log(t) + math.log(polynomial)
    if t > t_grid[-1]:
        raise ValueError(f"high-order density evaluation supports t <= {t_grid[-1]:g}")
    return float(_high_m_log_density_interpolator(m)(math.log(t)))


def _high_m_ell_tilde(m: int, t: float) -> float:
    return math.exp(_high_m_log_ell_tilde(m, t))


@lru_cache(maxsize=None)
def glaser_coefficients(
    m: int, n_terms: int = GLASER_TERMS
) -> tuple[float, float, np.ndarray]:
    """Return ``(nu_m, A_m, v)`` for the Glaser-type series."""
    if not 2 <= m <= MAX_M:
        raise ValueError(f"online density evaluation supports 2 <= m <= {MAX_M}")

    nu = 0.5 * (m - 1.0)
    log_leading = sum(
        special.gammaln(1.0 + index / m) for index in range(1, m)
    ) - special.gammaln(nu)
    leading = math.exp(log_leading)

    bernoulli = special.bernoulli(n_terms + 1)
    cumulants = np.zeros(n_terms + 1, dtype=float)
    for index in range(1, n_terms + 1):
        cumulants[index] = (
            (-1.0) ** (index + 1)
            * (m - m ** (-index))
            * bernoulli[index + 1]
            / (index * (index + 1.0))
        )

    exponential = np.zeros(n_terms + 1, dtype=float)
    exponential[0] = 1.0
    for index in range(1, n_terms + 1):
        orders = np.arange(1, index + 1, dtype=float)
        exponential[index] = (
            np.dot(
                orders,
                cumulants[1 : index + 1] * exponential[index - 1 :: -1],
            )
            / index
        )

    coefficients = np.empty(n_terms + 1, dtype=float)
    coefficients[0] = 1.0
    gamma_ratio = 1.0
    for index in range(1, n_terms + 1):
        gamma_ratio /= nu + index - 1.0
        coefficients[index] = exponential[index] * gamma_ratio
    return nu, leading, coefficients


def ell_tilde_glaser(m: int, t: float, n_terms: int = GLASER_TERMS) -> float:
    """Return ``exp(t) ell_m(t)`` from the Glaser-type series."""
    if t < 0.0:
        return 0.0
    if t == 0.0:
        if m == 2:
            return math.inf
        if m == 3:
            return glaser_coefficients(m, n_terms)[1]
        return 0.0

    nu, leading, coefficients = glaser_coefficients(m, n_terms)
    polynomial = float(np.polynomial.polynomial.polyval(t, coefficients))
    return leading * t ** (nu - 1.0) * polynomial


@lru_cache(maxsize=None)
def _factorial(degree: int) -> np.ndarray:
    return special.factorial(np.arange(degree + 1), exact=False)


def _exp_series_from_log_derivatives(log_derivatives: np.ndarray) -> np.ndarray:
    degree = len(log_derivatives) - 1
    coefficients = np.zeros(degree + 1, dtype=float)
    coefficients[0] = 1.0
    factorial = _factorial(degree)
    for order in range(degree):
        index = np.arange(order + 1)
        coefficients[order + 1] = np.sum(
            coefficients[index]
            * log_derivatives[order - index + 1]
            / factorial[order - index]
        ) / (order + 1.0)
    return coefficients


@lru_cache(maxsize=None)
def _gamma_ratio_series(m: int, index: int) -> np.ndarray:
    degree = m - 2
    if degree == 0:
        return np.ones(1, dtype=float)

    shape = index / m
    derivatives = np.zeros(degree + 1, dtype=float)
    for order in range(1, degree + 1):
        derivatives[order] = special.polygamma(order - 1, 1.0) - special.polygamma(
            order - 1, 1.0 + shape
        )
    return _exp_series_from_log_derivatives(derivatives)


@lru_cache(maxsize=None)
def _residue_factor_series(
    m: int, residue: int, index: int
) -> tuple[np.ndarray, float]:
    degree = m - 2
    shape = index / m
    gamma_ratio = _gamma_ratio_series(m, index)

    factor = shape * gamma_ratio.copy()
    if degree:
        factor[1:] += gamma_ratio[:-1]

    for pole in range(1, residue + 1):
        pole_float = float(pole)
        recurrence = np.empty(degree + 1, dtype=float)
        recurrence[0] = 1.0 - shape / pole_float
        if degree:
            powers = np.arange(1, degree + 1, dtype=float)
            recurrence[1:] = -shape / np.power(pole_float, powers + 1.0)
        factor = np.convolve(factor, recurrence)[: degree + 1]

    constant = float(factor[0])
    return factor / constant, math.log(constant)


@lru_cache(maxsize=None)
def residue_polynomial_coefficients(m: int, residue: int) -> np.ndarray:
    """Return ascending coefficients of the residue polynomial ``P_m,r``."""
    if not 2 <= m <= MAX_M or residue < 0:
        raise ValueError("invalid residue polynomial indices")

    degree = m - 2
    normalized = np.array([1.0], dtype=float)
    log_constant = 0.0
    for index in range(1, m):
        factor, factor_log_constant = _residue_factor_series(m, residue, index)
        normalized = np.convolve(normalized, factor)[: degree + 1]
        log_constant += factor_log_constant

    constant = math.exp(log_constant)
    factorial = _factorial(degree)
    return np.asarray(
        [
            constant * normalized[degree - power] / factorial[power]
            for power in range(degree + 1)
        ],
        dtype=float,
    )


def _distributed_residue_value(m: int, residue: int, t: float) -> float:
    degree = m - 2
    if degree == 0:
        return math.exp(_residue_factor_series(m, residue, 1)[1])

    powers = np.arange(degree + 1, dtype=float)
    local_t = t / (m - 1.0)
    exponential = np.power(local_t, powers) / _factorial(degree)
    product = np.array([1.0], dtype=float)
    log_constant = 0.0
    for index in range(1, m):
        factor, factor_log_constant = _residue_factor_series(m, residue, index)
        shifted = np.convolve(factor, exponential)[: degree + 1]
        product = np.convolve(product, shifted)[: degree + 1]
        log_constant += factor_log_constant
    return math.exp(log_constant) * float(product[degree])


def _residue_polynomial_value(m: int, residue: int, t: float) -> float:
    if m >= 35 and t < 12.0:
        return _distributed_residue_value(m, residue, t)
    coefficients = residue_polynomial_coefficients(m, residue)
    return float(np.polynomial.polynomial.polyval(t, coefficients))


def ell_tilde_residue(
    m: int,
    t: float,
    relative_tolerance: float = 5e-14,
    max_residues: int = 200,
) -> float:
    """Return ``exp(t) ell_m(t)`` from factorized float64 residues."""
    if t <= 0.0:
        return ell_tilde_glaser(m, t)

    total = 0.0
    negligible = 0
    for residue in range(max_residues):
        polynomial = _residue_polynomial_value(m, residue, t)
        term = math.exp(-residue * t) * polynomial
        total += term

        scale = max(abs(total), np.finfo(float).tiny)
        if residue > 0 and abs(term) <= relative_tolerance * scale:
            negligible += 1
            if negligible >= 3:
                break
        else:
            negligible = 0
    else:
        raise RuntimeError("residue expansion did not converge")

    if not math.isfinite(total) or total <= 0.0:
        raise ArithmeticError(f"invalid residue density for m={m}, t={t}: {total}")
    return total


def _glaser_is_converged(
    m: int, t: float, tolerance: float = 5e-11
) -> tuple[bool, float]:
    value_180 = ell_tilde_glaser(m, t, 180)
    if not (math.isfinite(value_180) and value_180 > 0.0):
        return False, value_180
    value_160 = ell_tilde_glaser(m, t, 160)
    relative_error = abs(value_180 - value_160) / max(
        abs(value_180), np.finfo(float).tiny
    )
    return relative_error <= tolerance, value_180


def ell_tilde(m: int, t: float) -> float:
    """Return ``exp(t) ell_m(t)`` using the production float64 backend."""
    if t < 0.0:
        return 0.0
    if m > 51:
        return _high_m_ell_tilde(m, t)
    if t <= 4.5:
        return ell_tilde_glaser(m, t, 180)
    if t < 10.0:
        converged, value = _glaser_is_converged(m, t)
        if converged:
            return value
    return ell_tilde_residue(m, t)


def log_ell_tilde(m: int, t: float) -> float:
    """Return ``log(exp(t) ell_m(t))`` without overflow or underflow."""
    if t < 0.0:
        return -math.inf
    if m > 51:
        return _high_m_log_ell_tilde(m, t)
    value = ell_tilde(m, t)
    return math.log(value)


def ell(m: int, t: float) -> float:
    """Return the density ``ell_m(t)``."""
    if t < 0.0:
        return 0.0
    return math.exp(-t) * ell_tilde(m, t)
