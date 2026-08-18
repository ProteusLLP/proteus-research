#!/usr/bin/env python3
"""
Large-n exact numerical evaluation for the Gamma prediction-interval construction.

This evaluates the exact conditional CDF directly by saddlepoint-centred
inverse Laplace inversion:

    F_n(q | t)
      = L^{-1}[ M_{n+1}(s) I_q(1-s, n(1-s)) ](t)
        / L^{-1}[ M_{n+1}(s) ](t),

where
    M_m(s) = exp(-s m log m) Gamma(m) Gamma(1-s)^m
             / Gamma(m(1-s)).

Only NumPy and SciPy are required.  The complex regularized incomplete beta
is evaluated with a modified-Lentz continued fraction.

The method is intended for moderate/large n.  For small n, the Glaser/residue
series implementation remains preferable.
"""

from __future__ import annotations
import math
import warnings
import numpy as np
from scipy import integrate, optimize, special


def J_from_z(n: int, z: float) -> float:
    """J_n(q) with z = y/xbar and q=z/(n+z)."""
    log_z = math.log(z)
    return (n + 1.0) * (
        float(np.logaddexp(math.log(n), log_z)) - math.log(n + 1.0)
    ) - log_z


def K(m: int, s: complex) -> complex:
    """log M_m(s)."""
    return (
        -s * m * math.log(m)
        + special.gammaln(m)
        + m * special.loggamma(1.0 - s)
        - special.loggamma(m * (1.0 - s))
    )


def Kprime_real(m: int, s: float) -> float:
    return m * (special.digamma(m * (1.0 - s)) - special.digamma(1.0 - s) - math.log(m))


def Ksecond_real(m: int, s: float) -> float:
    return m * (special.polygamma(1, 1.0 - s) - m * special.polygamma(1, m * (1.0 - s)))


def saddlepoint(m: int, t: float) -> float:
    """Solve K'_m(s)=t for s<1."""

    def objective(value: float) -> float:
        return Kprime_real(m, value) - t

    lo = -1.0
    while objective(lo) > 0.0:
        lo *= 2.0
    return optimize.brentq(objective, lo, 1.0 - 1e-10, xtol=1e-13)


def _betacf(
    a: complex, b: complex, x: float, maxiter: int = 5000, tol: float = 3e-14
) -> complex:
    """Continued fraction for the incomplete beta (modified Lentz algorithm)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    tiny = 1e-300

    c = 1.0 + 0.0j
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny + 0.0j
    d = 1.0 / d
    h = d

    for m in range(1, maxiter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny + 0.0j
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny + 0.0j
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny + 0.0j
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny + 0.0j
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < tol:
            return h

    raise RuntimeError("Complex incomplete-beta continued fraction failed.")


def _regularized_incomplete_beta_direct(x: float, a: complex, b: complex) -> complex:
    """Direct continued-fraction evaluation without complementary subtraction."""
    if x <= 0.0:
        return 0.0 + 0.0j
    if x >= 1.0:
        return 1.0 + 0.0j

    log_bt = (
        special.loggamma(a + b)
        - special.loggamma(a)
        - special.loggamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    return np.exp(log_bt) * _betacf(a, b, x) / a


def regularized_incomplete_beta_complex(x: float, a: complex, b: complex) -> complex:
    """I_x(a,b), for complex a,b and real 0<=x<=1."""
    if x > 0.5:
        return 1.0 - _regularized_incomplete_beta_direct(1.0 - x, b, a)
    return _regularized_incomplete_beta_direct(x, a, b)


def regularized_incomplete_beta_complement_complex(
    x: float, a: complex, b: complex
) -> complex:
    """Evaluate ``1-I_x(a,b)`` directly as ``I_(1-x)(b,a)``."""
    return _regularized_incomplete_beta_direct(1.0 - x, b, a)


def conditional_cdf_large_n(
    n: int,
    d: float,
    z: float,
    contour_widths: float = 14.0,
    epsabs: float = 2e-10,
    epsrel: float = 2e-10,
) -> float:
    """Exact conditional CDF at z=y/xbar by saddlepoint-centred inversion."""
    if n < 2 or d <= 0.0 or z <= 0.0:
        raise ValueError("Require n>=2, d>0 and z>0.")
    return conditional_cdf_large_n_log(
        n,
        d,
        math.log(z),
        contour_widths=contour_widths,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def conditional_cdf_large_n_log(
    n: int,
    d: float,
    log_multiplier: float,
    contour_widths: float = 14.0,
    epsabs: float = 2e-10,
    epsrel: float = 2e-10,
) -> float:
    """Exact conditional CDF evaluated directly at ``log(y/xbar)``."""
    if log_multiplier > 0.0:
        survival = conditional_survival_large_n_log(
            n,
            d,
            log_multiplier,
            contour_widths=contour_widths,
            epsabs=epsabs,
            epsrel=epsrel,
        )
        return min(1.0, max(0.0, 1.0 - survival))
    return _conditional_tail_large_n_log(
        n,
        d,
        log_multiplier,
        upper_tail=False,
        contour_widths=contour_widths,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def conditional_survival_large_n_log(
    n: int,
    d: float,
    log_multiplier: float,
    contour_widths: float = 14.0,
    epsabs: float = 2e-10,
    epsrel: float = 2e-10,
) -> float:
    """Exact upper-tail probability evaluated directly on the log scale."""
    return _conditional_tail_large_n_log(
        n,
        d,
        log_multiplier,
        upper_tail=True,
        contour_widths=contour_widths,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def _conditional_tail_large_n_log(
    n: int,
    d: float,
    log_multiplier: float,
    *,
    upper_tail: bool,
    contour_widths: float,
    epsabs: float,
    epsrel: float,
) -> float:
    if n < 2 or d <= 0.0:
        raise ValueError("Require n>=2 and d>0.")

    log_n = math.log(n)
    q = float(np.exp(log_multiplier - np.logaddexp(log_n, log_multiplier)))
    augmentation = (n + 1.0) * (
        float(np.logaddexp(log_n, log_multiplier)) - math.log(n + 1.0)
    ) - log_multiplier
    t = n * d + augmentation
    m = n + 1
    sigma = saddlepoint(m, t)
    k0 = float(np.real(K(m, sigma)))

    def base(omega: float) -> complex:
        s = sigma + 1j * omega
        return np.exp(K(m, s) - k0 - 1j * omega * t)

    def denominator_integrand(omega: float) -> float:
        return float(np.real(base(omega)))

    def numerator_integrand(omega: float) -> float:
        s = sigma + 1j * omega
        beta_factor = (
            regularized_incomplete_beta_complement_complex(q, 1.0 - s, n * (1.0 - s))
            if upper_tail
            else regularized_incomplete_beta_complex(q, 1.0 - s, n * (1.0 - s))
        )
        return float(np.real(base(omega) * beta_factor))

    # The saddlepoint-centred integrand is locally
    # exp{-K''(sigma) omega^2/2}; its natural width is 1/sqrt(K'').
    omega_scale = 1.0 / math.sqrt(Ksecond_real(m, sigma))
    omega_max = contour_widths * omega_scale

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The occurrence of roundoff error is detected.*",
            category=integrate.IntegrationWarning,
        )
        den = integrate.quad(
            denominator_integrand,
            0.0,
            omega_max,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=300,
        )[0]
        num = integrate.quad(
            numerator_integrand,
            0.0,
            omega_max,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=300,
        )[0]

    if not math.isfinite(den) or den <= 0.0:
        raise ArithmeticError("Contour denominator is non-positive or non-finite.")
    value = float(num / den)
    return min(1.0, max(0.0, value))


def prediction_log_multiplier_large_n(n: int, d: float, p: float) -> float:
    """Solve F_n(z)=p and return the log multiplier."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly between zero and one.")

    f1 = conditional_cdf_large_n_log(n, d, 0.0) - p
    if abs(f1) < 1e-12:
        return 0.0

    if f1 < 0.0:
        log_target = math.log1p(-p)
        lower = 0.0
        upper = 0.01
        for _ in range(30):
            survival = conditional_survival_large_n_log(n, d, upper)
            if survival <= 1.0 - p:
                break
            lower = upper
            upper *= 2.0
        else:
            raise RuntimeError("Could not bracket upper prediction multiplier.")

        def objective(log_multiplier: float) -> float:
            survival = max(
                conditional_survival_large_n_log(n, d, log_multiplier),
                np.finfo(float).tiny,
            )
            return math.log(survival) - log_target
    else:
        upper = 0.0
        lower = -1.0
        for _ in range(20):
            if conditional_cdf_large_n_log(n, d, lower) <= p:
                break
            upper = lower
            lower *= 2.0
        else:
            raise RuntimeError("Could not bracket lower prediction multiplier.")

        def objective(log_multiplier: float) -> float:
            return conditional_cdf_large_n_log(n, d, log_multiplier) - p

    return float(
        optimize.brentq(
            objective,
            lower,
            upper,
            xtol=2e-11,
        )
    )


def prediction_multiplier_large_n(n: int, d: float, p: float) -> float:
    """Solve F_n(z)=p for the multiplier z=Y/xbar."""
    return math.exp(prediction_log_multiplier_large_n(n, d, p))


if __name__ == "__main__":
    for n in (50, 100, 200, 1000, 10000, 100000):
        z = prediction_multiplier_large_n(n, 0.2, 0.95)
        print(f"n={n:6d}  d=0.2  p=.95  multiplier={z:.12f}")
