#!/usr/bin/env python3
"""
Exact frequentist prediction intervals for a two-parameter Gamma distribution.

This is the reference implementation accompanying the manuscript
"Exact Frequentist Prediction Intervals for a Gamma Distribution with Unknown
Shape and Scale".

It implements, directly from the paper:
  * the Glaser-type near-origin series for ell_m;
  * the exponential-polynomial residue expansion for ell_m;
  * the exponentially tilted branch integrals;
  * inversion on log(y / xbar) to obtain exact prediction multipliers;
  * the tabulated lower/upper critical values;
  * the Boeing 720 air-conditioning example, including the
    Krishnamoorthy-Mathew-Mukherjee cube-root approximation;
  * optional numerical checks and Monte Carlo audit helpers.

Runtime dependencies: numpy and scipy. Low-order density coefficients are
generated online in cached float64 arithmetic; high-order densities use a
packaged offline-generated log-density grid. No Fourier inversion or extended
arithmetic is used at runtime.

The implementation deliberately follows the notation of the paper.  The
single non-standard density is ell_m(t), the density of
    T_m = -log(m^m prod_i P_i),   P ~ Dirichlet_m(1,...,1).
For numerical work we use ell_tilde_m(t) = exp(t) ell_m(t).
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import integrate, optimize

from .density import (
    ell,
    ell_tilde,
    log_ell_tilde,
    ell_tilde_glaser,
    ell_tilde_residue,
    glaser_coefficients as _glaser_coefficients,
    residue_polynomial_coefficients as _residue_polynomial_coefficients,
)


# ---------------------------------------------------------------------------
# 1. Basic transformations
# ---------------------------------------------------------------------------


def J_from_w(n: int, w: float) -> float:
    """J_n(q) on the log-multiplier scale w = log(y / xbar).

    q = exp(w) / (n + exp(w)), and
    J_n(q) = (n+1) log((n+exp(w))/(n+1)) - w.

    The intermediate coordinate v=(n+1)q-1 is evaluated with expm1, then the
    cancellation-free J(v) expansion is used near the mode.
    """
    if w == 0.0:
        return 0.0
    if abs(w) >= 0.5:
        return float((n + 1.0) * (np.logaddexp(math.log(n), w) - math.log(n + 1.0)) - w)
    exp_w = math.exp(w)
    v = n * math.expm1(w) / (n + exp_w)
    if v <= -1.0:
        return float((n + 1.0) * math.log(n / (n + 1.0)) - w)
    if v >= n:
        return float(n * w - (n + 1.0) * math.log(n + 1.0))
    return _J_from_v(n, v)


def q_from_w(n: int, w: float) -> float:
    """q = y/(S_n+y) from w = log(y/xbar), evaluated stably."""
    x = w - math.log(n)
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def delta_from_w(n: int, w: float) -> float:
    """|(n+1)q-1|, also equal to |dJ_n/dw|."""
    return abs((n + 1) * q_from_w(n, w) - 1.0)


def _J_from_v(n: int, v: float) -> float:
    """J_n from v=(n+1)q-1, without cancellation near v=0."""
    if abs(v) < 0.1:
        term = v * v
        total = 0.5 * (1.0 + 1.0 / n) * term
        for k in range(3, 24):
            term *= v
            total += (((-1.0) ** k) + n ** (1 - k)) * term / k
        return float(total)
    return float(-math.log1p(v) - n * math.log1p(-v / n))


def dispersion(x: Sequence[float]) -> float:
    """D_n = log(xbar) - mean(log X_i)."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("All observations must be strictly positive.")
    return float(math.log(float(np.mean(x))) - float(np.mean(np.log(x))))


# ---------------------------------------------------------------------------
# 4. Branch roots and exact conditional quantile level G_{n,d}
# ---------------------------------------------------------------------------


def _branch_w_from_a(n: int, a: float, branch: str) -> float:
    """Solve J_n(q(w)) = a on the lower (w<0) or upper (w>0) branch."""
    if a < 0:
        raise ValueError("a must be non-negative")
    if a == 0.0:
        return 0.0

    if branch == "lower":
        if a > 30.0:
            return float((n + 1.0) * math.log(n / (n + 1.0)) - a)
        v = -optimize.brentq(
            lambda h: _J_from_v(n, -h) - a,
            0.0,
            np.nextafter(1.0, 0.0),
            xtol=np.nextafter(0.0, 1.0),
            rtol=4 * np.finfo(float).eps,
        )
    elif branch == "upper":
        if a > 30.0 * n:
            return float((a + (n + 1.0) * math.log(n + 1.0)) / n)
        v = optimize.brentq(
            lambda h: _J_from_v(n, h) - a,
            0.0,
            np.nextafter(float(n), 0.0),
            xtol=np.nextafter(0.0, 1.0),
            rtol=4 * np.finfo(float).eps,
        )
    else:
        raise ValueError("branch must be 'lower' or 'upper'")

    return float(math.log(n) + math.log1p(v) - math.log(n - v))


def _delta_branch(n: int, a: float, branch: str) -> float:
    w = _branch_w_from_a(n, a, branch)
    return delta_from_w(n, w)


def _branch_integral(
    n: int,
    s: float,
    a: float,
    branch: str,
    log_denominator: float = 0.0,
) -> float:
    """Integral in the tilted lower/upper branch formula.

    Uses r = s sin^2(phi), 0 < phi < pi/2.
    """
    if s < 0 or a < 0:
        raise ValueError
    if s == 0.0:
        return 0.0

    def integrand(phi: float) -> float:
        sp = math.sin(phi)
        cp = math.cos(phi)
        r = s * sp * sp
        t = s - r  # = s cos^2(phi)
        delta = _delta_branch(n, a + r, branch)
        jac = 2.0 * s * sp * cp
        density_ratio = math.exp(log_ell_tilde(n, t) - log_denominator)
        return density_ratio * jac / delta

    # QUADPACK's Gauss-Kronrod nodes lie in the interior; the transformation
    # makes the endpoint limits finite even for n=2 or a=0.
    value, err = integrate.quad(
        integrand,
        0.0,
        math.pi / 2.0,
        epsabs=2e-13,
        epsrel=2e-11,
        limit=160,
    )
    return float(value)


def branch_probability(n: int, d: float, w: float) -> float:
    """G_{n,d}(q(w)) for w<=0, or its upper-branch complement for w>0.

    This is the exact conditional quantile level used to invert prediction
    limits.  w = log(y / xbar).
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if d <= 0:
        raise ValueError("d must be strictly positive")

    s = n * d
    a = J_from_w(n, w)
    Cn = (n / (n + 1)) ** (n + 1)
    log_den = log_ell_tilde(n + 1, s + a)

    if w <= 0.0:
        scaled_num = _branch_integral(n, s, a, "lower", log_den)
        p = Cn * scaled_num
        return min(1.0, max(0.0, p))

    scaled_num = _branch_integral(n, s, a, "upper", log_den)
    surv = Cn * scaled_num
    p = 1.0 - surv
    return min(1.0, max(0.0, p))


def _prediction_log_multiplier_density(n: int, d: float, p: float) -> float:
    """Density-grid branch inversion for ``n <= 200``."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0,1)")
    if d <= 0:
        raise ValueError("d must be strictly positive")
    if n > 200:
        raise ValueError("density-grid scalar evaluation supports n <= 200")

    p0 = branch_probability(n, d, 0.0)

    if abs(p - p0) < 5e-15:
        return 0.0

    if p < p0:
        log_target = math.log(p)

        def f(w):
            val = branch_probability(n, d, w)
            return math.log(max(val, np.finfo(float).tiny)) - log_target

        hi = 0.0
        lo = -1.0
        while f(lo) > 0:
            lo *= 2.0
            if lo < -2e4:
                raise RuntimeError("Could not bracket lower prediction quantile")
        return float(
            optimize.brentq(
                f,
                lo,
                hi,
                xtol=1e-14,
                rtol=2e-13,
                maxiter=100,
            )
        )

    # Upper side: invert survival probability directly.
    log_target = math.log1p(-p)

    def f(w):
        val = branch_probability(n, d, w)
        surv = max(1.0 - val, np.finfo(float).tiny)
        return math.log(surv) - log_target

    lo = 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 2e4:
            raise RuntimeError("Could not bracket upper prediction quantile")
    return float(
        optimize.brentq(
            f,
            lo,
            hi,
            xtol=1e-14,
            rtol=2e-13,
            maxiter=100,
        )
    )


def prediction_log_multiplier(n: int, d: float, p: float) -> float:
    """Return log M_{n,p}(d), selecting a stable exact numerical backend."""
    if n > 200:
        from .large_n import prediction_log_multiplier_large_n

        return prediction_log_multiplier_large_n(n, d, p)
    try:
        return _prediction_log_multiplier_density(n, d, p)
    except ValueError as error:
        if "high-order density evaluation supports t" not in str(error):
            raise
        from .large_n import prediction_log_multiplier_large_n

        return prediction_log_multiplier_large_n(n, d, p)


@lru_cache(maxsize=None)
def _prediction_quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (math.pi / 4.0) * (nodes + 1.0), (math.pi / 4.0) * weights


@dataclass
class PredictionQuantileWorkspace:
    """Shared fixed-quadrature workspace for many probabilities at one ``n,d``.

    The conditioning value still changes with each candidate quantile, but the
    quadrature nodes, density values, and Jacobian depend only on ``n,d`` and
    are therefore cached here.
    """

    n: int
    d: float
    quadrature_order: int | None = None

    def __post_init__(self) -> None:
        if self.n < 2 or self.n > 200:
            raise ValueError("shared density-grid evaluation supports 2 <= n <= 200")
        if self.d <= 0.0:
            raise ValueError("d must be strictly positive")
        if self.quadrature_order is None:
            self.quadrature_order = 320 if self.n > 50 else 160
        if self.quadrature_order < 32:
            raise ValueError("quadrature_order must be at least 32")

        phi, weights = _prediction_quadrature_rule(self.quadrature_order)
        sine = np.sin(phi)
        cosine = np.cos(phi)
        self._s = self.n * self.d
        self._r = self._s * sine * sine
        t = self._s * cosine * cosine
        jacobian = 2.0 * self._s * sine * cosine
        log_density = np.asarray([log_ell_tilde(self.n, float(value)) for value in t])
        self._log_weighted_density = np.log(weights) + log_density + np.log(jacobian)
        self._constant = (self.n / (self.n + 1.0)) ** (self.n + 1)

    def probability(self, w: float) -> float:
        """Evaluate the conditional quantile level at log multiplier ``w``."""
        a = J_from_w(self.n, float(w))
        branch = "lower" if w <= 0.0 else "upper"
        log_denominator = log_ell_tilde(self.n + 1, self._s + a)
        delta = _branch_delta_array(self.n, a + self._r, branch)
        tail = self._constant * float(
            np.sum(np.exp(self._log_weighted_density - log_denominator) / delta)
        )
        value = tail if branch == "lower" else 1.0 - tail
        return min(1.0, max(0.0, value))

    def log_multipliers(self, probabilities: Sequence[float]) -> np.ndarray:
        """Invert all requested probabilities using the shared quadrature."""
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
            raise ValueError("probabilities must lie in (0,1)")

        result = np.empty_like(probabilities)
        p0 = self.probability(0.0)
        for index, p in np.ndenumerate(probabilities):
            if abs(float(p) - p0) < 5e-15:
                result[index] = 0.0
                continue
            if p < p0:
                target = math.log(float(p))

                def objective(w: float) -> float:
                    return (
                        math.log(max(self.probability(w), np.finfo(float).tiny))
                        - target
                    )

                lo, hi = -1.0, 0.0
                while objective(lo) > 0.0:
                    lo *= 2.0
                    if lo < -2e4:
                        raise RuntimeError(
                            "Could not bracket lower prediction quantile"
                        )
            else:
                target = math.log1p(-float(p))

                def objective(w: float) -> float:
                    survival = max(1.0 - self.probability(w), np.finfo(float).tiny)
                    return math.log(survival) - target

                lo, hi = 0.0, 1.0
                while objective(hi) > 0.0:
                    hi *= 2.0
                    if hi > 2e4:
                        raise RuntimeError(
                            "Could not bracket upper prediction quantile"
                        )
            result[index] = optimize.brentq(
                objective,
                lo,
                hi,
                xtol=1e-14,
                rtol=2e-13,
                maxiter=100,
            )
        return result


def prediction_log_multipliers(
    n: int,
    d: float,
    probabilities: Sequence[float],
    *,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Return many exact log multipliers with one shared quadrature workspace."""
    if n > 200:
        from .large_n import prediction_log_multiplier_large_n

        return np.asarray(
            [prediction_log_multiplier_large_n(n, d, p) for p in probabilities],
            dtype=float,
        )
    return PredictionQuantileWorkspace(n, d, quadrature_order).log_multipliers(
        probabilities
    )


def prediction_multiplier(n: int, d: float, p: float) -> float:
    """M_{n,p}(d)."""
    w = prediction_log_multiplier(n, d, p)
    return math.exp(w)


def prediction_quantile(x: Sequence[float], p: float) -> float:
    """Exact p-th prediction quantile for a future Gamma observation."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) * prediction_multiplier(len(x), dispersion(x), p)


def equal_tail_interval(x: Sequence[float], coverage: float = 0.95):
    """Exact equal-tail prediction interval for one future observation."""
    gamma = 1.0 - coverage
    return (
        prediction_quantile(x, gamma / 2.0),
        prediction_quantile(x, 1.0 - gamma / 2.0),
    )


# ---------------------------------------------------------------------------
# 5. Reproduction of manuscript tables and worked example
# ---------------------------------------------------------------------------

PAPER_N = (3, 5, 10, 20, 50)
PAPER_D = (0.05, 0.25, 1.00)
PAPER_P = (0.05, 0.95, 0.025, 0.975, 0.005, 0.995)


def paper_multiplier_table():
    """Reproduce the exact multiplier table in the manuscript."""
    rows = []
    for n in PAPER_N:
        for d in PAPER_D:
            vals = {p: prediction_multiplier(n, d, p) for p in PAPER_P}
            rows.append(
                {
                    "n": n,
                    "d": d,
                    "M_.05": vals[0.05],
                    "M_.95": vals[0.95],
                    "M_.025": vals[0.025],
                    "M_.975": vals[0.975],
                    "M_.005": vals[0.005],
                    "M_.995": vals[0.995],
                }
            )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


AIRCONDITIONING = np.array(
    [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487], dtype=float
)
AIR_P = np.array([0.005, 0.025, 0.050, 0.500, 0.950, 0.975, 0.995])


def kmm_quantile(x: Sequence[float], p: float) -> float:
    """Krishnamoorthy-Mathew-Mukherjee cube-root prediction approximation."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    w = np.cbrt(x)
    n = len(x)
    loc = float(np.mean(w))
    scale = float(np.std(w, ddof=1)) * math.sqrt(1.0 + 1.0 / n)
    z = loc + stats.t.ppf(p, df=n - 1) * scale
    return max(0.0, float(z)) ** 3


def airconditioning_example():
    """Reproduce the manuscript's Boeing 720 air-conditioning table."""
    from scipy import stats

    x = AIRCONDITIONING
    n = len(x)
    xbar = float(np.mean(x))
    d = dispersion(x)

    # Gamma MLE with location fixed at zero.
    alpha_hat, loc_hat, theta_hat = stats.gamma.fit(x, floc=0.0)

    rows = []
    for p in AIR_P:
        rows.append(
            {
                "p": p,
                "Exact": xbar * prediction_multiplier(n, d, float(p)),
                "KMM": kmm_quantile(x, float(p)),
                "Fitted Gamma": stats.gamma.ppf(
                    p, a=alpha_hat, loc=0.0, scale=theta_hat
                ),
            }
        )

    summary = {
        "n": n,
        "xbar": xbar,
        "D_n": d,
        "alpha_hat": float(alpha_hat),
        "theta_hat": float(theta_hat),
    }
    try:
        import pandas as pd

        return summary, pd.DataFrame(rows)
    except ImportError:
        return summary, rows


# ---------------------------------------------------------------------------
# 6. Numerical validation helpers
# ---------------------------------------------------------------------------


def check_series_overlap(m_values=(2, 3, 5, 10, 20, 51), t_values=(4.0, 4.5, 5.0)):
    """Compare the two exact density representations in their overlap."""
    rows = []
    for m in m_values:
        for t in t_values:
            g = ell_tilde_glaser(m, t)
            r = ell_tilde_residue(m, t)
            rows.append((m, t, g, r, (r / g - 1.0)))
    return rows


def audit_small(
    n_values=(2, 3, 5),
    alpha_values=(0.25, 1.0, 5.0),
    reps=250,
    seed=20260817,
):
    """A deliberately small, direct audit of U=G_{n,D_n}(Q).

    This calls the exact quadrature for every replication and is intended as
    a transparent correctness check, not as the production 24-million-run
    audit.  For the large audit, use an interpolated/compiled driver around
    the same branch_probability() routine.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        for alpha in alpha_values:
            U = np.empty(reps)
            for j in range(reps):
                z = rng.gamma(shape=alpha, scale=1.0, size=n + 1)
                x, y = z[:n], float(z[n])
                d = dispersion(x)
                w = math.log(y / float(np.mean(x)))
                U[j] = branch_probability(n, d, w)
            rows.append(
                {
                    "n": n,
                    "alpha": alpha,
                    "reps": reps,
                    "mean_U": float(np.mean(U)),
                    "var_U": float(np.var(U)),
                    "P(U<=.05)": float(np.mean(U <= 0.05)),
                    "P(U>=.95)": float(np.mean(U >= 0.95)),
                }
            )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


# ---------------------------------------------------------------------------
# 6b. Accelerated lookup for large Monte Carlo audits
# ---------------------------------------------------------------------------


def _branch_delta_array(n: int, a: np.ndarray, branch: str) -> np.ndarray:
    """Vectorised Newton solution for delta_+(a) or delta_-(a).

    Used only to build the Monte Carlo lookup surface.  The scalar production
    prediction calculation above continues to use bracketed root finding.
    """
    a = np.asarray(a, dtype=float)
    small = a < 0.75
    w = np.empty_like(a)
    root0 = np.sqrt(np.maximum(0.0, 2.0 * a * (n + 1.0) / n))
    if branch == "lower":
        c = (n + 1.0) * math.log(n / (n + 1.0))
        w[:] = c - a
        w[small] = -root0[small]
    else:
        w[:] = (a + (n + 1.0) * math.log(n + 1.0)) / n
        w[small] = root0[small]

    mask = a > 0.0
    logn = math.log(n)
    logn1 = math.log(n + 1)
    for _ in range(14):
        wm = w[mask]
        q = 1.0 / (1.0 + np.exp(np.clip(logn - wm, -700.0, 700.0)))
        J = (n + 1.0) * (np.logaddexp(logn, wm) - logn1) - wm
        der = (n + 1.0) * q - 1.0
        step = (J - a[mask]) / der
        new = wm - step
        if branch == "lower":
            bad = new >= 0.0
            new[bad] = 0.5 * wm[bad]
        else:
            bad = new <= 0.0
            new[bad] = 0.5 * wm[bad]
        w[mask] = new

    # exact a=0 is a branch point; delta=0 there.
    q = 1.0 / (1.0 + np.exp(np.clip(logn - w, -700.0, 700.0)))
    delta = np.abs((n + 1.0) * q - 1.0)
    delta[~mask] = 0.0
    return delta


def _ell_tilde_vector(m: int, t: np.ndarray, switch: float = 4.5) -> np.ndarray:
    """Vectorised evaluator used to construct lookup tables."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t >= 0.0
    central = pos & (t <= switch)
    if np.any(central):
        tc = t[central]
        nu, A, v = _glaser_coefficients(m, 180)
        vals = (
            A
            * np.power(tc, nu - 1.0, where=tc > 0, out=np.zeros_like(tc))
            * np.polynomial.polynomial.polyval(tc, v)
        )
        if np.any(tc == 0.0):
            z = tc == 0.0
            if m == 2:
                vals[z] = np.inf
            elif m == 3:
                vals[z] = A
            else:
                vals[z] = 0.0
        out[central] = vals

    tail = pos & (t > switch)
    if np.any(tail):
        tt = t[tail]
        total = np.zeros_like(tt)
        small_count = 0
        for r in range(60):
            coeff = _residue_polynomial_coefficients(m, r)
            term = np.exp(-r * tt) * np.polynomial.polynomial.polyval(tt, coeff)
            total += term
            if r > 0 and np.max(np.abs(term)) <= 2e-15 * max(
                1.0, float(np.max(np.abs(total)))
            ):
                small_count += 1
                if small_count >= 3:
                    break
            else:
                small_count = 0
        out[tail] = total
    return out


@dataclass
class ConditionalLookup:
    """Interpolated F_n(q|t) surface for fast Monte Carlo auditing.

    Coordinates are u=log(1+t) and phi=asin(sqrt(J_n(q)/t)).  Two surfaces
    are stored: lower-branch CDF and upper-branch survival probability.
    """

    n: int
    logt_grid: np.ndarray
    phi_grid: np.ndarray
    lower: np.ndarray
    upper_surv: np.ndarray

    def evaluate(self, t: np.ndarray, w: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

        t = np.asarray(t, dtype=float)
        w = np.asarray(w, dtype=float)
        a = (self.n + 1.0) * (
            np.logaddexp(math.log(self.n), w) - math.log(self.n + 1.0)
        ) - w
        ratio = np.clip(a / t, 0.0, 1.0)
        phi = np.arcsin(np.sqrt(ratio))
        pts = np.column_stack((np.log(t), phi))

        lo_int = RegularGridInterpolator(
            (self.logt_grid, self.phi_grid),
            self.lower,
            bounds_error=False,
            fill_value=np.nan,
        )
        up_int = RegularGridInterpolator(
            (self.logt_grid, self.phi_grid),
            self.upper_surv,
            bounds_error=False,
            fill_value=np.nan,
        )
        lower_mask = w <= 0.0
        U = np.empty_like(t)
        if np.any(lower_mask):
            U[lower_mask] = lo_int(pts[lower_mask])
        if np.any(~lower_mask):
            U[~lower_mask] = 1.0 - up_int(pts[~lower_mask])
        return U


def build_conditional_lookup(
    n: int,
    t_min: float = 1e-7,
    t_max: float = 700.0,
    n_t: int = 500,
    n_phi: int = 801,
) -> ConditionalLookup:
    """Build a fast conditional-CDF lookup surface for one sample size.

    This is used only for the large Monte Carlo audit.  Increase n_t and
    n_phi if you want a tighter interpolation audit.
    """
    if n_phi < 51 or n_t < 20:
        raise ValueError("Lookup grid is too coarse")

    logt_grid = np.linspace(math.log(t_min), math.log(t_max), n_t)
    t_grid = np.exp(logt_grid)
    phi = np.linspace(0.0, math.pi / 2.0, n_phi)
    sin2 = np.sin(phi) ** 2
    jac_base = np.sin(phi) * np.cos(phi)

    lower = np.empty((n_t, n_phi), dtype=float)
    upper_surv = np.empty_like(lower)
    Cn = (n / (n + 1.0)) ** (n + 1)

    for it, t in enumerate(t_grid):
        a = t * sin2
        tau = t - a
        et = _ell_tilde_vector(n, tau)
        den = ell_tilde(n + 1, float(t))
        jac = 2.0 * t * jac_base

        dminus = _branch_delta_array(n, a, "lower")
        dplus = _branch_delta_array(n, a, "upper")

        base_m = np.empty_like(phi)
        base_p = np.empty_like(phi)
        base_m[1:-1] = et[1:-1] * jac[1:-1] / dminus[1:-1]
        base_p[1:-1] = et[1:-1] * jac[1:-1] / dplus[1:-1]

        # phi -> 0 limit: delta ~ sqrt(2 n a/(n+1)).
        lim0 = ell_tilde(n, float(t)) * math.sqrt(2.0 * t * (n + 1.0) / n)
        base_m[0] = lim0
        base_p[0] = lim0

        # phi -> pi/2.  Only n=2 has a non-zero limit because ell_2(tau)
        # behaves like 1/(2 sqrt(tau)).
        if n == 2:
            base_m[-1] = math.sqrt(t) / _delta_branch(n, float(t), "lower")
            base_p[-1] = math.sqrt(t) / _delta_branch(n, float(t), "upper")
        else:
            base_m[-1] = 0.0
            base_p[-1] = 0.0

        cm = integrate.cumulative_trapezoid(base_m, phi, initial=0.0)
        cp = integrate.cumulative_trapezoid(base_p, phi, initial=0.0)
        lower[it, :] = Cn * (cm[-1] - cm) / den
        upper_surv[it, :] = Cn * (cp[-1] - cp) / den

        # Clip only interpolation-scale numerical noise.
        lower[it, :] = np.clip(lower[it, :], 0.0, 1.0)
        upper_surv[it, :] = np.clip(upper_surv[it, :], 0.0, 1.0)

    return ConditionalLookup(n, logt_grid, phi, lower, upper_surv)


def validate_lookup(lookup: ConditionalLookup, n_checks: int = 50, seed: int = 1234):
    """Compare the lookup with direct adaptive quadrature at random points."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(lookup.logt_grid[3], lookup.logt_grid[-3], size=n_checks)
    t = np.exp(u)
    # Draw phi away from exact branch/support boundaries.
    phi = rng.uniform(0.02, math.pi / 2 - 0.02, size=n_checks)
    a = t * np.sin(phi) ** 2
    lower = rng.random(n_checks) < 0.5
    w = np.empty(n_checks)
    for i in range(n_checks):
        w[i] = _branch_w_from_a(
            n=lookup.n, a=float(a[i]), branch="lower" if lower[i] else "upper"
        )
    approx = lookup.evaluate(t, w)
    exact = np.empty(n_checks)
    for i in range(n_checks):
        # For an actual augmented point, s=t-a and d=s/n.
        exact[i] = branch_probability(
            lookup.n, float((t[i] - a[i]) / lookup.n), float(w[i])
        )
    err = approx - exact
    return {
        "max_abs": float(np.max(np.abs(err))),
        "p95_abs": float(np.quantile(np.abs(err), 0.95)),
        "mean_abs": float(np.mean(np.abs(err))),
    }


def audit_large(
    n_values=(2, 3, 5, 10, 20, 30),
    alpha_values=(0.25, 1.0, 5.0, 20.0),
    reps_per_cell: int = 1_000_000,
    batch_size: int = 100_000,
    seed: int = 20260817,
    lookup_n_t: int = 500,
    lookup_n_phi: int = 901,
    t_max: float = 700.0,
    thresholds=(0.01, 0.025, 0.05, 0.10),
):
    """Accelerated version of the manuscript's 24-million-run audit.

    The expensive conditional distribution is tabulated once for each n and
    then bilinearly interpolated.  Samples falling outside the lookup range
    are evaluated by direct quadrature (normally extremely rare).
    """
    rng = np.random.default_rng(seed)
    thresholds = np.asarray(thresholds, dtype=float)
    if np.any((thresholds <= 0.0) | (thresholds >= 0.5)):
        raise ValueError("audit thresholds must lie in (0, 0.5)")
    cell_rows = []
    all_sum = 0.0
    all_sumsq = 0.0
    all_n = 0
    aggregate_lower = np.zeros(len(thresholds), dtype=np.int64)
    aggregate_upper = np.zeros(len(thresholds), dtype=np.int64)

    for n in n_values:
        lookup = build_conditional_lookup(
            n, t_max=t_max, n_t=lookup_n_t, n_phi=lookup_n_phi
        )
        for alpha in alpha_values:
            count = 0
            sum_u = 0.0
            sumsq_u = 0.0
            lower_counts = np.zeros(len(thresholds), dtype=np.int64)
            upper_counts = np.zeros(len(thresholds), dtype=np.int64)

            while count < reps_per_cell:
                b = min(batch_size, reps_per_cell - count)
                z = rng.gamma(shape=alpha, scale=1.0, size=(b, n + 1))
                z = np.maximum(z, np.finfo(float).tiny)
                total = np.sum(z, axis=1)
                xsum = np.sum(z[:, :-1], axis=1)
                xbar = xsum / n
                w = np.log(z[:, -1]) - np.log(xbar)
                # T_{n+1} = (n+1)log(S/(n+1)) - sum log X_i
                t = (n + 1.0) * np.log(total / (n + 1.0)) - np.sum(np.log(z), axis=1)

                U = lookup.evaluate(t, w)
                bad = ~np.isfinite(U)
                if np.any(bad):
                    # Rare lookup-range misses: revert to the exact scalar calculation.
                    for i in np.where(bad)[0]:
                        # s = T_n(R) = t - J(w); d=s/n.
                        a = J_from_w(n, float(w[i]))
                        d = max((float(t[i]) - a) / n, 1e-15)
                        U[i] = branch_probability(n, d, float(w[i]))

                sum_u += float(np.sum(U))
                sumsq_u += float(np.dot(U, U))
                for j, a0 in enumerate(thresholds):
                    lower_counts[j] += int(np.count_nonzero(U <= a0))
                    upper_counts[j] += int(np.count_nonzero(U >= 1.0 - a0))
                count += b

            cell_rows.append(
                {
                    "n": n,
                    "alpha": alpha,
                    "reps": count,
                    "mean_U": sum_u / count,
                    "var_U": sumsq_u / count - (sum_u / count) ** 2,
                    **{
                        f"lower_{a0:g}": lower_counts[j] / count
                        for j, a0 in enumerate(thresholds)
                    },
                    **{
                        f"upper_{a0:g}": upper_counts[j] / count
                        for j, a0 in enumerate(thresholds)
                    },
                }
            )
            all_sum += sum_u
            all_sumsq += sumsq_u
            all_n += count
            aggregate_lower += lower_counts
            aggregate_upper += upper_counts

    aggregate = {
        "N": all_n,
        "mean_U": all_sum / all_n,
        "var_U": all_sumsq / all_n - (all_sum / all_n) ** 2,
        "thresholds": thresholds,
        "lower": aggregate_lower / all_n,
        "upper": aggregate_upper / all_n,
        "central": 1.0 - (aggregate_lower + aggregate_upper) / all_n,
    }
    try:
        import pandas as pd

        cells = pd.DataFrame(cell_rows)
    except ImportError:
        cells = cell_rows
    return aggregate, cells


# ---------------------------------------------------------------------------
# 7. Command-line driver
# ---------------------------------------------------------------------------


def _print_table(obj, float_format="{:.8g}"):
    if hasattr(obj, "to_string"):
        print(obj.to_string(index=False, float_format=lambda x: float_format.format(x)))
    else:
        for row in obj:
            print(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        default="example",
        choices=(
            "example",
            "table",
            "overlap",
            "audit-small",
            "audit-large",
            "lookup-check",
            "quantile",
        ),
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--d", type=float, default=0.25)
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)

    if args.task == "example":
        summary, tab = airconditioning_example()
        print("Air-conditioning summary")
        for k, v in summary.items():
            print(f"  {k}: {v:.9g}" if isinstance(v, float) else f"  {k}: {v}")
        print("\nPrediction quantiles")
        _print_table(tab)
        return

    if args.task == "table":
        _print_table(paper_multiplier_table())
        return

    if args.task == "overlap":
        print("m  t   glaser_tilde   residue_tilde   relative_difference")
        for row in check_series_overlap():
            print(
                f"{row[0]:2d} {row[1]:3.1f} {row[2]:.12g} {row[3]:.12g} {row[4]: .3e}"
            )
        return

    if args.task == "audit-small":
        _print_table(
            audit_small(reps=250 if args.reps is None else args.reps, seed=args.seed)
        )
        return

    if args.task == "lookup-check":
        lookup = build_conditional_lookup(args.n, n_t=160, n_phi=501)
        print(validate_lookup(lookup, n_checks=30))
        return

    if args.task == "audit-large":
        agg, cells = audit_large(
            reps_per_cell=1_000_000 if args.reps is None else args.reps,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print("Aggregate:")
        print(agg)
        print("\nCells:")
        _print_table(cells)
        return

    if args.task == "quantile":
        w = prediction_log_multiplier(args.n, args.d, args.p)
        print(f"log M_{{n,p}}(d) = {w:.15g}")
        print(f"M_{{n,p}}(d)     = {math.exp(w):.15g}")
        return


if __name__ == "__main__":
    main()
