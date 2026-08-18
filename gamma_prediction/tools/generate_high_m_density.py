#!/usr/bin/env python3
"""Generate the packaged high-order tilted log-density interpolation grid.

This is an offline reproducibility tool. It uses saddlepoint-tilted pointwise
Fourier inversion to generate reference values. Runtime evaluation only loads
and interpolates the resulting float64 grid; it performs no Fourier inversion.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, loggamma, polygamma


def log_mgf(m: int, value: complex | np.ndarray) -> complex | np.ndarray:
    value = np.asarray(value, dtype=complex)
    return (
        -m * value * np.log(m)
        + gammaln(m)
        + m * loggamma(1.0 - value)
        - loggamma(m * (1.0 - value))
    )


def find_saddlepoint(m: int, t: float) -> float:
    def derivative(value: float) -> float:
        return m * (digamma(m * (1.0 - value)) - digamma(1.0 - value) - np.log(m))

    lower = min(-1.0, 1.0 - (m - 1.0) / max(t, 1e-300))
    while derivative(lower) > t:
        lower *= 2.0
    upper = 1.0 - min(1e-8, 1.0 / (t + 10.0))
    while derivative(upper) < t:
        upper = 1.0 - 0.1 * (1.0 - upper)
    return float(
        brentq(
            lambda value: derivative(value) - t,
            lower,
            upper,
            xtol=1e-13,
            rtol=1e-13,
        )
    )


@lru_cache(maxsize=32)
def quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return leggauss(order)


def corrected_saddlepoint_log_density(m: int, t: float) -> float:
    saddlepoint = find_saddlepoint(m, t)
    shape = 1.0 - saddlepoint
    cumulant = float(
        -saddlepoint * m * np.log(m)
        + gammaln(m)
        + m * gammaln(shape)
        - gammaln(m * shape)
    )
    second = float(m * (polygamma(1, shape) - m * polygamma(1, m * shape)))
    third = float(m * (-polygamma(2, shape) + m**2 * polygamma(2, m * shape)))
    fourth = float(m * (polygamma(3, shape) - m**3 * polygamma(3, m * shape)))
    standardized_third = third / second**1.5
    standardized_fourth = fourth / second**2
    correction = standardized_fourth / 8.0 - 5.0 * standardized_third**2 / 24.0
    base = cumulant - saddlepoint * t - 0.5 * np.log(2.0 * np.pi * second)
    return float(base + np.log1p(correction))


def pointwise_log_density(m: int, t: float) -> float:
    saddlepoint = find_saddlepoint(m, t)
    cumulant = float(np.real(log_mgf(m, complex(saddlepoint, 0.0))))
    second = float(
        m
        * (polygamma(1, 1.0 - saddlepoint) - m * polygamma(1, m * (1.0 - saddlepoint)))
    )
    frequency_scale = 1.0 / np.sqrt(max(second, 1e-30))
    frequency_max = 15.0 * frequency_scale
    order = max(64, int(2.0 * frequency_max * t / np.pi) + 20)
    nodes, weights = quadrature_rule(order)
    frequency = frequency_max * 0.5 * (nodes + 1.0)
    transformed_weights = frequency_max * 0.5 * weights
    values = log_mgf(m, saddlepoint + 1j * frequency)
    integrand = np.real(np.exp(values - cumulant - 1j * frequency * t))
    integral = float(np.sum(integrand * transformed_weights))
    if integral <= 0.0:
        return corrected_saddlepoint_log_density(m, t)
    return float(cumulant - saddlepoint * t + np.log(integral / np.pi))


def generate(output: Path, nodes: int, t_min: float, t_max: float) -> None:
    m_grid = np.arange(52, 202, dtype=int)
    t_grid = np.geomspace(t_min, t_max, nodes)
    log_ell_tilde = np.empty((len(m_grid), len(t_grid)), dtype=float)
    for index, m in enumerate(m_grid):
        print(f"m={m}", flush=True)
        log_ell_tilde[index] = [
            t + pointwise_log_density(int(m), float(t)) for t in t_grid
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        m_grid=m_grid,
        t_grid=t_grid,
        log_ell_tilde=log_ell_tilde,
        generator=np.array("saddlepoint-tilted pointwise Fourier inversion"),
        validation_seed=np.array(20260830),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "gamma_prediction/python/src/gamma_prediction/data/high_m_log_density.npz"
        ),
    )
    parser.add_argument("--nodes", type=int, default=1400)
    parser.add_argument("--t-min", type=float, default=5.5)
    parser.add_argument("--t-max", type=float, default=12000.0)
    args = parser.parse_args()
    generate(args.output, args.nodes, args.t_min, args.t_max)


if __name__ == "__main__":
    main()
