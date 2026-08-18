"""Chunked Monte Carlo validation for table-based prediction intervals."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .table import CriticalValueTable


def audit_coverage(
    table: CriticalValueTable,
    *,
    n_values: Iterable[int] = (3, 5, 10, 20, 50),
    alpha_values: Iterable[float] = (0.25, 1.0, 5.0, 20.0),
    p_values: Iterable[float] = (0.005, 0.025, 0.05, 0.95, 0.975, 0.995),
    replications: int = 100_000,
    batch_size: int = 10_000,
    seed: int = 20260817,
    exact_fallback: bool = True,
) -> list[dict[str, float]]:
    """Estimate coverage of table-based one-sided prediction limits."""
    if replications < 1 or batch_size < 1:
        raise ValueError("replications and batch_size must be positive.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for n in n_values:
        for alpha in alpha_values:
            for p in p_values:
                covered = 0
                total = 0
                while total < replications:
                    size = min(batch_size, replications - total)
                    sample = rng.gamma(alpha, 1.0, size=(size, n + 1))
                    observed = sample[:, :n]
                    future = sample[:, n]
                    xbar = np.mean(observed, axis=1)
                    d = np.log(xbar) - np.mean(np.log(observed), axis=1)
                    log_limits = table.log_multiplier_array(
                        n,
                        d,
                        p,
                        exact_fallback=exact_fallback,
                    )
                    limits = xbar * np.exp(log_limits)
                    if p < 0.5:
                        covered += int(np.count_nonzero(future >= limits))
                        nominal = 1.0 - p
                    else:
                        covered += int(np.count_nonzero(future <= limits))
                        nominal = p
                    total += size

                estimate = covered / total
                standard_error = np.sqrt(max(estimate * (1.0 - estimate), 0.0) / total)
                rows.append(
                    {
                        "n": float(n),
                        "alpha": float(alpha),
                        "p": float(p),
                        "nominal": nominal,
                        "coverage": estimate,
                        "mc_se": standard_error,
                        "z": (estimate - nominal) / standard_error
                        if standard_error > 0.0
                        else 0.0,
                    }
                )
    return rows
