"""Exact Gamma prediction intervals with unknown shape and scale."""

from .intervals import (
    AIRCONDITIONING,
    airconditioning_example,
    dispersion,
    equal_tail_interval,
    kmm_quantile,
    prediction_log_multiplier,
    prediction_log_multipliers,
    prediction_multiplier,
    prediction_quantile,
)
from .table import (
    CriticalValueTable,
    build_critical_value_table,
    load_default_table,
    refine_critical_value_table,
)
from .monte_carlo import audit_coverage
from .large_n import (
    conditional_cdf_large_n,
    conditional_cdf_large_n_log,
    conditional_survival_large_n_log,
    prediction_log_multiplier_large_n,
    prediction_multiplier_large_n,
)

__all__ = [
    "AIRCONDITIONING",
    "CriticalValueTable",
    "airconditioning_example",
    "audit_coverage",
    "build_critical_value_table",
    "dispersion",
    "equal_tail_interval",
    "kmm_quantile",
    "load_default_table",
    "prediction_log_multiplier",
    "prediction_log_multiplier_large_n",
    "prediction_log_multipliers",
    "prediction_multiplier",
    "prediction_multiplier_large_n",
    "prediction_quantile",
    "refine_critical_value_table",
    "conditional_cdf_large_n",
    "conditional_cdf_large_n_log",
    "conditional_survival_large_n_log",
]
