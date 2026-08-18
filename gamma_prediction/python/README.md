# gamma-prediction

This package implements the exact finite-sample prediction construction in
the accompanying paper. Public scalar functions select among the
Glaser/residue, packaged high-order density, and contour-inversion backends.
Repeated calculations can use a saved critical-value table, stored on the
`log M(n, p, d)` scale to preserve extreme tails.
Runtime calculations use standard float64 NumPy and SciPy arithmetic. For
`m<=51`, Glaser and factorized residue coefficients are generated online and
lazily cached. For `m=52..201`, the Glaser series is combined with a packaged
offline-generated tilted log-density grid through `t=12000`. This backend uses
no Fourier inversion or extended arithmetic at runtime.

The packaged density-grid backend supports sample sizes `2 <= n <= 200`.
Above 200, or beyond its density-resource range, the public scalar API
automatically uses saddlepoint-centred contour inversion. The explicit
`prediction_log_multiplier_large_n` and `conditional_cdf_large_n_log`
functions are also available for cross-checking and large repeated tasks.
Upper-tail contour inversion evaluates survival directly through the
complementary incomplete beta, avoiding cancellation when the CDF is near one.
A 297-cell overlap check through `n=200`, spanning `d=1e-6..50` and
`p=0.001..0.999`, had no failures and maximum relative multiplier disagreement
below `9e-7`.

## Install

From the research repository root:

```bash
python -m pip install -e ./gamma_prediction/python
```

```python
from gamma_prediction import equal_tail_interval, load_default_table

lower, upper = equal_tail_interval([3, 5, 7, 18, 43], coverage=0.95)
table = load_default_table()
multiplier = table.multiplier_at(100, 0.2, 0.95)
```

For repeated calculations, build a `CriticalValueTable` on a dense grid and
use `exact_fallback=True` outside the grid. The scalar API selects the
appropriate exact numerical backend automatically.
`load_default_table()` loads the included table for every `n=3..50` plus
`55,60,65,70,75,80,90,100`. The package also provides the chunked
`audit_coverage` Monte Carlo helper.

For many probabilities at one `(n,d)`, `prediction_log_multipliers` caches the
quadrature nodes, density values and Jacobian and then performs only weighted
sums during inversion. `refine_critical_value_table` uses this path and can
parallelise across sample sizes.

At every stored sample size through 100, the packaged grid has 50 dispersion
nodes and 54 probability nodes over `d in [1e-6, 50]` and
`p in [0.001, 0.999]`. Fixed-seed
lower and upper validation through `p=0.999` gives maximum relative multiplier
errors of `0.0103%` and `0.0039%` in the main domain; the maximum across six
independent dispersion-boundary strata is `0.0166%`. Inputs beyond the table
boundaries are rejected rather than extrapolated.

The offline high-order resource can be reproduced with:

```bash
python gamma_prediction/tools/generate_high_m_density.py
```

For very large samples, the large-$n$ asymptotic formulas in the paper remain
faster approximations. Contour inversion is retained as the calibrated exact
numerical reference rather than forcing an asymptotic fallback.

## Test

```bash
python -m pytest gamma_prediction/python/tests
```
