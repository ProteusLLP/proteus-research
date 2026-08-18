# Gamma Prediction Intervals

This folder considers the calculation exact frequentist prediction intervals for a two parameter Gamma distribution when
both parameters are unknown.

This directory contains the paper, reproduction code, numerical experiments,
diagnostic implementations, generated tables, and an installable Python
package.

The distributable package is isolated under [`python/`](python/):

```bash
python -m pip install -e ./gamma_prediction/python
```

```python
import gamma_prediction as gp

data = [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487]
upper = gp.prediction_quantile(data, p=0.95)
print(f"95% prediction quantile for {data} is {upper:.6g}")

```

## Reproduce the paper

The default command reproduces deterministic density checks, interval
multipliers, and the air-conditioning example:

```bash
python gamma_prediction/reproduce_paper.py
```

Run a practical fixed-seed Monte Carlo audit with:

```bash
python gamma_prediction/reproduce_paper.py --audit --reps-per-cell 100000
```

The paper's full $10^7$-per-cell, 28-cell audit is intentionally explicit:

```bash
python gamma_prediction/reproduce_paper.py --paper-audit
```

Outputs are written to `gamma_prediction/reproduced_results` by default.

The latest detailed comparison of manuscript values with reproduced results is
recorded in [`NUMERICAL_VERIFICATION.md`](NUMERICAL_VERIFICATION.md).
