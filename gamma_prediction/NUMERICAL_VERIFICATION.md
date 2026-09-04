# Numerical verification of the final paper

Date checked: 2026-08-18

Paper: `paper/gamma_prediction_interval_final.tex`

Implementation: `python/src/gamma_prediction`

Reproduction driver: `reproduce_paper.py`

## Confirmed directly

- All 90 displayed multiplier entries in Table 1 agree with current exact
  reproduction at their displayed precision.
- The highlighted `n=3`, `d=1` central 99% interval is reproduced as
  `[1.5103714e-26, 150533.6404] * Xbar`, which rounds to the paper's
  `[1.51e-26, 1.51e5] * Xbar`.
- The air-conditioning summaries reproduce as:
  - `n = 12`
  - `Xbar = 108.0833333333`
  - `D_n = 0.8543143234`
  - fitted shape `0.7064931748`
  - fitted scale `152.9856723149`
- All 21 predicted values in the air-conditioning table (exact, KMM and
  plug-in Gamma across seven probabilities) round to the displayed values.
- The reported 90%, 95% and 99% intervals reproduce as:
  - `[0.7549145007, 471.1258620]`
  - `[0.1632757274, 646.2862523]`
  - `[0.0027310617, 1199.4893447]`
- The large-dispersion transition for `p=0.95` is correct: above `p*_n` for
  `n<=18`, equal at `n=19`, and below for `n>=20`.
- Monte Carlo count arithmetic is correct: 28 cells x 10,000,000 gives
  280,000,000 augmented samples and 28 x 8 gives 224 coverage comparisons.
- The two-sided Bonferroni 99% threshold over 224 comparisons is
  `4.082036`, which rounds to `4.08`.
- The full fixed-seed Monte Carlo audit has been rerun and stored. All 224
  displayed coverage entries agree with the stored output at four decimals.
- The maximum absolute cell-level score is `2.6989817548`, attained at
  `n=3`, `alpha=10`, `p=0.995`; this rounds to the paper's `2.70`.
- The stored audit has mean `U = 0.5000096824` and variance
  `U = 0.0833285140`.

## Full stored Monte Carlo audit

The paper workload was rerun with 10,000,000 replications in each of 28 cells,
giving 280,000,000 augmented samples. The fixed-seed results are stored in
`reproduced_results/`:

- `monte_carlo_cells.csv`
- `coverage_table.csv`
- `coverage_table.tex`
- `monte_carlo_summary.json`
- `lookup_validation.json`

The stored summary is:

- mean U: `0.500009682434292`
- variance U: `0.0833285139675356`
- maximum absolute cell-level binomial z score: `2.6989817547645356`

All 224 manuscript table entries match `coverage_table.csv` at the displayed
four-decimal precision.

## Independent reduced audit

A fresh fixed-seed audit with 1,000,000 replications in each of the paper's 28
cells (28,000,000 augmented samples) produced:

- mean U: `0.4999639063`
- variance U: `0.0833337540`
- maximum absolute cell-level binomial z score: `2.8443`

All 224 fresh coverage estimates were within 2.72 combined Monte Carlo standard
errors of the values displayed in the paper, and therefore all were within the
paper's 4.08 simultaneous threshold.

Artifacts for this check were generated in `/tmp/gamma-paper-audit-1m` during
verification. They can be regenerated with:

```bash
python gamma_prediction/reproduce_paper.py \
  --audit --reps-per-cell 1000000
```

## Corrected during verification

The distributed table now uses 99 dispersion nodes and 17 probability levels for
every stored sample size through `n=100`, over `d in [1e-6, 50]` and
`p in [0.001, 0.999]`. The levels are `0.001`, `0.0025`, `0.005`, `0.01`,
`0.025`, `0.05`, `0.1`, `0.25`, `0.5`, `0.75`, `0.9`, `0.95`, `0.975`,
`0.99`, `0.995`, `0.9975`, and `0.999`. At these levels, the tensor-product
spline reduces to quintic interpolation in `log(d)`. In the lower tail, the
theoretically motivated
`log(-log M)` and `log(-log p)` transformation is used and blended smoothly
into the central surface on `p in [0.225, 0.25]`.

The production lookup is intended for these 17 stored probability levels; only
dispersion is interpolated in that use. Removing probability-refinement nodes
reduces the refined table by 84.1% relative to the former 107-level grid. The
stored values were also regenerated with adaptive scalar integration near the
branch junction, preventing inaccurate nodes from contaminating the
dispersion spline.

A reproducible fixed-seed check at off-grid dispersions in `[0.025, 1.5]`
gives:

- seven lower levels through `p=0.1`: median `0.0025 ppm`, 95th percentile
  `0.0795 ppm`, maximum `0.218 ppm`
- three central levels `p in {0.25, 0.5, 0.75}`: median `0.00042 ppm`,
  95th percentile `0.0073 ppm`, maximum `0.030 ppm`
- seven upper levels from `p=0.9`: median `0.00023 ppm`, 95th percentile
  `0.0092 ppm`, maximum `0.107 ppm`

An independent 300-point holdout samples all 17 levels across low, interior and
high dispersion ranges. The largest maximum among its three strata is `0.498 ppm`.
The full holdout is stored in `reproduced_results/interpolation_holdout.csv`.
A systematic 93,296-point check at the transformed midpoint of every
dispersion cell, for all 56 stored sample sizes and all 17 probability levels,
gave maximum error `3.18 ppm`, at `n=3`, `d=0.0118921`, and `p=0.0025`.
Thus, the random-holdout maximum is not presented as a global error bound.
Dense checks found no loss of monotonicity in probability. The table rejects
`d` or `p` outside its rectangular domain rather than extrapolating.

Table construction now caches the regularised quadrature nodes, density values
and Jacobian once for each `(n,d)`, then inverts all requested probabilities
using weighted sums. Existing exact rows are reused when a refinement adds only
new sample sizes; the eight high-order rows build in under 30 seconds with six
workers on the validation machine.

For `m=52..201`, the repaired backend uses the online Glaser series below
`t=5.5` and a packaged 1,400-node interpolation of the tilted log density up to
`t=12000`. The resource is generated offline by saddlepoint-tilted pointwise
inversion. Representative independent holdouts have maximum relative density
error below `2e-8`; runtime evaluation performs no Fourier inversion. The
small-`t` series is evaluated directly on the log scale when its density would
underflow in float64. For example, `log_ell_tilde(201, 0.0002)=-1218.6292`
remains finite.

Above `n=200`, or if a lower-order calculation exceeds the packaged density
range, scalar evaluation automatically uses saddlepoint-centred contour
inversion. Across overlap checks through `n=200`, relative multiplier
differences were below `1e-6`, including both tails at `d=50`. Contour-width
checks remained stable through `n=100000`.

The upper contour branch evaluates survival directly as the complementary
regularised incomplete beta, avoiding subtraction from one. A 297-cell matrix
over `n={100,150,200}`, nine dispersion values from `1e-6` to 50 and eleven
probabilities from 0.001 to 0.999 had zero failures; its maximum relative
multiplier difference was `8.8e-7`. The extreme case `n=100000`, `d=1e-6`,
`p=0.999` also completed, returning `log M=0.00436652`.

The exact workload can be rerun as:

```bash
python gamma_prediction/reproduce_paper.py --paper-audit
```

The seven independent fixed-seed blocks may be run in parallel without changing
their output; serial and seven-worker smoke-test artifacts were byte-for-byte
identical.
