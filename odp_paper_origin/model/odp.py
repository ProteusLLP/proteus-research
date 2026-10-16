"""Exact Bayesian predictive distributions for the ODP model with origin-period priors.

Companion code to:

    Norman, J. P. (2026). Exact Bayesian Predictive Distributions for the
    Over-dispersed Poisson Reserving Model with Origin-Period Priors.

Implements the posterior predictive simulation algorithm of Section 3.2:

    1. Sample xi_i | X  ~  Beta(a_i + s_i,  b_i + d_i)       [Theorem 3, eq. 18]
    2. Recover mu_i from xi via the inverse map                [eq. 14]
    3. Sample beta_j | X, mu  ~  Gamma(c_j, (1/phi) sum mu_r) [eq. 22]
    4. Sample X^fut_ij | mu, beta  ~  phi * Poi(mu_i beta_j / phi)

Three methods are obtained as special cases of the exposure-centred
one-parameter family (Section 4.1, eq. 59):

    - Chain Ladder:   tau = 0           (weak-prior limit,  Corollary 9)
    - Hierarchical:   tau = tau_EB      (empirical Bayes,   Section 4.4)
    - Cape Cod:       tau -> infinity   (strong-prior limit, Section 4.3)

The numerical example uses data from Saluz (2015), Tables 1--2.

Usage::

    python -m odp_paper_origin.model.odp
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pal
from pal import ProteusVariable, StochasticScalar, distributions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SIMS = 1_000_000
SEED = 42
PHI = 21_610.838_822_113
TAU_EB = 346.0

DATA_DIR = Path("odp_paper_origin/data")
OUTPUT_DIR = Path("odp_paper_origin/model")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriangleData:
    """Observed incremental triangle and premiums (Table 1 of the paper)."""

    triangle: np.ndarray  # (n, n) incremental; NaN in future cells
    premiums: np.ndarray  # (n,)   on-level premiums E_i
    phi: float  # ODP scale parameter

    @property
    def n(self) -> int:
        return self.triangle.shape[0]

    @property
    def d(self) -> np.ndarray:
        """Scaled origin-period row totals  d_i = D_i / phi  (eq. 3)."""
        return np.nansum(self.triangle, axis=1) / self.phi

    @property
    def c(self) -> np.ndarray:
        """Scaled development-period column totals  c_j = C_j / phi  (eq. 3)."""
        return np.nansum(self.triangle, axis=0) / self.phi

    @property
    def C(self) -> np.ndarray:
        """Unscaled column totals  C_j = phi * c_j  (eq. 5)."""
        return np.nansum(self.triangle, axis=0)

    @property
    def e(self) -> np.ndarray:
        """Relative exposure index  e_i = E_i / E_1  (eq. 58)."""
        return self.premiums / self.premiums[0]

    @property
    def cumulative_e(self) -> np.ndarray:
        """Cumulative exposure index  sum_{r=1}^{i} e_r."""
        return np.cumsum(self.e)

    @property
    def s(self) -> np.ndarray:
        """Common-age cumulative statistic  s_i  (eq. 16).

        s_i = sum_{k=1}^{i-1} d_{k, n-i+1},  i = 2, ..., n.
        Entry 0 is NaN (unused).
        """
        cumulative = np.nancumsum(self.triangle, axis=1) / self.phi
        s = np.full(self.n, np.nan)
        for i in range(1, self.n):
            s[i] = np.sum(cumulative[:i, self.n - i - 1])
        return s

    @property
    def xi0(self) -> np.ndarray:
        """Benchmark origin ratios  xi^0_i  (eq. 64)."""
        ce = self.cumulative_e
        xi0 = np.full(self.n, np.nan)
        for i in range(1, self.n):
            xi0[i] = ce[i - 1] / ce[i]
        return xi0


def load_saluz_data() -> TriangleData:
    """Load the Saluz (2015) triangle and premiums from CSV."""
    cum = pd.read_csv(DATA_DIR / "saluz_triangle.csv", index_col=0)
    inc = cum.diff(axis=1)
    inc.iloc[:, 0] = cum.iloc[:, 0]
    premiums = pd.read_csv(DATA_DIR / "premium.csv", index_col=0).squeeze()
    return TriangleData(
        triangle=inc.to_numpy(dtype=float),
        premiums=premiums.to_numpy(dtype=float),
        phi=PHI,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OriginPriorODP:
    """Exact Bayesian ODP model with origin-period priors (Sections 2--4).

    Parameters
    ----------
    data : TriangleData
    method : str
        Label used in output tables and figures.
    tau : float | None
        Concentration parameter of the exposure-centred prior (eq. 59).
        ``tau=0`` gives Chain Ladder; ``tau=None`` gives Cape Cod.
    """

    def __init__(self, data: TriangleData, method: str, tau: float | None) -> None:
        self.data = data
        self.method = method
        self.tau = tau
        self.n = data.n

        self.xi: ProteusVariable | None = None
        self.mu: ProteusVariable | None = None
        self.beta: ProteusVariable | None = None
        self.reserve_by_origin: ProteusVariable | None = None
        self.total_reserve: StochasticScalar | None = None

    def _sample_xi(self) -> ProteusVariable:
        """Posterior xi_i | X  (Theorem 3, eq. 18 / 62)."""
        xi = ProteusVariable("op", {})
        xi["1"] = StochasticScalar([1.0])

        s, d, e = self.data.s, self.data.d, self.data.e
        cumulative_e, xi0 = self.data.cumulative_e, self.data.xi0

        for i in range(2, self.n + 1):
            k = i - 1
            if self.tau is None:  # Cape Cod limit
                xi[str(i)] = xi0[k]
            else:
                tau = float(self.tau)
                u_i = 1.0 + s[k] + tau * cumulative_e[k - 1]  # eq. 63
                v_i = d[k] + tau * e[k]  # eq. 63
                xi[str(i)] = distributions.Beta(u_i, v_i).generate()

        self.xi = xi
        return xi

    def _recover_mu(self) -> ProteusVariable:
        """Recover mu_i from xi  (eq. 14)."""
        assert self.xi is not None
        mu = ProteusVariable("op", {})
        mu["1"] = StochasticScalar([1.0])

        cum_mu = mu["1"]
        for i in range(2, self.n + 1):
            cum_mu = cum_mu / self.xi[str(i)]
            mu[str(i)] = cum_mu - (cum_mu * self.xi[str(i)])
        self.mu = mu
        return mu

    def _sample_beta(self) -> ProteusVariable:
        """Posterior beta_j | X, mu  (eq. 22)."""
        assert self.mu is not None
        beta = ProteusVariable("dp", {})
        c, phi = self.data.c, self.data.phi

        for j in range(1, self.n + 1):
            mu_sum = sum(self.mu[str(r)] for r in range(1, self.n - j + 2))
            beta[str(j)] = distributions.Gamma(c[j - 1], phi / mu_sum).generate()

        self.beta = beta
        return beta

    def _simulate_future(self) -> StochasticScalar:
        """Simulate future incremental cells (Section 3.2, step 4)."""
        assert self.mu is not None and self.beta is not None
        phi = self.data.phi
        reserve = ProteusVariable("op", {})

        for i in range(1, self.n + 1):
            row_total = StochasticScalar([0.0])
            for j in range(self.n - i + 2, self.n + 1):
                lam = self.mu[str(i)] * self.beta[str(j)] / phi
                row_total = (
                    row_total + phi * distributions.Gamma(lam, 1).generate()
                )  # use Gamma instead of Poisson for forecasting to avoid discrete jumps in the predictive distribution
            reserve[str(i)] = row_total

        self.reserve_by_origin = reserve
        self.total_reserve = reserve.sum()
        return self.total_reserve

    def run(self) -> None:
        """Run the full posterior predictive simulation."""
        self._sample_xi()
        self._recover_mu()
        self._sample_beta()
        self._simulate_future()


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

PERCENTILES = (50, 75, 90, 95, 99, 99.5)


def total_summary(models: list[OriginPriorODP]) -> pd.DataFrame:
    """One row per method: mean, SD, CoV and selected percentiles."""
    rows = []
    for m in models:
        assert m.total_reserve is not None
        r = m.total_reserve
        row = {
            "Method": m.method,
            "Mean": r.mean(),
            "SD": r.std(),
            "CoV": r.std() / r.mean(),
        }
        for p, v in zip(PERCENTILES, r.percentile(list(PERCENTILES)), strict=False):
            row[f"{p}%"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def by_origin_summary(models: list[OriginPriorODP]) -> pd.DataFrame:
    """One row per (method, origin): mean, SD and selected percentiles."""
    rows = []
    for m in models:
        assert m.reserve_by_origin is not None
        for i in range(1, m.n + 1):
            r = m.reserve_by_origin[str(i)]
            row: dict = {
                "Method": m.method,
                "Origin": i,
                "Mean": r.mean(),
                "SD": r.std(),
            }
            for p, v in zip(PERCENTILES, r.percentile(list(PERCENTILES)), strict=False):
                row[f"{p}%"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def loss_ratio_summary(
    models: list[OriginPriorODP],
    premiums_pv: ProteusVariable,
) -> pd.DataFrame:
    """Posterior mean loss ratios by origin and method (Figure 3)."""
    rows = []
    for m in models:
        assert m.mu is not None and m.beta is not None
        beta_sum = m.beta.sum()
        for i in range(1, m.n + 1):
            lr = m.mu[str(i)] * beta_sum / premiums_pv[str(i)]
            rows.append({"Method": m.method, "Origin": i, "LR": lr.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _fmt(x: float, decimals: int = 0) -> str:
    """Format a number with thousands separators for LaTeX."""
    if decimals == 0:
        return f"{x:,.0f}"
    return f"{x:,.{decimals}f}"


def latex_by_origin_reserves(
    models: list[OriginPriorODP],
    lr_df: pd.DataFrame,
) -> str:
    """Table 3 -- posterior mean loss ratios and reserves by origin year."""
    n = models[0].n
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Origin & CL LR & CC LR & Cred LR "
        r"& CL reserve & CC reserve & Cred reserve \\",
        r"\midrule",
    ]
    res: dict[tuple[str, int], float] = {}
    for m in models:
        assert m.reserve_by_origin is not None
        for i in range(1, n + 1):
            res[(m.method, i)] = m.reserve_by_origin[str(i)].mean()

    lr = lr_df.set_index(["Method", "Origin"])["LR"]
    paid = np.nansum(models[0].data.triangle, axis=1)
    prem = models[0].data.premiums
    totals: dict[str, tuple[float, float]] = {m.method: (0.0, 0.0) for m in models}

    for i in range(1, n + 1):
        cl_lr = lr[("Chain Ladder", i)]
        cc_lr = lr[("Cape Cod", i)]
        hi_lr = lr[("Hierarchical", i)]
        cl_r = res[("Chain Ladder", i)]
        cc_r = res[("Cape Cod", i)]
        hi_r = res[("Hierarchical", i)]
        lines.append(
            f"{i} & {cl_lr:.4f} & {cc_lr:.4f} & {hi_lr:.4f} "
            f"& {_fmt(cl_r)} & {_fmt(cc_r)} & {_fmt(hi_r)} \\\\"
        )
        for method, rv in [
            ("Chain Ladder", cl_r),
            ("Cape Cod", cc_r),
            ("Hierarchical", hi_r),
        ]:
            tr, tu = totals[method]
            totals[method] = (tr + rv, tu + paid[i - 1] + rv)

    cl_tlr = totals["Chain Ladder"][1] / prem.sum()
    cc_tlr = totals["Cape Cod"][1] / prem.sum()
    hi_tlr = totals["Hierarchical"][1] / prem.sum()
    lines.append(r"\midrule")
    lines.append(
        f"Total & {cl_tlr:.4f} & {cc_tlr:.4f} & {hi_tlr:.4f} "
        f"& {_fmt(totals['Chain Ladder'][0])} "
        f"& {_fmt(totals['Cape Cod'][0])} "
        f"& {_fmt(totals['Hierarchical'][0])} \\\\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Posterior mean ultimate loss ratios and reserves by origin year "
        r"under Chain Ladder, Cape Cod and the exposure-centred credibility model.}",
        r"\label{tab:saluz_results}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_total_reserve_summary(df: pd.DataFrame) -> str:
    """Table 4 -- total reserve and reserve SD by method."""
    tau_labels = {
        "Chain Ladder": r"$\tau=0$",
        "Hierarchical": r"$\widehat{\tau}_{\mathrm{EB}}=" + str(int(TAU_EB)) + r"$",
        "Cape Cod": r"$\tau\to \infty$",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Selected concentration & Total reserve & Reserve SD \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        label = tau_labels.get(r["Method"], "")
        lines.append(
            f"{r['Method']} & {label} & {_fmt(r['Mean'])} & {_fmt(r['SD'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Total reserve and reserve standard deviation for the "
        r"three models, using the selected ODP scale parameter "
        r"$\phi=" + _fmt(PHI, 2) + r"$.}",
        r"\label{tab:saluz_totals}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_predictive_summary(df: pd.DataFrame) -> str:
    r"""Table 5 -- posterior predictive summaries for the total reserve."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Mean & SD & CoV (\%)& 75\% & 90\% & 95\% & 99\% & 99.5\% \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Method']} & {_fmt(r['Mean'])} & {_fmt(r['SD'])} "
            f"& {r['CoV'] * 100:.4f} & {_fmt(r['75%'])} & {_fmt(r['90%'])} "
            f"& {_fmt(r['95%'])} & {_fmt(r['99%'])} & {_fmt(r['99.5%'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Posterior predictive summaries for the total reserve, "
        r"based on 1,000,000 simulations using the simulation algorithm described in "
        r"Section~\ref{sec:predictive}.}",
        r"\label{tab:saluz_predictive_summary}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _ecdf_trace(x: StochasticScalar, name: str) -> go.Scatter:
    n_points = 10_000
    xs = np.sort(np.array(x.tolist(), dtype=float))[
        np.linspace(0, len(x) - 1, n_points, dtype=int)
    ]
    ys = np.linspace(0.0, 1.0, n_points)
    return go.Scatter(x=xs, y=ys, mode="lines", name=name)


def _hist_trace(x: StochasticScalar, name: str) -> go.Histogram:
    return go.Histogram(
        x=np.array(x.tolist(), dtype=float),
        name=name,
        histnorm="probability density",
        opacity=0.55,
        nbinsx=150,
    )


def cdf_figure(models: list[OriginPriorODP]) -> go.Figure:
    fig = go.Figure()
    for m in models:
        assert m.total_reserve is not None
        fig.add_trace(_ecdf_trace(m.total_reserve, m.method))
    fig.update_layout(
        title="Saluz total reserve predictive distributions",
        xaxis_title="Total reserve",
        yaxis_title="Cumulative probability",
        template="plotly_white",
    )
    return fig


def density_figure(models: list[OriginPriorODP]) -> go.Figure:
    fig = go.Figure()
    for m in models:
        assert m.total_reserve is not None
        fig.add_trace(_hist_trace(m.total_reserve, m.method))
    fig.update_layout(
        title="Saluz total reserve predictive distributions",
        xaxis_title="Total reserve",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
    )
    return fig


def lr_figure(lr_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for method, grp in lr_df.groupby("Method", sort=False):
        fig.add_trace(
            go.Scatter(
                x=grp["Origin"],
                y=grp["LR"],
                mode="markers+lines",
                name=str(method),
            )
        )
    fig.update_layout(
        title="Expected posterior mean loss ratios by origin period",
        xaxis_title="Origin period",
        yaxis_title="Expected posterior mean loss ratio",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pal.set_default_n_sims(N_SIMS)
    pal.set_random_seed(SEED)

    data = load_saluz_data()

    models = [
        OriginPriorODP(data, "Chain Ladder", tau=0.0),
        OriginPriorODP(data, "Hierarchical", tau=TAU_EB),
        OriginPriorODP(data, "Cape Cod", tau=None),
    ]
    for m in models:
        print(f"  Running {m.method} ...")
        m.run()

    premiums_pv = ProteusVariable(
        "op",
        {str(i): data.premiums[i - 1] for i in range(1, data.n + 1)},
    )

    # -- Summaries -----------------------------------------------------------

    total_df = total_summary(models)
    origin_df = by_origin_summary(models)
    lr_df = loss_ratio_summary(models, premiums_pv)

    # -- CSV -----------------------------------------------------------------

    total_df.to_csv(OUTPUT_DIR / "saluz_total_reserve_summary.csv", index=False)
    origin_df.to_csv(OUTPUT_DIR / "saluz_by_origin_reserve_summary.csv", index=False)

    # -- LaTeX ---------------------------------------------------------------

    tables = {
        "saluz_results.tex": latex_by_origin_reserves(models, lr_df),
        "saluz_totals.tex": latex_total_reserve_summary(total_df),
        "saluz_predictive_summary.tex": latex_predictive_summary(total_df),
    }
    for name, tex in tables.items():
        path = OUTPUT_DIR / name
        path.write_text(tex)
        print(f"  Wrote {path}")

    # -- Figures -------------------------------------------------------------

    cdf_figure(models).write_image(OUTPUT_DIR / "saluz_total_reserve_cdfs.svg")
    density_figure(models).write_image(OUTPUT_DIR / "saluz_total_reserve_density.svg")
    lr_figure(lr_df).write_image(OUTPUT_DIR / "saluz_posterior_lr_means.svg")

    # -- Console -------------------------------------------------------------

    print("\nTotal reserve summary")
    print(total_df.to_string(index=False))
    print(f"\nAll outputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
