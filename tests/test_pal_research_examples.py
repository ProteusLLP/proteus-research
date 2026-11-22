"""Small integration checks for PAL-based ODP research examples.

These tests exercise the public PAL API and pathwise aggregation with a small
simulation count. They are intentionally not statistical validation of the
underlying Bayesian claims-reserving research.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pal
from odp_paper.example_odp_reserve_risk import ODPModel
from pal._maths import asnumpy, xp
from pal.config import set_random_seed, set_sampling_method
from pal.types import SamplingMethod
from odp_paper_origin.model.odp import (
    OriginPriorODP,
    by_origin_summary,
    cdf_figure,
    density_figure,
    load_saluz_data,
    loss_ratio_summary,
    lr_figure,
    total_summary,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[SamplingMethod.MC, SamplingMethod.RQMC], ids=["MC", "RQMC"])
def small_pal_run(request):
    """Run every ODP integration test under both PAL sampling methods."""
    old = (
        pal.config.n_sims,
        pal.config.rng,
        pal.config.seed,
        pal.config.sampling_method,
        pal.config.qmc_dimensions,
    )
    pal.set_default_n_sims(512)
    set_sampling_method(request.param, dimensions=32)
    set_random_seed(20261122)
    try:
        yield request.param
    finally:
        (
            pal.config.n_sims,
            pal.config.rng,
            pal.config.seed,
            pal.config.sampling_method,
            pal.config.qmc_dimensions,
        ) = old


def test_taylor_ashe_reserve_simulation(small_pal_run):
    triangle = pd.read_csv(
        ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv", index_col=0
    ).to_numpy()
    model = ODPModel(triangle)
    reserves = model.simulate_reserves()

    assert reserves.n_sims == 512
    assert isinstance(reserves.values, xp.ndarray)
    assert np.isfinite(model.phi) and model.phi > 0.0
    assert bool(xp.all(xp.isfinite(reserves.values)))
    assert bool(xp.all(reserves.values >= 0.0))
    assert reserves.std() > 0.0

    origin_total = model.total_future_claims_by_origin.sum()
    np.testing.assert_allclose(asnumpy(reserves.values), asnumpy(origin_total.values))
    assert np.isclose(model.total_future_claims_by_origin["1"].mean(), 0.0)
    assert model.total_future_claims_by_origin.cdf_plot().data
    assert model.total_future_claims_by_origin.percentile_fan_plot().data


def test_taylor_ashe_posterior_preserves_stochastic_ancestry(small_pal_run):
    triangle = pd.read_csv(
        ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv", index_col=0
    ).to_numpy()
    model = ODPModel(triangle)
    model.estimate_phi()
    model.build_posterior()
    assert (
        model.mu["10"].coupled_variable_group
        is model.betas["10"].coupled_variable_group
    )


@pytest.mark.parametrize(
    ("method", "tau"),
    [("Chain Ladder", 0.0), ("Hierarchical", 346.0), ("Cape Cod", None)],
)
def test_saluz_origin_prior_reserves(method, tau, small_pal_run):
    data = load_saluz_data()
    model = OriginPriorODP(data, method, tau)
    model.run()

    assert model.mu is not None
    assert model.beta is not None
    assert model.reserve_by_origin is not None
    assert model.total_reserve is not None

    total = model.total_reserve
    assert total.n_sims == 512
    assert isinstance(total.values, xp.ndarray)
    assert bool(xp.all(xp.isfinite(total.values)))
    assert bool(xp.all(total.values >= 0.0))
    assert total.std() > 0.0
    assert total.percentile(99.5) >= total.percentile(50)

    origin_total = model.reserve_by_origin.sum()
    np.testing.assert_allclose(asnumpy(total.values), asnumpy(origin_total.values))
    assert np.isclose(model.reserve_by_origin["1"].mean(), 0.0)


def test_origin_prior_model_is_seed_reproducible(small_pal_run):
    data = load_saluz_data()
    first = OriginPriorODP(data, "Hierarchical", tau=346.0)
    first.run()
    assert first.total_reserve is not None
    first_samples = asnumpy(first.total_reserve.values).copy()

    pal.set_random_seed(20261122)
    second = OriginPriorODP(data, "Hierarchical", tau=346.0)
    second.run()
    assert second.total_reserve is not None
    np.testing.assert_array_equal(first_samples, asnumpy(second.total_reserve.values))


def test_origin_prior_tables_and_figures(small_pal_run):
    data = load_saluz_data()
    model = OriginPriorODP(data, "Hierarchical", tau=346.0)
    model.run()

    totals = total_summary([model])
    by_origin = by_origin_summary([model])
    premiums = pal.ProteusVariable(
        "op", {str(i): data.premiums[i - 1] for i in range(1, data.n + 1)}
    )
    lr = loss_ratio_summary([model], premiums)

    assert len(totals) == 1
    assert len(by_origin) == data.n
    assert len(lr) == data.n
    assert cdf_figure([model]).data
    assert density_figure([model]).data
    assert lr_figure(lr).data


@pytest.mark.parametrize("example", ["Taylor-Ashe", "Saluz"])
def test_mc_and_rqmc_predictive_results_are_consistent(example):
    """Different random designs must agree within conservative sampling error.

    We compare estimated means and upper percentiles, never individual paths.
    """
    old = (
        pal.config.n_sims,
        pal.config.rng,
        pal.config.seed,
        pal.config.sampling_method,
        pal.config.qmc_dimensions,
    )
    results = {}
    try:
        pal.set_default_n_sims(2048)
        for method in (SamplingMethod.MC, SamplingMethod.RQMC):
            set_sampling_method(method, dimensions=32)
            set_random_seed(20261122)
            if example == "Taylor-Ashe":
                triangle = pd.read_csv(
                    ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv", index_col=0
                ).to_numpy()
                values = ODPModel(triangle).simulate_reserves()
            else:
                model = OriginPriorODP(load_saluz_data(), "Hierarchical", tau=346.0)
                model.run()
                assert model.total_reserve is not None
                values = model.total_reserve
            results[method] = (
                float(values.mean()),
                float(values.std()),
                float(values.percentile(90.0)),
            )
    finally:
        (
            pal.config.n_sims,
            pal.config.rng,
            pal.config.seed,
            pal.config.sampling_method,
            pal.config.qmc_dimensions,
        ) = old
    mc_mean, mc_sd, mc_p90 = results[SamplingMethod.MC]
    rqmc_mean, rqmc_sd, rqmc_p90 = results[SamplingMethod.RQMC]
    assert mc_mean > 0 and rqmc_mean > 0
    # The designs are not pathwise equivalent. Allow MC sampling error in
    # the first moment, and a wider tolerance for the estimated 90th percentile.
    se_difference = (mc_sd**2 + rqmc_sd**2) ** 0.5 / (2048**0.5)
    assert abs(mc_mean - rqmc_mean) < 7 * se_difference + 0.02 * max(mc_mean, rqmc_mean)
    assert abs(mc_p90 - rqmc_p90) < 0.18 * max(mc_p90, rqmc_p90)
