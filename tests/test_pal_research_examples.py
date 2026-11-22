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
from odp_paper_origin.model.odp import OriginPriorODP, load_saluz_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_pal_run():
    """Keep the research's one-million-path defaults out of smoke tests."""
    original_n_sims = pal.config.n_sims
    original_rng = pal.config.rng
    pal.set_default_n_sims(512)
    pal.set_random_seed(20261122)
    try:
        yield
    finally:
        pal.config.n_sims = original_n_sims
        pal.config.rng = original_rng


def test_taylor_ashe_reserve_simulation(small_pal_run):
    triangle = pd.read_csv(
        ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv", index_col=0
    ).to_numpy()
    model = ODPModel(triangle)
    reserves = model.simulate_reserves()

    assert reserves.n_sims == 512
    assert np.isfinite(model.phi) and model.phi > 0.0
    assert np.all(np.isfinite(reserves.values))
    assert np.all(reserves.values >= 0.0)
    assert reserves.std() > 0.0

    origin_total = model.total_future_claims_by_origin.sum()
    np.testing.assert_allclose(reserves.values, origin_total.values)
    assert np.isclose(model.total_future_claims_by_origin["1"].mean(), 0.0)


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
    assert np.all(np.isfinite(total.values))
    assert np.all(total.values >= 0.0)
    assert total.std() > 0.0
    assert total.percentile(99.5) >= total.percentile(50)

    origin_total = model.reserve_by_origin.sum()
    np.testing.assert_allclose(total.values, origin_total.values)
    assert np.isclose(model.reserve_by_origin["1"].mean(), 0.0)


def test_origin_prior_model_is_seed_reproducible(small_pal_run):
    data = load_saluz_data()
    first = OriginPriorODP(data, "Hierarchical", tau=346.0)
    first.run()
    assert first.total_reserve is not None
    first_samples = first.total_reserve.values.copy()

    pal.set_random_seed(20261122)
    second = OriginPriorODP(data, "Hierarchical", tau=346.0)
    second.run()
    assert second.total_reserve is not None
    np.testing.assert_array_equal(first_samples, second.total_reserve.values)
