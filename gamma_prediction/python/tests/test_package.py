import math

import numpy as np

import gamma_prediction as gp
from gamma_prediction.monte_carlo import audit_coverage
from gamma_prediction.table import CriticalValueTable, build_critical_value_table
from gamma_prediction.intervals import check_series_overlap
from gamma_prediction.density import (
    _ell_tilde_glaser_adaptive,
    ell_tilde,
    log_ell_tilde,
)


AIR = [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487]


def test_paper_center_quantile_is_tightly_inverted():
    value = gp.prediction_log_multiplier(50, 0.05, 0.5)
    assert math.isclose(value, -0.0331459448703, rel_tol=0.0, abs_tol=3e-12)


def test_batched_exact_inversion_matches_scalar_extreme_tails():
    probabilities = np.array([0.001, 0.025, 0.5, 0.975, 0.999])
    batched = gp.prediction_log_multipliers(3, 8.0, probabilities)
    scalar = np.array([gp.prediction_log_multiplier(3, 8.0, p) for p in probabilities])
    assert np.max(np.abs(np.expm1(batched - scalar))) < 3e-8


def test_batched_exact_inversion_matches_scalar_near_branch_junction():
    batched = gp.prediction_log_multipliers(50, 27.5, [0.9])[0]
    scalar = gp.prediction_log_multiplier(50, 27.5, 0.9)
    assert abs(batched - scalar) < 1e-12


def test_large_log_multiplier_transform_remains_smooth():
    from gamma_prediction.intervals import J_from_w

    values = np.array([J_from_w(3, w) for w in np.linspace(35.0, 40.0, 21)])
    assert np.all(np.diff(values) > 0.0)


def test_paper_example_matches_reproduced_values():
    lower, upper = gp.equal_tail_interval(AIR, coverage=0.95)
    assert math.isclose(lower, 0.1632757274, rel_tol=2e-9)
    assert math.isclose(upper, 646.2862523, rel_tol=2e-9)


def test_density_representations_overlap():
    rows = check_series_overlap(m_values=(3, 5, 10, 20), t_values=(4.0, 4.5, 5.0))
    assert max(abs(row[-1]) for row in rows) < 1e-12


def test_glaser_series_stops_adaptively():
    converged, value, terms = _ell_tilde_glaser_adaptive(20, 1.0)
    assert converged
    assert terms < 40
    assert math.isclose(value, ell_tilde(20, 1.0), rel_tol=3e-14)

    converged, _, _ = _ell_tilde_glaser_adaptive(3, 2.0 * math.pi)
    assert not converged


def test_online_density_handover_anchors():
    assert math.isclose(ell_tilde(3, 5.0), 1.84545461340647, rel_tol=3e-13)
    assert math.isclose(ell_tilde(20, 5.0), 3.35622225949113, rel_tol=3e-13)
    assert math.isclose(ell_tilde(51, 8.0), 4.58273027275238e-4, rel_tol=2e-10)


def test_high_order_density_resource_anchors():
    anchors = (
        (52, 5.5, 1.6513768666061761e-8),
        (76, 10.0, 1.5433738963374223e-8),
        (100, 45.0, 5.6460446216183661e17),
        (101, 14.644660940672622, 6.6457184219940807e-9),
        (101, 5379.0, 7.2470549240662981e173),
    )
    for m, t, expected in anchors:
        assert math.isclose(ell_tilde(m, t), expected, rel_tol=6e-8)

    log_anchors = (
        (201, 2e-4, -1218.6292380531224),
        (151, 10.0, -87.76253693080771),
        (201, 10.0, -145.81037170753842),
        (201, 100.0, 95.76745449470269),
        (201, 10000.0, 792.8797798582182),
    )
    for m, t, expected in log_anchors:
        assert math.isclose(log_ell_tilde(m, t), expected, abs_tol=3e-8)


def test_contour_backend_overlaps_density_grid_through_n200():
    checks = (
        (200, 1e-6, 0.999, 3e-8),
        (100, 0.2, 0.95, 3e-8),
        (150, 1.0, 0.975, 3e-8),
        (200, 0.2, 0.001, 3e-8),
        (200, 50.0, 0.025, 1e-6),
        (200, 50.0, 0.999, 3e-8),
    )
    for n, d, p, tolerance in checks:
        density_grid = gp.prediction_log_multiplier(n, d, p)
        contour = gp.prediction_log_multiplier_large_n(n, d, p)
        assert abs(math.expm1(contour - density_grid)) < tolerance


def test_scalar_backend_routes_to_contour_above_n200():
    direct = gp.prediction_log_multiplier_large_n(1000, 0.2, 0.95)
    automatic = gp.prediction_log_multiplier(1000, 0.2, 0.95)
    assert math.isclose(automatic, direct, rel_tol=0.0, abs_tol=1e-12)

    beyond_resource = gp.prediction_log_multiplier(200, 70.0, 0.95)
    contour = gp.prediction_log_multiplier_large_n(200, 70.0, 0.95)
    assert math.isclose(beyond_resource, contour, rel_tol=0.0, abs_tol=1e-12)


def test_table_interpolation_and_round_trip(tmp_path):
    table = build_critical_value_table(
        n_grid=[3, 5],
        d_grid=[0.025, 0.05, 0.25, 1.0],
        p_grid=[0.005, 0.025, 0.5, 0.95, 0.975, 0.995],
        progress=False,
    )
    direct = gp.prediction_log_multiplier(3, 0.05, 0.95)
    assert abs(table.log_multiplier_at(3, 0.05, 0.95) - direct) < 1e-12
    path = tmp_path / "critical-values.npz"
    table.save(path)
    restored = CriticalValueTable.load(path)
    assert np.array_equal(restored.n_grid, table.n_grid)
    assert np.allclose(restored.log_multiplier, table.log_multiplier)
    text_path = tmp_path / "critical-values.txt"
    table.save_text(text_path)
    text = text_path.read_text(encoding="ascii")
    assert "n\td\tp\tlog_multiplier\tmultiplier" in text
    assert "3\t0.025" in text


def test_packaged_table_loads():
    table = gp.load_default_table()
    assert table.log_multiplier.shape == (56, 22, 17)
    assert table.refined_log_multiplier.shape == (56, 99, 17)
    assert table.refined_p_grid[0] == 0.001
    assert table.refined_p_grid[-1] == 0.999
    assert np.array_equal(table.refined_p_grid, table.p_grid)
    assert set((55, 60, 65, 70, 75, 80, 90, 100)).issubset(table.n_grid)
    assert math.isclose(
        table.multiplier_at(3, 0.05, 0.95),
        gp.prediction_multiplier(3, 0.05, 0.95),
        rel_tol=2e-8,
    )
    assert math.isclose(
        table.log_multiplier_at(100, 0.2, 0.95),
        gp.prediction_log_multiplier(100, 0.2, 0.95),
        rel_tol=0.0,
        abs_tol=3e-8,
    )
    assert math.isclose(
        table.log_multiplier_at(3, 8.0, 0.999),
        gp.prediction_log_multiplier(3, 8.0, 0.999),
        rel_tol=0.0,
        abs_tol=3e-8,
    )


def test_packaged_table_off_grid_interpolation_accuracy():
    table = gp.load_default_table()
    interpolated = table.log_multiplier_at(3, 1.3881283432695093, 0.95)
    exact = gp.prediction_log_multiplier(3, 1.3881283432695093, 0.95)
    assert abs(math.expm1(interpolated - exact)) < 1e-6


def test_lower_tail_scalar_and_vector_interpolation_agree():
    table = gp.load_default_table()
    dispersions = np.array([0.075228, 0.5, 8.973543, 39.491044])
    scalar = np.array(
        [table.log_multiplier_at(3, float(d), 0.025) for d in dispersions]
    )
    vector = table.log_multiplier_array(3, dispersions, 0.025)
    assert np.allclose(vector, scalar, rtol=0.0, atol=1e-13)


def test_lower_tail_blend_is_monotone():
    table = gp.load_default_table()
    probabilities = np.linspace(0.22, 0.255, 141)
    for n in (3, 20, 100):
        for d in (0.001, 0.25, 5.0, 50.0):
            values = np.array(
                [table.log_multiplier_at(n, d, float(p)) for p in probabilities]
            )
            assert np.all(np.diff(values) > 0.0)


def test_high_n_exact_backend_matches_table():
    table = gp.load_default_table()
    assert table.multiplier_at(100, 0.2, 0.95) > 0.0
    assert math.isclose(
        gp.prediction_log_multiplier(100, 0.2, 0.95),
        table.log_multiplier_at(100, 0.2, 0.95),
        rel_tol=0.0,
        abs_tol=3e-8,
    )
    assert table.multiplier_at(100, 60.0, 0.95, exact_fallback=True) > 0.0


def test_small_table_audit_is_calibrated():
    table = build_critical_value_table(
        n_grid=[3],
        d_grid=np.geomspace(0.001, 50.0, 25),
        p_grid=[0.025, 0.975],
        progress=False,
    )
    rows = audit_coverage(
        table,
        n_values=[3],
        alpha_values=[0.25, 1.0, 5.0],
        p_values=[0.025, 0.975],
        replications=500,
        batch_size=500,
        seed=123,
    )
    assert len(rows) == 6
    assert max(abs(row["z"]) for row in rows) < 4.5


if __name__ == "__main__":
    from scipy.stats import gamma

    data = gamma(a=0.5, scale=2.0).rvs(size=2, random_state=1234)
    upper = gp.prediction_quantile(data, p=0.95)
    print(f"95% prediction quantile for {data} is {upper:.6g}")
    print(f"True 95% quantile is {gamma.ppf(0.95, a=0.5, scale=2.0):.6g}")
