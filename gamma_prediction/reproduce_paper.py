#!/usr/bin/env python3
"""Reproduce numerical results for the Gamma prediction-interval paper.

The default run writes deterministic series checks, multiplier tables, and the
air-conditioning example. Use ``--audit`` for a configurable Monte Carlo audit
or ``--paper-audit`` for the paper's 10,000,000 replications per cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = PROJECT_ROOT / "python" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from gamma_prediction.intervals import (  # noqa: E402
    audit_large,
    airconditioning_example,
    build_conditional_lookup,
    check_series_overlap,
    paper_multiplier_table,
    prediction_log_multiplier,
    prediction_multiplier,
    validate_lookup,
)
from gamma_prediction.density import ell_tilde  # noqa: E402
from gamma_prediction import load_default_table  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "reproduced_results"
PAPER_ALPHAS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
PAPER_THRESHOLDS = (0.005, 0.01, 0.05, 0.10)


def records(value: Any) -> list[dict[str, Any]]:
    """Convert a list or optional pandas DataFrame to records."""
    if hasattr(value, "to_dict"):
        return list(value.to_dict(orient="records"))
    return [dict(row) for row in value]


def scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [scalar(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(scalar(value), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: float) -> str:
    if value == 0.0:
        return "0"
    absolute = abs(value)
    if absolute < 1e-4 or absolute >= 1e5:
        mantissa, exponent = f"{value:.5e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{value:.6g}"


def write_multiplier_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ("M_.05", "M_.95", "M_.025", "M_.975", "M_.005", "M_.995")
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$n$ & $d$ & $M_{.05}$ & $M_{.95}$ & $M_{.025}$ & $M_{.975}$ & $M_{.005}$ & $M_{.995}$ \\",
        r"\midrule",
    ]
    for row in rows:
        values = " & ".join(latex_number(float(row[column])) for column in columns)
        lines.append(f"{int(row['n'])} & {float(row['d']):g} & {values} " + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_air_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$p$ & Exact & KMM & Plug-in Gamma \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['p']):g} & {float(row['Exact']):.6g} & "
            f"{float(row['KMM']):.6g} & {float(row['Fitted Gamma']):.6g} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_paper_anchors() -> None:
    anchors = (
        (3, 0.05, 0.05, 0.17959271),
        (3, 1.00, 0.005, 1.5103714e-26),
        (3, 1.00, 0.995, 150533.64),
        (10, 0.25, 0.95, 2.8115385),
        (20, 1.00, 0.975, 5.5696683),
        (50, 0.05, 0.995, 2.0695907),
    )
    for n, d, p, expected in anchors:
        actual = prediction_multiplier(n, d, p)
        if not math.isclose(actual, expected, rel_tol=5e-7, abs_tol=1e-13):
            raise AssertionError((n, d, p, actual, expected))


def reproduce_deterministic(output: Path) -> None:
    print("1. Checking overlap of the density representations...")
    overlap = [
        {
            "m": m,
            "t": t,
            "glaser_tilde": glaser,
            "residue_tilde": residue,
            "relative_difference": relative,
        }
        for m, t, glaser, residue, relative in check_series_overlap(
            m_values=(2, 3, 5, 10, 20)
        )
    ]
    write_csv(output / "series_overlap.csv", overlap)
    print(
        f"   max relative difference: {max(abs(row['relative_difference']) for row in overlap):.3g}"
    )

    density_anchors = []
    for m, t, expected in (
        (3, 5.0, 1.84545461340647),
        (20, 5.0, 3.35622225949113),
        (51, 4.5, 2.61082730955885e-10),
        (51, 8.0, 4.582730260064e-4),
    ):
        actual = ell_tilde(m, t)
        relative_error = actual / expected - 1.0
        density_anchors.append(
            {
                "m": m,
                "t": t,
                "expected_ell_tilde": expected,
                "actual_ell_tilde": actual,
                "relative_error": relative_error,
            }
        )
        if abs(relative_error) > 2e-10:
            raise AssertionError((m, t, actual, expected))
    write_csv(output / "density_anchors.csv", density_anchors)

    print("2. Reproducing the manuscript multiplier table...")
    multipliers = records(paper_multiplier_table())
    write_csv(output / "prediction_multipliers.csv", multipliers)
    write_multiplier_latex(output / "prediction_multipliers.tex", multipliers)
    assert_paper_anchors()
    print("   anchor checks passed")

    print("3. Reproducing the air-conditioning example...")
    summary, table = airconditioning_example()
    air_rows = records(table)
    write_json(output / "airconditioning_summary.json", summary)
    write_csv(output / "airconditioning_prediction_quantiles.csv", air_rows)
    write_air_latex(output / "airconditioning_prediction_quantiles.tex", air_rows)

    print("4. Validating packaged-table interpolation...")
    validate_table_interpolation(output)


def validate_table_interpolation(output: Path) -> None:
    rng = np.random.default_rng(20260818)
    table = load_default_table()
    validation_n = (3, 5, 10, 20, 50, 55, 75, 100)
    rows = []
    domains = (
        ("lower", 160, table.p_grid[table.p_grid <= 0.1]),
        ("central", 80, table.p_grid[(table.p_grid > 0.1) & (table.p_grid < 0.9)]),
        ("upper", 160, table.p_grid[table.p_grid >= 0.9]),
    )
    for domain, checks, probabilities in domains:
        for _ in range(checks):
            n = int(rng.choice(validation_n))
            d = float(np.exp(rng.uniform(np.log(0.025), np.log(1.5))))
            p = float(rng.choice(probabilities))
            interpolated = table.log_multiplier_at(n, d, p)
            exact = prediction_log_multiplier(n, d, p)
            relative_error = abs(math.expm1(interpolated - exact))
            rows.append(
                {
                    "domain": domain,
                    "n": n,
                    "d": d,
                    "p": p,
                    "interpolated_log_multiplier": interpolated,
                    "exact_log_multiplier": exact,
                    "relative_multiplier_error": relative_error,
                }
            )
    write_csv(output / "interpolation_validation.csv", rows)
    summary = {
        "seed": 20260818,
        "checks": len(rows),
        "n": list(validation_n),
        "d": [0.025, 1.5],
        "domains": {},
    }
    for domain, checks, probabilities in domains:
        errors = np.asarray(
            [
                row["relative_multiplier_error"]
                for row in rows
                if row["domain"] == domain
            ]
        )
        summary["domains"][domain] = {
            "checks": checks,
            "p": [float(p) for p in probabilities],
            "median_relative_error": float(np.quantile(errors, 0.5)),
            "p95_relative_error": float(np.quantile(errors, 0.95)),
            "max_relative_error": float(np.max(errors)),
        }
    write_json(output / "interpolation_validation.json", summary)
    for domain in summary["domains"].values():
        print(
            "   median / p95 / max relative error: "
            f"{100 * domain['median_relative_error']:.6f}% / "
            f"{100 * domain['p95_relative_error']:.6f}% / "
            f"{100 * domain['max_relative_error']:.6f}%"
        )

    holdout_rng = np.random.default_rng(20260819)
    holdout_rows = []
    holdout_domains = (
        ("low_d", 100, 1e-6, 0.025),
        ("interior_d", 100, 0.025, 10.0),
        ("high_d", 100, 10.0, 50.0),
    )
    for domain, checks, d_min, d_max in holdout_domains:
        for _ in range(checks):
            n = int(holdout_rng.choice(validation_n))
            d = float(np.exp(holdout_rng.uniform(np.log(d_min), np.log(d_max))))
            p = float(holdout_rng.choice(table.p_grid))
            interpolated = table.log_multiplier_at(n, d, p)
            exact = prediction_log_multiplier(n, d, p)
            holdout_rows.append(
                {
                    "domain": domain,
                    "n": n,
                    "d": d,
                    "p": p,
                    "interpolated_log_multiplier": interpolated,
                    "exact_log_multiplier": exact,
                    "relative_multiplier_error": abs(math.expm1(interpolated - exact)),
                }
            )
    write_csv(output / "interpolation_holdout.csv", holdout_rows)
    holdout_summary = {"seed": 20260819, "domains": {}}
    for domain, checks, d_min, d_max in holdout_domains:
        errors = np.asarray(
            [
                row["relative_multiplier_error"]
                for row in holdout_rows
                if row["domain"] == domain
            ]
        )
        holdout_summary["domains"][domain] = {
            "checks": checks,
            "d": [d_min, d_max],
            "p": [float(p) for p in table.p_grid],
            "median_relative_error": float(np.quantile(errors, 0.5)),
            "p95_relative_error": float(np.quantile(errors, 0.95)),
            "max_relative_error": float(np.max(errors)),
        }
    write_json(output / "interpolation_holdout.json", holdout_summary)


def audit_blocks() -> tuple[tuple[Any, ...], ...]:
    return (
        ((3,), PAPER_ALPHAS, 20260817, 500, 901, 250_000),
        ((5,), PAPER_ALPHAS, 20260822, 300, 601, 250_000),
        ((10,), (0.1, 0.5, 1.0), 202608271, 250, 501, 250_000),
        ((10,), (2.0, 5.0, 10.0, 100.0), 202608272, 250, 501, 250_000),
        ((20,), (0.1, 0.5), 202608371, 220, 451, 200_000),
        ((20,), (1.0, 2.0), 202608372, 220, 451, 200_000),
        ((20,), (5.0, 10.0, 100.0), 202608373, 220, 451, 200_000),
    )


def run_audit(output: Path, reps_per_cell: int, batch_size: int) -> None:
    print(
        f"5. Running Monte Carlo audit with {reps_per_cell:,} replications per cell..."
    )
    cells: list[dict[str, Any]] = []
    for n_values, alphas, seed, n_t, n_phi, suggested_batch in audit_blocks():
        _, block_cells = audit_large(
            n_values=n_values,
            alpha_values=alphas,
            reps_per_cell=reps_per_cell,
            batch_size=min(batch_size, suggested_batch),
            seed=seed,
            lookup_n_t=n_t,
            lookup_n_phi=n_phi,
            t_max=700.0,
            thresholds=PAPER_THRESHOLDS,
        )
        cells.extend(records(block_cells))
        print(f"   completed n={n_values[0]}, alpha={tuple(alphas)}")
    cells.sort(key=lambda row: (int(row["n"]), float(row["alpha"])))
    write_csv(output / "monte_carlo_cells.csv", cells)

    coverage_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for row in cells:
        coverage = {"n": int(row["n"]), "alpha": float(row["alpha"])}
        for threshold in PAPER_THRESHOLDS:
            observed = 1.0 - float(row[f"lower_{threshold:g}"])
            coverage[f"coverage_{threshold:g}"] = observed
            nominal = 1.0 - threshold
            standard_error = math.sqrt(nominal * (1.0 - nominal) / reps_per_cell)
            scores.append(abs(observed - nominal) / standard_error)
        for probability in (0.90, 0.95, 0.99, 0.995):
            threshold = 1.0 - probability
            observed = 1.0 - float(row[f"upper_{threshold:g}"])
            coverage[f"coverage_{probability:g}"] = observed
            standard_error = math.sqrt(
                probability * (1.0 - probability) / reps_per_cell
            )
            scores.append(abs(observed - probability) / standard_error)
        coverage_rows.append(coverage)
    write_csv(output / "coverage_table.csv", coverage_rows)
    write_coverage_latex(output / "coverage_table.tex", coverage_rows, reps_per_cell)

    total = sum(int(row["reps"]) for row in cells)
    weighted_mean = (
        sum(float(row["mean_U"]) * int(row["reps"]) for row in cells) / total
    )
    second_moment = (
        sum(
            (float(row["var_U"]) + float(row["mean_U"]) ** 2) * int(row["reps"])
            for row in cells
        )
        / total
    )
    summary = {
        "reps_per_cell": reps_per_cell,
        "cells": len(cells),
        "total_augmented_samples": total,
        "mean_U": weighted_mean,
        "var_U": second_moment - weighted_mean**2,
        "max_abs_binomial_z": max(scores),
    }
    write_json(output / "monte_carlo_summary.json", summary)

    lookup_validation = {}
    for n in (3, 5, 10, 20):
        lookup = build_conditional_lookup(n, n_t=500, n_phi=901, t_max=700.0)
        lookup_validation[str(n)] = validate_lookup(
            lookup, n_checks=50, seed=20260817 + n
        )
    write_json(output / "lookup_validation.json", lookup_validation)
    print(json.dumps(summary, indent=2))


def write_coverage_latex(
    path: Path, rows: list[dict[str, Any]], reps_per_cell: int
) -> None:
    columns = (0.005, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 0.995)
    lines = [
        r"\begin{tabular}{rrrrrrrrrr}",
        r"\toprule",
        r"$n$ & $\alpha$ & $0.005$ & $0.01$ & $0.05$ & $0.1$ & $0.9$ & $0.95$ & $0.99$ & $0.995$ \\",
        r"\midrule",
    ]
    previous_n = None
    for row in rows:
        n = int(row["n"])
        if previous_n is not None and n != previous_n:
            lines.append(r"\addlinespace")
        n_text = str(n) if n != previous_n else ""
        values = " & ".join(
            f"{float(row[f'coverage_{probability:g}']):.4f}" for probability in columns
        )
        lines.append(f"{n_text} & {float(row['alpha']):g} & {values} " + r"\\")
        previous_n = n
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    lines.insert(0, f"% {reps_per_cell:,} simulations per (n, alpha) cell")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audit", action="store_true")
    group.add_argument(
        "--paper-audit",
        action="store_true",
        help="run 10,000,000 simulations for each of the paper's 28 cells",
    )
    parser.add_argument("--reps-per-cell", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.reps_per_cell < 1 or args.batch_size < 1:
        raise ValueError("replications and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    reproduce_deterministic(args.output)
    if args.audit or args.paper_audit:
        reps = 10_000_000 if args.paper_audit else args.reps_per_cell
        run_audit(args.output, reps, args.batch_size)
    else:
        print("5. Monte Carlo audit skipped (use --audit or --paper-audit).")
    manifest = {
        "implementation": "Glaser/residue, packaged high-m density, and contour inversion",
        "audit_mode": "paper"
        if args.paper_audit
        else ("custom" if args.audit else "none"),
        "reps_per_cell": 10_000_000
        if args.paper_audit
        else (args.reps_per_cell if args.audit else None),
        "files": sorted(
            path.name
            for path in args.output.iterdir()
            if path.is_file() and path.name != "manifest.json"
        ),
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
