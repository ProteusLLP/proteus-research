"""Compare published ODP reserve figures with PAL's MC and RQMC estimators.

Run from any working directory:
    python -m odp_paper_origin.model.compare_sampling --sims 131072

The prior Saluz results are saved from a 1,000,000-path MC experiment.
This analysis reports Monte Carlo estimates with their own sampling error; it
does not expect identical random paths or exact equality across methods.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pal
from pal.config import set_random_seed, set_sampling_method
from pal.types import SamplingMethod
from odp_paper.example_odp_reserve_risk import ODPModel
from odp_paper_origin.model.odp import OriginPriorODP, TAU_EB, load_saluz_data

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "odp_paper_origin" / "model" / "saluz_total_reserve_summary.csv"
TAYLOR_ASHE = ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv"
FIELDS = ("Mean", "SD", "50%", "75%", "90%", "95%", "99%", "99.5%")


def _stats(reserve) -> dict[str, float]:
    return {
        "Mean": float(reserve.mean()),
        "SD": float(reserve.std()),
        **{f"{p}%": float(reserve.percentile(p)) for p in (50, 75, 90, 95, 99, 99.5)},
    }


def _saluz(sims: int, seed: int, method: SamplingMethod) -> dict[str, dict[str, float]]:
    pal.set_default_n_sims(sims)
    set_sampling_method(method, dimensions=32)
    set_random_seed(seed)
    data = load_saluz_data()
    values = {}
    for name, tau in (("Chain Ladder", 0.0), ("Hierarchical", TAU_EB), ("Cape Cod", None)):
        model = OriginPriorODP(data, name, tau)
        model.run()
        assert model.total_reserve is not None
        values[name] = _stats(model.total_reserve)
    return values


def _taylor_ashe(sims: int, seed: int, method: SamplingMethod) -> tuple[float, dict[str, float]]:
    pal.set_default_n_sims(sims)
    set_sampling_method(method, dimensions=32)
    set_random_seed(seed)
    tri = pd.read_csv(TAYLOR_ASHE, index_col=0).to_numpy()
    model = ODPModel(tri)
    return float(model.estimate_phi()), _stats(model.simulate_reserves())


def compare(sims: int, seed: int, replicate_sims: int, replicates: int) -> dict:
    reference = pd.read_csv(PRIOR).set_index("Method")
    output: dict = {"sims": sims, "seed": seed, "previous_sims": 1000000, "models": {}}
    for method in (SamplingMethod.MC, SamplingMethod.RQMC):
        started = time.perf_counter()
        saluz = _saluz(sims, seed, method)
        phi, ta = _taylor_ashe(sims, seed, method)
        elapsed = time.perf_counter() - started
        for name, stats in {**saluz, "Taylor-Ashe": ta}.items():
            rec = output["models"].setdefault(name, {})
            deviations = (
                {field: 100.0 * (stats[field] / float(reference.loc[name, field]) - 1.0) for field in FIELDS}
                if name in reference.index else None
            )
            rec[method.value] = {
                "stats": stats,
                "deviation_from_previous_percent": deviations,
                "elapsed_total_method_seconds": round(elapsed, 2),
            }
        output["taylor_ashe_phi"] = phi

    # Independent scrambled Sobol replicates estimate the variability of RQMC
    # estimators; MC replicates are assessed on the same basis.
    if replicates:
        stability = {}
        for method in (SamplingMethod.MC, SamplingMethod.RQMC):
            draws = {"Chain Ladder": [], "Hierarchical": [], "Cape Cod": []}
            for i in range(replicates):
                saluz = _saluz(replicate_sims, 2140 + i * 377, method)
                for name, vals in saluz.items():
                    draws[name].append([vals["Mean"], vals["99.5%"]])
            for name, values in draws.items():
                data = np.asarray(values)
                stability.setdefault(name, {})[method.value] = {
                    "replicates": replicates, "sims_per_replicate": replicate_sims,
                    "sd_of_mean_estimate": float(np.std(data[:, 0], ddof=1)),
                    "sd_of_p99_5_estimate": float(np.std(data[:, 1], ddof=1)),
                }
        output["replication_stability"] = stability
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicate-sims", type=int, default=8192)
    parser.add_argument("--replicates", type=int, default=4)
    args = parser.parse_args()
    previous = (
        pal.config.n_sims, pal.config.rng, pal.config.seed,
        pal.config.sampling_method, pal.config.qmc_dimensions,
    )
    try:
        outcome = compare(args.sims, args.seed, args.replicate_sims, args.replicates)
        print("ODP_NUMERICAL_COMPARISON_BEGIN", flush=True)
        print(json.dumps(outcome, sort_keys=True, indent=2), flush=True)
        print("ODP_NUMERICAL_COMPARISON_END", flush=True)
    finally:
        (
            pal.config.n_sims, pal.config.rng, pal.config.seed,
            pal.config.sampling_method, pal.config.qmc_dimensions,
        ) = previous


if __name__ == "__main__":
    main()
