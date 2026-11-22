"""Profile the public PAL ODP research models under MC and RQMC.

The profiler reports cumulative/self time by function, without changing the
model. Repeatable representative workloads use 2^N scenarios.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path
from time import perf_counter

import pandas as pd

import pal
from pal.config import set_sampling_method, set_random_seed
from pal.types import SamplingMethod
from odp_paper.example_odp_reserve_risk import ODPModel
from odp_paper_origin.model.odp import OriginPriorODP, TAU_EB, load_saluz_data

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=65536)
    parser.add_argument("--lines", type=int, default=35)
    args = parser.parse_args()
    prior = (pal.config.n_sims, pal.config.rng, pal.config.seed, pal.config.sampling_method, pal.config.qmc_dimensions)
    saluz = load_saluz_data()
    triangle = pd.read_csv(ROOT / "odp_paper" / "taylor_ashe_claims_triangle.csv", index_col=0).to_numpy()
    try:
        pal.set_default_n_sims(args.sims)
        for method in (SamplingMethod.MC, SamplingMethod.RQMC):
            for name, make_model in [
                ("Taylor-Ashe", lambda: ODPModel(triangle)),
                ("Chain Ladder", lambda: OriginPriorODP(saluz, "Chain Ladder", 0.0)),
                ("Hierarchical", lambda: OriginPriorODP(saluz, "Hierarchical", TAU_EB)),
                ("Cape Cod", lambda: OriginPriorODP(saluz, "Cape Cod", None)),
            ]:
                set_sampling_method(method, dimensions=32)
                set_random_seed(20261122)
                model = make_model()
                pr = cProfile.Profile()
                t0 = perf_counter()
                pr.enable()
                if name == "Taylor-Ashe":
                    reserve = model.simulate_reserves()
                else:
                    model.run()
                    reserve = model.total_reserve
                pr.disable()
                elapsed = perf_counter() - t0
                print(f"ODP_PROFILE method={method.value} model={name} n={args.sims} elapsed={elapsed:.6f} mean={reserve.mean():.6f}", flush=True)
                for sorting in ("cumulative", "tottime"):
                    buf = io.StringIO()
                    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(sorting).print_stats(args.lines)
                    print(f"ODP_PROFILE_BEGIN {method.value} {name} {sorting}", flush=True)
                    print(buf.getvalue(), flush=True)
                    print(f"ODP_PROFILE_END {method.value} {name} {sorting}", flush=True)
    finally:
        (pal.config.n_sims, pal.config.rng, pal.config.seed, pal.config.sampling_method, pal.config.qmc_dimensions) = prior


if __name__ == "__main__":
    main()
