"""Critical-value tables for fast exact Gamma prediction."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .intervals import (
    dispersion,
    prediction_log_multiplier,
    prediction_log_multipliers,
)


DEFAULT_TABLE_PATH = (
    Path(__file__).resolve().parent / "tables" / "gamma_prediction_critical_values.npz"
)

REFINED_D_ADDITIONS = np.array(
    [
        2e-6,
        5e-6,
        2e-5,
        5e-5,
        2e-4,
        5e-4,
        0.003,
        0.004,
        0.03,
        0.075,
        0.15,
        0.3,
        0.75,
        1.5,
        3.0,
        6.0,
        7.0,
        8.0,
        12.0,
        18.0,
        22.5,
        27.5,
        32.5,
        35.0,
        37.5,
        42.5,
        45.0,
        47.5,
    ]
)
LOWER_TAIL_MAX_P = 0.25
LOWER_TAIL_BLEND_MIN_P = 0.225


@dataclass
class CriticalValueTable:
    """Interpolated table of ``log M(n, p, d)`` values.

    A quintic tensor-product spline is evaluated in ``log(d)`` and logit
    probability.
    Values outside the stored grid are rejected by default; callers can request
    the exact scalar fallback explicitly.
    """

    n_grid: np.ndarray
    d_grid: np.ndarray
    p_grid: np.ndarray
    log_multiplier: np.ndarray
    refined_n_grid: np.ndarray | None = None
    refined_d_grid: np.ndarray | None = None
    refined_p_grid: np.ndarray | None = None
    refined_log_multiplier: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.n_grid = np.asarray(self.n_grid, dtype=int)
        self.d_grid = np.asarray(self.d_grid, dtype=float)
        self.p_grid = np.asarray(self.p_grid, dtype=float)
        self.log_multiplier = np.asarray(self.log_multiplier, dtype=float)
        expected = (len(self.n_grid), len(self.d_grid), len(self.p_grid))
        if self.log_multiplier.shape != expected:
            raise ValueError(f"log_multiplier must have shape {expected}")
        if np.any(np.diff(self.n_grid) <= 0):
            raise ValueError("n_grid must be strictly increasing")
        if np.any(self.d_grid <= 0) or np.any(np.diff(self.d_grid) <= 0):
            raise ValueError("d_grid must be positive and increasing")
        if np.any((self.p_grid <= 0) | (self.p_grid >= 1)) or np.any(
            np.diff(self.p_grid) <= 0
        ):
            raise ValueError("p_grid must increase within (0, 1)")
        self._log_d_grid = np.log(self.d_grid)
        self._p_axis = np.log(self.p_grid) - np.log1p(-self.p_grid)
        self._interpolators = [
            RectBivariateSpline(
                self._log_d_grid,
                self._p_axis,
                values,
                kx=min(5, len(self.d_grid) - 1),
                ky=min(5, len(self.p_grid) - 1),
                s=0.0,
            )
            for values in self.log_multiplier
        ]
        self._lower_interpolators = self._build_lower_interpolators(
            self.d_grid,
            self.p_grid,
            self.log_multiplier,
        )
        refined = (
            self.refined_n_grid,
            self.refined_d_grid,
            self.refined_p_grid,
            self.refined_log_multiplier,
        )
        if any(value is not None for value in refined):
            if not all(value is not None for value in refined):
                raise ValueError("all refined grids and values must be provided")
            self.refined_n_grid = np.asarray(self.refined_n_grid, dtype=int)
            self.refined_d_grid = np.asarray(self.refined_d_grid, dtype=float)
            self.refined_p_grid = np.asarray(self.refined_p_grid, dtype=float)
            self.refined_log_multiplier = np.asarray(
                self.refined_log_multiplier, dtype=float
            )
            expected = (
                len(self.refined_n_grid),
                len(self.refined_d_grid),
                len(self.refined_p_grid),
            )
            if self.refined_log_multiplier.shape != expected:
                raise ValueError(f"refined_log_multiplier must have shape {expected}")
            if not np.all(np.isin(self.refined_n_grid, self.n_grid)):
                raise ValueError("refined_n_grid must be a subset of n_grid")
            if np.any(np.diff(self.refined_n_grid) <= 0):
                raise ValueError("refined_n_grid must be strictly increasing")
            if np.any(self.refined_d_grid <= 0) or np.any(
                np.diff(self.refined_d_grid) <= 0
            ):
                raise ValueError("refined_d_grid must be positive and increasing")
            if np.any(
                (self.refined_p_grid <= 0) | (self.refined_p_grid >= 1)
            ) or np.any(np.diff(self.refined_p_grid) <= 0):
                raise ValueError("refined_p_grid must increase within (0, 1)")
            self._refined_log_d_grid = np.log(self.refined_d_grid)
            self._refined_p_axis = np.log(self.refined_p_grid) - np.log1p(
                -self.refined_p_grid
            )
            self._refined_interpolators = [
                RectBivariateSpline(
                    self._refined_log_d_grid,
                    self._refined_p_axis,
                    values,
                    kx=min(5, len(self.refined_d_grid) - 1),
                    ky=min(5, len(self.refined_p_grid) - 1),
                    s=0.0,
                )
                for values in self.refined_log_multiplier
            ]
            self._refined_lower_interpolators = self._build_lower_interpolators(
                self.refined_d_grid,
                self.refined_p_grid,
                self.refined_log_multiplier,
            )
        else:
            self._refined_interpolators = []
            self._refined_lower_interpolators = []

    @staticmethod
    def _build_lower_interpolators(
        d_grid: np.ndarray,
        p_grid: np.ndarray,
        values: np.ndarray,
    ) -> list[RectBivariateSpline | None]:
        """Build the transformed lower-tail surfaces suggested by the tail law."""
        mask = p_grid <= LOWER_TAIL_MAX_P
        if np.count_nonzero(mask) < 2:
            return [None] * len(values)
        lower_values = values[:, :, mask]
        if np.any(lower_values >= 0.0):
            raise ValueError("lower-tail log multipliers must be negative")
        tail_axis = np.log(-np.log(p_grid[mask]))[::-1]
        return [
            RectBivariateSpline(
                np.log(d_grid),
                tail_axis,
                np.log(-surface[:, ::-1]),
                kx=min(5, len(d_grid) - 1),
                ky=min(5, np.count_nonzero(mask) - 1),
                s=0.0,
            )
            for surface in lower_values
        ]

    def save(self, path: str | Path) -> None:
        values = dict(
            n_grid=self.n_grid,
            d_grid=self.d_grid,
            p_grid=self.p_grid,
            log_multiplier=self.log_multiplier,
        )
        if self.refined_n_grid is not None:
            values.update(
                refined_n_grid=self.refined_n_grid,
                refined_d_grid=self.refined_d_grid,
                refined_p_grid=self.refined_p_grid,
                refined_log_multiplier=self.refined_log_multiplier,
            )
        np.savez_compressed(Path(path), **values)

    def save_text(self, path: str | Path) -> None:
        """Write a tab-separated table with robust extreme-tail multipliers."""
        from decimal import Decimal, localcontext

        with Path(path).open("w", encoding="ascii", newline="\n") as stream:
            stream.write("# gamma-prediction critical values\n")
            stream.write("# log_multiplier is the canonical stored value.\n")
            stream.write("# multiplier is exp(log_multiplier), evaluated in Decimal.\n")
            stream.write("n\td\tp\tlog_multiplier\tmultiplier\n")
            with localcontext() as context:
                context.prec = 18
                context.Emin = -999999
                context.Emax = 999999
                for n in self.n_grid:
                    _, _, d_grid, p_grid, log_values, _, _ = self._surface(int(n))
                    for d_index, d in enumerate(d_grid):
                        for p_index, p in enumerate(p_grid):
                            log_value = float(log_values[d_index, p_index])
                            multiplier = Decimal(str(log_value)).exp()
                            stream.write(
                                f"{int(n)}\t{float(d):.12g}\t{float(p):.12g}\t"
                                f"{log_value:.17g}\t{multiplier:.16E}\n"
                            )

    @classmethod
    def load(cls, path: str | Path) -> "CriticalValueTable":
        with np.load(Path(path), allow_pickle=False) as values:
            names = set(values.files)
            if "log_multiplier" in names:
                log_values = values["log_multiplier"]
            elif "log_c" in names:
                log_values = values["log_c"]
            else:
                raise ValueError("table archive has no log multiplier array")
            return cls(
                values["n_grid"],
                values["d_grid"],
                values["p_grid"],
                log_values,
                values["refined_n_grid"] if "refined_n_grid" in names else None,
                values["refined_d_grid"] if "refined_d_grid" in names else None,
                values["refined_p_grid"] if "refined_p_grid" in names else None,
                values["refined_log_multiplier"]
                if "refined_log_multiplier" in names
                else None,
            )

    def _n_index(self, n: int) -> int:
        index = int(np.searchsorted(self.n_grid, int(n)))
        if index >= len(self.n_grid) or self.n_grid[index] != int(n):
            raise ValueError(f"n={n} is outside the table grid")
        return index

    def _surface(self, n: int):
        n_index = self._n_index(n)
        if self.refined_n_grid is not None:
            refined_index = int(np.searchsorted(self.refined_n_grid, n))
            if (
                refined_index < len(self.refined_n_grid)
                and self.refined_n_grid[refined_index] == n
            ):
                return (
                    self._refined_interpolators[refined_index],
                    self._refined_lower_interpolators[refined_index],
                    self.refined_d_grid,
                    self.refined_p_grid,
                    self.refined_log_multiplier[refined_index],
                    self._refined_log_d_grid,
                    self._refined_p_axis,
                )
        return (
            self._interpolators[n_index],
            self._lower_interpolators[n_index],
            self.d_grid,
            self.p_grid,
            self.log_multiplier[n_index],
            self._log_d_grid,
            self._p_axis,
        )

    def _interpolate(self, n: int, d: float, p: float) -> float:
        if d <= 0:
            raise ValueError("d must be positive")
        (
            interpolator,
            lower_interpolator,
            d_grid,
            _,
            _,
            _,
            p_axis,
        ) = self._surface(n)
        if d < d_grid[0] or d > d_grid[-1]:
            raise ValueError("d is outside the table grid")
        target_p = math.log(p) - math.log1p(-p)
        if target_p < p_axis[0] or target_p > p_axis[-1]:
            raise ValueError("p is outside the table grid")
        log_d = math.log(float(d))
        central_value = float(interpolator(log_d, target_p)[0, 0])
        if p > LOWER_TAIL_MAX_P or lower_interpolator is None:
            return central_value

        lower_value = -math.exp(
            float(lower_interpolator(log_d, math.log(-math.log(p)))[0, 0])
        )
        if p < LOWER_TAIL_BLEND_MIN_P:
            return lower_value

        position = (p - LOWER_TAIL_BLEND_MIN_P) / (
            LOWER_TAIL_MAX_P - LOWER_TAIL_BLEND_MIN_P
        )
        weight = position * position * (3.0 - 2.0 * position)
        return (1.0 - weight) * lower_value + weight * central_value

    def log_multiplier_at(
        self,
        n: int,
        d: float,
        p: float,
        *,
        exact_fallback: bool = False,
    ) -> float:
        """Return ``log M(n, p, d)`` by interpolation or exact fallback."""
        if not (0.0 < p < 1.0):
            raise ValueError("p must lie strictly between 0 and 1")
        try:
            _, _, d_grid, p_grid, log_values, _, _ = self._surface(n)
            d_index = int(np.searchsorted(d_grid, d))
            p_index = int(np.searchsorted(p_grid, p))
            if (
                d_index < len(d_grid)
                and d_grid[d_index] == d
                and p_index < len(p_grid)
                and p_grid[p_index] == p
            ):
                return float(log_values[d_index, p_index])
            return self._interpolate(n, d, p)
        except ValueError:
            if not exact_fallback:
                raise
            return prediction_log_multiplier(int(n), float(d), float(p))

    def multiplier_at(self, n: int, d: float, p: float, **kwargs) -> float:
        return float(np.exp(self.log_multiplier_at(n, d, p, **kwargs)))

    def log_multiplier_array(
        self,
        n: int,
        d: np.ndarray,
        p: float,
        *,
        exact_fallback: bool = False,
    ) -> np.ndarray:
        """Vectorized log multipliers for an array of dispersions at fixed p."""
        dispersions = np.asarray(d, dtype=float)
        if np.any(dispersions <= 0.0):
            raise ValueError("d must be positive")

        (
            interpolator,
            lower_interpolator,
            d_grid,
            _,
            _,
            _,
            p_axis,
        ) = self._surface(n)
        in_range = (dispersions >= d_grid[0]) & (dispersions <= d_grid[-1])
        if not exact_fallback and not np.all(in_range):
            raise ValueError("d is outside the table grid")

        result = np.empty_like(dispersions)
        target_p = math.log(p) - math.log1p(-p)
        if target_p < p_axis[0] or target_p > p_axis[-1]:
            if not exact_fallback:
                raise ValueError("p is outside the table grid")
            in_range[:] = False
        if np.any(in_range):
            log_d = np.log(dispersions[in_range])
            central_values = interpolator.ev(
                log_d,
                np.full(np.count_nonzero(in_range), target_p),
            )
            if p > LOWER_TAIL_MAX_P or lower_interpolator is None:
                result[in_range] = central_values
            else:
                lower_values = -np.exp(
                    lower_interpolator.ev(
                        log_d,
                        np.full(
                            np.count_nonzero(in_range), math.log(-math.log(p))
                        ),
                    )
                )
                if p < LOWER_TAIL_BLEND_MIN_P:
                    result[in_range] = lower_values
                else:
                    position = (p - LOWER_TAIL_BLEND_MIN_P) / (
                        LOWER_TAIL_MAX_P - LOWER_TAIL_BLEND_MIN_P
                    )
                    weight = position * position * (3.0 - 2.0 * position)
                    result[in_range] = (
                        (1.0 - weight) * lower_values + weight * central_values
                    )
        for index in zip(*np.where(~in_range)):
            result[index] = prediction_log_multiplier(
                int(n), float(dispersions[index]), float(p)
            )
        return result

    def interval_for_sample(
        self,
        sample: Iterable[float],
        p_lower: float = 0.025,
        p_upper: float = 0.975,
        *,
        exact_fallback: bool = False,
    ) -> tuple[float, float]:
        values = np.asarray(list(sample), dtype=float)
        if len(values) < 2:
            raise ValueError("at least two observations are required")
        xbar = float(np.mean(values))
        d = dispersion(values)
        return (
            xbar
            * self.multiplier_at(
                len(values), d, p_lower, exact_fallback=exact_fallback
            ),
            xbar
            * self.multiplier_at(
                len(values), d, p_upper, exact_fallback=exact_fallback
            ),
        )


def build_critical_value_table(
    n_grid: Iterable[int],
    d_grid: Iterable[float],
    p_grid: Iterable[float],
    *,
    progress: bool = True,
) -> CriticalValueTable:
    """Build an exact table using the corrected paper implementation."""
    ns = np.asarray(sorted(set(int(n) for n in n_grid)), dtype=int)
    ds = np.asarray(list(d_grid), dtype=float)
    ps = np.asarray(list(p_grid), dtype=float)
    if len(ns) == 0 or len(ds) == 0 or len(ps) == 0:
        raise ValueError("all table grids must be non-empty")
    values = np.empty((len(ns), len(ds), len(ps)), dtype=float)
    for i, n in enumerate(ns):
        for j, d in enumerate(ds):
            if progress:
                print(f"n={n} d={d:g}", flush=True)
            values[i, j, :] = prediction_log_multipliers(int(n), float(d), ps)
            values[i, j, :] = np.maximum.accumulate(values[i, j, :])
    return CriticalValueTable(ns, ds, ps, values)


def _build_refined_n(arguments):
    n, d_grid, p_grid = arguments
    values = np.empty((len(d_grid), len(p_grid)), dtype=float)
    for index, d in enumerate(d_grid):
        values[index] = prediction_log_multipliers(int(n), float(d), p_grid)
    return values


def _insert_transformed_midpoints(
    grid: np.ndarray,
    transform,
    inverse,
) -> np.ndarray:
    """Bisect every grid cell in the coordinate used for interpolation."""
    transformed = transform(np.asarray(grid, dtype=float))
    midpoints = inverse(0.5 * (transformed[:-1] + transformed[1:]))
    return np.sort(np.r_[grid, midpoints])


def refine_critical_value_table(
    table: CriticalValueTable,
    *,
    d_additions: Iterable[float] = REFINED_D_ADDITIONS,
    p_additions: Iterable[float] = (),
    workers: int = 1,
    progress: bool = True,
) -> CriticalValueTable:
    """Add the production lookup grid for every stored ``n <= 100``.

    Dispersion cells are bisected in ``log(d)``.  By default, the probability
    grid contains only the table's 17 critical probability levels, because the
    production lookup is intended for those levels rather than interpolation
    between them.  Additional probability levels can be requested explicitly.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    n_grid = table.n_grid[table.n_grid <= 100]
    d_grid = np.unique(np.r_[table.d_grid, np.asarray(list(d_additions), dtype=float)])
    p_grid = np.unique(np.r_[table.p_grid, np.asarray(list(p_additions), dtype=float)])
    d_grid = _insert_transformed_midpoints(d_grid, np.log, np.exp)
    reusable: dict[int, np.ndarray] = {}
    if (
        table.refined_n_grid is not None
        and np.array_equal(table.refined_d_grid, d_grid)
        and np.array_equal(table.refined_p_grid, p_grid)
    ):
        reusable = {
            int(n): table.refined_log_multiplier[index]
            for index, n in enumerate(table.refined_n_grid)
        }
    missing = [int(n) for n in n_grid if int(n) not in reusable]
    arguments = [(n, d_grid, p_grid) for n in missing]
    if workers == 1:
        generated = []
        for arguments_for_n in arguments:
            generated.append(_build_refined_n(arguments_for_n))
            if progress:
                print(f"refined n={arguments_for_n[0]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = list(executor.map(_build_refined_n, arguments))
    generated_by_n = dict(zip(missing, generated))
    refined_values = np.asarray(
        [reusable.get(int(n), generated_by_n.get(int(n))) for n in n_grid]
    )
    refined_values = np.maximum.accumulate(refined_values, axis=2)
    return CriticalValueTable(
        table.n_grid,
        table.d_grid,
        table.p_grid,
        table.log_multiplier,
        n_grid,
        d_grid,
        p_grid,
        refined_values,
    )


def load_default_table() -> CriticalValueTable:
    """Load the packaged lookup table through n=100, d=1e-6..50, p=.001..999."""
    return CriticalValueTable.load(DEFAULT_TABLE_PATH)
