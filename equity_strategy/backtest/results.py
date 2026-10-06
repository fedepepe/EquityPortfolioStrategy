"""Persistence and reshaping of backtest results.

Results map ``(n_stocks, window_length, rebalance_interval, algorithm, weight_method)``
to the scalar performance metrics of that run.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from equity_strategy.portfolio.metrics import MetricResult, PerfMetric

ResultKey = tuple[int, int, int, str, str]
Results = dict[ResultKey, dict[PerfMetric, MetricResult]]
GridKey = tuple[int, str, str]  # (n_stocks, algorithm, weight_method)

# Classes referenced by results pickled before the package restructuring
_LEGACY_CLASSES = {
    ("portfolio_metrics", "EnumPerfMetrics"): PerfMetric,
    ("portfolio_metrics", "PerfMetric"): MetricResult,
}


class _ResultsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        legacy_class = _LEGACY_CLASSES.get((module, name))
        if legacy_class is not None:
            return legacy_class
        return super().find_class(module, name)


def save_results(results: Results, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(path: Path) -> Results:
    with open(path, "rb") as handle:
        return _ResultsUnpickler(handle).load()


def latest_results_file(directory: Path) -> Path:
    """The results pickle with the greatest file name (i.e. the latest date for a given tag)."""
    candidates = sorted(directory.glob("*.pkl"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No results files found in {directory}.")
    return candidates[0]


def results_grid(results: Results, metric: PerfMetric) -> dict[GridKey, pd.DataFrame]:
    """One (rebalance interval x window length) table of ``metric`` per (n_stocks, algorithm, weight method)."""
    values = defaultdict(dict)
    for (n_stocks, window_length, rebalance_interval, algorithm, method), metrics in results.items():
        values[(n_stocks, algorithm, method)][(rebalance_interval, window_length)] = metrics[metric].value
    return {key: pd.Series(grid, dtype=float).unstack() for key, grid in values.items()}
