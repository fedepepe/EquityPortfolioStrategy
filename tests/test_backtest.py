import math
import pickle
import sys
import types
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pytest

from equity_strategy.backtest import results as results_io
from equity_strategy.backtest.backtest import MARKET, PortfolioBacktest
from equity_strategy.definitions import Algorithms, WeightMethods
from equity_strategy.portfolio.metrics import MetricResult, PerfMetric

WEIGHT_METHODS = [WeightMethods.EQ, WeightMethods.RISK_PARITY, WeightMethods.METRIC]


def _backtest(frames, tmp_path, **options):
    defaults = dict(n_stocks=5, window_lengths=30, rebalance_intervals=20, algorithms=Algorithms.LOMTM,
                    weight_methods=WEIGHT_METHODS, date_start=frames["prices"].index[100], parallel=False,
                    results_dir=tmp_path, endowment=1e5)
    defaults.update(options)
    return PortfolioBacktest(dataset="TEST", **frames, **defaults)


def test_single_backtest_tracks_every_portfolio(market_frames, tmp_path):
    backtest = _backtest(market_frames, tmp_path)
    assert not backtest.parametric_sweep

    backtest.backtest()

    assert set(backtest.portfolios) == {*WEIGHT_METHODS, MARKET}
    n_dates = len(market_frames["prices"]) - 100 + 1
    for portfolio in backtest.portfolios.values():
        assert len(portfolio.value_history) == n_dates
        assert math.isfinite(portfolio.metrics[PerfMetric.SHARPE].value)
    # fees make every strategy lose some value on the first rebalancing day
    assert backtest.portfolios[WeightMethods.EQ].value_history.iloc[1] < 1e5


def test_allocate_returns_one_target_per_weight_method(market_frames, tmp_path):
    backtest = _backtest(market_frames, tmp_path)

    targets = backtest.allocate()

    assert set(targets) == set(WEIGHT_METHODS)
    for allocation in targets.values():
        assert {"Weight", "Quantity", "Price"} <= set(allocation.columns)


def test_grid_sweep_saves_results_that_can_be_reloaded(market_frames, tmp_path):
    backtest = _backtest(market_frames, tmp_path, window_lengths=[20, 30], rebalance_intervals=[10, 20])
    assert backtest.parametric_sweep

    backtest.backtest()

    reloaded = results_io.load_results(backtest.results_pickle_path)
    grids = results_io.results_grid(reloaded, PerfMetric.SHARPE)
    grid = grids[(5, Algorithms.LOMTM, WeightMethods.EQ)]
    assert grid.shape == (2, 2)
    assert list(grid.index) == [10, 20] and list(grid.columns) == [20, 30]
    assert (tmp_path / f"sw_{backtest.results_date}_5_{Algorithms.LOMTM}_{WeightMethods.EQ}.txt").is_file()


def test_multi_start_backtest_writes_alpha_files(market_frames, tmp_path):
    backtest = _backtest(market_frames, tmp_path, date_start=None, weight_methods=[WeightMethods.EQ])

    backtest.backtest_multi_start()

    ic_file = tmp_path / f"_{backtest.results_date}_5_{Algorithms.LOMTM}_{WeightMethods.EQ}_ic.txt"
    assert len(ic_file.read_text().splitlines()) == 1


def test_results_pickled_before_restructuring_can_be_loaded(tmp_path, monkeypatch):
    legacy_module = types.ModuleType("portfolio_metrics")
    legacy_enum = Enum("EnumPerfMetrics", [metric.name for metric in PerfMetric], module="portfolio_metrics")

    @dataclass
    class LegacyMetric:
        label: str
        value: float = None
        format: str = ".2f"
        rev_color_scale: bool = False

    LegacyMetric.__module__, LegacyMetric.__qualname__ = "portfolio_metrics", "PerfMetric"
    legacy_module.EnumPerfMetrics, legacy_module.PerfMetric = legacy_enum, LegacyMetric
    monkeypatch.setitem(sys.modules, "portfolio_metrics", legacy_module)

    path = tmp_path / "sw_2022-01-01.pkl"
    legacy_results = {(10, 50, 20, "sev", "equal"): {legacy_enum.SHARPE: LegacyMetric(label="sharpe", value=1.5)}}
    path.write_bytes(pickle.dumps(legacy_results))
    monkeypatch.delitem(sys.modules, "portfolio_metrics")

    loaded = results_io.load_results(path)

    metric = loaded[(10, 50, 20, "sev", "equal")][PerfMetric.SHARPE]
    assert isinstance(metric, MetricResult)
    assert metric.value == 1.5


def test_latest_results_file_requires_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        results_io.latest_results_file(tmp_path)


def test_results_grid_ignores_unrelated_runs():
    results = {(5, 30, 20, "lomtm", "equal"): {PerfMetric.SHARPE: MetricResult("sharpe", 1.0)},
               (5, 30, 20, "lomtm", "mkt"): {PerfMetric.SHARPE: MetricResult("sharpe", 0.5)}}
    grids = results_io.results_grid(results, PerfMetric.SHARPE)
    assert grids[(5, "lomtm", "mkt")].equals(pd.DataFrame({30: [0.5]}, index=[20]))
