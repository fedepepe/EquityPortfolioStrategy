import numpy as np
import pandas as pd
import pytest

from equity_strategy.definitions import WeightMethods
from equity_strategy.portfolio.metrics import PerfMetric
from equity_strategy.portfolio.portfolio import CASH, Portfolio


def _target(quantities):
    return pd.DataFrame({"Quantity": quantities})


def test_rebalance_charges_fixed_and_proportional_fees():
    portfolio = Portfolio(endowment=10_000)

    portfolio.rebalance(_target({"A": 10.7}), {"A": 100.0}, fee_fixed=1.0, fee_proportional=0.001)

    assert portfolio.holdings["A"] == 10
    assert portfolio.holdings[CASH] == pytest.approx(10_000 - 1_000 - 1.0 - 1.0)
    assert portfolio.total_value == pytest.approx(10_000 - 2.0)
    assert portfolio.traded_value == pytest.approx(1_000)


def test_rebalance_skips_trades_of_zero_shares():
    portfolio = Portfolio(endowment=10_000)
    portfolio.rebalance(_target({"A": 10}), {"A": 100.0}, fee_fixed=1.0, fee_proportional=0.0)
    cash_before = portfolio.holdings[CASH]

    portfolio.rebalance(_target({"A": 10}), {"A": 100.0}, fee_fixed=1.0, fee_proportional=0.0)

    assert portfolio.holdings[CASH] == cash_before


def test_rebalance_closes_positions_outside_target_and_skips_unpriced_stocks():
    portfolio = Portfolio(endowment=10_000)
    portfolio.rebalance(_target({"A": 10, "B": 5}), {"A": 100.0, "B": 50.0}, 0.0, 0.0)

    portfolio.rebalance(_target({"C": 1}), {"A": 110.0, "B": np.nan, "C": 20.0}, 0.0, 0.0)

    assert "A" not in portfolio.holdings
    assert portfolio.holdings["B"] == 5  # no price, position kept
    assert portfolio.holdings["C"] == 1


def test_update_value_and_history():
    portfolio = Portfolio(endowment=1_000)
    portfolio.rebalance(_target({"A": 5}), {"A": 100.0}, 0.0, 0.0)
    date = pd.Timestamp("2021-01-04")

    portfolio.update_value({"A": 120.0})
    portfolio.record(date, save_positions=True)

    assert portfolio.total_value == pytest.approx(500 + 5 * 120)
    assert portfolio.value_history[date] == pytest.approx(1_100)
    assert portfolio.holdings_history.loc[date, "A"] == 5


def _selection():
    return pd.DataFrame({"Metric": [0.5, 0.3], "Pos": [1, 1], "Price": [10.0, 20.0]}, index=["A", "B"])


@pytest.mark.parametrize("method", [WeightMethods.EQ, WeightMethods.MKTCAP, WeightMethods.RISK_PARITY])
def test_compute_weights_are_fully_invested(method):
    volatility = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02]})
    allocation = Portfolio(endowment=1_000).compute_weights(
        _selection(), method, risk_aversion=None, max_leverage=1.0,
        market_cap=pd.Series({"A": 3e9, "B": 1e9}), returns=volatility, volatility=volatility)

    assert allocation["Weight"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(allocation["Quantity"], 1_000 * allocation["Weight"] / allocation["Price"])


def test_risk_parity_weights_are_inversely_proportional_to_volatility():
    volatility = pd.DataFrame({"A": [0.01], "B": [0.02]})
    allocation = Portfolio().compute_weights(_selection(), WeightMethods.RISK_PARITY, None, 1.0,
                                             returns=volatility, volatility=volatility)
    assert allocation.loc["A", "Weight"] == pytest.approx(2 * allocation.loc["B", "Weight"])


def test_compute_weights_rejects_unknown_method():
    with pytest.raises(ValueError):
        Portfolio().compute_weights(_selection(), "unknown", None, 1.0)


def test_analyze_computes_scalar_metrics(market_frames):
    values = market_frames["benchmark_index"] * 1_000
    portfolio = Portfolio.from_value_history(values)

    portfolio.analyze(market_returns=market_frames["benchmark_index"].pct_change())

    scalars = portfolio.scalar_metrics()
    assert PerfMetric.SHARPE in scalars
    assert scalars[PerfMetric.BETA].value == pytest.approx(1.0)
    assert scalars[PerfMetric.TURNOVER].value == 0
