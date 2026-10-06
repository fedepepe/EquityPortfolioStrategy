import pandas as pd
import pytest

from equity_strategy.definitions import Algorithms, WeightMethods
from equity_strategy.portfolio.portfolio import Portfolio
from equity_strategy.selection.stock_picker import MIN_DEPENDENCE_SCORE, StockPicker


def _picker(frames, algorithm, n_stocks=5, window=60):
    def window_of(frame):
        return frame.tail(window)

    return StockPicker(n_stocks=n_stocks, algorithm=algorithm,
                       returns=window_of(frames["returns"]),
                       volatility=window_of(frames["volatility"]),
                       sharpe=window_of(frames["returns"] / frames["volatility"]),
                       market_cap=window_of(frames["market_cap"]))


def test_long_only_momentum_selects_highest_mean_returns(market_frames):
    selection = _picker(market_frames, Algorithms.LOMTM).pick_stocks(parallel=False)

    expected = market_frames["returns"].tail(60).mean().nlargest(5).index
    assert set(selection.index) == set(expected)
    assert list(selection.columns) == ["Metric", "Pos"]
    assert (selection["Pos"] == 1).all()


def test_long_only_momentum_supports_metric_weighting(market_frames):
    selection = _picker(market_frames, Algorithms.LORMMTM).pick_stocks(parallel=False)
    selection["Price"] = 10.0

    allocation = Portfolio().compute_weights(selection, WeightMethods.METRIC, None, 1.0,
                                             returns=market_frames["returns"],
                                             volatility=market_frames["volatility"])

    pd.testing.assert_series_equal(allocation["Weight"], selection["Metric"], check_names=False)


def test_long_only_dependence_selection_applies_threshold(market_frames):
    selection = _picker(market_frames, Algorithms.LOLIN).pick_stocks(parallel=False)

    assert len(selection) <= 5
    assert (selection["Metric"] > MIN_DEPENDENCE_SCORE).all()
    assert (selection["Pos"] == 1).all()


@pytest.mark.parametrize("algorithm", [Algorithms.MTM, Algorithms.RMLIN])
def test_long_short_selection_assigns_positions(market_frames, algorithm):
    selection = _picker(market_frames, algorithm, n_stocks=15).pick_stocks(parallel=False)

    assert len(selection) == 15
    assert set(selection["Pos"].dropna()) <= {1, -1}
    assert (selection["Pos"] == -1).any()


def test_unknown_algorithm_raises(market_frames):
    with pytest.raises(ValueError):
        _picker(market_frames, "unknown").pick_stocks(parallel=False)
