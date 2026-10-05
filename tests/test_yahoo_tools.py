import numpy as np
import pandas as pd
import pytest

from equity_strategy.data import yahoo_tools
from equity_strategy.data.universes import to_yahoo_symbol


@pytest.mark.parametrize("symbol, expected", [
    ("NESN.S", "NESN.SW"),
    ("ERIC-B.ST", "ERIC-B.ST"),
    ("ASML.AS", "ASML.AS"),
])
def test_to_yahoo_symbol_only_maps_swiss_suffix(symbol, expected):
    assert to_yahoo_symbol(symbol) == expected


def test_reindex_by_date_series_matches_calendar_days():
    old = pd.Series([1.0, 2.0, 3.0],
                    index=pd.to_datetime(["2021-01-04 16:00", "2021-01-05 16:00", "2021-01-07 16:00"]))
    new_index = pd.DatetimeIndex(pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]))

    result = yahoo_tools.reindex_by_date(old, new_index)

    assert result.index.equals(new_index)
    np.testing.assert_array_equal(result.values, [1.0, 2.0, np.nan, 3.0])


def test_reindex_by_date_dataframe_interpolates_and_backfills():
    old = pd.DataFrame({"a": [2.0, 4.0]}, index=pd.to_datetime(["2021-01-05", "2021-01-07"]))
    new_index = pd.date_range("2021-01-04", "2021-01-07", freq="D")

    result = yahoo_tools.reindex_by_date(old, new_index, bfill=True)

    np.testing.assert_allclose(result["a"].values, [2.0, 2.0, 3.0, 4.0])


def test_reindex_by_date_rejects_other_types():
    with pytest.raises(TypeError):
        yahoo_tools.reindex_by_date([1, 2], pd.DatetimeIndex([]))


def test_correct_data_anomalies_fixes_misplaced_decimal_point():
    dates = pd.bdate_range("2021-01-01", periods=40)
    prices = pd.DataFrame({"X.L": np.full(40, 100.0)}, index=dates)
    prices.iloc[30, 0] = 10_000.0  # quoted in pence instead of pounds

    corrected = yahoo_tools.correct_data_anomalies(prices)

    assert corrected.iloc[30, 0] == pytest.approx(100.0)


def test_correct_data_anomalies_terminates_when_jump_cannot_be_corrected():
    dates = pd.bdate_range("2021-01-01", periods=40)
    prices = pd.DataFrame({"X.L": np.full(40, 100.0)}, index=dates)
    prices.iloc[5, 0] = 10_000.0  # too early for a reference price level

    corrected = yahoo_tools.correct_data_anomalies(prices)

    assert corrected.iloc[5, 0] == 10_000.0


def test_realized_volatility_penalizes_missing_observations():
    index = pd.date_range("2021-01-04 09:30", periods=11, freq="min", tz="UTC")
    prices = pd.DataFrame({"full": np.exp(np.linspace(0, 0.01, 11)), "gappy": np.exp(np.linspace(0, 0.01, 11))},
                          index=index)
    prices.iloc[5, 1] = np.nan

    volatility = yahoo_tools.realized_volatility(prices, typical_n_obs=11)

    assert volatility["gappy"] > volatility["full"] > 0


def test_adjust_stk_prices_scales_float16_days_to_daily_close():
    index = pd.date_range("2021-08-11 13:30", periods=4, freq="min", tz="UTC")
    intraday = pd.DataFrame({"A": [10.0, 11.0, np.nan, np.nan], "B": [20.0, 20.0, 20.0, 20.0]},
                            index=index).astype(np.float16)
    close_daily = pd.DataFrame({"A": [5.5], "B": [40.0]}, index=pd.DatetimeIndex(["2021-08-11"], tz="UTC"))

    adjusted = yahoo_tools.adjust_stk_prices({"2021-08-11": intraday}, close_daily)["2021-08-11"]

    assert adjusted["A"].iloc[1] == pytest.approx(5.5)
    assert adjusted["B"].iloc[-1] == pytest.approx(40.0)
