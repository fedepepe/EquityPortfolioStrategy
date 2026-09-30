"""Download, merge, clean and resample Yahoo Finance price data."""
import functools
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

from equity_strategy.config import YMD_DATE_FORMAT
from equity_strategy.data.universes import get_tickers, to_yahoo_symbol

logger = logging.getLogger(__name__)

# Intraday prices, keyed by trading day ('YYYY-MM-DD')
IntradayPrices = dict[str, pd.DataFrame]
PandasObject = TypeVar("PandasObject", pd.Series, pd.DataFrame)

# Tickers with genuine large price jumps that must not be treated as data errors
ANOMALY_CHECK_EXCLUDED_TICKERS = frozenset({"CFEB.BR", "DPH.L", "FNOX.ST", "ORRON.ST", "SIOFF.OL", "SOFF.OL"})
# A daily price ratio beyond this factor (or below its inverse) is treated as a misplaced decimal point
PRICE_JUMP_FACTOR = 8.0
PRICE_LEVEL_WINDOW = 20  # trading days used for the reference (median) price level
MAX_CORRECTION_PASSES = 10

# Rows with fewer valid prices than these fractions of the tickers are discarded
MIN_VALID_FRACTION_DAILY = 0.1
MIN_VALID_FRACTION_INTRADAY = 0.05


def log_step(message: str):
    """Decorator that logs the start and the end of a long-running step."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            logger.info("%s...", message)
            result = function(*args, **kwargs)
            logger.info("%s: done.", message)
            return result

        return wrapper

    return decorator


def day_labels(index: pd.DatetimeIndex) -> pd.Index:
    return index.strftime(YMD_DATE_FORMAT)


@log_step("Downloading data from Yahoo Finance")
def download_stock_data(dataset: str, date_start: datetime, date_end: datetime, interval: str) -> pd.DataFrame:
    """OHLCV history of every constituent, with (field, ticker) MultiIndex columns."""
    frames = []
    for symbol in map(to_yahoo_symbol, get_tickers(dataset)):
        history = yf.Ticker(ticker=symbol).history(start=date_start, end=date_end,
                                                   interval=interval, auto_adjust=True)
        time.sleep(0.1)  # throttle requests to Yahoo Finance
        if history.empty:
            logger.warning("No %s data for %s.", interval, symbol)
            continue
        history.columns = pd.MultiIndex.from_tuples((field, symbol) for field in history.columns)
        frames.append(history)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")


def reindex_by_date(data: PandasObject, new_index: pd.DatetimeIndex, bfill: bool = False) -> PandasObject:
    """Align ``data`` to ``new_index`` by calendar day, ignoring the time of day.

    DataFrames are additionally interpolated in time (and optionally back-filled); Series are not.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected a pandas Series or DataFrame, got {type(data).__name__}.")

    old_days = pd.Index(data.index.date)
    is_last_of_day = ~old_days.duplicated(keep="last")
    by_day = data[is_last_of_day]
    by_day.index = old_days[is_last_of_day]

    aligned = by_day.reindex(pd.Index(new_index.date))
    aligned.index = new_index

    if isinstance(aligned, pd.Series):
        return aligned.astype(float)

    aligned = aligned.apply(pd.to_numeric).interpolate(method="time")
    if bfill:
        aligned = aligned.bfill()
    return aligned


def _count_valid(frame: pd.DataFrame) -> int:
    return int(frame.notna().to_numpy().sum())


@log_step("Building price dataset")
def merge_stock_prices(close_intraday_file: Path, prices_file_prefix: Path) -> IntradayPrices:
    """Merge the downloaded intraday price files into the per-day intraday dataset.

    For a day present in several files, the version with more valid prices wins.
    """
    close_by_day = pd.read_pickle(close_intraday_file) if close_intraday_file.is_file() else {}

    price_files = sorted(prices_file_prefix.parent.glob(f"{prices_file_prefix.name}*.pkl"), reverse=True)
    for price_file in price_files:
        close = pd.read_pickle(price_file)["Close"]
        close = close.dropna(axis=0, thresh=int(MIN_VALID_FRACTION_DAILY * len(close.columns)))
        close = close.replace([np.inf, -np.inf], np.nan)

        labels = day_labels(close.index)
        for day in sorted(set(labels)):
            close_day = close[labels == day]
            if day not in close_by_day or _count_valid(close_day) > _count_valid(close_by_day[day]):
                close_by_day[day] = close_day

    return close_by_day


@log_step("Back-adjusting stock prices")
def adjust_stk_prices(close_by_day: IntradayPrices, close_daily: pd.DataFrame) -> IntradayPrices:
    """Scale each day's intraday prices so the last price matches the dividend/split-adjusted daily close."""
    close_daily = close_daily.interpolate(limit=1)
    daily_labels = day_labels(close_daily.index)

    for day in sorted(close_by_day):
        reference_rows = close_daily[daily_labels == day]
        if reference_rows.empty:
            logger.warning("No daily close for %s: intraday prices left unadjusted.", day)
            continue

        intraday = close_by_day[day]
        intraday = intraday[~intraday.index.duplicated(keep="first")]
        last_valid_price = intraday.ffill().iloc[-1]
        reference_close = reference_rows.iloc[0].reindex(intraday.columns).astype(float)
        # tickers without a reference close or without intraday data keep their raw prices
        adjustment = (reference_close / last_valid_price).fillna(1.0)
        close_by_day[day] = intraday * adjustment

    return close_by_day


@log_step("Building stock information dataset")
def merge_stk_data(stock_info_file: Path, stock_info_prefix: Path) -> dict[str, pd.DataFrame]:
    stock_info_by_day = pd.read_pickle(stock_info_file) if stock_info_file.is_file() else {}

    info_files = sorted(stock_info_prefix.parent.glob(f"{stock_info_prefix.name}*.pkl"), reverse=True)
    for info_file in info_files:
        info, download_date = pd.read_pickle(info_file)
        stock_info_by_day[download_date.strftime(YMD_DATE_FORMAT)] = info
    return stock_info_by_day


def merge_mktcap_data(market_cap_file: Path, market_cap_prefix: Path) -> pd.DataFrame:
    """Merge daily market-capitalization snapshots into a (date x ticker) DataFrame."""
    market_cap = pd.read_pickle(market_cap_file) if market_cap_file.is_file() else pd.DataFrame()

    for snapshot_file in sorted(market_cap_prefix.parent.glob(f"{market_cap_prefix.name}*.pkl")):
        snapshot, snapshot_date = pd.read_pickle(snapshot_file)
        if pd.Timestamp(snapshot_date) not in market_cap.index:
            market_cap = pd.concat([market_cap, snapshot.rename(snapshot_date).to_frame().T])

    market_cap.index = pd.DatetimeIndex(pd.to_datetime(market_cap.index.to_series()))
    return market_cap


def realized_volatility(close_intraday: pd.DataFrame, typical_n_obs: int) -> pd.Series:
    """Daily realized volatility from intraday prices.

    Missing observations inflate the estimate through a penalty factor, and the estimate is
    rescaled for days with an unusual number of intraday samples.
    """
    squared_log_returns = np.log(close_intraday).diff().dropna(how="all").pow(2)
    missing_fraction = squared_log_returns.isna().sum() / squared_log_returns.shape[0]
    penalty = 1.0 / (1.0 - missing_fraction) ** 2

    realized_variance = squared_log_returns.interpolate(method="time").sum()
    realized_variance *= typical_n_obs / close_intraday.shape[0]
    return (np.sqrt(realized_variance) * penalty).replace(0, np.nan)


def get_returns_volat(close_by_day: IntradayPrices, close_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns (from daily closes) and daily realized volatility (from intraday prices)."""
    days = sorted(close_by_day)
    typical_n_obs = statistics.mode(close_by_day[day].shape[0] for day in days)

    volatility_rows = []
    for day in days:
        intraday = close_by_day[day]
        intraday = intraday.dropna(axis=0, thresh=int(MIN_VALID_FRACTION_INTRADAY * len(intraday.columns))).copy()
        intraday.index = pd.DatetimeIndex(pd.to_datetime(intraday.index, utc=True))

        volatility = realized_volatility(intraday, typical_n_obs).to_frame().T
        volatility.index = pd.to_datetime([day])
        if volatility.shape[1] > 0:
            volatility_rows.append(volatility)

    volatility = reindex_by_date(pd.concat(volatility_rows), close_daily.index)
    close_daily = close_daily.dropna(axis=1, how="all")
    volatility = volatility.reindex_like(close_daily)

    returns = close_daily.pct_change(fill_method=None).dropna(axis=0, how="all")
    volatility = volatility.loc[returns.index]
    return returns, volatility


def _price_jump_dates(prices: pd.Series) -> list[pd.Timestamp]:
    change = prices.pct_change(fill_method=None)
    is_jump = (change > PRICE_JUMP_FACTOR - 1) | (change < 1.0 / PRICE_JUMP_FACTOR - 1)
    return change.index[is_jump].tolist()


def correct_data_anomalies(prices: pd.DataFrame) -> pd.DataFrame:
    """Fix prices off by a power-of-ten-like factor (e.g. pence quoted as pounds). Modifies ``prices`` in place."""
    for ticker in prices.columns:
        if ticker in ANOMALY_CHECK_EXCLUDED_TICKERS:
            continue

        for _ in range(MAX_CORRECTION_PASSES):
            jump_dates = _price_jump_dates(prices[ticker])
            if not jump_dates:
                break

            price_level = prices[ticker].rolling(PRICE_LEVEL_WINDOW).median()
            corrected = False
            for date in jump_dates:
                price_ratio = prices.loc[date, ticker] / price_level[date]
                if price_ratio > PRICE_JUMP_FACTOR:
                    scale = np.round(price_ratio)
                elif price_ratio < 1.0 / PRICE_JUMP_FACTOR:
                    scale = 1.0 / np.round(1.0 / price_ratio)
                else:
                    continue
                logger.warning("Correcting price of %s at %s by a factor %s.", ticker, date, scale)
                prices.loc[date, ticker] /= scale
                corrected = True

            if not corrected:
                break

    return prices


def squeeze_to_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    """Return the first column of a single-column DataFrame (e.g. from ``yf.download``)."""
    return data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
