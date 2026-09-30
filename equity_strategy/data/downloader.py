"""Download Yahoo Finance data for a stock universe and load it from the on-disk cache.

Cached files, all under ``<PROJECT_ROOT>/<dataset>/``:

- ``prices_intraday_<date>.pkl``  raw intraday OHLCV downloads
- ``close_intraday_adj.pkl``      merged, back-adjusted intraday closes, keyed by day
- ``prices_daily.pkl``            daily OHLCV (dividend/split adjusted)
- ``stk_data_<date>.pkl`` / ``stk_data.pkl``  ``yf.Ticker.info`` snapshots
- ``mktcap.pkl``                  market capitalization history
- ``retvol_daily_<date>.pkl``     daily returns / realized volatility / market cap (recomputed daily)
- ``mkt_data_<date>.pkl``         benchmark index and returns (re-downloaded daily)
"""
import datetime as dt
import logging
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import yfinance as yf

from equity_strategy.config import YMD_DATE_FORMAT, dataset_dir
from equity_strategy.data import yahoo_tools
from equity_strategy.data.universes import get_tickers, to_yahoo_symbol
from equity_strategy.definitions import StockUniverses

logger = logging.getLogger(__name__)

BENCHMARK_SYMBOLS = {
    StockUniverses.SP500: "^SP500TR",
    StockUniverses.STOXXE600: "^STOXX",
}

# Yahoo Finance serves 1-minute bars for the last 30 days only, in chunks of at most 7 days
ONE_MINUTE_HISTORY_DAYS = 30
INTRADAY_CHUNK_DAYS = 6
DEFAULT_INTRADAY_LOOKBACK_DAYS = 60
# Days with a smaller fraction of valid intraday prices are discarded
MIN_VALID_INTRADAY_FRACTION = 0.4
STOCK_INFO_DOWNLOAD_ATTEMPTS = 5


class StockData(NamedTuple):
    close: pd.DataFrame  # daily adjusted closes
    returns: pd.DataFrame  # daily simple returns
    volatility: pd.DataFrame  # daily realized volatility
    market_cap: pd.DataFrame
    close_intraday: yahoo_tools.IntradayPrices


class MarketData(NamedTuple):
    returns: pd.Series
    index: pd.Series


def _save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


class YahooDataDownloader:
    def __init__(self, dataset: str, data_dir: Path | None = None):
        self.dataset = dataset
        self.data_dir = data_dir if data_dir is not None else dataset_dir(dataset)

        self.close_intraday_file = self.data_dir / "close_intraday_adj.pkl"
        self.prices_daily_file = self.data_dir / "prices_daily.pkl"
        self.stock_info_file = self.data_dir / "stk_data.pkl"
        self.market_cap_file = self.data_dir / "mktcap.pkl"

        self.prices_intraday_prefix = self.data_dir / "prices_intraday_"
        self.stock_info_prefix = self.data_dir / "stk_data_"
        self.market_cap_prefix = self.data_dir / "mktcap_"
        self.returns_volatility_prefix = self.data_dir / "retvol_daily_"
        self.market_data_prefix = self.data_dir / "mkt_data_"

        self.start_date: dt.datetime | None = None
        self.end_date: dt.datetime | None = None

    @staticmethod
    def _dated_file(prefix: Path, date: dt.date | None = None) -> Path:
        date = date if date is not None else dt.date.today()
        return prefix.with_name(f"{prefix.name}{date.strftime(YMD_DATE_FORMAT)}.pkl")

    @staticmethod
    def _delete_stale_files(prefix: Path, keep: Path) -> None:
        for path in prefix.parent.glob(f"{prefix.name}*.pkl"):
            if path.resolve() != keep.resolve():
                path.unlink()

    # ----- Downloading -----

    def download_latest_data(self, last_download_date: str | dt.datetime | None = None) -> None:
        """Download new intraday prices and stock info, then rebuild the adjusted price datasets."""
        prices = self.download_intraday_prices(last_download_date)
        if prices is not None:
            tickers = list(prices["Close"].columns)
        else:
            tickers = [to_yahoo_symbol(symbol) for symbol in get_tickers(self.dataset)]
        self.download_stock_info(tickers)
        self.update_daily_prices()

        # force a fresh benchmark download on the next load
        self._dated_file(self.market_data_prefix).unlink(missing_ok=True)

    def download_intraday_prices(self,
                                 last_download_date: str | dt.datetime | None = None
                                 ) -> pd.DataFrame | None:
        """Download intraday prices since the last download; ``None`` if already up to date."""
        if last_download_date is None:
            if self.close_intraday_file.is_file():
                last_day = max(pd.read_pickle(self.close_intraday_file))
                last_download_date = dt.datetime.strptime(last_day, YMD_DATE_FORMAT)
            else:
                last_download_date = dt.datetime.today() - dt.timedelta(days=DEFAULT_INTRADAY_LOOKBACK_DAYS)
        elif isinstance(last_download_date, str):
            last_download_date = dt.datetime.strptime(last_download_date, YMD_DATE_FORMAT)

        today = dt.datetime.today()
        if last_download_date >= today - dt.timedelta(hours=12):
            logger.info("Intraday prices are up to date.")
            return None

        logger.info("Downloading latest intraday stock prices...")
        chunks = []
        end_date = today
        while last_download_date < end_date:
            start_date = end_date - dt.timedelta(days=INTRADAY_CHUNK_DAYS)
            recent = start_date >= today - dt.timedelta(days=ONE_MINUTE_HISTORY_DAYS)
            interval = "1m" if recent else "5m"
            chunks.append(yahoo_tools.download_stock_data(self.dataset, start_date, end_date, interval))
            end_date = start_date

        prices = pd.concat(reversed(chunks))
        prices.index = pd.to_datetime(prices.index.to_series(), utc=True)
        prices = prices.sort_index()

        _save_pickle(prices, self._dated_file(self.prices_intraday_prefix))
        return prices

    def update_daily_prices(self) -> None:
        """Download daily prices over the intraday history and back-adjust the intraday prices with them."""
        close_by_day = yahoo_tools.merge_stock_prices(self.close_intraday_file, self.prices_intraday_prefix)

        days = sorted(close_by_day)
        self.start_date = dt.datetime.strptime(days[0], YMD_DATE_FORMAT)
        self.end_date = dt.datetime.strptime(days[-1], YMD_DATE_FORMAT) + dt.timedelta(days=1)
        prices_daily = yahoo_tools.download_stock_data(self.dataset, self.start_date, self.end_date, "1d")

        prices_daily.index = pd.to_datetime(prices_daily.index.to_series(), utc=True)
        prices_daily = prices_daily.sort_index().ffill(limit=3)
        _save_pickle(prices_daily, self.prices_daily_file)

        close_adjusted = yahoo_tools.adjust_stk_prices(close_by_day, prices_daily["Close"])
        _save_pickle(close_adjusted, self.close_intraday_file)

    def download_stock_info(self, tickers: Sequence[str]) -> None:
        """Download a ``yf.Ticker.info`` snapshot for each ticker and merge it into the history."""
        logger.info("Downloading stock information for %d tickers...", len(tickers))
        rows = []
        for ticker in tickers:
            for _ in range(STOCK_INFO_DOWNLOAD_ATTEMPTS):
                try:
                    rows.append(pd.Series(yf.Ticker(ticker).info).rename(ticker))
                    time.sleep(0.5)  # throttle requests to Yahoo Finance
                    break
                except Exception as error:  # yfinance raises a variety of network/parsing errors
                    logger.warning("Stock info download for %s failed: %s", ticker, error)
        stock_info = pd.concat(rows, axis=1).transpose() if rows else pd.DataFrame()

        today = dt.date.today()
        _save_pickle([stock_info, today], self._dated_file(self.stock_info_prefix, today))

        merged = yahoo_tools.merge_stk_data(self.stock_info_file, self.stock_info_prefix)
        _save_pickle(merged, self.stock_info_file)

    def update_market_cap(self) -> None:
        """Extract today's market capitalizations from the stock info snapshot and merge them into ``mktcap.pkl``.

        Requires :meth:`download_stock_info` to have run today. Not part of :meth:`download_latest_data`.
        """
        today = dt.date.today()
        stock_info, snapshot_date = pd.read_pickle(self._dated_file(self.stock_info_prefix, today))
        if "marketCap" in stock_info.columns:
            _save_pickle([stock_info["marketCap"], snapshot_date],
                         self._dated_file(self.market_cap_prefix, snapshot_date))

        market_cap = yahoo_tools.merge_mktcap_data(self.market_cap_file, self.market_cap_prefix)
        _save_pickle(market_cap, self.market_cap_file)

    # ----- Loading -----

    def get_stock_data(self) -> StockData:
        """Load prices and derived daily returns/volatility, recomputing the latter once per day."""
        close_intraday = pd.read_pickle(self.close_intraday_file)
        close = yahoo_tools.correct_data_anomalies(pd.read_pickle(self.prices_daily_file)["Close"])

        cache_file = self._dated_file(self.returns_volatility_prefix)
        if cache_file.is_file():
            logger.info("Loading stock data from %s.", cache_file.name)
            returns, volatility, market_cap = pd.read_pickle(cache_file)
        else:
            logger.info("Computing daily returns and realized volatility...")
            close_intraday = {day: prices for day, prices in close_intraday.items()
                              if prices.size == 0 or prices.notna().to_numpy().mean() >= MIN_VALID_INTRADAY_FRACTION}
            returns, volatility = yahoo_tools.get_returns_volat(close_intraday, close)
            market_cap = yahoo_tools.reindex_by_date(pd.read_pickle(self.market_cap_file), returns.index, bfill=True)
            _save_pickle([returns, volatility, market_cap], cache_file)

        self._delete_stale_files(self.returns_volatility_prefix, keep=cache_file)

        if self.start_date is None:
            self.start_date = close.index[0]
        if self.end_date is None:
            self.end_date = close.index[-1] + dt.timedelta(days=1)

        return StockData(close, returns, volatility, market_cap, close_intraday)

    def get_market_data(self, index: pd.DatetimeIndex) -> MarketData:
        """Benchmark index level and daily returns aligned to ``index``, downloaded once per day."""
        cache_file = self._dated_file(self.market_data_prefix)

        if cache_file.is_file():
            logger.info("Loading market data from %s.", cache_file.name)
            returns, index_level = pd.read_pickle(cache_file)
        else:
            if self.dataset not in BENCHMARK_SYMBOLS:
                raise ValueError(f"No benchmark index defined for {self.dataset!r}.")
            start_date = index[0] - dt.timedelta(days=3)
            end_date = index[-1] + dt.timedelta(days=1)

            logger.info("Downloading market data...")
            data = yf.download(BENCHMARK_SYMBOLS[self.dataset], start=start_date, end=end_date, interval="1d",
                               auto_adjust=False, actions=False, threads=False)
            index_level_raw = yahoo_tools.squeeze_to_series(data["Close"])
            returns_raw = index_level_raw.pct_change(fill_method=None)

            index_level = yahoo_tools.reindex_by_date(index_level_raw, index).ffill()
            returns = yahoo_tools.reindex_by_date(returns_raw, index).fillna(0)
            _save_pickle([returns, index_level], cache_file)

        self._delete_stale_files(self.market_data_prefix, keep=cache_file)
        return MarketData(returns, index_level)
