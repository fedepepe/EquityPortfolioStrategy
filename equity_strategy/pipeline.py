"""High-level workflows: download data, build backtests from the cache, compute allocations."""
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from equity_strategy import config
from equity_strategy.backtest.backtest import PortfolioBacktest
from equity_strategy.data.downloader import YahooDataDownloader
from equity_strategy.data.universes import get_tickers, to_yahoo_symbol
from equity_strategy.strategies import Strategy

logger = logging.getLogger(__name__)

DEFAULT_ENDOWMENT = 1e5


def download_latest_data(datasets: Iterable[str]) -> None:
    for dataset in datasets:
        YahooDataDownloader(dataset).download_latest_data()


def download_stock_info(datasets: Iterable[str]) -> None:
    for dataset in datasets:
        tickers = [to_yahoo_symbol(symbol) for symbol in get_tickers(dataset)]
        YahooDataDownloader(dataset).download_stock_info(tickers)


def create_backtest(dataset: str, **backtest_options) -> PortfolioBacktest:
    """Build a backtest on the cached data of ``dataset``, with the fees and lag from ``parameters.pkl``.

    ``backtest_options`` are forwarded to :class:`PortfolioBacktest`.
    """
    downloader = YahooDataDownloader(dataset)
    stock_data = downloader.get_stock_data()
    market_data = downloader.get_market_data(stock_data.returns.index)
    parameters = config.load_backtest_parameters(dataset)

    options = dict(endowment=DEFAULT_ENDOWMENT, lag=parameters.data_lag,
                   fee_fixed=parameters.fee_fixed, fee_proportional=parameters.fee_proportional)
    options.update(backtest_options)
    return PortfolioBacktest(dataset=dataset,
                             prices=stock_data.close,
                             returns=stock_data.returns,
                             volatility=stock_data.volatility,
                             market_cap=stock_data.market_cap,
                             benchmark_index=market_data.index,
                             **options)


def allocation_table(allocation: pd.DataFrame) -> pd.DataFrame:
    """Allocation of one weighting method as a table sorted by decreasing weight."""
    return (allocation.reset_index()
            .rename(columns={"index": "Ticker"})
            .sort_values("Weight", ascending=False))


def run_sweep(strategies: Iterable[Strategy], rebalance_intervals, date_start: str,
              multi_start: bool = True) -> list[PortfolioBacktest]:
    backtests = []
    for strategy in strategies:
        backtest = create_backtest(strategy.dataset,
                                   n_stocks=strategy.n_stocks,
                                   window_lengths=strategy.window_length,
                                   rebalance_intervals=rebalance_intervals,
                                   algorithms=strategy.algorithm,
                                   weight_methods=strategy.weight_methods,
                                   date_start=date_start)
        if multi_start:
            backtest.backtest_multi_start()
        else:
            backtest.backtest()
        backtests.append(backtest)
    return backtests


def save_allocations(strategies: Iterable[Strategy], output_dir: Path = config.PREDICTIONS_DIR) -> list[Path]:
    """Compute the current allocation of each strategy and save it as an Excel file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for strategy in strategies:
        backtest = create_backtest(strategy.dataset,
                                   n_stocks=strategy.n_stocks,
                                   window_lengths=strategy.window_length,
                                   algorithms=strategy.algorithm,
                                   weight_methods=strategy.weight_methods)
        date = backtest.prices.index[-1]
        for method, allocation in backtest.allocate().items():
            table = allocation_table(allocation)
            logger.info("%s %s allocation at %s:\n%s", strategy.dataset, method, date, table)
            file_name = (f"{strategy.dataset}_{date.strftime('%Y_%m_%d')}_{strategy.algorithm}_{method}_"
                         f"{strategy.n_stocks}_{strategy.window_length}.xlsx")
            table.to_excel(output_dir / file_name)
            saved_files.append(output_dir / file_name)
    return saved_files
