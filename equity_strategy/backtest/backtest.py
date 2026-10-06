"""Rolling-window backtesting of stock-selection and weighting strategies.

At every rebalancing date, the stock picker selects stocks using the last ``window_length``
observations available ``lag`` days before, and every weighting method gets its own portfolio.
Parameters can be swept over grids of values; results are written to ``<dataset>/results/``.
"""
import functools
import itertools
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from equity_strategy import config, plotting
from equity_strategy.backtest import results as results_io
from equity_strategy.portfolio.metrics import PerfMetric, metric_template
from equity_strategy.portfolio.portfolio import Portfolio
from equity_strategy.selection.stock_picker import StockPicker

logger = logging.getLogger(__name__)

MARKET = "mkt"
# Start shifts (in trading days, roughly one quarter apart) of the multi-start robustness check
MULTI_START_OFFSETS = (0, 65, 130, 195, 261)
PROGRESS_LOG_INTERVAL = 50

GridValues = int | Sequence[int] | np.ndarray | None
DateLike = str | datetime | pd.Timestamp | None


def _as_grid(values: GridValues) -> np.ndarray | None:
    return None if values is None else np.atleast_1d(np.asarray(values))


def _as_list(values: str | Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    return [values] if isinstance(values, str) else list(values)


def _single_value(grid: Sequence | None):
    return grid[0] if grid is not None and len(grid) == 1 else None


def _drop_last_rows(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    return frame.iloc[:len(frame) - n_rows]


def _align_start_offset(offset: int, rebalance_interval: int) -> int:
    """Shift ``offset`` by at most one day, to the neighbour with the largest LCM with the rebalancing interval."""
    candidates = [offset - 1, offset, offset + 1]
    return max(candidates[int(np.argmax(np.lcm(rebalance_interval, candidates)))], 0)


def _log_duration(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        logger.info("Simulation time: %s", time.strftime("%Mm %Ss", time.gmtime(time.perf_counter() - start)))
        return result

    return wrapper


class PortfolioBacktest:
    def __init__(self,
                 dataset: str,
                 prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 volatility: pd.DataFrame,
                 market_cap: pd.DataFrame | None = None,
                 benchmark_index: pd.Series | pd.DataFrame | None = None,
                 *,
                 n_stocks: GridValues = None,
                 window_lengths: GridValues = None,
                 rebalance_intervals: GridValues = None,
                 algorithms: str | Sequence[str] | None = None,
                 weight_methods: str | Sequence[str] | None = None,
                 risk_free_return: float | pd.Series | None = None,
                 date_start: DateLike = None,
                 date_stop: DateLike = None,
                 endowment: float = 1e6,
                 lag: int = 1,
                 benchmark_fraction: float = 0.1,
                 fee_fixed: float = 0.0,
                 fee_proportional: float = 1e-3,
                 risk_aversion: float | None = None,
                 max_leverage: float = 3.0,
                 parallel: bool = False,
                 cv_bandwidth: bool = False,
                 results_dir: Path | None = None,
                 results_tag: str | None = None,
                 results_date: DateLike = None,
                 overwrite_results: bool = False,
                 save_positions: bool = False,
                 figure_format: str | None = None):
        """
        :param prices, returns, volatility, market_cap: daily (date x ticker) data; ``returns``,
            ``volatility`` and ``market_cap`` must share the same index
        :param benchmark_index: market index level, used as the ``mkt`` reference portfolio
        :param n_stocks, window_lengths, rebalance_intervals, algorithms: a value or a grid of values to sweep
        :param lag: days between the last observation used and the rebalancing date
        :param benchmark_fraction: see :class:`StockPicker`
        :param figure_format: 'png' or 'pdf' to save heatmaps next to the results
        """
        self.dataset = dataset
        self.prices = prices
        self.returns = returns
        self.volatility = volatility
        self.market_cap = market_cap
        if isinstance(benchmark_index, pd.DataFrame):
            benchmark_index = benchmark_index.iloc[:, 0]
        self.benchmark_index = benchmark_index

        excess_returns = returns if risk_free_return is None else returns.subtract(risk_free_return, axis=0)
        self.sharpe = excess_returns / volatility
        self.risk_free_return = 0 if risk_free_return is None else risk_free_return

        self.n_stocks_grid = _as_grid(n_stocks)
        self.window_grid = _as_grid(window_lengths)
        self.rebalance_grid = _as_grid(rebalance_intervals)
        self.algorithms = _as_list(algorithms)
        self.weight_methods = _as_list(weight_methods)

        # parameters of the current run; preset when a single value is given, e.g. for allocate()
        self.n_stocks = _single_value(self.n_stocks_grid)
        self.window_length = _single_value(self.window_grid)
        self.rebalance_interval = _single_value(self.rebalance_grid)
        self.algorithm = _single_value(self.algorithms)

        self.date_start = pd.Timestamp(date_start) if date_start is not None else None
        self.date_stop = pd.Timestamp(date_stop) if date_stop is not None else None
        self.endowment = endowment
        self.lag = lag
        self.benchmark_fraction = benchmark_fraction
        self.fee_fixed = fee_fixed
        self.fee_proportional = fee_proportional
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        self.parallel = parallel
        self.cv_bandwidth = cv_bandwidth
        self.save_positions = save_positions
        self.figure_format = figure_format

        if self.date_start is not None:
            n_before_start = (prices.index.tz_localize(None) < self.date_start).sum()
            self.start_index = n_before_start + lag - 1
        elif self.window_grid is not None:
            # start all runs of a window-length sweep on the same trading day
            self.start_index = max(self.window_grid) + lag - 1
        else:
            self.start_index = None

        grids = [self.n_stocks_grid, self.window_grid, self.rebalance_grid, self.algorithms]
        self.parametric_sweep = int(np.prod([len(grid) for grid in grids if grid is not None])) > 1

        self.results: results_io.Results = {}
        self.results_dir = Path(results_dir) if results_dir is not None else config.results_dir(dataset)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if results_tag is None:
            results_tag = "sw" if self.parametric_sweep else ""
        self.results_tag = results_tag
        results_date = pd.Timestamp(results_date) if results_date is not None else datetime.today()
        self.results_date = results_date.strftime(config.YMD_DATE_FORMAT)
        self.overwrite_results = overwrite_results

        self.portfolios: dict[str, Portfolio] = {}
        self.dates: pd.DatetimeIndex | None = None  # dates of the last run, starting from the initial one
        # target weights set at each rebalancing date of the last run, per weighting method
        self._weight_history: dict[str, dict[pd.Timestamp, pd.Series]] = {}
        if self.weight_methods:
            self._init_portfolios(prices.index[0])

    # ----- Configuration -----

    def _results_path(self, suffix: str = "") -> Path:
        return self.results_dir / f"{self.results_tag}_{self.results_date}{suffix}"

    @property
    def results_pickle_path(self) -> Path:
        return self._results_path(".pkl")

    def _results_text_path(self, method: str) -> Path:
        if method == MARKET:
            return self._results_path(f"_{MARKET}.txt")
        return self._results_path(f"_{self.n_stocks}_{self.algorithm}_{method}.txt")

    def set_run(self, n_stocks: int, window_length: int, algorithm: str,
                rebalance_interval: int | None = None) -> None:
        self.n_stocks = n_stocks
        self.window_length = window_length
        self.algorithm = algorithm
        self.rebalance_interval = rebalance_interval

    def _init_portfolios(self, date: pd.Timestamp) -> None:
        self.portfolios = {method: Portfolio(self.endowment) for method in self.weight_methods}
        self._weight_history = {method: {} for method in self.weight_methods}
        for portfolio in self.portfolios.values():
            portfolio.record(date, self.save_positions)

    # ----- Simulation -----

    def allocate(self, date: pd.Timestamp | None = None,
                 prices: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
        """Target allocation of every weighting method at ``date`` (default: last available date)."""
        if None in (self.n_stocks, self.window_length, self.algorithm):
            raise RuntimeError("Set n_stocks, window_length and algorithm (see set_run) before allocating.")

        dates = self.returns.index
        if date is None:
            date = dates[-1]
        if prices is None:
            prices = self.prices.loc[date].to_dict()

        # the observation window ends `lag` days before the rebalancing date
        observable = dates <= date - timedelta(days=self.lag)

        def window(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[observable].tail(self.window_length)

        returns_window = window(self.returns)
        if len(returns_window) < self.window_length:
            logger.warning("Not enough samples for a window of %d observations.", self.window_length)

        picker = StockPicker(n_stocks=self.n_stocks,
                             algorithm=self.algorithm,
                             returns=returns_window,
                             volatility=window(self.volatility),
                             sharpe=window(self.sharpe),
                             market_cap=None if self.market_cap is None else window(self.market_cap),
                             benchmark_fraction=self.benchmark_fraction)
        selection = picker.pick_stocks(parallel=self.parallel, cv_bandwidth=self.cv_bandwidth)
        selection["Price"] = [prices[ticker] for ticker in selection.index]

        # full history of the selected stocks for the optimization-based weighting methods
        up_to_date = dates <= date
        returns_history = _drop_last_rows(self.returns.loc[up_to_date, selection.index], self.lag)
        volatility_history = _drop_last_rows(self.volatility.loc[up_to_date, selection.index], self.lag)
        market_cap_now = None if self.market_cap is None else self.market_cap.loc[date]

        return {method: portfolio.compute_weights(selection=selection,
                                                  method=method,
                                                  risk_aversion=self.risk_aversion,
                                                  max_leverage=self.max_leverage,
                                                  market_cap=market_cap_now,
                                                  returns=returns_history,
                                                  volatility=volatility_history,
                                                  fee_fixed=self.fee_fixed,
                                                  fee_proportional=self.fee_proportional)
                for method, portfolio in self.portfolios.items() if method != MARKET}

    @_log_duration
    def run_single(self, start_offset: int = 0) -> None:
        """Simulate the current run parameters, starting ``start_offset`` trading days after the default start."""
        start = self.start_index + start_offset
        dates = self.prices.index
        if self.date_stop is not None:
            dates = dates[dates.tz_localize(None) <= self.date_stop]
        if len(dates) <= self.window_length:
            raise ValueError("Not enough data samples for backtesting.")

        # portfolios are initialized the day before the first trading day
        self.dates = dates[start - 1:]
        rebalance_dates = set(dates[start::self.rebalance_interval])

        logger.info("Algorithm: %s, window: %s, rebalancing every %s days, %s stocks. Time steps: %d.",
                    self.algorithm, self.window_length, self.rebalance_interval, self.n_stocks, len(self.dates) - 1)

        # valuations use the last available price; trades only use prices quoted on the day
        filled_prices = self.prices.ffill()
        self._init_portfolios(self.dates[0])

        for step, date in enumerate(self.dates[1:], start=1):
            valuation_prices = filled_prices.loc[date].to_dict()
            for portfolio in self.portfolios.values():
                portfolio.update_value(valuation_prices)

            if date in rebalance_dates:
                trade_prices = self.prices.loc[date].to_dict()
                targets = self.allocate(date, trade_prices)
                for method, portfolio in self.portfolios.items():
                    portfolio.rebalance(targets[method], trade_prices, self.fee_fixed, self.fee_proportional)
                    self._weight_history[method][date] = targets[method]["Weight"]

            for portfolio in self.portfolios.values():
                portfolio.record(date, self.save_positions)

            if step % PROGRESS_LOG_INTERVAL == 0:
                logger.info("Step %d of %d done.", step, len(self.dates) - 1)

        if self.benchmark_index is not None:
            benchmark = self.benchmark_index.reindex(self.dates)
            self.portfolios[MARKET] = Portfolio.from_value_history(benchmark / benchmark.iloc[0] * self.endowment)

    def backtest(self) -> None:
        """Run every combination of the parameter grids and record the performance of each run."""
        if self.overwrite_results:
            self.results_pickle_path.unlink(missing_ok=True)

        for n_stocks, algorithm in itertools.product(self.n_stocks_grid, self.algorithms):
            self.set_run(n_stocks=n_stocks, window_length=self.window_grid[0], algorithm=algorithm)
            if self.parametric_sweep:
                self._clear_results_text_files()

            configurations = list(itertools.product(self.window_grid, self.rebalance_grid))
            start = time.perf_counter()
            for n_done, (window_length, rebalance_interval) in enumerate(configurations, start=1):
                logger.info("Running backtest %d of %d.", n_done, len(configurations))
                self.set_run(n_stocks, window_length, algorithm, rebalance_interval)
                self.run_single()
                self.analyze()

                elapsed = time.perf_counter() - start
                logger.info("Total time elapsed: %s (average per simulation: %s).",
                            time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed)),
                            time.strftime("%Mm %Ss", time.gmtime(elapsed / n_done)))

                self._store_results()
                self.print_results(to_file=self.parametric_sweep)
                if not self.parametric_sweep:
                    for path in self.save_allocation_history():
                        logger.info("Allocation history saved to %s.", path)

            if self.parametric_sweep:
                results_io.save_results(self.results, self.results_pickle_path)

    def backtest_multi_start(self) -> None:
        """Robustness check: run each configuration from several start dates and regress the pooled
        weekly returns on the market's, appending annualized alpha and beta to ``*_ic.txt`` files.

        Only the first value of the ``n_stocks`` grid is used.
        """
        if self.benchmark_index is None:
            raise ValueError("The multi-start backtest requires a benchmark index.")
        if self.overwrite_results:
            self.results_pickle_path.unlink(missing_ok=True)

        n_stocks = self.n_stocks_grid[0]
        for window_length, rebalance_interval, algorithm in itertools.product(
                self.window_grid, self.rebalance_grid, self.algorithms):
            self.set_run(n_stocks, window_length, algorithm, rebalance_interval)

            weekly_returns = []
            for n_done, offset in enumerate(MULTI_START_OFFSETS, start=1):
                logger.info("Running backtest %d of %d.", n_done, len(MULTI_START_OFFSETS))
                self.run_single(start_offset=_align_start_offset(offset, rebalance_interval))
                weekly_returns.append(pd.DataFrame({
                    method: portfolio.value_history.resample("W").last().ffill().pct_change(fill_method=None).dropna()
                    for method, portfolio in self.portfolios.items()
                }))
            pooled = pd.concat(weekly_returns, axis=0)

            for method in self.portfolios:
                if method == MARKET:
                    continue
                y = np.asarray(pooled[method].values, dtype=float).reshape(-1, 1)
                x = np.asarray(pooled[MARKET].values, dtype=float).reshape(-1, 1)
                fit = sm.OLS(y, sm.add_constant(x, prepend=True)).fit()
                alpha, beta = fit.params
                line = (f"{n_stocks}\t{window_length}\t{rebalance_interval}\t{algorithm:<7}\t{method:<7}\t"
                        f"alpha: {config.WEEKS_IN_YEAR * alpha:.2%}, pval: {fit.pvalues[0]:.3f}, beta: {beta:.2f}")
                logger.info(line)
                with open(self._results_path(f"_{n_stocks}_{algorithm}_{method}_ic.txt"), "a") as text_file:
                    text_file.write(f"{line}\n")

    # ----- Analysis and reporting -----

    def analyze(self) -> None:
        market_returns = None
        if self.benchmark_index is not None:
            market_returns = self.benchmark_index.ffill().pct_change(fill_method=None)
        for portfolio in self.portfolios.values():
            portfolio.analyze(market_returns=market_returns, risk_free_return=self.risk_free_return)

    def allocation_history(self, method: str) -> pd.DataFrame:
        """Target weights of ``method`` at each rebalancing date of the last run (date x ticker).

        Tickers not selected on a date have weight 0; a date where nothing was selected is a row of zeros.
        """
        weights_by_date = self._weight_history[method]
        history = pd.DataFrame(list(weights_by_date.values()), index=list(weights_by_date.keys()))
        history = history.reindex(columns=sorted(history.columns)).fillna(0.0)
        history.index = pd.Index(pd.DatetimeIndex(history.index).date, name="Date")
        return history

    def save_allocation_history(self) -> list[Path]:
        """Write the allocation history of each weighting method of the last run to an Excel file
        in the results directory; the first column is the rebalancing date, the others are tickers.
        """
        paths = []
        for method in self._weight_history:
            path = self._results_path(f"_{self.n_stocks}_{self.algorithm}_{method}_{self.window_length}_"
                                      f"{self.rebalance_interval}_allocation.xlsx")
            self.allocation_history(method).to_excel(path)
            paths.append(path)
        return paths

    def compute_ff_factors(self, n_factors: int) -> None:
        for portfolio in self.portfolios.values():
            portfolio.compute_ff_factors(n_factors)

    def _result_row(self, method: str, portfolio: Portfolio) -> str:
        name = MARKET if method == MARKET else self.algorithm
        row = f"{self.n_stocks}\t{self.window_length}\t{self.rebalance_interval}\t{name:<7}\t{method:<7}\t"
        for metric in PerfMetric:
            result = portfolio.metrics[metric]
            if result.is_scalar:
                row += f"\t{result.value:{result.format}}"
        return row

    def print_results(self, to_file: bool = False) -> None:
        """Print one row of scalar metrics per portfolio, or write them to the results text files."""
        for portfolio in self.portfolios.values():
            if portfolio.metrics[PerfMetric.SHARPE].value is None:
                raise RuntimeError("Run analyze() before printing the results.")

        if not to_file:
            first_portfolio = next(iter(self.portfolios.values()))
            header = "stk\tobs\treb\talgo\tweight  "
            for metric in PerfMetric:
                if first_portfolio.metrics[metric].is_scalar:
                    header += f"\t{first_portfolio.metrics[metric].label}"
            print(header)

        for method, portfolio in self.portfolios.items():
            row = self._result_row(method, portfolio)
            if not to_file:
                print(row)
                continue
            # the market row is identical across runs, so its file keeps only the latest one
            mode = "w" if method == MARKET else "a"
            with open(self._results_text_path(method), mode) as text_file:
                text_file.write(f"{row}\n")

    def print_results_for_latex(self) -> None:
        for method, portfolio in self.portfolios.items():
            metrics = portfolio.metrics
            values = portfolio.value_history
            print(f"{self.algorithm}-{method} & "
                  f"{100 * metrics[PerfMetric.ALPHA].value:.2f} & "
                  f"{metrics[PerfMetric.BETA].value:.2f} & "
                  f"{100 * metrics[PerfMetric.ANN_MEAN_RET].value:.2f} & "
                  f"{100 * metrics[PerfMetric.VOLATILITY].value:.2f} & "
                  f"{100 * (values.iloc[-1] / values.iloc[0]):.2f} & "
                  f"{100 * metrics[PerfMetric.MAXDD].value:.2f} & "
                  f"{100 * metrics[PerfMetric.TURNOVER].value:.2f} & "
                  f"{metrics[PerfMetric.SHARPE].value:.4f} & "
                  f"{metrics[PerfMetric.SORTINO].value:.4f} & "
                  f"{metrics[PerfMetric.STAR].value:.4f} \\\\")

    def _store_results(self) -> None:
        for method, portfolio in self.portfolios.items():
            key = (self.n_stocks, self.window_length, self.rebalance_interval, self.algorithm, method)
            self.results[key] = portfolio.scalar_metrics()

    def _clear_results_text_files(self) -> None:
        for method in self.weight_methods:
            self._results_text_path(method).write_text("")

    def load_results(self, path: Path | None = None) -> None:
        """Load results from ``path``, this backtest's results file, or else the latest one in the results folder."""
        if path is None:
            path = self.results_pickle_path
            if not path.is_file():
                path = results_io.latest_results_file(self.results_dir)
        logger.info("Loading results from %s.", path)
        self.results = results_io.load_results(path)

    # ----- Plots -----

    def plot_cum_wealth(self) -> plt.Figure:
        figure = None
        for method, portfolio in self.portfolios.items():
            if method == MARKET:
                description, linestyle = "market", "--"
            else:
                description, linestyle = f"algo={self.algorithm} wght={method}", "-"
            figure = plotting.plot_cum_wealth(portfolio.value_history, figure=figure,
                                              description=description, linestyle=linestyle)
        figure.axes[0].legend(loc="best")
        figure.suptitle(str(self.dataset))
        return figure

    def plot_heatmap(self, metric: PerfMetric) -> dict[int, plt.Figure]:
        """One heatmap mosaic of ``metric`` per number of stocks in the results."""
        grids = results_io.results_grid(self.results, metric)
        if not grids:
            logger.warning("No results to plot.")
            return {}

        template = metric_template(metric)
        figures = {}
        for n_stocks in sorted({key[0] for key in grids}):
            grids_n_stocks = {key: grid for key, grid in grids.items() if key[0] == n_stocks}
            figure = plotting.plot_heatmap_mosaic(grids_n_stocks, title=f"{self.dataset} - {template.label}",
                                                  reverse=template.rev_color_scale)
            figures[n_stocks] = figure
            if self.figure_format == "png":
                figure.savefig(self._results_path(f"_{n_stocks}_{template.label}.png"), dpi=600)
            elif self.figure_format == "pdf":
                figure.savefig(self._results_path(f"_{n_stocks}_{template.label}.pdf"), format="pdf",
                               bbox_inches="tight")
        return figures
