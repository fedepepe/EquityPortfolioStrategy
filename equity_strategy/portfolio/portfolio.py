"""Portfolio bookkeeping: positions, cash, trading costs, target weights and performance metrics."""
import logging
import math

import numpy as np
import pandas as pd

from equity_strategy.definitions import WeightMethods
from equity_strategy.portfolio import analysis
from equity_strategy.portfolio.analysis import RiskFreeReturn
from equity_strategy.portfolio.metrics import MetricResult, PerfMetric, empty_metrics
from equity_strategy.portfolio.optimization import get_opt_weights

logger = logging.getLogger(__name__)

CASH = "cash"
DEFAULT_ENDOWMENT = 1e5


class Portfolio:
    """A cash-and-stocks portfolio traded in whole shares with fixed plus proportional fees."""

    def __init__(self, endowment: float = DEFAULT_ENDOWMENT):
        self.holdings: dict[str, float] = {CASH: endowment}  # shares held per ticker; cash amount for CASH
        self.position_values: dict[str, float] = {CASH: endowment}
        self.total_value = endowment
        self.traded_value = 0.0  # gross value traded since the last valuation
        self.metrics: dict[PerfMetric, MetricResult] = empty_metrics()

        self._dates: list[pd.Timestamp] = []
        self._value_history: list[float] = []
        self._traded_value_history: list[float] = []
        self._holdings_history: list[dict[str, float]] = []
        self._position_values_history: list[dict[str, float]] = []

    @classmethod
    def from_value_history(cls, values: pd.Series) -> "Portfolio":
        """A passive portfolio (e.g. the market benchmark) defined by its value history only."""
        portfolio = cls(endowment=float(values.iloc[0]))
        portfolio._dates = list(values.index)
        portfolio._value_history = values.tolist()
        portfolio._traded_value_history = [0.0] * len(values)
        return portfolio

    # ----- History -----

    @property
    def value_history(self) -> pd.Series:
        return pd.Series(self._value_history, index=pd.DatetimeIndex(self._dates), dtype=float)

    @property
    def traded_value_history(self) -> pd.Series:
        return pd.Series(self._traded_value_history, index=pd.DatetimeIndex(self._dates), dtype=float)

    @property
    def holdings_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._holdings_history, index=pd.DatetimeIndex(self._dates[:len(self._holdings_history)]))

    @property
    def position_values_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._position_values_history,
                            index=pd.DatetimeIndex(self._dates[:len(self._position_values_history)]))

    def record(self, date: pd.Timestamp, save_positions: bool = False) -> None:
        """Append the current state to the history."""
        self._dates.append(date)
        self._value_history.append(self.total_value)
        self._traded_value_history.append(self.traded_value)
        if save_positions:
            self._holdings_history.append(dict(self.holdings))
            self._position_values_history.append(dict(self.position_values))

    # ----- Trading -----

    def _trade(self, ticker: str, price: float, quantity: float, fee_fixed: float, fee_proportional: float) -> None:
        """Buy (quantity > 0) or sell (quantity < 0) shares, paying for them and the fees in cash."""
        trade_value = quantity * price
        fees = fee_fixed + fee_proportional * abs(trade_value)
        self.traded_value += abs(trade_value)

        self.holdings[CASH] -= trade_value + fees
        self.position_values[CASH] -= trade_value + fees
        self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
        self.position_values[ticker] = self.position_values.get(ticker, 0) + trade_value

        if self.holdings[ticker] == 0:
            del self.holdings[ticker]
            del self.position_values[ticker]

        # exchanging cash for stock does not change the value; only the fees do
        self.total_value -= fees

    def rebalance(self, target: pd.DataFrame, prices: dict[str, float],
                  fee_fixed: float, fee_proportional: float) -> None:
        """Trade towards the target ``Quantity`` of each ticker in ``target``, closing all other positions.

        Tickers without a valid price are left untouched.
        """
        def trade_if_priced(ticker: str, quantity: float) -> None:
            if quantity != 0 and not np.isnan(prices[ticker]):
                self._trade(ticker, prices[ticker], quantity, fee_fixed, fee_proportional)

        target_tickers = set(target.index)
        held_tickers = [ticker for ticker in self.holdings if ticker != CASH]

        for ticker in held_tickers:
            if ticker not in target_tickers:
                trade_if_priced(ticker, -self.holdings[ticker])

        for ticker in target.index:
            target_quantity = math.floor(target.loc[ticker, "Quantity"])
            trade_if_priced(ticker, target_quantity - self.holdings.get(ticker, 0))

    def update_value(self, prices: dict[str, float]) -> None:
        """Mark positions to market and start a new trading period."""
        total_value = 0.0
        for ticker, quantity in self.holdings.items():
            if ticker == CASH:
                total_value += quantity
            elif ticker in prices:
                if np.isnan(prices[ticker]):
                    logger.warning("Missing price for %s.", ticker)
                    continue
                self.position_values[ticker] = quantity * prices[ticker]
                total_value += self.position_values[ticker]

        self.total_value = total_value
        self.traded_value = 0.0

    # ----- Allocation -----

    def compute_weights(self,
                        selection: pd.DataFrame,
                        method: str,
                        risk_aversion: float | None,
                        max_leverage: float,
                        market_cap: pd.Series | None = None,
                        returns: pd.DataFrame | None = None,
                        volatility: pd.DataFrame | None = None,
                        fee_fixed: float = 0.0,
                        fee_proportional: float = 0.0) -> pd.DataFrame:
        """Target ``Weight`` and share ``Quantity`` for the selected stocks.

        :param selection: stock-picker output with ``Metric``, ``Pos`` and ``Price`` columns
        :param market_cap: current market capitalizations (``mktcap`` method)
        :param returns, volatility: history of the selected stocks (``riskpar``, ``lotp``, ``ilotp`` methods)
        """
        allocation = selection.copy()
        if method == WeightMethods.MKTCAP:
            allocation["MktCap"] = market_cap.reindex(allocation.index).fillna(0)
        elif method == WeightMethods.RISK_PARITY:
            allocation["Volat"] = np.sqrt((volatility[allocation.index] ** 2).sum(skipna=False))

        # discard stocks without a price (or without the data the weighting method needs)
        if allocation.isnull().values.any():
            allocation = allocation.dropna().copy()
            returns = returns[allocation.index]
            volatility = volatility[allocation.index]

        allocation["Weight"] = self._target_weights(allocation, method, risk_aversion, max_leverage,
                                                    returns, volatility)

        if allocation.isnull().values.any():
            missing = ", ".join(allocation[allocation.isnull().any(axis=1)].index)
            logger.warning("Could not allocate %s with %s.", missing, method)
            allocation["Weight"] = allocation["Weight"].fillna(0)

        # wealth allocated to each stock, divided by the price including trading costs
        allocation["Quantity"] = (self.total_value * allocation["Weight"]
                                  / (fee_fixed + (1.0 + fee_proportional) * allocation["Price"]))
        return allocation

    @staticmethod
    def _target_weights(allocation: pd.DataFrame, method: str, risk_aversion: float | None,
                        max_leverage: float, returns: pd.DataFrame, volatility: pd.DataFrame) -> pd.Series:
        if method == WeightMethods.EQ:
            return allocation["Pos"] / len(allocation)
        if method == WeightMethods.METRIC:
            return allocation["Metric"]
        if method == WeightMethods.RISK_PARITY:
            inverse_volatility = 1.0 / allocation["Volat"]
            return allocation["Pos"] * inverse_volatility / inverse_volatility.sum()
        if method == WeightMethods.MKTCAP:
            return allocation["Pos"] * allocation["MktCap"] / allocation["MktCap"].sum()
        if method in (WeightMethods.LOTP, WeightMethods.ILOTP):
            return get_opt_weights(returns=returns, volatility=volatility, allow_short_sell=False,
                                   risk_aversion=risk_aversion, max_leverage=max_leverage,
                                   volatility_scaled=(method == WeightMethods.ILOTP))
        raise ValueError(f"Unknown weighting method: {method!r}")

    # ----- Performance -----

    def analyze(self,
                market_returns: pd.Series | None = None,
                risk_free_return: RiskFreeReturn = 0,
                min_acceptable_return: float = 0,
                tail_level: float = 0.01) -> None:
        """Compute all performance metrics (IC, alpha and beta only if ``market_returns`` is given)."""
        values = self.value_history
        stats = analysis.compute_returns_volat_sharpe(values, risk_free_return)
        self.metrics[PerfMetric.RETS].value = stats.returns
        self.metrics[PerfMetric.LOG_RETS].value = stats.log_returns
        self.metrics[PerfMetric.EXC_RETS].value = stats.excess_returns
        self.metrics[PerfMetric.ANN_MEAN_RET].value = stats.annual_mean_return
        self.metrics[PerfMetric.ANN_MEAN_EXC_RET].value = stats.annual_mean_excess_return
        self.metrics[PerfMetric.VOLATILITY].value = stats.volatility
        self.metrics[PerfMetric.SHARPE].value = stats.sharpe

        self.metrics[PerfMetric.SORTINO].value = analysis.compute_sortino(values, risk_free_return,
                                                                          min_acceptable_return)
        self.metrics[PerfMetric.STAR].value = analysis.compute_star(values, risk_free_return, tail_level)
        self.metrics[PerfMetric.TURNOVER].value = analysis.compute_turnover(values, self.traded_value_history)

        drawdown = analysis.compute_drawdown(values)
        self.metrics[PerfMetric.DRAWDOWN].value = drawdown
        self.metrics[PerfMetric.MAXDD].value = -drawdown.min()

        if market_returns is not None:
            self.metrics[PerfMetric.IC].value = analysis.compute_info_ratio(values, market_returns)
            alpha_beta = analysis.compute_alpha_beta(values, market_returns, risk_free_return)
            self.metrics[PerfMetric.ALPHA].value = alpha_beta.alpha
            self.metrics[PerfMetric.BETA].value = alpha_beta.beta
            self.metrics[PerfMetric.PVAL].value = alpha_beta.alpha_pvalue

    def compute_ff_factors(self, n_factors: int) -> None:
        self.metrics[PerfMetric.FF_FACTORS].value = analysis.compute_ff_factors(self.value_history, n_factors)

    def scalar_metrics(self) -> dict[PerfMetric, MetricResult]:
        return {metric: result for metric, result in self.metrics.items() if result.is_scalar}
