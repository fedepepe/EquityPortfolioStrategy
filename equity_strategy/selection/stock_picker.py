"""Stock-selection algorithms.

Every algorithm returns a DataFrame indexed by ticker with two columns:

- ``Metric``: the selection score (for ``max_sr``, the optimized weight)
- ``Pos``: +1 for a long position, -1 for a short position
"""
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize as opt

from equity_strategy.definitions import Algorithms, CorrelationMethods
from equity_strategy.portfolio.optimization import leverage
from equity_strategy.selection.dependence import dependence_scores

logger = logging.getLogger(__name__)

# Long-only dependence-based selection discards stocks scoring at or below this value
MIN_DEPENDENCE_SCORE = 0.2

# Maximum-Sharpe selection: candidates are the largest stocks by average market cap
MAX_SHARPE_UNIVERSE_SIZE = 100
MAX_SHARPE_MAX_WEIGHT = 0.5
MAX_SHARPE_MAX_LEVERAGE = 1.0
MAX_SHARPE_RISK_AVERSION = 3.0


@dataclass(frozen=True)
class _SelectionRule:
    method: str  # 'momentum', 'dependence' or 'max_sharpe'
    long_only: bool
    risk_managed: bool = False
    correlation: CorrelationMethods = CorrelationMethods.SEV


_SELECTION_RULES = {
    Algorithms.MTM: _SelectionRule("momentum", long_only=False),
    Algorithms.LOMTM: _SelectionRule("momentum", long_only=True),
    Algorithms.RMMTM: _SelectionRule("momentum", long_only=False, risk_managed=True),
    Algorithms.LORMMTM: _SelectionRule("momentum", long_only=True, risk_managed=True),
    Algorithms.SEV: _SelectionRule("dependence", long_only=True),
    Algorithms.RMSEV: _SelectionRule("dependence", long_only=True, risk_managed=True),
    Algorithms.LIN: _SelectionRule("dependence", long_only=False, correlation=CorrelationMethods.LINEAR),
    Algorithms.LOLIN: _SelectionRule("dependence", long_only=True, correlation=CorrelationMethods.LINEAR),
    Algorithms.RMLIN: _SelectionRule("dependence", long_only=False, risk_managed=True,
                                     correlation=CorrelationMethods.LINEAR),
    Algorithms.LORMLIN: _SelectionRule("dependence", long_only=True, risk_managed=True,
                                       correlation=CorrelationMethods.LINEAR),
    Algorithms.MAX_SHARPE: _SelectionRule("max_sharpe", long_only=True),
}


def _long_short_selection(scores: pd.Series, n_stocks: int) -> pd.DataFrame:
    """Take the ``n_stocks`` highest scores; go long above the cross-sectional median and short below it.

    Since the candidates are the highest scores, short positions only occur when
    ``n_stocks`` exceeds half of the universe.
    """
    median = np.quantile(scores, 0.5)
    selection = scores.head(n_stocks).to_frame("Metric")
    selection.loc[selection["Metric"] > median, "Pos"] = 1
    selection.loc[selection["Metric"] < median, "Pos"] = -1
    return selection


def _portfolio_returns(weights: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
    return returns.add(1).values.dot(weights) - weights.sum()


def _neg_mean_variance_utility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    portfolio_returns = _portfolio_returns(weights, returns)
    return -(portfolio_returns.mean() - MAX_SHARPE_RISK_AVERSION * portfolio_returns.var())


def _smooth_nonzero_count(weights: np.ndarray) -> float:
    """Differentiable proxy for the number of non-zero weights."""
    return (1 - np.exp(-np.square(10.0 * weights * len(weights)))).sum()


class StockPicker:
    def __init__(self,
                 n_stocks: int,
                 algorithm: str,
                 returns: pd.DataFrame,
                 volatility: pd.DataFrame,
                 sharpe: pd.DataFrame,
                 market_cap: pd.DataFrame | None = None,
                 benchmark_fraction: float = 0.1):
        """
        :param returns, volatility, sharpe, market_cap: (date x ticker) data over the observation window
        :param benchmark_fraction: fraction of the universe (highest Sharpe ratios) forming the
            benchmark index of the dependence-based algorithms
        """
        self.n_stocks = n_stocks
        self.algorithm = algorithm
        self.returns = returns
        self.volatility = volatility
        self.sharpe = sharpe
        self.market_cap = market_cap
        self.benchmark_fraction = benchmark_fraction

    def pick_stocks(self, parallel: bool = False, cv_bandwidth: bool = False) -> pd.DataFrame:
        try:
            rule = _SELECTION_RULES[self.algorithm]
        except KeyError:
            raise ValueError(f"Unknown stock-selection algorithm: {self.algorithm!r}") from None

        if rule.method == "momentum":
            return self._select_by_momentum(rule)
        if rule.method == "dependence":
            return self._select_by_dependence(rule, parallel=parallel, cv_bandwidth=cv_bandwidth)
        return self._select_max_sharpe()

    def _select_by_momentum(self, rule: _SelectionRule) -> pd.DataFrame:
        features = self.sharpe if rule.risk_managed else self.returns
        scores = features.mean(skipna=False).sort_values(ascending=False).dropna()

        if not rule.long_only:
            return _long_short_selection(scores, self.n_stocks)
        selection = scores.head(self.n_stocks).to_frame("Metric")
        selection["Pos"] = 1
        return selection

    def _select_by_dependence(self, rule: _SelectionRule, parallel: bool, cv_bandwidth: bool) -> pd.DataFrame:
        """Select stocks whose excess features best explain a high-Sharpe benchmark index."""
        benchmark = self._high_sharpe_index(rule.risk_managed)
        features = self.sharpe if rule.risk_managed else self.returns
        features = features.subtract(benchmark, axis=0)

        scores = dependence_scores(features, benchmark, method=rule.correlation,
                                   cv_bandwidth=cv_bandwidth, parallel=parallel)
        scores = scores.sort_values(ascending=False).dropna()

        if not rule.long_only:
            return _long_short_selection(scores, self.n_stocks)
        selection = scores.head(self.n_stocks).to_frame("Metric")
        selection["Pos"] = 1
        return selection[selection["Metric"] > MIN_DEPENDENCE_SCORE]

    def _high_sharpe_index(self, risk_managed: bool) -> pd.Series:
        """Cap-weighted daily index of the stocks with the highest Sharpe ratio on each day.

        The index averages Sharpe ratios if ``risk_managed``, returns otherwise.
        """
        if self.market_cap is None:
            raise ValueError("Market-cap data is required by the dependence-based algorithms.")

        index_values = []
        for date, sharpe_row in self.sharpe.iterrows():
            n_top = round(self.benchmark_fraction * sharpe_row.size)
            top_sharpe = sharpe_row.sort_values(ascending=False).head(n_top).fillna(0)
            caps = self.market_cap.loc[date].reindex(top_sharpe.index).fillna(0)

            values = top_sharpe if risk_managed else self.returns.loc[date, top_sharpe.index]
            index_values.append((values * caps).sum(skipna=False) / caps.sum())

        return pd.Series(index_values, index=self.sharpe.index, dtype=float)

    def _select_max_sharpe(self) -> pd.DataFrame:
        """Long-only mean-variance optimization over the largest stocks, with at most ``n_stocks`` holdings."""
        if self.market_cap is None:
            raise ValueError("Market-cap data is required by the max-Sharpe algorithm.")

        largest = self.market_cap.mean().sort_values(ascending=False).head(MAX_SHARPE_UNIVERSE_SIZE).index
        candidates = [ticker for ticker in largest if ticker in self.returns.columns]
        returns = self.returns.loc[:, candidates].dropna(axis=1)
        tickers = returns.columns

        ones = pd.Series(1.0, index=tickers)
        constraints = (
            opt.NonlinearConstraint(leverage, 0, MAX_SHARPE_MAX_LEVERAGE),
            opt.NonlinearConstraint(_smooth_nonzero_count, 0, self.n_stocks),
        )
        bounds = opt.Bounds(0 * ones, MAX_SHARPE_MAX_WEIGHT * ones)
        equal_weights = ones / ones.sum()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="delta_grad == 0.0")
            warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
            result = opt.minimize(_neg_mean_variance_utility, equal_weights, args=(returns,),
                                  method="trust-constr", options={"maxiter": 1_000_000_000},
                                  constraints=constraints, bounds=bounds)
        if not result.success:
            logger.warning("Max-Sharpe optimization did not converge (%s); using the last iterate.",
                           result.message)

        selection = pd.DataFrame({"Metric": result.x}, index=tickers)
        selection["Pos"] = np.where(selection["Metric"] >= 0, 1, -1)
        return selection
