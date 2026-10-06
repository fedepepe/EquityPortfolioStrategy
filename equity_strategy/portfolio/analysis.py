"""Performance statistics computed from a portfolio's value history."""
import urllib.request
import zipfile
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from equity_strategy.config import TRADING_DAYS_IN_YEAR
from equity_strategy.data.yahoo_tools import reindex_by_date

RiskFreeReturn = float | pd.Series

FRENCH_LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF_FACTORS_URL = f"{FRENCH_LIBRARY_URL}/F-F_Research_Data_Factors_daily_CSV.zip"
FF_MOMENTUM_URL = f"{FRENCH_LIBRARY_URL}/F-F_Momentum_Factor_daily_CSV.zip"
MIN_TRACKING_ERROR = 1e-4  # below this, the portfolio is considered identical to the benchmark


class ReturnStats(NamedTuple):
    returns: pd.Series
    log_returns: pd.Series
    excess_returns: pd.Series
    annual_mean_return: float
    annual_mean_excess_return: float
    volatility: float
    sharpe: float


class AlphaBeta(NamedTuple):
    alpha: float  # annualized
    beta: float
    alpha_pvalue: float


def compute_returns(value_history: pd.Series) -> pd.Series:
    """Simple daily returns; missing values are forward-filled first."""
    return value_history.ffill().pct_change(fill_method=None).dropna()


def compute_exc_ret(value_history: pd.Series, benchmark_returns: RiskFreeReturn = 0) -> pd.Series:
    returns = compute_returns(value_history)
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.loc[returns.index]
    return returns - benchmark_returns


def compute_returns_volat_sharpe(value_history: pd.Series, risk_free_return: RiskFreeReturn = 0) -> ReturnStats:
    returns = compute_returns(value_history)
    excess_returns = compute_exc_ret(value_history, risk_free_return)
    annual_mean_excess_return = TRADING_DAYS_IN_YEAR * excess_returns.mean()
    volatility = np.sqrt(TRADING_DAYS_IN_YEAR) * excess_returns.std()
    return ReturnStats(
        returns=returns,
        log_returns=np.log(1.0 + returns),
        excess_returns=excess_returns,
        annual_mean_return=TRADING_DAYS_IN_YEAR * returns.mean(),
        annual_mean_excess_return=annual_mean_excess_return,
        volatility=volatility,
        sharpe=annual_mean_excess_return / volatility,
    )


def compute_info_ratio(value_history: pd.Series, benchmark_returns: pd.Series) -> float:
    active_returns = compute_exc_ret(value_history, benchmark_returns)
    if active_returns.std() < MIN_TRACKING_ERROR:
        return 0.0
    return np.sqrt(TRADING_DAYS_IN_YEAR) * active_returns.mean() / active_returns.std()


def compute_sortino(value_history: pd.Series, risk_free_return: RiskFreeReturn = 0,
                    min_acceptable_return: float = 0) -> float:
    excess_returns = compute_exc_ret(value_history, risk_free_return)
    annual_mean = TRADING_DAYS_IN_YEAR * excess_returns.mean()
    downside_deviation = np.sqrt(TRADING_DAYS_IN_YEAR) * excess_returns[excess_returns < min_acceptable_return].std()
    return annual_mean / downside_deviation


def compute_star(value_history: pd.Series, risk_free_return: RiskFreeReturn = 0, level: float = 0.05) -> float:
    """STARR ratio: mean excess return over the expected tail loss at ``level``."""
    excess_returns = compute_exc_ret(value_history, risk_free_return)
    value_at_risk = excess_returns.quantile(level)
    expected_tail_loss = excess_returns[excess_returns < value_at_risk].mean()
    return -excess_returns.mean() / expected_tail_loss


def compute_turnover(value_history: pd.Series, traded_value_history: pd.Series) -> float:
    return traded_value_history.divide(value_history).mean()


def compute_alpha_beta(value_history: pd.Series, benchmark_returns: pd.Series,
                       risk_free_return: RiskFreeReturn = 0) -> AlphaBeta:
    """CAPM regression of the portfolio excess returns on the benchmark excess returns."""
    excess_returns = compute_exc_ret(value_history, risk_free_return)
    benchmark_window = benchmark_returns.loc[excess_returns.index]
    if isinstance(risk_free_return, pd.Series):
        risk_free_return = risk_free_return.loc[excess_returns.index]

    y = np.asarray(excess_returns.values, dtype=float).reshape(-1, 1)
    x = np.asarray(benchmark_window.subtract(risk_free_return, axis=0).values, dtype=float).reshape(-1, 1)
    fit = sm.OLS(y, sm.add_constant(x, prepend=True)).fit()
    return AlphaBeta(alpha=TRADING_DAYS_IN_YEAR * fit.params[0], beta=fit.params[1], alpha_pvalue=fit.pvalues[0])


def compute_drawdown(value_history: pd.Series) -> pd.Series:
    previous_peak = value_history.cummax()
    return (value_history - previous_peak) / previous_peak


def _download_french_factors(url: str, skip_rows: int) -> pd.DataFrame:
    """Daily factor returns (as fractions) from the Kenneth French data library."""
    local_file, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(local_file) as archive, archive.open(archive.namelist()[0]) as csv_file:
        factors = pd.read_csv(csv_file, skiprows=skip_rows).dropna()

    dates = pd.to_datetime(factors.iloc[:, 0].astype(str).str.strip(), format="%Y%m%d", utc=True)
    factors = factors.drop(columns=factors.columns[0])
    factors.index = pd.DatetimeIndex(dates)
    return factors / 100


def compute_ff_factors(value_history: pd.Series, n_factors: int,
                       risk_free_return: RiskFreeReturn = 0) -> dict[str, dict[str, float]]:
    """Regress excess returns on the first ``n_factors`` Fama-French factors (4 adds momentum).

    Returns annualized coefficients (in %) with HAC t-values and p-values, keyed by factor.
    """
    factors = _download_french_factors(FF_FACTORS_URL, skip_rows=3).drop(columns="RF")
    if n_factors > 3:
        factors = pd.concat([factors, _download_french_factors(FF_MOMENTUM_URL, skip_rows=12)], axis=1)

    excess_returns = compute_exc_ret(value_history, risk_free_return)
    factors = reindex_by_date(factors, excess_returns.index).loc[excess_returns.index]

    y = np.asarray(excess_returns.values, dtype=float).reshape(-1, 1)
    x = np.asarray(factors.iloc[:, :n_factors].values, dtype=float).reshape(len(y), -1)
    x = sm.add_constant(x, prepend=True)
    max_lag = int(4 * (x.shape[0] / 100) ** (2 / 9))  # Newey-West rule of thumb
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})

    labels = ["constant", *factors.columns]
    return {labels[i]: {"coeff": 100.0 * TRADING_DAYS_IN_YEAR * fit.params[i],
                        "t_value": fit.tvalues[i],
                        "p_value": fit.pvalues[i]}
            for i in range(n_factors + 1)}
