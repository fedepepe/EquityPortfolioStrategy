"""Mean-variance portfolio optimization."""
import logging
import math
import warnings

import numpy as np
import pandas as pd
import scipy.optimize as opt

logger = logging.getLogger(__name__)

MAX_ABS_WEIGHT = 0.5  # maximum single-stock weight, to promote diversification
MAX_OPTIMIZATION_ATTEMPTS = 10
INITIAL_MAX_ITERATIONS = 100


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return np.sqrt(weights.dot(cov_matrix).dot(weights))


def leverage(weights: np.ndarray) -> float:
    return abs(weights).sum()


def mean_variance_objective(weights: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray,
                            risk_aversion: float | None = None) -> float:
    """Objective to minimize for a given risk aversion.

    - ``None``: tangency portfolio (variance of the weights rescaled to unit expected return)
    - ``inf``: minimum-variance portfolio
    - otherwise: mean-variance utility ``risk_aversion / 2 * variance - expected return``
    """
    if risk_aversion is None:
        unit_return_weights = weights / weights.dot(mu)
        return unit_return_weights.dot(cov_matrix).dot(unit_return_weights)
    if math.isinf(risk_aversion):
        return weights.dot(cov_matrix).dot(weights)
    return risk_aversion / 2.0 * weights.dot(cov_matrix).dot(weights) - weights.dot(mu)


def volatility_scaled_moments(returns: pd.DataFrame, volatility: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of volatility-scaled returns (daily Sharpe ratios)."""
    scaled_returns = returns / volatility
    return scaled_returns.mean().values, scaled_returns.cov().values


def get_opt_weights(returns: pd.DataFrame,
                    volatility: pd.DataFrame,
                    allow_short_sell: bool = True,
                    risk_aversion: float | None = None,
                    max_leverage: float = 3.0,
                    volatility_scaled: bool = True,
                    tol: float = 1e-8,
                    verbose: bool = False,
                    rng: np.random.Generator | None = None) -> pd.Series:
    """Fully invested mean-variance optimal weights.

    Stocks with missing or constant returns are excluded. If the optimization fails, the
    equally-weighted portfolio is returned.
    """
    returns = returns.dropna(axis=0)
    if returns.empty:
        return pd.Series(dtype=float)
    returns = returns.loc[:, (returns != returns.iloc[0]).any()]

    if volatility_scaled:
        volatility = volatility.dropna(axis=0)
        tickers = [ticker for ticker in returns.columns if ticker in volatility.columns]
        returns, volatility = returns[tickers], volatility[tickers]

    n_stocks = len(returns.columns)
    if n_stocks == 0:
        return pd.Series(dtype=float)

    ones = pd.Series(1.0, index=returns.columns)
    equal_weights = ones / n_stocks
    constraints = (
        opt.LinearConstraint(ones, 1, 1),  # fully invested
        opt.NonlinearConstraint(leverage, 0, max_leverage),
    )
    lower_bound = -MAX_ABS_WEIGHT if allow_short_sell else 0.0
    bounds = opt.Bounds(lower_bound * ones, MAX_ABS_WEIGHT * ones)
    rng = rng if rng is not None else np.random.default_rng()

    try:
        if volatility_scaled:
            mu, cov_matrix = volatility_scaled_moments(returns, volatility)
        else:
            mu, cov_matrix = returns.mean().values, returns.cov().values

        initial_weights = equal_weights
        max_iterations = INITIAL_MAX_ITERATIONS
        for _ in range(MAX_OPTIMIZATION_ATTEMPTS):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="delta_grad == 0.0")
                warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
                result = opt.minimize(mean_variance_objective, initial_weights,
                                      args=(mu, cov_matrix, risk_aversion), method="trust-constr",
                                      options={"verbose": verbose, "xtol": tol, "gtol": tol,
                                               "barrier_tol": tol, "maxiter": max_iterations},
                                      constraints=constraints, bounds=bounds)
            if result.success:
                return pd.Series(result.x, index=returns.columns)

            # retry from a random fully-invested portfolio with a larger iteration budget
            initial_weights = rng.uniform(low=0.0, high=MAX_ABS_WEIGHT, size=n_stocks) * ones
            initial_weights /= initial_weights.sum()
            max_iterations += INITIAL_MAX_ITERATIONS

        logger.warning("Mean-variance optimization did not converge; using the equally-weighted portfolio.")
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        logger.warning("Mean-variance optimization failed (%s); using the equally-weighted portfolio.", error)
    return equal_weights
