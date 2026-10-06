"""Dependence measures between a stock's features and a benchmark series.

The SEV (share of explained variance) of Y given X is ``Var(E[Y | X]) / Var(Y)``, where the
conditional expectation is estimated with a Gaussian-kernel Nadaraya-Watson regression.
Unlike linear correlation, it captures non-linear dependence.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy import stats

from equity_strategy.definitions import CorrelationMethods

KERNEL_SUPPORT_CUT = 3.0  # the density support extends this many bandwidths beyond the data range
MAD_TO_STD = 0.6745  # median absolute deviation of a standard normal variable
SCOTT_FACTOR = 1.059
CV_BANDWIDTH_POINTS = 21
CV_BANDWIDTH_RANGE = 10.0  # candidates span [bw_ref / range, bw_ref * range]

_pandarallel_initialized = False


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def _kernel_matrix(points: np.ndarray, samples: np.ndarray, bandwidth: float) -> np.ndarray:
    """Scaled kernel K((point - sample) / bw) / bw for every (point, sample) pair."""
    return gaussian_kernel((points[:, None] - samples[None, :]) / bandwidth) / bandwidth


def kernel_density(samples: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray:
    """Kernel density estimate of ``samples`` evaluated at ``points``."""
    return _kernel_matrix(points, samples, bandwidth).mean(axis=1)


def kernel_density_support(samples: np.ndarray, bandwidth: float) -> tuple[np.ndarray, float]:
    """Evaluation grid (power-of-two size) covering the data plus a margin, and its step."""
    grid_size = int(2 ** np.ceil(np.log2(10.0 * len(samples))))
    lower = np.min(samples) - KERNEL_SUPPORT_CUT * bandwidth
    upper = np.max(samples) + KERNEL_SUPPORT_CUT * bandwidth
    grid, step = np.linspace(lower, upper, grid_size, retstep=True)
    return grid, step


def nadaraya_watson(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                    bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Kernel regression E[Y | X = point] and the density of X at each point."""
    kernel = _kernel_matrix(points, x, bandwidth)
    density = kernel.mean(axis=1)
    weighted_y = (kernel * y).mean(axis=1)
    density[density == 0] = sys.float_info.min
    return weighted_y / density, density


def scott_bandwidth(samples: np.ndarray) -> float:
    """Scott's rule of thumb with a robust (MAD-based) scale estimate; NaN if the scale is undefined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sigma = np.nanmedian(np.abs(samples - np.nanmedian(samples))) / MAD_TO_STD

    if sigma <= 0:
        sigma = np.ptp(samples)

    if sigma > 0:
        return SCOTT_FACTOR * sigma * len(samples) ** (-0.2)
    return np.nan


def cross_validated_bandwidth(samples: np.ndarray, bandwidth_ref: float,
                              n_candidates: int = CV_BANDWIDTH_POINTS) -> float:
    """Bandwidth minimizing the leave-one-out cross-validation risk of the density estimate."""
    candidates = np.logspace(np.log10(bandwidth_ref / CV_BANDWIDTH_RANGE),
                             np.log10(bandwidth_ref * CV_BANDWIDTH_RANGE), n_candidates)
    n_obs = len(samples)

    risks = np.empty(n_candidates)
    for i, bandwidth in enumerate(candidates):
        grid, step = kernel_density_support(samples, bandwidth)
        squared_density_integral = np.sum(kernel_density(samples, grid, bandwidth) ** 2) * step

        # density at each sample estimated without that sample
        pairwise = _kernel_matrix(samples, samples, bandwidth)
        leave_one_out = (pairwise.sum(axis=1) - gaussian_kernel(0.0) / bandwidth) / (n_obs - 1)

        risks[i] = squared_density_integral - 2.0 / n_obs * leave_one_out.sum()

    return candidates[np.argmin(risks)]


def dependence_score(x: np.ndarray, y: np.ndarray,
                     method: CorrelationMethods = CorrelationMethods.SEV,
                     cv_bandwidth: bool = False) -> float:
    """Dependence of ``y`` on ``x``: linear/Spearman correlation or the SEV measure."""
    if method == CorrelationMethods.LINEAR:
        return np.corrcoef(x, y)[0, 1]
    if method == CorrelationMethods.SPEARMAN:
        return stats.spearmanr(x, y).correlation
    if method != CorrelationMethods.SEV:
        raise ValueError(f"Correlation method not implemented: {method}")

    bandwidth = scott_bandwidth(x)
    if np.isnan(bandwidth):
        return np.nan
    if cv_bandwidth:
        bandwidth = cross_validated_bandwidth(x, bandwidth)

    grid, step = kernel_density_support(x, bandwidth)
    conditional_mean, density = nadaraya_watson(x, y, grid, bandwidth)

    explained_second_moment = np.sum(conditional_mean ** 2 * density) * step
    return float((explained_second_moment - y.mean() ** 2) / y.var(ddof=1))


def _column_dependence_score(column: pd.Series, target: pd.Series,
                             method: CorrelationMethods, cv_bandwidth: bool) -> float:
    with warnings.catch_warnings():
        # NaNs in the input make the variance undefined; the score is then NaN and discarded
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        return dependence_score(column.values, target.values, method, cv_bandwidth)


def dependence_scores(features: pd.DataFrame, target: pd.Series,
                      method: CorrelationMethods = CorrelationMethods.SEV,
                      cv_bandwidth: bool = False, parallel: bool = False) -> pd.Series:
    """Dependence score of ``target`` on each column of ``features``.

    ``parallel`` uses pandarallel. It pays off only with fork-based multiprocessing (Linux/WSL):
    on Windows, starting the workers costs seconds per call, far more than the scores themselves.
    """
    args = (target, method, cv_bandwidth)
    if not parallel:
        return features.apply(_column_dependence_score, args=args)

    global _pandarallel_initialized
    if not _pandarallel_initialized:
        pandarallel.initialize(verbose=1)
        _pandarallel_initialized = True
    return features.parallel_apply(_column_dependence_score, args=args)
