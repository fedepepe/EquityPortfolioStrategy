import numpy as np
import pandas as pd
import pytest

from equity_strategy.definitions import CorrelationMethods
from equity_strategy.selection import dependence


def test_linear_method_matches_pearson_correlation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = 0.5 * x + rng.normal(size=100)
    score = dependence.dependence_score(x, y, CorrelationMethods.LINEAR)
    assert score == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_sev_is_high_for_nonlinear_dependence_and_low_for_independence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    dependent = x ** 2 + 0.1 * rng.normal(size=300)
    independent = rng.normal(size=300)

    assert dependence.dependence_score(x, dependent) > 0.6
    assert dependence.dependence_score(x, independent) < 0.2
    # linear correlation misses the quadratic dependence
    assert abs(dependence.dependence_score(x, dependent, CorrelationMethods.LINEAR)) < 0.3


def _sev_reference(x, y):
    """Point-by-point implementation of the SEV estimator, as originally written."""
    bandwidth = dependence.scott_bandwidth(x)
    grid, step = dependence.kernel_density_support(x, bandwidth)
    conditional_mean, density = np.empty(len(grid)), np.empty(len(grid))
    for i, point in enumerate(grid):
        kernel = dependence.gaussian_kernel((point - x) / bandwidth) / bandwidth
        density[i] = sum(kernel) / len(x)
        conditional_mean[i] = sum(kernel * y) / len(x) / density[i]
    return (sum(conditional_mean ** 2 * density) * step - y.mean() ** 2) / y.var(ddof=1)


def test_sev_matches_reference_implementation():
    rng = np.random.default_rng(6)
    x = rng.normal(size=80)
    y = np.sin(2 * x) + 0.3 * rng.normal(size=80)
    assert dependence.dependence_score(x, y) == pytest.approx(_sev_reference(x, y), rel=1e-10)


def test_sev_is_nan_for_constant_input():
    assert np.isnan(dependence.dependence_score(np.ones(50), np.arange(50.0)))


def test_kernel_density_integrates_to_one():
    samples = np.random.default_rng(3).normal(size=200)
    bandwidth = dependence.scott_bandwidth(samples)
    grid, step = dependence.kernel_density_support(samples, bandwidth)
    assert np.sum(dependence.kernel_density(samples, grid, bandwidth)) * step == pytest.approx(1.0, abs=1e-3)


def test_cross_validated_bandwidth_matches_explicit_leave_one_out():
    samples = np.random.default_rng(4).normal(size=40)
    bandwidth_ref = dependence.scott_bandwidth(samples)

    candidates = np.logspace(np.log10(bandwidth_ref / 10), np.log10(bandwidth_ref * 10), 21)
    risks = []
    for bandwidth in candidates:
        grid, step = dependence.kernel_density_support(samples, bandwidth)
        integral = np.sum(dependence.kernel_density(samples, grid, bandwidth) ** 2) * step
        leave_one_out = sum(dependence.kernel_density(np.delete(samples, i), samples[i:i + 1], bandwidth)[0]
                            for i in range(len(samples)))
        risks.append(integral - 2 / len(samples) * leave_one_out)

    assert dependence.cross_validated_bandwidth(samples, bandwidth_ref) == pytest.approx(candidates[np.argmin(risks)])


def test_dependence_scores_scores_every_column():
    rng = np.random.default_rng(5)
    target = pd.Series(rng.normal(size=60))
    features = pd.DataFrame({"a": target + 0.1 * rng.normal(size=60), "b": rng.normal(size=60)})
    scores = dependence.dependence_scores(features, target, CorrelationMethods.LINEAR, parallel=False)
    assert list(scores.index) == ["a", "b"]
    assert scores["a"] > scores["b"]
