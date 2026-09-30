import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def market_frames():
    """Synthetic daily prices, returns, volatility, market cap and benchmark for 20 tickers."""
    rng = np.random.default_rng(0)
    n_days, n_tickers = 450, 20
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    drift = np.linspace(-0.0005, 0.001, n_tickers)
    returns = pd.DataFrame(rng.normal(drift, 0.01, size=(n_days, n_tickers)), index=dates, columns=tickers)
    prices = 100 * (1 + returns).cumprod()
    volatility = pd.DataFrame(rng.uniform(0.005, 0.02, size=(n_days, n_tickers)), index=dates, columns=tickers)
    market_cap = pd.DataFrame(np.tile(np.arange(1, n_tickers + 1) * 1e9, (n_days, 1)), index=dates, columns=tickers)
    benchmark = prices.mean(axis=1)

    return dict(prices=prices, returns=returns, volatility=volatility, market_cap=market_cap,
                benchmark_index=benchmark)
