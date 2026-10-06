# Equity Portfolio Strategy

Research framework for building and backtesting equity portfolio strategies on the **S&P 500** and **STOXX Europe 600**.

Each strategy:
1. **Selects** `n_stocks` stocks from the universe, using a rolling window of `window_length` daily observations and a stock-selection algorithm.
2. **Weights** the selected stocks with one or more weighting methods.
3. **Rebalances** every `rebalance_interval` trading days. The backtest includes fixed and proportional transaction fees and a data lag.

The framework reports performance against the market index: annualized return, volatility, Sharpe, Sortino, STARR, information coefficient, alpha/beta, drawdown, turnover and Fama-French factors. It can also compute the current allocation of the production strategies.

## Stock-selection algorithms (`definitions.Algorithms`)

| Value | Description |
|---|---|
| `mtm`, `lomtm`, `rmmtm`, `lormmtm` | Momentum: mean return (or Sharpe ratio for `rm`) over the window |
| `sev`, `rmsev` | Non-linear dependence (share of explained variance, kernel regression) on a high-Sharpe benchmark index, long-only |
| `lin`, `lolin`, `rmlin`, `lormlin` | Same procedure with linear correlation |
| `max_sr` | Long-only mean-variance optimization over the 100 largest stocks |

Prefixes: `lo` = long-only, `rm` = risk-managed (Sharpe ratios in place of returns).

## Weighting methods (`definitions.WeightMethods`)

| Value | Description |
|---|---|
| `metric` | Weights equal to the selection score |
| `equal` | Equal weights |
| `mktcap` | Market-cap weights |
| `riskpar` | Inverse-volatility (risk parity) weights |
| `lotp` / `ilotp` | Long-only mean-variance optimization on returns / on volatility-scaled returns |

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt       # pinned runtime dependencies
pip install -e .[dev]                 # the package itself, plus pytest and ruff
```

Downloading the S&P 500 constituents uses Selenium with Chrome.

## Usage

```bash
python -m equity_strategy <command> [options]      # or: equity-strategy <command>
```

| Command | What it does |
|---|---|
| `download [--datasets ...]` | Download the latest intraday and daily prices, and stock info, from Yahoo Finance |
| `download-info [--datasets ...]` | Download per-ticker info (`yf.Ticker.info`) only |
| `backtest` | Backtest one strategy, plot cumulative wealth and print the latest allocation. Options: `--dataset --algorithm --weight-method --n-stocks --window --rebalance --start` |
| `sweep [--method multi-start\|grid] [--download]` | Parameter sweep over `strategies.SWEEP_STRATEGIES`. `multi-start` writes alpha/beta over 5 start dates to `*_ic.txt`; `grid` saves all metrics to a results pickle and plots Sharpe and IC heatmaps |
| `plot-sweep [--dataset] [--results-file]` | Re-plot heatmaps of saved grid-sweep results (default: latest file) |
| `allocate` | Compute the current weights of `strategies.TOP_STRATEGIES` and save them to `predictions/*.xlsx` |

Global options: `-v` for debug logging, `--no-show` to skip plot windows.

Example:

```bash
python -m equity_strategy backtest --dataset SP500 --algorithm sev --weight-method riskpar equal --n-stocks 10 --window 90
```

From Python:

```python
from equity_strategy.pipeline import create_backtest

backtest = create_backtest("SP500", n_stocks=10, window_lengths=90, rebalance_intervals=20,
                           algorithms="sev", weight_methods=["riskpar", "equal"], date_start="2022-06-30")
backtest.backtest()
backtest.plot_cum_wealth()
```

## Data and outputs

Cached data lives in `SP500/` and `STOXXE600/` (see `equity_strategy/data/downloader.py` for the file layout). Transaction fees and the data lag per universe are read from `parameters.pkl`. Backtest results are written to `<dataset>/results/`, and allocations to `predictions/`. A single (non-sweep) backtest also saves, per weighting method, the target weights at every rebalancing date to `<dataset>/results/_<date>_<n_stocks>_<algorithm>_<method>_<window>_<rebalance>_allocation.xlsx` (first column the rebalancing date, then one column per ticker; 0 = not held).

## Project structure

| Module | Role |
|---|---|
| `equity_strategy/cli.py` | Command-line entry point |
| `equity_strategy/pipeline.py` | High-level workflows: download, build a backtest from cached data, sweeps, allocations |
| `equity_strategy/definitions.py`, `strategies.py`, `config.py` | Enums, strategy sets, paths and constants |
| `equity_strategy/data/` | Yahoo Finance download and caching (`downloader.py`), price cleaning and realized volatility (`yahoo_tools.py`), index constituents (`universes.py`) |
| `equity_strategy/selection/` | `StockPicker` algorithms and the SEV dependence measure |
| `equity_strategy/portfolio/` | `Portfolio` bookkeeping and weighting, mean-variance optimization, performance metrics |
| `equity_strategy/backtest/` | `PortfolioBacktest` engine and results persistence |
| `equity_strategy/plotting.py` | Wealth curves and sweep heatmaps |
| `tests/` | Unit and end-to-end tests on synthetic data |
| `legacy/` | Archived thesis and experiment scripts (not maintained) |

## Development

```bash
pytest               # run the test suite
ruff check .         # lint
```
