"""Command-line interface: ``python -m equity_strategy <command> [options]``."""
import argparse
import logging

import matplotlib.pyplot as plt

from equity_strategy import pipeline, strategies
from equity_strategy.backtest.backtest import PortfolioBacktest
from equity_strategy.definitions import ALL_WEIGHT_METHODS, Algorithms, StockUniverses
from equity_strategy.portfolio.metrics import PerfMetric

DATASETS = [universe.value for universe in StockUniverses]
ALGORITHMS = [algorithm.value for algorithm in Algorithms]
SWEEP_HEATMAP_METRICS = (PerfMetric.SHARPE, PerfMetric.IC)


def _download(args: argparse.Namespace) -> None:
    pipeline.download_latest_data(args.datasets)


def _download_info(args: argparse.Namespace) -> None:
    pipeline.download_stock_info(args.datasets)


def _backtest(args: argparse.Namespace) -> None:
    backtest = pipeline.create_backtest(args.dataset,
                                        n_stocks=args.n_stocks,
                                        window_lengths=args.window,
                                        rebalance_intervals=args.rebalance,
                                        algorithms=args.algorithm,
                                        weight_methods=args.weight_method,
                                        date_start=args.start)
    backtest.backtest()
    backtest.plot_cum_wealth()

    print(f"Allocation at {backtest.prices.index[-1]}:")
    for allocation in backtest.allocate().values():
        print(pipeline.allocation_table(allocation))


def _plot_heatmaps(backtest: PortfolioBacktest) -> None:
    for metric in SWEEP_HEATMAP_METRICS:
        backtest.plot_heatmap(metric=metric)


def _sweep(args: argparse.Namespace) -> None:
    if args.download:
        pipeline.download_latest_data(sorted({strategy.dataset for strategy in strategies.SWEEP_STRATEGIES}))
    backtests = pipeline.run_sweep(strategies.SWEEP_STRATEGIES,
                                   rebalance_intervals=strategies.SWEEP_REBALANCE_INTERVALS,
                                   date_start=strategies.SWEEP_START_DATE,
                                   multi_start=(args.method == "multi-start"))
    if args.method == "grid":
        for backtest in backtests:
            _plot_heatmaps(backtest)


def _plot_sweep(args: argparse.Namespace) -> None:
    backtest = pipeline.create_backtest(args.dataset)
    backtest.load_results(args.results_file)
    _plot_heatmaps(backtest)


def _allocate(args: argparse.Namespace) -> None:
    for path in pipeline.save_allocations(strategies.TOP_STRATEGIES):
        print(f"Saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equity_strategy", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--no-show", action="store_true", help="do not open plot windows")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download the latest prices from Yahoo Finance")
    download.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    download.set_defaults(handler=_download)

    download_info = commands.add_parser("download-info", help="download per-ticker information (yf.Ticker.info)")
    download_info.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    download_info.set_defaults(handler=_download_info)

    backtest = commands.add_parser("backtest", help="backtest a single strategy and print its latest allocation")
    backtest.add_argument("--dataset", choices=DATASETS, default=StockUniverses.SP500)
    backtest.add_argument("--algorithm", choices=ALGORITHMS, default=Algorithms.LOLIN)
    backtest.add_argument("--weight-method", nargs="+", choices=ALL_WEIGHT_METHODS, default=["riskpar"])
    backtest.add_argument("--n-stocks", type=int, default=5)
    backtest.add_argument("--window", type=int, default=80, help="observation window length (trading days)")
    backtest.add_argument("--rebalance", type=int, default=10, help="rebalancing interval (trading days)")
    backtest.add_argument("--start", default="2022-06-30", help="first trading date (YYYY-MM-DD)")
    backtest.set_defaults(handler=_backtest)

    sweep = commands.add_parser("sweep", help="parameter sweep over the strategies in strategies.SWEEP_STRATEGIES")
    sweep.add_argument("--method", choices=["multi-start", "grid"], default="multi-start",
                       help="multi-start: alpha/beta over 5 start dates (text files); "
                            "grid: all metrics per configuration (pickle + heatmaps)")
    sweep.add_argument("--download", action="store_true", help="download the latest data first")
    sweep.set_defaults(handler=_sweep)

    plot_sweep = commands.add_parser("plot-sweep", help="plot heatmaps of saved grid-sweep results")
    plot_sweep.add_argument("--dataset", choices=DATASETS, default=StockUniverses.SP500)
    plot_sweep.add_argument("--results-file", default=None, help="results pickle (default: latest)")
    plot_sweep.set_defaults(handler=_plot_sweep)

    allocate = commands.add_parser("allocate", help="save the current allocation of strategies.TOP_STRATEGIES")
    allocate.set_defaults(handler=_allocate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.handler(args)
    if not args.no_show and plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
