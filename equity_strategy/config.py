"""Project-wide paths and constants.

All paths are resolved relative to the repository root, so the package works
regardless of the current working directory.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PARAMETERS_FILE = PROJECT_ROOT / "parameters.pkl"

YMD_DATE_FORMAT = "%Y-%m-%d"
TRADING_DAYS_IN_YEAR = 252.0
WEEKS_IN_YEAR = 52.0


def dataset_dir(dataset: str) -> Path:
    """Directory holding the cached market data of a stock universe."""
    return PROJECT_ROOT / dataset


def results_dir(dataset: str) -> Path:
    """Directory where backtest results of a stock universe are written."""
    return dataset_dir(dataset) / "results"


@dataclass(frozen=True)
class BacktestParameters:
    """Per-universe trading frictions stored in ``parameters.pkl``."""

    fee_fixed: float
    fee_proportional: float
    data_lag: int


def load_backtest_parameters(dataset: str, path: Path = PARAMETERS_FILE) -> BacktestParameters:
    with open(path, "rb") as f:
        parameters = pickle.load(f)[dataset]
    return BacktestParameters(
        fee_fixed=parameters["trsctn_fee_fix"],
        fee_proportional=parameters["trsctn_fee_prop"],
        data_lag=parameters["data_lag"],
    )
