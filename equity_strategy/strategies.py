"""Strategy definitions: the production strategies and the parameter-sweep set."""
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from equity_strategy.definitions import ALL_WEIGHT_METHODS, Algorithms, StockUniverses, WeightMethods


class Strategy(NamedTuple):
    dataset: str
    algorithm: str
    weight_methods: str | Sequence[str]
    n_stocks: int | None
    window_length: int | np.ndarray | None = None


TOP_STRATEGIES = [
    Strategy(dataset=StockUniverses.SP500, algorithm=Algorithms.SEV,
             weight_methods=WeightMethods.RISK_PARITY, n_stocks=10, window_length=90),
    Strategy(dataset=StockUniverses.STOXXE600, algorithm=Algorithms.RMLIN,
             weight_methods=WeightMethods.RISK_PARITY, n_stocks=10, window_length=60),
]

SWEEP_WINDOW_LENGTHS = np.arange(20, 95, 10)
SWEEP_REBALANCE_INTERVALS = np.arange(20, 65, 5)
SWEEP_START_DATE = "2021-02-01"

SWEEP_STRATEGIES = [
    Strategy(dataset=StockUniverses.SP500, algorithm=Algorithms.MAX_SHARPE,
             weight_methods=ALL_WEIGHT_METHODS, n_stocks=10, window_length=SWEEP_WINDOW_LENGTHS),
    Strategy(dataset=StockUniverses.SP500, algorithm=Algorithms.LOLIN,
             weight_methods=ALL_WEIGHT_METHODS, n_stocks=10, window_length=SWEEP_WINDOW_LENGTHS),
]
