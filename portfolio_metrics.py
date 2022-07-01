import pandas as pd
from enum import Enum, unique, auto
from dataclasses import dataclass
from typing import Union


@unique
class EnumPerfMetrics(Enum):
    ALPHA = auto()
    PVAL = auto()
    BETA = auto()
    RETS = auto()
    LOG_RETS = auto()
    EXC_RETS = auto()
    ANN_MEAN_RET = auto()
    ANN_MEAN_EXC_RET = auto()
    VOLATILITY = auto()
    SHARPE = auto()
    IC = auto()
    SORTINO = auto()
    STAR = auto()
    DRAWDOWN = auto()
    MAXDD = auto()
    TURNOVER = auto()
    FF_FACTORS = auto()


@dataclass
class PerfMetric:
    label: str
    value: Union[float, pd.Series] = None
    format: str = '.2f'
    rev_color_scale: bool = False
