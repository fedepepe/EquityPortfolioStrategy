"""Performance-metric identifiers and their display settings."""
import dataclasses
from dataclasses import dataclass
from enum import Enum, auto, unique

import pandas as pd


@unique
class PerfMetric(Enum):
    # Member order defines the values and must not change: results pickles store them.
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
class MetricResult:
    label: str
    value: float | pd.Series | dict | None = None
    format: str = ".2f"
    rev_color_scale: bool = False  # True if lower values are better (e.g. volatility)

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.value, float)


_METRIC_TEMPLATES = {
    PerfMetric.RETS: MetricResult(label="returns"),
    PerfMetric.LOG_RETS: MetricResult(label="log_returns"),
    PerfMetric.EXC_RETS: MetricResult(label="exc_returns"),
    PerfMetric.ANN_MEAN_RET: MetricResult(label="return", format="+.2%"),
    PerfMetric.ANN_MEAN_EXC_RET: MetricResult(label="xret", format="+.2%"),
    PerfMetric.VOLATILITY: MetricResult(label="volat", format=".2%", rev_color_scale=True),
    PerfMetric.SHARPE: MetricResult(label="sharpe"),
    PerfMetric.IC: MetricResult(label="i.c."),
    PerfMetric.SORTINO: MetricResult(label="sortino"),
    PerfMetric.STAR: MetricResult(label="starr", format=".4f"),
    PerfMetric.TURNOVER: MetricResult(label="trnovr", format=".2%", rev_color_scale=True),
    PerfMetric.ALPHA: MetricResult(label="alpha", format="+.2%"),
    PerfMetric.BETA: MetricResult(label="beta", rev_color_scale=True),
    PerfMetric.PVAL: MetricResult(label="pval", rev_color_scale=True),
    PerfMetric.DRAWDOWN: MetricResult(label="drawdown"),
    PerfMetric.MAXDD: MetricResult(label="maxdd", format=".2%", rev_color_scale=True),
    PerfMetric.FF_FACTORS: MetricResult(label="ff_factors"),
}


def empty_metrics() -> dict[PerfMetric, MetricResult]:
    """A fresh set of metrics with display settings and no values."""
    return {metric: dataclasses.replace(template) for metric, template in _METRIC_TEMPLATES.items()}


def metric_template(metric: PerfMetric) -> MetricResult:
    return _METRIC_TEMPLATES[metric]
