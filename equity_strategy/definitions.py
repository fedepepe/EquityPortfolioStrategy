"""Enumerations shared across the package.

The string enums use :class:`_ValueEnumMeta`, so attribute access returns the
plain string value: ``Algorithms.SEV == 'sev'`` and ``type(Algorithms.SEV) is str``.
This keeps comparisons, dictionary keys and result file names simple, and lets
users pass either the enum attribute or its string value.
"""
from enum import Enum, EnumMeta, auto


class _ValueEnumMeta(EnumMeta):
    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class StockUniverses(Enum, metaclass=_ValueEnumMeta):
    SP500 = "SP500"
    STOXXE600 = "STOXXE600"


class WeightMethods(Enum, metaclass=_ValueEnumMeta):
    METRIC = "metric"  # weights equal to the selection score
    EQ = "equal"
    MKTCAP = "mktcap"
    RISK_PARITY = "riskpar"
    LOTP = "lotp"  # long-only mean-variance optimized portfolio
    ILOTP = "ilotp"  # LOTP estimated on volatility-scaled returns


class Algorithms(Enum, metaclass=_ValueEnumMeta):
    """Stock-selection algorithms.

    Prefixes: ``lo`` = long-only, ``rm`` = risk-managed (uses Sharpe ratios instead of returns).
    """

    SEV = "sev"
    RMSEV = "rmsev"

    LIN = "lin"
    RMLIN = "rmlin"
    LOLIN = "lolin"
    LORMLIN = "lormlin"

    MTM = "mtm"
    RMMTM = "rmmtm"
    LOMTM = "lomtm"
    LORMMTM = "lormmtm"

    MAX_SHARPE = "max_sr"


class CorrelationMethods(Enum):
    LINEAR = auto()
    SPEARMAN = auto()
    SEV = auto()


ALL_WEIGHT_METHODS = [method.value for method in WeightMethods]
